from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Generator, Iterable, List, Literal, Optional, Tuple

import streamlit as st
from streamlit_agraph import Config, Edge, Node, agraph


NodeStatus = Literal["pending", "active", "completed"]
EventStatus = Literal["start", "end"]


@dataclass(frozen=True)
class GraphEvent:
    node: str
    status: EventStatus
    data: Optional[Dict[str, Any]] = None


PIPELINE_NODES: List[Tuple[str, str]] = [
    ("planner", "🧠 planner"),
    ("load_memory", "🧩 memory"),
    ("memory_planner", "🧩 memory_planner"),
    ("router", "🚦 router"),
    ("research_agent", "🔍 research"),
    ("rag_agent", "📚 rag"),
    ("memory_rag", "🧩 memory_rag"),
    ("synth_agent", "✍️ synth"),
    ("critic", "🛡️ critic"),
    ("store_memory", "💾 memory_store"),
]

PIPELINE_EDGES: List[Tuple[str, str]] = [
    ("planner", "load_memory"),
    ("load_memory", "memory_planner"),
    ("memory_planner", "router"),
    # conditional fanout (router)
    ("router", "research_agent"),
    ("router", "rag_agent"),
    # research/rag chain
    ("research_agent", "rag_agent"),
    ("rag_agent", "memory_rag"),
    ("memory_rag", "synth_agent"),
    ("synth_agent", "critic"),
    ("critic", "store_memory"),
]


def build_graph_nodes() -> List[str]:
    return [n for n, _label in PIPELINE_NODES]


def _color_for(status: NodeStatus) -> str:
    if status == "active":
        return "#ff4b4b"  # red
    if status == "completed":
        return "#2ecc71"  # green
    return "#9aa0a6"  # gray


def update_graph_state(
    *,
    current_node: Optional[str],
    completed_nodes: Iterable[str],
    all_nodes: Optional[Iterable[str]] = None,
) -> Dict[str, NodeStatus]:
    nodes = list(all_nodes) if all_nodes is not None else build_graph_nodes()
    completed = set(str(x) for x in completed_nodes)
    cur = str(current_node or "").strip()
    out: Dict[str, NodeStatus] = {}
    for n in nodes:
        if n in completed:
            out[n] = "completed"
        elif cur and n == cur:
            out[n] = "active"
        else:
            out[n] = "pending"
    return out


def render_graph(
    *,
    graph_state: Dict[str, NodeStatus],
    height: int = 440,
    key: str = "agraph",
) -> None:
    nodes: List[Node] = []
    for node_id, label in PIPELINE_NODES:
        stt = graph_state.get(node_id, "pending")
        nodes.append(
            Node(
                id=node_id,
                label=label,
                size=24,
                color=_color_for(stt),
            )
        )

    edges: List[Edge] = [Edge(source=s, target=t) for s, t in PIPELINE_EDGES]

    cfg = Config(
        width="100%",
        height=height,
        directed=True,
        physics=False,
        hierarchical=True,
        nodeHighlightBehavior=True,
        highlightColor="#F7A7A6",
        collapsible=False,
    )
    agraph(nodes=nodes, edges=edges, config=cfg, key=key)


def render_logs(log_lines: List[str], *, height: int = 260) -> None:
    # Scrollable log area without flicker.
    text = "\n".join(log_lines[-400:])
    st.text_area("Execution logs", value=text, height=height, key="exec_logs", disabled=True)


def render_decision_panel(state: Dict[str, Any]) -> None:
    mem_quality = float(state.get("memory_quality") or 0.0)
    mem_conflict = bool(state.get("memory_conflict") or False)
    route = str(state.get("route") or "").strip()

    st.metric("memory_quality", f"{mem_quality:.3f}")
    st.write(f"memory_conflict: `{mem_conflict}`")
    st.write(f"route: `{route or '(unknown)'}`")


def _extract_node_from_stream_chunk(chunk: Any) -> Optional[str]:
    """
    LangGraph `.stream()` yields dict-ish updates. We infer the "current node"
    by taking the first key that looks like a node name (not a reserved key).
    """
    if not isinstance(chunk, dict) or not chunk:
        return None
    reserved = {"__end__", "__interrupt__", "__metadata__", "messages"}
    for k in chunk.keys():
        ks = str(k)
        if ks and ks not in reserved:
            return ks
    return None


def run_stream_execution(app: Any, inputs: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
    """
    iterate over app.stream(inputs)
    yield structured events:
      { "node": str, "status": "start|end", "data": optional }
    """
    last_node: Optional[str] = None
    for chunk in app.stream(inputs):
        node = _extract_node_from_stream_chunk(chunk)
        if node and node != last_node:
            yield {"node": node, "status": "start", "data": None}
        if node:
            yield {"node": node, "status": "end", "data": chunk.get(node) if isinstance(chunk, dict) else None}
        else:
            yield {"node": "unknown", "status": "end", "data": {"chunk": chunk}}
        last_node = node or last_node


def replay_execution(trace: List[Dict[str, Any]], *, delay_s: float = 0.35) -> Generator[Dict[str, Any], None, None]:
    """
    trace = list of {node, message}
    simulate delay (time.sleep)
    update graph + logs
    """
    for item in trace:
        node = str(item.get("node") or "").strip() or "unknown"
        msg = str(item.get("message") or "").strip()
        yield {"node": node, "status": "start", "data": None}
        time.sleep(max(0.0, float(delay_s)))
        yield {"node": node, "status": "end", "data": {"message": msg, "state": item.get("state")}}


def mock_trace_example() -> List[Dict[str, Any]]:
    return [
        {"node": "planner", "message": "Plan generated: ['search_web','search_paper','retrieve','synthesize']"},
        {"node": "load_memory", "message": "Loaded history + retrieved memories", "state": {"memory_quality": 0.72, "memory_conflict": False}},
        {"node": "router", "message": "route=hybrid_path", "state": {"route": "hybrid_path"}},
        {"node": "research_agent", "message": "Web/paper search complete"},
        {"node": "rag_agent", "message": "Retrieved 3 docs"},
        {"node": "synth_agent", "message": "Drafted answer"},
        {"node": "critic", "message": "Accepted"},
        {"node": "store_memory", "message": "Stored memory + updated usage"},
    ]


def render_execution_viewer(
    *,
    app: Any | None,
    inputs: Dict[str, Any],
    mode: Literal["Live", "Replay"],
    replay_trace: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """
    Streamlit execution viewer.
    Keeps history in st.session_state to avoid flicker and preserve logs.
    """
    if "gv_completed" not in st.session_state:
        st.session_state.gv_completed = []
    if "gv_current" not in st.session_state:
        st.session_state.gv_current = None
    if "gv_logs" not in st.session_state:
        st.session_state.gv_logs = []
    if "gv_state" not in st.session_state:
        st.session_state.gv_state = {}

    left, right = st.columns([0.52, 0.48], gap="large")

    with left:
        gph = st.empty()

    with right:
        top = st.container()
        bottom = st.container()

    run_btn = st.button("Run execution", type="primary", key="gv_run")
    reset_btn = st.button("Reset viewer", key="gv_reset")

    if reset_btn:
        st.session_state.gv_completed = []
        st.session_state.gv_current = None
        st.session_state.gv_logs = []
        st.session_state.gv_state = {}
        st.rerun()

    # Initial render
    with left:
        graph_state = update_graph_state(
            current_node=st.session_state.gv_current,
            completed_nodes=st.session_state.gv_completed,
        )
        with gph:
            render_graph(graph_state=graph_state, key="agraph_main")

    with top:
        render_logs(st.session_state.gv_logs)

    with bottom:
        st.subheader("Decision panel")
        render_decision_panel(st.session_state.gv_state)

    if not run_btn:
        return

    # Stream execution events
    if mode == "Live":
        if app is None:
            st.error("Live mode requires a local LangGraph app instance.")
            return
        event_iter = run_stream_execution(app, inputs)
    else:
        event_iter = replay_execution(replay_trace or mock_trace_example())

    for ev in event_iter:
        node = str(ev.get("node") or "").strip() or "unknown"
        status = str(ev.get("status") or "end").strip().lower()
        data = ev.get("data") if isinstance(ev, dict) else None

        # Update state snapshot if present
        if isinstance(data, dict):
            # Allow either {state:{...}} or direct state dict
            st_candidate = data.get("state") if isinstance(data.get("state"), dict) else None
            if st_candidate:
                st.session_state.gv_state.update(st_candidate)
            else:
                # best effort: if chunk looks like state delta
                for k in ("memory_quality", "memory_conflict", "route"):
                    if k in data:
                        st.session_state.gv_state[k] = data.get(k)

        if status == "start":
            st.session_state.gv_current = node
            st.session_state.gv_logs.append(f"[{node}] start")
        else:
            if node not in st.session_state.gv_completed:
                st.session_state.gv_completed.append(node)
            st.session_state.gv_current = None
            if isinstance(data, dict) and "message" in data:
                st.session_state.gv_logs.append(f"[{node}] {data.get('message')}")
            else:
                st.session_state.gv_logs.append(f"[{node}] end")

        # Re-render panels smoothly
        with left:
            graph_state = update_graph_state(
                current_node=st.session_state.gv_current,
                completed_nodes=st.session_state.gv_completed,
            )
            with gph:
                render_graph(graph_state=graph_state, key="agraph_main")
        with top:
            render_logs(st.session_state.gv_logs)
        with bottom:
            st.subheader("Decision panel")
            render_decision_panel(st.session_state.gv_state)

