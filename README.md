# AI Research Assistant (LangGraph + MCP + RAG + SQLite + Streamlit) — Dockerized

This repository contains:

- **`mcp`**: FastAPI MCP tool server (tool layer + orchestration endpoints)
- **`app`**: LangGraph orchestrator + agents that call MCP (via a `ToolClient` abstraction)
- **`ui`**: Streamlit demo UI (single pipeline execution)
- **Chroma** persistence via a Docker named volume
- **SQLite** persistence via a Docker named volume (`chat_messages`)

## What’s included (demo-safe stack)

- **Web search**: hybrid
  - **Primary**: Tavily (if `TAVILY_API_KEY` is set)
  - **Fallback**: DuckDuckGo (always available)
- **Paper search**: Semantic Scholar public API (no API key required)
- **Embeddings (local)**: `sentence-transformers/all-MiniLM-L6-v2`
  - cached under **`/app/cache`** in Docker
- **LLM**: OpenRouter (if configured) with a guaranteed fallback response
- **Adaptive graph**:
  - heuristic **router** (`fast_path` / `research_path`) to reduce unnecessary tool calls
  - linear pipeline after routing: RAG → synthesis → memory store (no validation retry loop)
- **Observability**:
  - structured `errors` + `observations` attached to runs (available in API responses)
- **Session memory (NEW)**:
  - short-term memory: SQLite chat history (`chat_messages`)
  - long-term memory: Chroma collection `research_assistant_memory`
  - optional `session_id` can be passed to `/run`
  - memory retrieval query: `query + last user messages` (from last 5 turns)
  - memory ranking (importance + usage aware):
    - per-memory `final_score = 0.4*similarity + 0.2*recency + 0.3*importance + 0.1*log(1+usage_count)`
    - memories are sorted by `final_score` and truncated to top-k (3–5)
    - `memory_quality`: average `final_score` of the selected memories
    - `memory_sufficient`: `memory_quality > 0.6`
  - derived routing signals (STRICT):
    - `memory_conflict`: simple contradiction / low-variance heuristic
  - storage rules:
    - store ONLY when `len(answer) > 100`, answer not empty, and query is not a greeting
    - stored memory includes full answer + `summary` (first 2 sentences)
    - stored metadata includes: `importance`, `usage_count`, `created_at`, `last_used`, `type`
- **Graph viewer (NEW)**:
  - (removed) previously included a Streamlit graph-visualization tab

## Project layout

```
.
├── agents/
│   ├── router.py            # heuristic router (fast_path vs research_path)
│   ├── memory_nodes.py      # load_memory / memory_rag / store_memory
├── app/                # Dockerfile only
├── graph/
├── mcp_client/
├── mcp_server/
├── rag/
├── docker-compose.yml
├── main.py
├── session_manager.py
├── memory_store.py
├── requirements.txt
└── .env.example
```

## Environment variables

Copy the example file and fill in keys as needed:

```bash
copy .env.example .env
```

Supported variables:

- **`TAVILY_API_KEY`**: optional; enables Tavily as primary web search
- **`OPENROUTER_API_KEY`**: optional; enables OpenRouter for synthesis
- **`OPENROUTER_MODEL`**: optional; defaults to `meta-llama/llama-3-8b-instruct`
- **`MCP_URL`**: MCP base URL
  - Docker (internal): `http://mcp:8000`
  - Local (host): `http://localhost:8001` (default compose publishing)
- **`RESEARCH_DB_PATH`**: optional; defaults to `/app/data/research.db`
- **`TOOL_CACHE_TTL_SECONDS`**: optional; SQLite tool-cache TTL (default `600`)
- **`MCP_CACHE_TTL_SECONDS`**: optional; in-memory TTL cache in MCP (default `300`)
- **`CHROMA_PERSIST_DIR`**: optional; defaults to `/app/chroma_db`
- **`RAG_CHUNK_MAX_CHARS`**: optional; chunk size for seed docs (default `360`)

UI dependency: Streamlit only.

## Run with Docker Compose

Build and start:

```bash
docker-compose up --build
```

> First build can take longer because `sentence-transformers` pulls in CPU PyTorch and the model is downloaded on first run.
> The images are configured to prefer **CPU-only PyTorch wheels** for demo safety.

### Ports / URLs

Inside Docker networking:

- **App calls MCP at** `http://mcp:8000`

On your host machine:

- MCP is published as **`http://localhost:8001`** (container port 8000 → host port 8001)
  - Health check: `http://localhost:8001/health`
- Streamlit UI is published as **`http://localhost:8501`**

> Note: `docker-compose.yml` maps `8001:8000` to avoid conflicts if host port 8000 is already used.
> If port 8000 is free on your machine and you prefer `localhost:8000`, change the compose mapping to `8000:8000`.

### Chroma persistence

Chroma persists to:

- container path: **`/app/chroma_db`**
- Docker volume: **`chroma_data`**

### Embedding/model cache (Docker)

HuggingFace/SentenceTransformers caches model files under:

- **`/app/cache`**

To remove persisted data:

```bash
docker-compose down -v
```

## Run locally (without Docker)

Start MCP server:

```bash
python -m uvicorn mcp_server.server:app --host 127.0.0.1 --port 8000
```

In a second terminal, run the app:

```bash
set MCP_URL=http://localhost:8000
python main.py --mode planner --start-server 0
```

Run the UI locally:

```bash
set MCP_URL=http://localhost:8000
streamlit run ui.py
```

## Troubleshooting

- **App fails with “Connection refused” to MCP**:
  - Make sure MCP is running/reachable at `MCP_URL`.
  - In Docker Compose, `mcp` has a healthcheck and `app` waits until it is healthy.

- **No Tavily / no OpenRouter keys**:
  - This is supported. The system will:
    - use **DuckDuckGo fallback** for web search
    - return a **safe fallback summary** if OpenRouter is not configured or times out

- **First run feels slow**:
  - The embedding model may download on first use (HuggingFace/SentenceTransformers).
  - Optional: set `HF_TOKEN` to reduce rate-limit issues.

## 🔍 Observability (Optional)

LangSmith tracing can be enabled by setting `LANGCHAIN_API_KEY` (and optionally `LANGCHAIN_PROJECT`).
If `LANGCHAIN_API_KEY` is not set, the system runs normally with no tracing.

## 🤗 HuggingFace Token (Optional)

You can provide a HuggingFace token to improve model download reliability/speed and avoid rate limits:

- Set `HF_TOKEN` (optional)
- Recommended: set `HF_HUB_DISABLE_TELEMETRY=1`

If `HF_TOKEN` is not set, the system uses public access and still works normally.

## Session-aware usage (optional)

The UI supports `session_id` (optional). You can also call the API directly:

```bash
curl -X POST http://localhost:8001/run ^
  -H "Content-Type: application/json" ^
  -d "{\"q\":\"What did we decide last time?\",\"session_id\":\"demo-session-1\"}"
```
