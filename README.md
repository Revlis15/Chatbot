# AI Research Assistant (LangGraph + MCP + RAG + SQLite + Streamlit) — Dockerized

This repository contains:
- **`mcp`**: FastAPI MCP tool server (tool layer + orchestration endpoints)
- **`app`**: LangGraph orchestrator + agents that call MCP (via a `ToolClient` abstraction)
- **`ui`**: Streamlit demo UI (compare patterns + history)
- **Chroma** persistence via a Docker named volume
- **SQLite** persistence via a Docker named volume (`papers`, `history`, `tool_cache`)

## What’s included (demo-safe stack)

- **Web search**: hybrid
  - **Primary**: Tavily (if `TAVILY_API_KEY` is set)
  - **Fallback**: DuckDuckGo (always available)
- **Paper search**: Semantic Scholar public API (no API key required)
- **Embeddings (local)**: `sentence-transformers/all-MiniLM-L6-v2`
  - cached under **`/app/cache`** in Docker
- **LLM**: OpenRouter (if configured) with a guaranteed fallback response
- **Adaptive graph**:
  - heuristic **router** (`fast_path` vs `research_path`) to reduce unnecessary tool calls
  - bounded **critic** retry (1 extra RAG pass) before returning a final answer
- **Observability**:
  - structured `errors` + `observations` attached to runs (surfaced in UI)

## Project layout

```
.
├── agents/
│   ├── router.py            # heuristic router (fast_path vs research_path)
│   ├── critic.py            # guardrail + bounded retry signal
│   ├── context_compress.py  # rule-based context compression
├── app/                # Dockerfile only
├── graph/
├── mcp_client/
├── mcp_server/
├── rag/
├── docker-compose.yml
├── main.py
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

