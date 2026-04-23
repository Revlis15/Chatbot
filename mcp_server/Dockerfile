FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt /app/requirements.txt
# Ensure CPU-only PyTorch wheels (demo-safe) for sentence-transformers.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r /app/requirements.txt

COPY . /app

EXPOSE 8000

# Use multiple workers because /run executes a graph that makes internal HTTP
# tool calls (requests) back into this same FastAPI app.
CMD ["uvicorn", "mcp_server.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
