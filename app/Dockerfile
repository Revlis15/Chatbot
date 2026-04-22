FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt /app/requirements.txt
# Ensure CPU-only PyTorch wheels (demo-safe) for sentence-transformers.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r /app/requirements.txt

COPY . /app

CMD ["python", "main.py", "--mode", "planner", "--start-server", "0"]
