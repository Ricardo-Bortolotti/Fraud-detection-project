# =========================
# BASE
# =========================
FROM python:3.11-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

# >>> CACHE INTELIGENTE AQUI <<<
ENV UV_CACHE_DIR=/root/.cache/uv

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# =========================
# API
# =========================
FROM base AS api

COPY . .

CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# =========================
# APP
# =========================
FROM base AS app

COPY . .

CMD ["uv", "run", "streamlit", "run", "app/streamlit_app.py", "--server.address", "0.0.0.0"]