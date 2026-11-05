# The API image (Render free web service: 512 MB RAM, 0.1 CPU).
# The index lives in Postgres (Neon); the image holds code only, no data.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.15 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

RUN useradd --create-home app
USER app

# Render passes PORT (default 10000); locally: docker run -e PORT=8000 -p 8000:8000
ENV PORT=10000
EXPOSE 10000
CMD ["sh", "-c", "exec /app/.venv/bin/uvicorn lexrag.api.app:app --host 0.0.0.0 --port ${PORT} --proxy-headers"]
