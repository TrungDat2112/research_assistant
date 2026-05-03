FROM python:3.11-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY configs ./configs
COPY ui ./ui


RUN uv sync --frozen --no-dev

RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 7860

CMD ["uv", "run", "streamlit", "run", "ui/app.py", \
    "--server.port=7860", \
    "--server.address=0.0.0.0", \
    "--server.headless=true"]
