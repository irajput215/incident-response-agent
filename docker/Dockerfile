# =============================================================================
# Multi-stage build.
#
# The builder resolves dependencies into a virtualenv; the runtime stage copies
# only that venv and the application. No compilers, no uv cache, no build
# toolchain in the shipped image — the final image has no way to install
# anything, which is a feature.
# =============================================================================
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first, on their own layer: they change far less often than the
# source, so an application edit does not re-resolve the whole dependency tree.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev

COPY app ./app
RUN uv sync --frozen --no-dev


FROM python:3.12-slim AS runtime

# A non-root user. An agent that can be asked to investigate a production estate
# should not be running as root when it does.
RUN useradd --create-home --uid 10001 app

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_ENV=production

COPY --from=builder --chown=app:app /app /app

# Writable runtime directories, owned by the app user.
RUN mkdir -p /app/data/logs && chown -R app:app /app/data

USER app

EXPOSE 8000

# /health reports capability, so an unhealthy container says *why* it is
# unhealthy rather than just failing the probe.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://127.0.0.1:8000/health', timeout=4).status_code == 200 else 1)"

CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
