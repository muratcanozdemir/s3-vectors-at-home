# syntax=docker/dockerfile:1
# The API server: `docker compose up` (see compose.yaml) runs it with MinIO.
#
# Base images are pinned by digest (Dependabot proposes bumps). The
# embedding model is downloaded at build time at the revision pinned in
# vectorstore/embedding.py, so the container never needs Hugging Face at
# runtime (HF_HUB_OFFLINE=1).

FROM ghcr.io/astral-sh/uv:0.13.10@sha256:a3ea1357342c1a4bf9a0c53a3deb1b55379aefa037338d8113d5771efa01774f AS uv

FROM python:3.12-slim@sha256:191083cdf6c7bda8c76158983163ead5d5cf7b77cf3033e805bb0f5296f51b4b AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    HF_HOME=/app/hf
WORKDIR /src
# Dependencies first, from the lockfile, so code changes don't reinstall them.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --no-install-project
COPY pyproject.toml uv.lock README.md ./
COPY api ./api
COPY vectorstore ./vectorstore
COPY s3_vectors_at_home ./s3_vectors_at_home
# --no-editable installs the project into the venv, so /src isn't needed after.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable
RUN /app/.venv/bin/python -c "from vectorstore.embedding import OnnxEmbedder; OnnxEmbedder.from_hub()"

FROM python:3.12-slim@sha256:191083cdf6c7bda8c76158983163ead5d5cf7b77cf3033e805bb0f5296f51b4b
RUN useradd --system --uid 10001 --no-create-home app
COPY --from=build /app /app
ENV PATH=/app/.venv/bin:$PATH \
    HF_HOME=/app/hf \
    HF_HUB_OFFLINE=1 \
    PYTHONUNBUFFERED=1
USER app
WORKDIR /app
EXPOSE 8000
# /status reads the default index, so this also checks MinIO is reachable.
HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/status', timeout=4)"]
# More workers: set WEB_CONCURRENCY (read by uvicorn) and S3VAH_EMBED_THREADS
# to about cores / workers.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
