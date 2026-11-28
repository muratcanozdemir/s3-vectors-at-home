# Development

## Running the tests

<!-- docs-test: skip -->
```bash
docker compose -f minio/docker-compose.yaml up -d
export MINIO_ENDPOINT=localhost:9000 MINIO_ACCESS_KEY=minioadmin MINIO_SECRET_KEY=minioadmin123
uv sync --all-extras
uv run pytest
```

**Fast tests.** These need no MinIO and no model: they use a fake MinIO client and a fake embedder, with no network access.

| File | Covers |
|---|---|
| `tests/test_vectorstore_golden.py` | the store's core behaviour |
| `tests/test_segments.py` | segment storage and compaction |
| `tests/test_filters.py` | the filter language |
| `tests/test_edge_cases.py` | boundaries and odd inputs |
| `tests/test_properties.py` | randomized checks against a dict + numpy brute-force model |
| `tests/test_s3vectors_api.py` | boto3 against an in-process server |

**Tests against real MinIO:**

- `tests/test_concurrency_minio.py` runs several writer processes at once. Set `CAS_WORKERS=16` for a heavier run.
- The rest are end-to-end checks with the real MinIO and embedding model.

## Documentation

The docs are these Markdown files under `docs/`, published with Backstage TechDocs (`mkdocs.yml`). Every `bash` and `python` block in them is run in CI against the Docker stack, so the examples can't go stale. To run them yourself:

<!-- docs-test: skip -->
```bash
docker compose up -d --build
export MINIO_ENDPOINT=localhost:9000 MINIO_ACCESS_KEY=minioadmin MINIO_SECRET_KEY=minioadmin123
uv sync --all-extras
PATH=$PWD/.venv/bin:$PATH uv run python scripts/run_doc_examples.py docs/*.md
```

Write examples that check what they show: `curl -sf … | jq -e '…'` for HTTP, and `assert` in Python. Put `<!-- docs-test: skip -->` on the line before a block that shouldn't run, such as one that starts the stack.

## Embeddings

Embeddings run on ONNX Runtime, with no PyTorch.

- **Compatibility:** `vectorstore/embedding.py` reproduces sentence-transformers' `all-MiniLM-L6-v2` pipeline from the model files, pinned to an exact Hugging Face revision. `tests/test_embedding.py` checks it against reference vectors made by sentence-transformers, so new vectors stay compatible with ones stored earlier.
- **Changing the model or revision:**
  1. Update both `MODEL_REVISION` and the CI cache key.
  2. Regenerate the reference vectors:

<!-- docs-test: skip -->
```bash
uv run --with sentence-transformers==6.0.0 python scripts/make_embedding_fixture.py
```

## Dependencies

Dependencies are locked in `uv.lock`, which is committed, so `uv sync` is reproducible. Dependabot proposes updates for GitHub Actions, the Docker base images and the pinned MinIO image. Python dependencies are bumped by hand with `uv lock --upgrade`.

## Releasing

Releases are automated with [release-please](https://github.com/googleapis/release-please).

1. **Write [Conventional Commits](https://www.conventionalcommits.org/).** `feat:`, `fix:`, `perf:` and `deps:` appear in the changelog. `docs:`, `chore:`, `ci:`, `test:` and `refactor:` don't.
2. **Every push to `main` updates a release PR.** It bumps the version in `pyproject.toml`, `uv.lock` and `.release-please-manifest.json`, and adds the new commits to `CHANGELOG.md`.
3. **Merging the release PR ships it.** CI tags `v<version>` and creates the GitHub release. It then publishes the wheel to PyPI with Trusted Publishing, and pushes the image to `ghcr.io` as `<version>`, `<major.minor>` and `latest`, after the tests and the image smoke test pass.

Don't edit the version by hand: CI fails if the three files disagree.

PRs are merged with merge commits. **Give PRs plain titles, without a `feat:`-style prefix.** The merge commit's message repeats the PR title, so a prefixed title would put each change in the changelog twice. The changelog comes from the commits inside the PR.
