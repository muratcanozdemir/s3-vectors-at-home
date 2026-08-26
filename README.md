- What is this?

“Like AWS S3 Vectors, but it’s just MinIO + FAISS + FastAPI.”

# You guessed it
- Mom, can I have S3 vectors please?
- We have S3 vectors at home.

- This repo is meant to be a PoC of sorts, maybe will take it up a few levels of professionalism at some point.

## How to run
```
docker compose -f minio/docker-compose.yaml up -d
export MINIO_ENDPOINT=localhost:9000
export MINIO_ACCESS_KEY=minioadmin
export MINIO_SECRET_KEY=minioadmin123
uv sync
uvicorn api.main:app
```
- Enjoy.

## Sample CLI commands
### Single upload
`s3-vectors-cli upload --doc-id foo --text "My text"`

### Bulk upload
`s3-vectors-cli bulk-upload --file docs.json`
// where docs.json is:
// `[{"doc_id": "a", "text": "abc"}, {"doc_id": "b", "text": "xyz"}]`

### Search
`s3-vectors-cli search --query "my search text" --top-k 3`

### Get doc
`s3-vectors-cli get --doc-id foo`

### List docs
`s3-vectors-cli list --skip 0 --limit 10`

### Delete doc
`s3-vectors-cli delete --doc-id foo`

### Status
`s3-vectors-cli status`

## Running tests
```
docker compose -f minio/docker-compose.yaml up -d
export MINIO_ENDPOINT=localhost:9000
export MINIO_ACCESS_KEY=minioadmin
export MINIO_SECRET_KEY=minioadmin123
uv sync --all-extras
uv run pytest
```
`tests/test_vectorstore_golden.py` is fast and needs none of the above (fake MinIO client + fake embedder, no network); the rest are end-to-end sanity checks against the real MinIO + embedding model.

# Contributing
- Dependencies are locked (`uv.lock`, committed) and `torch` is pinned to the CPU wheel index (`tool.uv.sources` in `pyproject.toml`) so `uv sync` stays reproducible and doesn't pull CUDA packages nobody here needs.