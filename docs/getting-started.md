# Getting started

## With Docker (recommended)

From a clone of the repository:

<!-- docs-test: skip -->
```bash
docker compose up -d --build
```

This starts:

- **MinIO** on <http://localhost:9000> (console on <http://localhost:9001>, user `minioadmin`, password `minioadmin123`);
- **the API** on <http://localhost:8000>.

The API image has the embedding model built in, so it never downloads anything at runtime. It runs as a non-root user and reports healthy once `GET /status` answers. Check it:

```bash
curl -sf localhost:8000/status | jq -e '.status == "ok"'
```

```json
{"status": "ok", "index": "default", "document_count": 0, "segments": 0, "storage_bytes": 0, "embedding_model": "all-MiniLM-L6-v2"}
```

### Using the published image

Releases publish the image to GitHub Container Registry, tagged with the version (`0.1.0`), the minor version (`0.1`) and `latest`. To run it against your own MinIO:

<!-- docs-test: skip -->
```bash
docker run -d -p 8000:8000 \
  -e MINIO_ENDPOINT=minio.example.internal:9000 \
  -e MINIO_ACCESS_KEY=... -e MINIO_SECRET_KEY=... \
  ghcr.io/muratcanozdemir/s3-vectors-at-home:0.1
```

See [Configuration](configuration.md) for every setting.

## With pip

<!-- docs-test: skip -->
```bash
pip install s3-vectors-at-home
export MINIO_ENDPOINT=localhost:9000 MINIO_ACCESS_KEY=minioadmin MINIO_SECRET_KEY=minioadmin123
uvicorn api.main:app --port 8000      # the server
s3-vectors-cli status                 # or the CLI
```

You need a MinIO server (or another S3-compatible store that enforces conditional writes; see [Operations](operations.md#requirements)). `docker compose -f minio/docker-compose.yaml up -d` from the repository starts one. The first start downloads the embedding model (~90 MB) from Hugging Face and caches it.

## From source

<!-- docs-test: skip -->
```bash
docker compose -f minio/docker-compose.yaml up -d
export MINIO_ENDPOINT=localhost:9000 MINIO_ACCESS_KEY=minioadmin MINIO_SECRET_KEY=minioadmin123
uv sync
uv run uvicorn api.main:app --reload
```

## Your first index

Create an index, add a few documents, and search them by meaning:

```bash
curl -sf -X POST localhost:8000/indexes -H 'content-type: application/json' -d '{"name": "first-steps"}'

curl -sf -X POST localhost:8000/indexes/first-steps/upload/bulk -H 'content-type: application/json' -d '{
  "docs": [
    {"doc_id": "paris",  "text": "Paris is the capital of France.",         "metadata": {"continent": "europe"}},
    {"doc_id": "tokyo",  "text": "Tokyo is the largest city in Japan.",     "metadata": {"continent": "asia"}},
    {"doc_id": "baking", "text": "Preheat the oven before baking bread.",   "metadata": {"topic": "cooking"}}
  ]}'

curl -sf -X POST localhost:8000/indexes/first-steps/search -H 'content-type: application/json' \
  -d '{"query": "French city", "top_k": 1}' | jq -e '.matches == ["paris"]'

curl -sf -X DELETE localhost:8000/indexes/first-steps
```

Next: the [Text API](text-api.md) in full, or the [S3 Vectors API](s3-vectors.md) if you have your own vectors.
