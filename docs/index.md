# s3-vectors-at-home

> "Mom, can I have S3 Vectors?" "We have S3 Vectors at home."

A self-hosted vector store: **MinIO + FAISS + FastAPI**. One server, two APIs:

- **A text API.** Send text, search by meaning. Embeddings are built in (`all-MiniLM-L6-v2` on ONNX Runtime), so you never handle vectors.
- **The Amazon S3 Vectors API.** Point `boto3` or any AWS SDK at it and bring your own vectors.

Both store everything in a MinIO bucket, are safe with many processes writing at once, and give exact (not approximate) search results, metadata filters included.

## Five-minute tour

```bash
# Start MinIO + the API (http://localhost:8000):
#   docker compose up -d --build
curl -sf -X POST localhost:8000/indexes -H 'content-type: application/json' -d '{"name": "tour"}'
curl -sf -X POST localhost:8000/indexes/tour/upload/bulk -H 'content-type: application/json' -d '{"docs": [
  {"doc_id": "cat",    "text": "a cat sat on the mat", "metadata": {"kind": "pet"}},
  {"doc_id": "rocket", "text": "rocket engine thrust", "metadata": {"kind": "space"}}]}'
curl -sf -X POST localhost:8000/indexes/tour/search -H 'content-type: application/json' \
  -d '{"query": "kitten on a rug", "top_k": 1}' | jq -e '.matches == ["cat"]'
curl -sf -X DELETE localhost:8000/indexes/tour
```

## Where to go next

| I want to… | Read |
|---|---|
| run it (Docker, pip, from source) | [Getting started](getting-started.md) |
| store and search text over HTTP | [Text API](text-api.md) |
| filter on metadata | [Filters](filters.md) |
| use boto3 / the AWS SDKs | [S3 Vectors API](s3-vectors.md) |
| script it from a shell | [CLI](cli.md) |
| set environment variables | [Configuration](configuration.md) |
| deploy, scale, back up, upgrade | [Operations](operations.md) |
| run the tests, contribute, release | [Development](development.md) |

Interactive API docs (OpenAPI) are served by the running server at <http://localhost:8000/docs>.

!!! warning "No authentication"
    Neither API checks credentials or request signatures. Run the server on a network you trust, or behind a proxy that authenticates.
