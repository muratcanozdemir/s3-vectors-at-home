# s3-vectors-at-home

> "Mom, can I have S3 Vectors?" "We have S3 Vectors at home."

A self-hosted vector store: **MinIO + FAISS + FastAPI**. One server, two APIs:

- **A text API.** Send text, search by meaning. Embeddings are built in (`all-MiniLM-L6-v2` on ONNX Runtime).
- **The [Amazon S3 Vectors](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors.html) API.** Point `boto3` or any AWS SDK at it and bring your own vectors.

Search is exact, metadata filters included. Any number of processes can safely write to the same bucket at once. And a write costs the same whether an index holds 50 documents or 50,000.

## Quick start

<!-- docs-test: skip -->
```bash
docker compose up -d --build        # MinIO on :9000, the API on http://localhost:8000
```

```bash
curl -X POST localhost:8000/upload/bulk -H 'content-type: application/json' -d '{"docs": [
  {"doc_id": "cat",    "text": "a cat sat on the mat", "metadata": {"kind": "pet"}},
  {"doc_id": "rocket", "text": "rocket engine thrust", "metadata": {"kind": "space"}}]}'

curl -X POST localhost:8000/search -H 'content-type: application/json' \
  -d '{"query": "kitten on a rug", "top_k": 1}'
# {"matches": ["cat"], "results": [{"doc_id": "cat", "score": 0.6022, "metadata": {"kind": "pet"}}]}
```

With your own vectors, through the AWS SDK:

<!-- docs-test: skip -->
```python
import boto3

s3v = boto3.client("s3vectors", endpoint_url="http://localhost:8000", region_name="us-east-1",
                   aws_access_key_id="any", aws_secret_access_key="any")
s3v.create_vector_bucket(vectorBucketName="media")
s3v.create_index(vectorBucketName="media", indexName="movies", dataType="float32",
                 dimension=4, distanceMetric="cosine")
s3v.put_vectors(vectorBucketName="media", indexName="movies", vectors=[
    {"key": "star-wars", "data": {"float32": [0.1, 0.9, 0.2, 0.0]}, "metadata": {"year": 1977}}])
s3v.query_vectors(vectorBucketName="media", indexName="movies", topK=3,
                  queryVector={"float32": [0.1, 0.8, 0.2, 0.1]}, filter={"year": {"$lt": 1980}},
                  returnDistance=True)
```

Also available: `pip install s3-vectors-at-home`, which installs the server and the `s3-vectors-cli` command, and the image `ghcr.io/muratcanozdemir/s3-vectors-at-home:0.1`.

## Documentation

| | |
|---|---|
| [Getting started](docs/getting-started.md) | Docker, pip or source; your first index |
| [Text API](docs/text-api.md) | every route, with examples and errors |
| [Filters](docs/filters.md) | the metadata filter language |
| [S3 Vectors API](docs/s3-vectors.md) | boto3 examples, limits, differences from AWS |
| [CLI](docs/cli.md) | `s3-vectors-cli` commands and exit codes |
| [Configuration](docs/configuration.md) | environment variables |
| [Operations](docs/operations.md) | deploying, storage, concurrency, performance, backups, upgrading |
| [Development](docs/development.md) | tests, docs examples, releasing |

Every example in the docs (and the `curl` quick start above) runs in CI against the real stack. The boto3 snippet is a short version of the tested one in [S3 Vectors API](docs/s3-vectors.md).

The server doesn't authenticate requests. Run it on a network you trust.

## License

[MIT](LICENSE)
