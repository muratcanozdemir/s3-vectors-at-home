# Configuration

Everything is set with environment variables. The same variables apply to the server, the CLI and the Docker image.

## Object store

| Variable | Default | |
|---|---|---|
| `MINIO_ENDPOINT` | *(required)* | `host:port` of MinIO or another S3-compatible store, e.g. `localhost:9000` |
| `MINIO_ACCESS_KEY` | *(required)* | access key |
| `MINIO_SECRET_KEY` | *(required)* | secret key |
| `MINIO_SECURE` | `false` | `true` to connect over HTTPS |

- **The bucket** is named `vectors` and is created on first start if it doesn't exist. Text-API indexes live at its root (`indexes/`, `segments/`); S3 Vectors data lives under `s3vectors/`.
- **Startup checks:** the server checks at startup that the store enforces conditional writes, and refuses to start if it doesn't. See [Operations](operations.md#requirements).

## Server

| Variable | Default | |
|---|---|---|
| `WEB_CONCURRENCY` | `1` | API worker processes (read by uvicorn). Workers share the bucket safely. |
| `S3VAH_EMBED_THREADS` | ONNX Runtime's default (all cores) | threads per worker for embedding. Set it to about cores ÷ workers. |
| `S3VAH_CACHE_MB` | no limit | memory for loaded indexes, per worker process, in MiB. Past it, the least recently used indexes are dropped and reloaded when next used. See [Operations](operations.md#memory). |
| `S3VAH_CHUNK_OVERLAP` | `32` | tokens shared by consecutive chunks of a long text in the text API; `off` embeds only the first 256 tokens, as earlier versions did. See [Long documents](text-api.md#long-documents). |
| `S3VAH_S3VECTORS_API` | `true` | `false` removes the [S3 Vectors API](s3-vectors.md) routes |
| `S3VAH_REGION` | `us-east-1` | region shown in S3 Vectors ARNs |
| `S3VAH_ACCOUNT_ID` | `000000000000` | account shown in S3 Vectors ARNs |

## Embedding model

The model is `sentence-transformers/all-MiniLM-L6-v2`, pinned to an exact revision and run on ONNX Runtime. Outside Docker it's downloaded from Hugging Face on first use and cached; the standard Hugging Face variables apply:

| Variable | |
|---|---|
| `HF_HOME` | cache directory (default `~/.cache/huggingface`) |
| `HF_HUB_OFFLINE=1` | never download; use the cache. The Docker image sets this, and has the model built in. |

## Docker Compose

`compose.yaml` and `minio/docker-compose.yaml` read these from your shell or from an `.env` file next to `compose.yaml`. MinIO and the API get the same credentials:

<!-- docs-test: skip -->
```bash
# .env
MINIO_ACCESS_KEY=change-me
MINIO_SECRET_KEY=change-me-too-at-least-8-chars
WEB_CONCURRENCY=4
S3VAH_EMBED_THREADS=2
```

MinIO sets its root user from the credentials only when it first creates its data volume. To change them later, recreate the volume with `docker compose down -v`, which deletes the stored data, or change the user in MinIO itself.
