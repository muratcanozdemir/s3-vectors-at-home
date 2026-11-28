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

## Indexes, metadata and filters
A store holds any number of named **indexes**; `default` always exists. Every document command works on `default` unless told otherwise.

Each document can carry **metadata**: a flat JSON object whose values are strings, numbers, booleans or lists of those (up to 32 keys and 4 KB). **Search filters** on it with the MongoDB-style operators S3 Vectors uses:

| Filter | Matches |
|---|---|
| `{"lang": "en"}` | `lang` equals `"en"` (a list field matches if any element does) |
| `{"year": {"$gte": 2020, "$lt": 2025}}` | several operators on one field are ANDed |
| `{"tags": {"$in": ["ml", "ai"]}}` / `{"$nin": [...]}` | any / none of the values |
| `{"draft": {"$ne": true}}` | not equal (also true when the field is missing) |
| `{"source": {"$exists": false}}` | field absent |
| `{"$or": [{...}, {...}]}`, `{"$and": [...]}` | combine filters; listing several fields also ANDs them |

`$gt`/`$gte`/`$lt`/`$lte` compare numbers only, and booleans never equal numbers. Filtered search is exact: you get the true top-k among the matching documents.

## HTTP API
The original routes work on the `default` index. The same routes under `/indexes/{name}` work on a named one.

| Route | |
|---|---|
| `GET /indexes`, `POST /indexes {"name"}`, `DELETE /indexes/{name}` | list / create (409 if it exists) / delete an index |
| `POST /upload` `{"doc_id", "text", "metadata"?}` | add or replace a doc |
| `POST /upload/bulk` `{"docs": [...]}` | many at once; one bad doc rejects the whole batch |
| `POST /search` `{"query", "top_k"?, "filter"?}` | `matches` (doc_ids, closest first) and `results` (`[{"doc_id", "score", "metadata"}]`) |
| `GET /document/{doc_id}`, `DELETE /document/{doc_id}` | `{"doc_id", "text", "metadata"}` / delete |
| `GET /documents?skip&limit`, `GET /status` | list docs (`text_preview`, `metadata`) / counts |

For example, `POST /indexes/papers/search` with `{"query": "graph neural networks", "filter": {"year": {"$gte": 2022}}}`. `score` is cosine similarity (−1…1, higher is closer). Invalid input gets 400, unknown index or doc 404, and a write that lost to other writers too many times 503 with `Retry-After`.

## Sample CLI commands
Every document command takes `--index NAME` (default `default`), after the command name.

```
s3-vectors-cli indexes create papers
s3-vectors-cli indexes list
s3-vectors-cli upload --index papers --doc-id foo --text "My text" --metadata '{"year": 2024, "tags": ["ml"]}'
s3-vectors-cli bulk-upload --index papers --file docs.json   # [{"doc_id", "text", "metadata"?}, ...]
s3-vectors-cli search --index papers --query "my search text" --top-k 3 --filter '{"year": {"$gte": 2020}}'
s3-vectors-cli search --query "my search text" --scores       # prints "score  doc_id" lines
s3-vectors-cli get --doc-id foo
s3-vectors-cli list --skip 0 --limit 10
s3-vectors-cli delete --doc-id foo
s3-vectors-cli status --index papers
s3-vectors-cli indexes delete papers
```

## Behaviour worth knowing
- **Storage.** Each index is one object, `indexes/<name>.v3.bin`, holding every document's text, metadata and vector together with the FAISS index. So a write replaces a document's text, metadata and vector together in one atomic PUT, and reads (get, list, count, filtering) are served from memory.
- **Caching.** A loaded index is cached in memory. Each operation does a cheap `HEAD` on the index object and only reloads if its ETag changed, so writes from other processes are still picked up.
- **Bulk upload is all-or-nothing on validation.** Every doc_id and metadata value is checked before anything is written, so one bad doc rejects the whole batch. The batch is embedded in one call and the index is written once. If a doc_id repeats within a batch, the last one wins.
- **Many processes, one bucket.** Any number of API workers and CLI runs can read and write the same bucket at once. Index updates use compare-and-swap: each write is a conditional PUT (`If-Match` on the ETag it read, or `If-None-Match: *` to create), and a writer that loses the race reloads, re-applies its change on top and retries, so no update is lost. Because a doc is a single entry in that one object, concurrent writes to the *same* doc_id are safe too: the last successful write wins, as a whole. Requirements and limits:
  - The object store must enforce conditional PUTs (MinIO does, in releases since late 2024). The store checks this at startup and refuses to run if it doesn't, rather than quietly risk lost updates.
  - Under extreme write contention a write can give up after 25 attempts: HTTP 503 with `Retry-After`, or CLI exit 1, with nothing half-applied. In testing, 32 processes writing flat out needed at most 16 attempts.
  - A deleted index stays deleted: a write racing the delete fails with "index not found" instead of re-creating it.
  - An index is written as a single PUT, so it is capped at 5 GiB of vectors plus text plus metadata.
- **Cost of a write.** Every write re-uploads its whole index (vectors, texts and metadata). That's fine for thousands to tens of thousands of short documents (1,000 docs of ~200 characters are 1.7 MB). For much larger corpora, split them across indexes.
- **Performance** (1,000 docs, laptop, local MinIO): bulk upload ~3.8 s, almost all of it the embedding model; single add ~20 ms; search ~6 ms; filtered search ~12 ms; get/list/count < 1 ms. The API loads the model and the default index at startup, so the first request isn't slow, and it fails at boot if MinIO is unreachable.
- **Limits.** `top_k` is 1–1000, `skip` ≥ 0, `limit` 0–1000 (out-of-range gets 422). Index names are 1–63 characters of `a-z`, `0-9` and `-`. A doc_id is up to 512 characters with no `/`.
- **Upgrading from older versions.** The first time the default index is used, data in the old layouts (`index.v2.bin` or `faiss.index` + `index.ids.json`, with per-document `<id>.npy` / `<id>.meta.json` objects) is migrated into it, with empty metadata. The old objects are left untouched. Nothing reads them again, so delete them once you're sure you won't roll back. Don't run old and new versions against the same bucket at once: the old code wouldn't see writes made by the new.

## Running tests
```
docker compose -f minio/docker-compose.yaml up -d
export MINIO_ENDPOINT=localhost:9000
export MINIO_ACCESS_KEY=minioadmin
export MINIO_SECRET_KEY=minioadmin123
uv sync --all-extras
uv run pytest
```
`tests/test_vectorstore_golden.py` and `tests/test_filters.py` are fast and need none of the above (fake MinIO client + fake embedder, no network). `tests/test_concurrency_minio.py` runs several writer processes against the real MinIO (`CAS_WORKERS=16` for a heavier run). The rest are end-to-end checks against the real MinIO and embedding model.

# Contributing
- Dependencies are locked (`uv.lock`, committed) and `torch` is pinned to the CPU wheel index (`tool.uv.sources` in `pyproject.toml`) so `uv sync` stays reproducible and doesn't pull CUDA packages nobody here needs.