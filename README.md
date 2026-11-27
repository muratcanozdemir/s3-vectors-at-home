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

Add `--scores` to print `score  doc_id` per match. Over HTTP, `POST /search` returns `matches` (doc_ids, closest first) and `results` (`[{"doc_id", "score"}]`), where `score` is cosine similarity (−1…1, higher is closer).

### Get doc
`s3-vectors-cli get --doc-id foo`

### List docs
`s3-vectors-cli list --skip 0 --limit 10`

### Delete doc
`s3-vectors-cli delete --doc-id foo`

### Status
`s3-vectors-cli status`

## Behaviour worth knowing
- **Bulk upload is all-or-nothing on validation.** Every `doc_id` is checked before anything is written, so one bad id rejects the whole batch (HTTP 400 / CLI exit 1) instead of leaving it half-applied. The batch is embedded in one call and the FAISS index is written once, not once per document. If a `doc_id` repeats within a batch, the last one wins.
- **The index is cached in memory.** Searches don't download the index from MinIO each time; they do a cheap `HEAD` on the index object and reload only if its ETag changed, so writes from other processes are still picked up.
- **Performance.** Independent object-store calls (a bulk upload's per-document objects, a page of `list` reads, an index rebuild) run on a small thread pool, and bulk uploads embed in chunks of 256 while the previous chunk uploads. `count` and `list` read the cached index rather than listing the bucket. The API loads the model and index at startup, so the first request isn't slow, and it fails at boot if MinIO is unreachable. Rough numbers for 1,000 docs on a laptop against local MinIO: bulk upload 6.2 s → 3.6 s (about 2.1 s of that is the model), list 100: 172 → 58 ms, count: 87 → 1 ms, rebuild: 850 → 520 ms. Search is ~1–2 ms on top of embedding the query.
- **Limits.** `top_k` is 1–1000, `skip` ≥ 0, `limit` 0–1000; out-of-range values get HTTP 422.
- **The index is one object** (`index.v2.bin`: the FAISS index plus its id → `doc_id` map), so each index write is a single atomic PUT. Buckets on the old two-object layout (`faiss.index` + `index.ids.json`) are migrated automatically on first use: the index is rebuilt from the per-document vectors and the old objects are deleted.
- **Single writer.** The index lock is per-process. Several API/CLI processes *reading* at once is fine, but concurrent *writes* from different processes can still lose an index update (the per-document `.npy`/`.meta.json` objects are kept, and the index rebuilds from them if it goes missing).

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