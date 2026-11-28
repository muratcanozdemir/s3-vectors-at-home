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
| `GET /documents?skip&limit`, `GET /status` | list docs (`text_preview`, `metadata`) / document count, segments, storage bytes |
| `POST /compact` | merge all segments and reclaim space (normally automatic) |

For example, `POST /indexes/papers/search` with `{"query": "graph neural networks", "filter": {"year": {"$gte": 2022}}}`. `score` is cosine similarity (−1…1, higher is closer). Invalid input gets 400, unknown index or doc 404, and a write that lost to other writers too many times 503 with `Retry-After`.

## S3 Vectors API
The server also speaks the [Amazon S3 Vectors](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors.html) API, so the AWS SDKs and tools built on them work against it. You bring your own vectors (any dimension up to 4,096, cosine or euclidean); this part doesn't use the built-in embedding model.

```python
import boto3

s3v = boto3.client("s3vectors", endpoint_url="http://localhost:8000", region_name="us-east-1",
                   aws_access_key_id="any", aws_secret_access_key="any")
s3v.create_vector_bucket(vectorBucketName="media")
s3v.create_index(vectorBucketName="media", indexName="movies", dataType="float32", dimension=4,
                 distanceMetric="cosine", metadataConfiguration={"nonFilterableMetadataKeys": ["plot"]})
s3v.put_vectors(vectorBucketName="media", indexName="movies", vectors=[
    {"key": "star-wars", "data": {"float32": [0.1, 0.9, 0.2, 0.0]}, "metadata": {"genre": "scifi", "year": 1977}},
])
s3v.query_vectors(vectorBucketName="media", indexName="movies", queryVector={"float32": [0.1, 0.8, 0.2, 0.1]},
                  topK=3, filter={"genre": "scifi", "year": {"$lt": 1980}}, returnMetadata=True, returnDistance=True)
```

- **Operations:**
  - `CreateVectorBucket`, `GetVectorBucket`, `ListVectorBuckets`, `DeleteVectorBucket`;
  - `CreateIndex`, `GetIndex`, `ListIndexes`, `DeleteIndex`;
  - `PutVectors`, `GetVectors`, `ListVectors` (including parallel `segmentCount`/`segmentIndex` listing), `DeleteVectors`, `QueryVectors`.

  These take the request and response shapes, limits and error codes of the service model. Resources can be addressed by name or by ARN (`S3VAH_REGION` and `S3VAH_ACCOUNT_ID` set the ARNs' region and account, default `us-east-1` / `000000000000`). The SDK paginators work.
- **Search is exact.** The top-k is the true top-k among the vectors that match the filter, always. Filters support `$eq $ne $gt $gte $lt $lte $in $nin $exists $startsWith $and $or`. Metadata limits are S3 Vectors': 50 keys; 100 KB in all; 40 KB filterable, with each filterable string up to 2 KB. `nonFilterableMetadataKeys` are stored and returned but can't be filtered on.
- **Where AWS's docs leave it open, this server chooses:**
  - cosine `distance` is 1 − cosine similarity, and euclidean is the plain (not squared) distance, closest first;
  - `GetVectors` leaves out keys that don't exist, and `DeleteVectors` ignores them;
  - a filter on a non-filterable key is a `ValidationException`;
  - deleting a vector bucket that still has indexes is a `ConflictException`;
  - NaN, infinity, a wrong dimension, or an all-zero vector on a cosine index are rejected.
- **Not supported** (each answered with a clear `ValidationException`):
  - full-text and hybrid search (`queryTexts`, `textSearchConfigurations`);
  - KMS encryption;
  - bucket policies, tags, and index-mode updates.
- **No request signatures are checked:** any credentials work, as with the rest of this API. Run it on a network you trust. `S3VAH_S3VECTORS_API=false` turns the S3 Vectors routes off.
- **Storage.** Vector buckets live under `s3vectors/` in the same MinIO bucket, on the same segmented, compare-and-swap storage as the text API, so the same concurrency guarantees apply. Vectors are stored and returned exactly as the float32 values you sent.

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
s3-vectors-cli indexes compact papers                          # optional: merges happen automatically
s3-vectors-cli indexes delete papers
```

## Behaviour worth knowing
- **Storage.** An index is a small manifest, `indexes/<name>.manifest.json`, plus immutable *segments* under `segments/<name>/`. A write uploads one segment holding just its documents (text, metadata and vector together, or a delete marker), then appends it to the manifest. So a write costs the same at 50 documents or 50,000, and reads (get, list, count, filtering) are served from memory.
- **Caching.** A loaded index is cached in memory. Each operation does a cheap `HEAD` on the manifest. When another process has written, only the new manifest and the segments it hasn't seen are downloaded, not the whole index.
- **Compaction.** Segments are merged in the background after writes, size-tiered, so an index keeps a few dozen segments at most and each document is rewritten about log₂(n) times over its life. `POST /compact` (or `s3-vectors-cli indexes compact`) merges everything into one and reclaims the space of overwritten and deleted documents. You never need to run it, but it's there for a tidy index before a backup. It also deletes *orphaned* segments, ones a crash or failed write left behind that nothing references, once they are an hour old.
- **Bulk upload is all-or-nothing on validation.** Every doc_id and metadata value is checked before anything is written, so one bad doc rejects the whole batch. The batch is embedded in one call and written as one segment. If a doc_id repeats within a batch, the last one wins.
- **Many processes, one bucket.** Any number of API workers and CLI runs can read and write the same bucket at once. Each manifest update is compare-and-swap: a conditional PUT (`If-Match` on the ETag it read, or `If-None-Match: *` to create). A writer that loses the race reloads and appends the same segment on top, so no update is lost, and merges use the same mechanism. Every record gets a sequence number from the manifest, and a document's newest record wins, so concurrent writes to the *same* doc_id are safe too: the last one appended wins, as a whole. Requirements and limits:
  - The object store must enforce conditional PUTs (MinIO does, in releases since late 2024). The store checks this at startup and refuses to run if it doesn't, rather than quietly risk lost updates.
  - All writes to one index pass through its manifest, so they top out at roughly 100+ per second across all processes. That's about 120/s with 32 processes writing flat out against local MinIO; every write succeeded, the slowest after 36 attempts. Batch writes (`/upload/bulk`) for more throughput. A write that keeps losing for 30 seconds gives up with HTTP 503 and `Retry-After`, or CLI exit 1, and nothing is applied.
  - A deleted index stays deleted: a write racing the delete fails with "index not found" instead of re-creating it. Re-creating a name gives a fresh, empty index.
  - Orphan cleanup assumes the clocks of the machines writing to a bucket agree to within ~45 minutes.
- **Performance** (laptop, local MinIO). With 1,000 docs:
  - bulk upload takes ~1.3 s, almost all of it the embedding model;
  - search ~3 ms (about 2 ms embedding the query, 1 ms checking the manifest's ETag), filtered search ~4 ms;
  - get, list and count take under 1 ms.

  A single write stays ~6–12 ms and uploads ~6 KB whatever the index size. With the previous one-object layout it took 13 ms / 1.8 MB at 1,000 docs, 71 ms / 18 MB at 10,000 and 565 ms / 88 MB at 50,000. Loading a 50,000-doc index from scratch takes ~0.3 s. The API loads the model and the default index at startup, so the first request isn't slow, and it fails at boot if MinIO is unreachable.
- **Limits.** `top_k` is 1–1000, `skip` ≥ 0, `limit` 0–1000 (out-of-range gets 422). Index names are 1–63 characters of `a-z`, `0-9` and `-`. A doc_id is up to 512 characters with no `/`.
- **Upgrading from older versions.** Each index is migrated the first time it's used:
  - from the previous one-object layout (`indexes/<name>.v3.bin`);
  - for the default index, from the original layouts (`index.v2.bin` or `faiss.index` + `index.ids.json`, with per-document `<id>.npy` / `<id>.meta.json` objects), with empty metadata.

  The old objects are left untouched (deleting an index also deletes its `.v3.bin`). Nothing reads them again, so delete them once you're sure you won't roll back. Don't run old and new versions against the same bucket at once: the old code wouldn't see writes made by the new.

## Running tests
```
docker compose -f minio/docker-compose.yaml up -d
export MINIO_ENDPOINT=localhost:9000
export MINIO_ACCESS_KEY=minioadmin
export MINIO_SECRET_KEY=minioadmin123
uv sync --all-extras
uv run pytest
```
`tests/test_vectorstore_golden.py`, `tests/test_segments.py`, `tests/test_filters.py`, `tests/test_edge_cases.py`, `tests/test_properties.py` (randomised checks against a dict + numpy model) and `tests/test_s3vectors_api.py` (boto3 against an in-process server) are fast and need none of the above (fake MinIO client + fake embedder, no network). `tests/test_concurrency_minio.py` runs several writer processes against the real MinIO (`CAS_WORKERS=16` for a heavier run). The rest are end-to-end checks against the real MinIO and embedding model.

# Contributing
- Dependencies are locked (`uv.lock`, committed), so `uv sync` stays reproducible.
- Embeddings run on ONNX Runtime, with no PyTorch. `vectorstore/embedding.py` reproduces sentence-transformers' `all-MiniLM-L6-v2` pipeline from the model files, pinned to an exact Hugging Face revision. `tests/test_embedding.py` checks it against reference vectors made by sentence-transformers, so vectors stay compatible with ones stored earlier. To change the model or revision, update both `MODEL_REVISION` and the CI cache key, then regenerate the reference: `uv run --with sentence-transformers==6.0.0 python scripts/make_embedding_fixture.py`.