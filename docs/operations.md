# Operations

## Deploying

- **One container per API process group.** The image runs uvicorn with `WEB_CONCURRENCY` workers. For more throughput, add workers or replicas; they all share one MinIO bucket safely (see [below](#many-processes-one-bucket)).
- **Health:** `GET /status` answers once the model and the default index are loaded. The image's `HEALTHCHECK` uses it, and so should your orchestrator's readiness probe.
- **Startup is fail-fast.** The server exits if MinIO is unreachable or doesn't enforce conditional writes. `compose.yaml` restarts it until MinIO is up.
- **Image tags:** `ghcr.io/muratcanozdemir/s3-vectors-at-home:<version>` (e.g. `0.1.0`), `:<major.minor>` (e.g. `0.1`, which moves with patch releases), and `:latest`.
- **Security:** neither API authenticates, and the S3 Vectors API doesn't check request signatures. Keep the server on a trusted network, or put an authenticating proxy in front of it.

## Requirements

The object store must **enforce conditional writes**: `If-Match` and `If-None-Match: *` on PUT. All concurrency safety rests on them.

- **MinIO:** enforces them in releases since late 2024; `minio/docker-compose.yaml` pins one.
- **Other stores:** the server checks at startup and refuses to run on a store that ignores them, rather than quietly risk lost updates.

## Storage layout

An index is a small **manifest** plus immutable **segments**:

```
vectors/                                   # the MinIO bucket
├── indexes/<name>.manifest.json           # which segments make up the index; the only object rewritten
├── segments/<name>/<incarnation>/<id>.seg # text, metadata and vectors (or delete markers) of one write
└── s3vectors/
    ├── _buckets/<vector-bucket>.json      # S3 Vectors bucket registry
    └── <vector-bucket>/indexes/…, segments/…   # the same layout, per vector bucket
```

- **Writes** upload one segment holding just that write's documents, then append it to the manifest. A write therefore costs the same whether the index holds 50 documents or 50,000.
- **Reads** (get, list, count, search, filtering) are served from memory.
- **Caching:** every operation does a cheap `HEAD` on the manifest. When another process has written, only the new manifest and the segments not yet seen are downloaded.

## Memory

Every worker process keeps each index it has used in memory: vectors, text and metadata. Roughly, a document takes 4.4 bytes per dimension per vector, plus its text, plus ~200 bytes; a 384-dimension text index needs about 2.7 KB per short document, so 100,000 documents take ~270 MB, in every worker.

- **The budget:** `S3VAH_CACHE_MB` caps what one process holds across all its indexes, text and S3 Vectors alike. When the total goes over, the indexes used least recently are dropped. The next request for a dropped index reloads it from MinIO (about 0.3 s for 50,000 documents), so set the budget above what your busy indexes need, and let it trim the ones used rarely.
- **The index in use** is never dropped. An index bigger than the whole budget still works; it's just the only one kept.
- **Without a budget** (the default), nothing is dropped, as in earlier versions.
- **Approximate indexes** count their graph too, and the field indexes filters build.
- **Watching it:** `GET /status` reports this worker's `cache`: how many indexes it holds, their estimated `bytes`, the `max_bytes` budget, and the `evictions` so far. Frequent evictions mean the budget is too small for the working set. The size is an estimate: it came within about 10% of measured memory for 384- and 1,024-dimension indexes.
- **With several workers,** each has its own budget: total memory is up to `WEB_CONCURRENCY` × `S3VAH_CACHE_MB`, plus about 240 MB per worker for the server and the model before any index is loaded.

## Compaction

- **Automatic:** segments are merged in the background after writes, size-tiered. An index keeps a few dozen segments at most, and each document is rewritten about log₂(n) times over its life.
- **Manual:** `POST /indexes/{name}/compact` (or `s3-vectors-cli indexes compact NAME`) merges everything into one segment. It reclaims the space of overwritten and deleted documents, and it's useful before a backup.
- **Orphans:** compaction also deletes segments that a crash or failed write left behind and nothing references, once they're an hour old. This assumes the clocks of the machines writing to a bucket agree to within about 45 minutes.

## Many processes, one bucket

Any number of API workers, replicas and CLI runs can read and write the same bucket at once.

- **Compare-and-swap:** every manifest update is a conditional PUT, so a writer that loses a race reloads and appends its segment on top. No update is lost, and merges use the same mechanism.
- **Same document from several writers:** every record gets a sequence number, and a document's newest record wins as a whole. Concurrent writes to one `doc_id` are safe.
- **Throughput ceiling:** all writes to one index pass through its manifest, so they top out at roughly 100+ per second across all processes. With 32 processes writing flat out against local MinIO, about 120 writes/s went through, and every write succeeded, the slowest after 36 attempts.
- **Batch for more:** `/upload/bulk` and `PutVectors` write many documents per manifest update.
- **Write timeout:** a write that keeps losing for 30 seconds gives up without being applied. The API answers 503 with `Retry-After`, S3 Vectors answers `ServiceUnavailableException`, and the CLI exits 1.
- **Deletes are final:** a write racing an index delete fails with "index not found" rather than re-creating the index. Re-creating the name gives a fresh, empty index.

## Performance

Measured on a laptop against local MinIO.

**With 1,000 documents:**

| Operation | Time |
|---|---|
| bulk upload | ~1.3 s (almost all of it the embedding model) |
| search | ~3 ms (about 2 ms embedding the query, 1 ms checking the manifest) |
| filtered search | ~4 ms |
| get, list, count | under 1 ms |

**A single write** stays at about 6–12 ms and uploads about 6 KB, whatever the index size. The old one-object layout cost more as the index grew:

| Index size | Old layout | Now |
|---|---|---|
| 1,000 docs | 13 ms / 1.8 MB | ~6–12 ms / ~6 KB |
| 10,000 docs | 71 ms / 18 MB | ~6–12 ms / ~6 KB |
| 50,000 docs | 565 ms / 88 MB | ~6–12 ms / ~6 KB |

**Loading** a 50,000-document index from scratch takes about 0.3 s. The API loads the model and the default index at startup, so the first request isn't slow.

## Approximate search

Every index searches exactly unless it's created as approximate: with `"search": "approximate"` in the text API, `--approximate` in the CLI, or `S3VAH_S3VECTORS_SEARCH=approximate` for new S3 Vectors indexes. An approximate index searches an HNSW graph instead of comparing the query with every vector.

Measured on 100,000 clustered 384-dimension vectors, top 10:

| Search | Exact index | Approximate index | Recall |
|---|---|---|---|
| no filter | 7.6 ms | 0.2 ms | 100% |
| filter matching 1% | 0.3 ms | 0.3 ms | 100% |
| filter matching 25% | 8 ms | 5 ms | 100% |
| filter matching 60% | 15 ms | 11 ms | 100% |

- **When it's worth it:** for large indexes searched often, without a filter or with broad filters. Below 2,000 vectors no graph is built at all, since exact search is already well under a millisecond.
- **Recall:** the true top 10 came back for every test query here, but approximate means not guaranteed. Real embeddings are clustered, which suits graphs; data spread uniformly in many dimensions is much harder. Measure with your own data: send the same search with `"exact": true` and compare. Raising `ef_search` improves recall.
- **Filters:** a filter that matches at most 20,000 vectors or 5% of the index is searched exactly over just those vectors, which is both fast and exact. A broader one searches the graph, skipping documents that don't match.
- **Building the graph:** the graph is built in the background, with `S3VAH_GRAPH_BUILD_THREADS` threads, when the index is loaded and again after many deletes. Until it's built, searches are exact, and `/status` shows `"graph_ready": false`. For 100,000 vectors, loading took 1.6 s and the graph was ready 3 s later, using 8 threads. The graph isn't stored in MinIO, so each worker builds its own after a restart.
- **Writes** go into the graph as they arrive: about 0.1 ms a vector on one thread, with bigger writes spread over the build threads. Bulk loading 100,000 vectors in batches of 5,000 took about 5 s, against 1.3 s for an exact index.
- **Deletes** leave a gap in the graph that searches skip. Once more than a quarter of it is gaps, the index is compacted and the graph rebuilt in the background.
- **Memory:** the graph adds about 400–500 bytes a vector (with the default `m` of 32). While a graph is being built, the index's vectors are held twice.

## Limits

| | |
|---|---|
| Text API `top_k` | 1–1,000 |
| Text API paging | `skip` ≥ 0, `limit` 0–1,000 |
| Text API index names | 1–63 characters: `a-z`, `0-9` and `-` |
| `doc_id` | up to 512 characters, no `/` |
| Text API metadata | 32 keys, 4 KB |

The S3 Vectors limits are in [S3 Vectors API](s3-vectors.md#limits).

## Backups

`s3-vectors-cli backup DIR` copies every index, text API and S3 Vectors alike, into a new local directory. `s3-vectors-cli restore DIR` puts them back. Neither needs the server to be stopped.

```bash
dir=$(mktemp -d)/vectors-backup
curl -sf -X POST localhost:8000/indexes -H 'content-type: application/json' -d '{"name": "backup-demo"}'
curl -sf -X POST localhost:8000/indexes/backup-demo/upload -H 'content-type: application/json' \
  -d '{"doc_id": "note", "text": "remember the milk"}'

s3-vectors-cli backup "$dir"
curl -sf -X DELETE localhost:8000/indexes/backup-demo          # disaster strikes
s3-vectors-cli restore "$dir" --index backup-demo
curl -sf localhost:8000/indexes/backup-demo/document/note | jq -e '.text == "remember the milk"'
curl -sf -X DELETE localhost:8000/indexes/backup-demo
```

- **Consistent while writes go on:** an index is copied as of one moment. The backup reads its manifest, copies the segments it lists, and saves that same manifest, so concurrent writes are either all in the copy or not in it at all. If compaction removes a segment mid-copy, the backup follows the newer manifest.
- **The directory** holds one file per object, at its object key, plus `backup.json`, which lists every file with its size and MD5. It's written last, so a directory without it is an unfinished backup. Copy the directory anywhere: it's plain files.
- **Restores check first:** every file is verified against `backup.json` before anything is written. Segments go up before their manifest, so a running server never sees a half-restored index.
- **Restores never overwrite** an existing index unless you pass `--replace`. Running servers then reload the replaced index on their next request, and compaction deletes its old segments later. A fresh server creates an empty `default` index on startup, so restoring `default` into it needs `--replace`.
- **Choosing indexes:** `--index NAME` (repeatable) restores some of them: text API indexes by name, S3 Vectors ones as `BUCKET/INDEX`. `--dry-run` checks the backup and reports what would be restored.
- **Not included:** indexes still in a layout older than manifests, because nothing has used them since an upgrade. The backup lists any it finds; any request to such an index migrates it, and then it can be backed up. Approximate indexes' graphs aren't stored, so they're rebuilt after a restore, as after any restart.
- **In Docker,** back up to a mounted directory, for example `docker compose run --rm --user "$(id -u):$(id -g)" -v "$PWD/backups:/backups" api s3-vectors-cli backup /backups/today`. A path inside the container disappears with it, and `--user` lets the container's process write to your directory (it runs as UID 10001 otherwise).
- **Compact first** (optional) for a smaller copy without overwritten or deleted documents.

A bucket copy, for example with `mc mirror`, also works if writes are paused, or if it copies manifests last: a manifest copied before its newest segments points at objects the copy doesn't have.

## Upgrading

Each index is migrated the first time it's used:

- from the previous one-object layout (`indexes/<name>.v3.bin`);
- for the default index, also from the original layouts (`index.v2.bin`, or `faiss.index` + `index.ids.json` with per-document `<id>.npy` / `<id>.meta.json` objects), with empty metadata.

The old objects are left in place, and nothing reads them again, so delete them once you're sure you won't roll back. Don't run old and new versions against the same bucket at the same time: the old code doesn't see writes made by the new.

**Long documents (chunking).** Segments from earlier versions read unchanged, and documents in them keep their single vector until they are written again. Once a document longer than 256 tokens has been written, though, its index has segments that versions without chunking can't read: they report the index as unreadable. To roll back, delete or re-upload those documents with `S3VAH_CHUNK_OVERLAP=off` and compact first.
