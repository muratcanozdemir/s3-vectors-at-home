# CLI

`s3-vectors-cli` works on the same storage as the [text API](text-api.md), directly, with no server needed. It's installed with the package (`pip install s3-vectors-at-home`) and reads the [MinIO settings](configuration.md) from the environment.

<!-- docs-test: skip -->
```bash
export MINIO_ENDPOINT=localhost:9000 MINIO_ACCESS_KEY=minioadmin MINIO_SECRET_KEY=minioadmin123
```

Inside the Docker stack it's already configured: `docker compose exec api s3-vectors-cli status`.

The CLI and any number of API servers can use the same bucket at the same time; see [Operations](operations.md#many-processes-one-bucket). The first command loads the embedding model, about a second, plus a one-time download outside Docker.

## Indexes

```bash
s3-vectors-cli indexes create cli-demo
s3-vectors-cli indexes list
```

```
Created index cli-demo
cli-demo
default
```

`indexes delete NAME` deletes an index. `indexes compact [NAME]` merges an index's segments, which normally happens by itself.

## Documents

Every document command takes `--index NAME` after the command name. Without it, commands use the `default` index.

```bash
s3-vectors-cli upload --index cli-demo --doc-id note-1 --text "Buy oat milk and coffee" \
  --metadata '{"list": "groceries", "priority": 2}'

cat > docs.json <<'JSON'
[
  {"doc_id": "note-2", "text": "Call the dentist about Tuesday", "metadata": {"list": "errands"}},
  {"doc_id": "note-3", "text": "Pick up bread and eggs",          "metadata": {"list": "groceries", "priority": 1}}
]
JSON
s3-vectors-cli bulk-upload --index cli-demo --file docs.json

s3-vectors-cli search --index cli-demo --query "shopping for breakfast" --top-k 2 \
  --filter '{"list": "groceries"}' --scores

s3-vectors-cli get --index cli-demo --doc-id note-2
s3-vectors-cli list --index cli-demo --limit 10
s3-vectors-cli delete --index cli-demo --doc-id note-2
s3-vectors-cli status --index cli-demo
```

```
Uploaded note-1
Bulk uploaded 2 documents.
0.5551  note-3
0.4888  note-1
{
  "doc_id": "note-2",
  "text": "Call the dentist about Tuesday",
  "metadata": {
    "list": "errands"
  }
}
note-1 - Buy oat milk and coffee
note-2 - Call the dentist about Tuesday
note-3 - Pick up bread and eggs
Deleted note-2
Index: cli-demo
Documents: 2 (2 vectors)
Segments: 3 (4,976 bytes)
Embedding model: all-MiniLM-L6-v2
```

- **The bulk file** is a JSON list of `{"doc_id", "text", "metadata"?}` objects. As with the API, one bad entry rejects the whole file and nothing is uploaded.
- **`indexes create NAME --approximate`** creates an index searched through an HNSW graph, with optional `--m`, `--ef-construction` and `--ef-search` (see [Text API](text-api.md#indexes)). `search --exact` compares against every vector, even on an approximate index. `status` shows whether an approximate index's graph is ready.
- **`backup DIR`** and **`restore DIR`** (with `--index NAME`, `--replace`, `--dry-run`) copy indexes to a local directory and back; see [Operations](operations.md#backups).
- **`search` without `--scores`** prints `Matches: [...]`. With `--passages` it prints each match's score, doc_id and `[start:end]` span, then the best-matching passage, indented.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | success |
| 1 | an error, printed as one `Error: ...` line on stderr: invalid JSON, a bad doc_id, metadata or filter, an unknown index, a missing file, an unreachable or refusing object store, unreadable stored data, or a write that lost to concurrent writers for 30 seconds. `get` and `delete` of a missing document also exit 1 (`Not found.`). |
| 2 | invalid command-line usage (argparse) |

```bash
# Errors are one line on stderr, with exit code 1: safe to use in scripts.
if s3-vectors-cli get --index cli-demo --doc-id no-such-note 2>/dev/null; then exit 1; fi
s3-vectors-cli indexes delete cli-demo
```
