# Text API

Store text with optional metadata; search it by meaning. The server embeds text itself, with `all-MiniLM-L6-v2` (384 dimensions, cosine similarity).

Every document route exists twice:

- at the root (`/upload`, `/search`, …), working on the `default` index, which always exists;
- under `/indexes/{name}` (`/indexes/papers/upload`, …), working on a named index.

The examples below use a named index, `papers`. All request and response bodies are JSON.

## Indexes

```bash
# Create (201; 409 if it already exists)
curl -sf -X POST localhost:8000/indexes -H 'content-type: application/json' -d '{"name": "papers"}'
# List
curl -sf localhost:8000/indexes | jq -e '.indexes | index("papers")'
```

```json
{"status": "created", "index": "papers"}
{"indexes": ["default", "papers"]}
```

Index names are 1–63 characters of `a-z`, `0-9` and `-`, and can't start or end with `-`. `DELETE /indexes/{name}` deletes an index and all its documents; `default` can't be deleted.

## Adding documents

One document; adding an existing `doc_id` replaces it:

```bash
curl -sf -X POST localhost:8000/indexes/papers/upload -H 'content-type: application/json' -d '{
  "doc_id": "attention",
  "text": "Attention is all you need: the Transformer, built on self-attention.",
  "metadata": {"year": 2017, "tags": ["nlp", "transformers"]}
}'
```

Many at once, which is much faster: the batch is embedded in one call and written as one object.

```bash
curl -sf -X POST localhost:8000/indexes/papers/upload/bulk -H 'content-type: application/json' -d '{
  "docs": [
    {"doc_id": "resnet", "text": "Deep residual learning for image recognition.",
     "metadata": {"year": 2015, "tags": ["vision"]}},
    {"doc_id": "gcn", "text": "Semi-supervised classification with graph convolutional networks.",
     "metadata": {"year": 2016, "tags": ["graphs"]}},
    {"doc_id": "vit", "text": "An image is worth 16x16 words: transformers for image recognition at scale.",
     "metadata": {"year": 2020, "tags": ["vision", "transformers"]}}
  ]}' | jq -e '.count == 3'
```

```json
{"status": "ok", "count": 3}
```

A bulk upload is all-or-nothing on validation: every document is checked first, and one bad `doc_id` or metadata value rejects the whole batch with a 400, writing nothing. If a `doc_id` repeats within a batch, the last one wins.

**Rules:**

- **`doc_id`:** a non-empty string of up to 512 characters, without `/`.
- **`text`:** any string, stored in full. A text longer than the model's 256-token window is split into chunks, and all of it is searchable; see [Long documents](#long-documents).
- **`metadata`:** a flat object of up to 32 keys and 4 KB. Values are strings, numbers, booleans, or lists of those. Metadata is optional.

## Searching

```bash
curl -sf -X POST localhost:8000/indexes/papers/search -H 'content-type: application/json' \
  -d '{"query": "neural networks for pictures", "top_k": 2}' | jq -e '.matches == ["resnet", "vit"]'
```

```json
{
  "matches": ["resnet", "vit"],
  "results": [
    {"doc_id": "resnet", "score": 0.5405, "metadata": {"year": 2015, "tags": ["vision"]}},
    {"doc_id": "vit",    "score": 0.5345, "metadata": {"year": 2020, "tags": ["vision", "transformers"]}}
  ]
}
```

- `matches` lists doc_ids, closest first. `results` has the same order, adding `score` (cosine similarity, −1 to 1, higher is closer) and the metadata, and with `"passages": true` the best-matching passage (see [Long documents](#long-documents)).
- `top_k` defaults to 5 and can be 1–1000. Fewer results come back if the index has fewer documents.

Add a `filter` to search only documents whose metadata matches. Results stay exact: you get the true top-k among the matching documents.

```bash
curl -sf -X POST localhost:8000/indexes/papers/search -H 'content-type: application/json' \
  -d '{"query": "neural networks for pictures", "top_k": 2, "filter": {"year": {"$gte": 2018}}}' \
  | jq -e '.matches == ["vit"]'
```

The filter language is in [Filters](filters.md).

## Long documents

The model reads at most 256 tokens (roughly 150–200 words) at a time. A longer text is split into chunks that fit, each sharing 32 tokens with the one before and never cutting a word in half, and every chunk gets its own vector. The document is still one document: getting, listing, counting, replacing and deleting work on the whole of it.

- **Scoring:** a document scores as its best-matching chunk, and appears at most once in the results. `top_k` counts documents, not chunks.
- **Passages:** add `"passages": true` to a search, and each result also has `passage`, the text of that best chunk, and `span`, its `[start, end)` character offsets in the document's text. For a document that wasn't split, the passage is the whole text.

```bash
curl -sf -X POST localhost:8000/indexes -H 'content-type: application/json' -d '{"name": "manuals"}'
filler=$(for i in $(seq 1 60); do printf 'Section %s covers invoicing, tax codes and ledger reconciliation. ' "$i"; done)
jq -n --arg f "$filler" '{docs: [
  {doc_id: "handbook", text: ($f + "The lighthouse keeper feeds the seagulls sardines at dawn. " + $f)},
  {doc_id: "ledger",   text: ($f + $f)}]}' \
  | curl -sf -X POST localhost:8000/indexes/manuals/upload/bulk -H 'content-type: application/json' -d @-

curl -sf -X POST localhost:8000/indexes/manuals/search -H 'content-type: application/json' \
  -d '{"query": "lighthouse keeper feeding seagulls", "top_k": 1, "passages": true}' \
  | jq -e '.matches == ["handbook"] and (.results[0].passage | contains("feeds the seagulls"))'
curl -sf localhost:8000/indexes/manuals/status | jq -e '.document_count == 2 and .vector_count > 2'
curl -sf -X DELETE localhost:8000/indexes/manuals
```

```json
{"matches": ["handbook"], "results": [{"doc_id": "handbook", "score": 0.2221, "metadata": {},
  "passage": "invoicing, tax codes and ledger reconciliation. Section 53 … The lighthouse keeper feeds the seagulls sardines at dawn. Section 1 … Section 10 covers invoicing, tax codes",
  "span": [3375, 4633]}]}
{"status": "ok", "index": "manuals", "document_count": 2, "vector_count": 14, "segments": 1, "storage_bytes": 37620, "embedding_model": "all-MiniLM-L6-v2", "cache": {...}}
```

The fact is in the middle of the handbook, far past its first 256 tokens. It's found in the chunk that contains it, which scores 0.22, while the best chunk of `ledger`, the same text without the fact, scores −0.09. Without chunking, both documents would have been embedded from the same opening and scored alike. (The passage is shortened here with `…`.)

`S3VAH_CHUNK_OVERLAP` sets the overlap, or turns chunking off (see [Configuration](configuration.md#server)). The setting applies to documents as they are written: documents written earlier keep the vectors they have until they are written again.

## Reading, listing and deleting

```bash
curl -sf localhost:8000/indexes/papers/document/gcn | jq -e '.metadata.year == 2016'
curl -sf 'localhost:8000/indexes/papers/documents?skip=0&limit=2' | jq -e '.documents | length == 2'
curl -sf -X DELETE localhost:8000/indexes/papers/document/gcn
```

```json
{"doc_id": "gcn", "text": "Semi-supervised classification with graph convolutional networks.", "metadata": {"year": 2016, "tags": ["graphs"]}}
{"documents": [{"doc_id": "attention", "text_preview": "Attention is all you need: the Transformer, built on self-attent", "metadata": {...}}, ...]}
{"status": "deleted"}
```

- `GET /documents` lists in `doc_id` order (by code point). `text_preview` is the first 64 characters.
- `skip` (≥ 0) and `limit` (0–1000, default 100) page through the list.
- Getting or deleting a missing document is a 404.

## Status and compaction

```bash
curl -sf localhost:8000/indexes/papers/status | jq -e '.document_count == 3'
curl -sf -X POST localhost:8000/indexes/papers/compact | jq -e '.segments == 1'
```

```json
{"status": "ok", "index": "papers", "document_count": 3, "vector_count": 3, "segments": 3, "storage_bytes": 6771, "embedding_model": "all-MiniLM-L6-v2",
 "cache": {"indexes": 2, "bytes": 11623, "max_bytes": null, "evictions": 0}}
{"status": "ok", "index": "papers", "document_count": 3, "segments": 1, "storage_bytes": 5008}
```

`cache` is this worker process's memory for loaded indexes, all of them; see [Operations](operations.md#memory).

Each write adds a small segment (three here: one upload, one bulk upload, one delete). Compaction normally runs by itself in the background, once there are enough segments to be worth merging. Calling it merges everything into one segment and reclaims the space held by overwritten and deleted documents. See [Operations](operations.md#compaction).

## From Python

Any HTTP client works. With `httpx`, which the package already depends on:

```python
import httpx

api = httpx.Client(base_url="http://localhost:8000/indexes/papers", timeout=30)

api.post("/upload/bulk", json={"docs": [
    {"doc_id": "bert", "text": "BERT: pre-training of deep bidirectional transformers for language understanding.",
     "metadata": {"year": 2018, "tags": ["nlp", "transformers"]}},
]}).raise_for_status()

hits = api.post("/search", json={
    "query": "language model pre-training",
    "top_k": 3,
    "filter": {"tags": "nlp"},
}).json()["results"]

for hit in hits:
    print(f'{hit["score"]:.3f}  {hit["doc_id"]}  {hit["metadata"]["year"]}')
assert hits[0]["doc_id"] == "bert"
```

```bash
# Clean up the example index.
curl -sf -X DELETE localhost:8000/indexes/papers
```

## Errors

Errors come back as `{"detail": ...}` with these status codes:

| Status | When |
|---|---|
| 400 | invalid input: a bad `doc_id`, metadata or filter (`{"detail": "unknown operator '$regex' for 'year'"}`); a bad index name; deleting `default` |
| 404 | an unknown index or document |
| 409 | creating an index that already exists |
| 422 | a malformed body or a parameter out of range, such as `top_k: 0` (FastAPI's validation format) |
| 500 | the stored data is unreadable (`IndexCorruptError`); this is a server-side problem, not a bad request |
| 503 | a write lost to concurrent writers for 30 seconds. It wasn't applied, so retry after `Retry-After` seconds. |
