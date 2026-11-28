# Filters

Both APIs filter on metadata with the same MongoDB-style language that Amazon S3 Vectors uses. A filter is a JSON object, passed as `filter` to `POST /search`, to `QueryVectors`, or to `s3-vectors-cli search --filter`.

## Operators

| Filter | Matches documents where… |
|---|---|
| `{"lang": "en"}` | `lang` equals `"en"`; shorthand for `{"lang": {"$eq": "en"}}` |
| `{"lang": {"$ne": "en"}}` | `lang` isn't `"en"`, **or is missing** |
| `{"year": {"$gt": 2020}}` | `year` > 2020; also `$gte`, `$lt`, `$lte` (numbers only) |
| `{"year": {"$gte": 2020, "$lt": 2025}}` | 2020 ≤ `year` < 2025: operators on one field are ANDed |
| `{"tags": {"$in": ["ml", "ai"]}}` | `tags` equals any of the values |
| `{"tags": {"$nin": ["draft"]}}` | `tags` equals none of the values (true if missing) |
| `{"source": {"$exists": true}}` | the field is present (`false`: absent) |
| `{"path": {"$startsWith": "docs/"}}` | a string field starts with the prefix |
| `{"lang": "en", "year": 2024}` | several fields: all must match |
| `{"$and": [{...}, {...}]}` | every sub-filter matches |
| `{"$or": [{...}, {...}]}` | at least one sub-filter matches |

## How values compare

- **Lists match on any element.** With `{"tags": ["ml", "nlp"]}` stored, `{"tags": "ml"}` matches, `{"tags": {"$in": ["nlp", "cv"]}}` matches, and `{"tags": {"$nin": ["ml"]}}` doesn't.
- **Range operators compare numbers only.** `{"year": {"$gt": 2020}}` never matches `"2021"` stored as a string.
- **Booleans aren't numbers.** `{"flag": 1}` doesn't match `true`. Numbers compare by value, so `1` matches `1.0`.
- **A missing field** matches only `$ne`, `$nin` and `{"$exists": false}`.
- **Unknown operators are rejected** with a 400 (`ValidationException` in S3 Vectors), never silently ignored. That includes `$regex`, `$not`, `$nor`, and nested objects as values.

Filtered search is **exact** on an exact index (the default): the results are the true top-k among the documents that match, however few match. A filter doesn't make results approximate or short. On an [approximate index](operations.md#approximate-search), a filter that matches few documents is still searched exactly; a broad one goes through the graph.

**Speed.** Each field a filter uses is indexed the first time a filter mentions it: which documents hold each value, and the values in sorted order for ranges and prefixes. Fields nobody filters on cost nothing. With 100,000 documents, a filter matching 1% of them takes under 1 ms, one matching 25% about 5–8 ms, and one matching 60% about 11–15 ms. Before, every filtered search checked every document, which took about 110 ms.

## Examples

```bash
curl -sf -X POST localhost:8000/indexes -H 'content-type: application/json' -d '{"name": "filter-demo"}'
curl -sf -X POST localhost:8000/indexes/filter-demo/upload/bulk -H 'content-type: application/json' -d '{"docs": [
  {"doc_id": "a", "text": "intro to machine learning", "metadata": {"year": 2019, "tags": ["ml"], "lang": "en"}},
  {"doc_id": "b", "text": "deep learning in practice", "metadata": {"year": 2023, "tags": ["ml", "dl"], "lang": "en"}},
  {"doc_id": "c", "text": "apprentissage automatique", "metadata": {"year": 2024, "tags": ["ml"], "lang": "fr", "draft": true}}
]}'

search() {  # search <filter>: print the matching doc_ids, sorted
  curl -sf -X POST localhost:8000/indexes/filter-demo/search -H 'content-type: application/json' \
    -d "{\"query\": \"machine learning\", \"top_k\": 10, \"filter\": $1}" | jq -c '.matches | sort'
}

test "$(search '{"lang": "en"}')"                                       = '["a","b"]'
test "$(search '{"year": {"$gte": 2020, "$lt": 2024}}')"                = '["b"]'
test "$(search '{"tags": "dl"}')"                                        = '["b"]'
test "$(search '{"draft": {"$ne": true}}')"                              = '["a","b"]'
test "$(search '{"$or": [{"lang": "fr"}, {"year": {"$lt": 2020}}]}')"    = '["a","c"]'
test "$(search '{"draft": {"$exists": false}, "tags": {"$nin": ["dl"]}}')" = '["a"]'

curl -sf -X DELETE localhost:8000/indexes/filter-demo
```

## Limits

- **Text API:** filters can nest `$and`/`$or` up to 8 levels deep.
- **S3 Vectors API:** a filter may have up to 100 constraints, each operand value counting once. Keys listed in an index's `nonFilterableMetadataKeys` can't be filtered on.
