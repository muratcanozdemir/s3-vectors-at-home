# S3 Vectors API

The server speaks the [Amazon S3 Vectors](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors.html) API. `boto3`, the other AWS SDKs, and tools built on them work against it unchanged: point them at the server with an endpoint URL.

Unlike the [text API](text-api.md), you bring your own vectors: float32, any dimension from 1 to 4,096, with cosine or euclidean distance. The built-in embedding model isn't used here.

## Connecting

```python
import boto3

s3v = boto3.client(
    "s3vectors",
    endpoint_url="http://localhost:8000",
    region_name="us-east-1",
    aws_access_key_id="any",       # credentials aren't checked, but the SDK wants some
    aws_secret_access_key="any",
)
assert "vectorBuckets" in s3v.list_vector_buckets()
```

The AWS CLI works the same way: `aws s3vectors list-vector-buckets --endpoint-url http://localhost:8000`.

## A complete example

Create a vector bucket and an index, store vectors with metadata, query with a filter, page through everything, then clean up:

```python
import boto3

s3v = boto3.client("s3vectors", endpoint_url="http://localhost:8000", region_name="us-east-1",
                   aws_access_key_id="any", aws_secret_access_key="any")

s3v.create_vector_bucket(vectorBucketName="media")
s3v.create_index(
    vectorBucketName="media",
    indexName="movies",
    dataType="float32",
    dimension=4,
    distanceMetric="cosine",
    # Stored and returned, but not filterable (and not counted against the 40 KB filterable limit).
    metadataConfiguration={"nonFilterableMetadataKeys": ["plot"]},
)

movies = {
    "star-wars":   ([0.1, 0.9, 0.2, 0.0], {"genre": "scifi", "year": 1977, "plot": "A farm boy joins a rebellion."}),
    "alien":       ([0.2, 0.8, 0.0, 0.3], {"genre": "scifi", "year": 1979, "plot": "The crew of a towing ship meets a predator."}),
    "arrival":     ([0.1, 0.7, 0.3, 0.1], {"genre": "scifi", "year": 2016, "plot": "A linguist talks to visitors."}),
    "casablanca":  ([0.9, 0.1, 0.1, 0.0], {"genre": "drama", "year": 1942, "plot": "Of all the gin joints..."}),
}
s3v.put_vectors(vectorBucketName="media", indexName="movies", vectors=[
    {"key": key, "data": {"float32": vec}, "metadata": meta} for key, (vec, meta) in movies.items()
])

# The three nearest sci-fi films from before 1980.
hits = s3v.query_vectors(
    vectorBucketName="media", indexName="movies",
    queryVector={"float32": [0.1, 0.8, 0.2, 0.1]},
    topK=3,
    filter={"genre": "scifi", "year": {"$lt": 1980}},
    returnMetadata=True, returnDistance=True,
)["vectors"]
for h in hits:
    print(f'{h["distance"]:.4f}  {h["key"]}  {h["metadata"]["year"]}')
assert [h["key"] for h in hits] == ["star-wars", "alien"]

# Fetch by key; keys that don't exist are left out.
got = s3v.get_vectors(vectorBucketName="media", indexName="movies", keys=["alien", "missing"],
                      returnData=True, returnMetadata=True)["vectors"]
assert [v["key"] for v in got] == ["alien"]
assert got[0]["data"]["float32"] == [0.20000000298023224, 0.800000011920929, 0.0, 0.30000001192092896]  # float32, exactly

# List everything with the SDK's paginator.
pages = s3v.get_paginator("list_vectors").paginate(vectorBucketName="media", indexName="movies",
                                                   PaginationConfig={"PageSize": 2})
assert sorted(v["key"] for page in pages for v in page["vectors"]) == sorted(movies)

# Clean up: indexes first, then the bucket.
s3v.delete_vectors(vectorBucketName="media", indexName="movies", keys=list(movies))
s3v.delete_index(vectorBucketName="media", indexName="movies")
s3v.delete_vector_bucket(vectorBucketName="media")
```

## Semantic search with your own embeddings

Any embedding model works, as long as one index always gets vectors from the same model. This example uses the same model as the text API, from this package, so it needs `pip install s3-vectors-at-home` (or a source checkout). `sentence-transformers`, OpenAI or Bedrock embeddings work the same way.

```python
import boto3
from vectorstore.embedding import OnnxEmbedder

embed = OnnxEmbedder.from_hub()          # all-MiniLM-L6-v2: 384 dimensions, unit length
s3v = boto3.client("s3vectors", endpoint_url="http://localhost:8000", region_name="us-east-1",
                   aws_access_key_id="any", aws_secret_access_key="any")

s3v.create_vector_bucket(vectorBucketName="help-center")
s3v.create_index(vectorBucketName="help-center", indexName="faq", dataType="float32",
                 dimension=embed.get_sentence_embedding_dimension(), distanceMetric="cosine")

faq = {
    "reset-password": "How do I reset my password?",
    "refund":         "Can I get my money back after buying?",
    "shipping":       "How long does delivery take?",
}
vectors = embed.encode(list(faq.values()))
s3v.put_vectors(vectorBucketName="help-center", indexName="faq", vectors=[
    {"key": key, "data": {"float32": vec.tolist()}, "metadata": {"question": text}}
    for (key, text), vec in zip(faq.items(), vectors)
])

query = embed.encode(["I forgot my login"])[0]
best = s3v.query_vectors(vectorBucketName="help-center", indexName="faq", queryVector={"float32": query.tolist()},
                         topK=1, returnMetadata=True)["vectors"][0]
print(best["key"], "->", best["metadata"]["question"])
assert best["key"] == "reset-password"

s3v.delete_index(vectorBucketName="help-center", indexName="faq")
s3v.delete_vector_bucket(vectorBucketName="help-center")
```

## Supported operations

| Area | Operations |
|---|---|
| Vector buckets | `CreateVectorBucket`, `GetVectorBucket`, `ListVectorBuckets`, `DeleteVectorBucket` |
| Indexes | `CreateIndex`, `GetIndex`, `ListIndexes`, `DeleteIndex` |
| Vectors | `PutVectors`, `GetVectors`, `ListVectors` (including parallel `segmentCount`/`segmentIndex` listing), `DeleteVectors`, `QueryVectors` |

- **Request and response shapes, limits and error codes** follow the service model.
- **Addressing:** resources can be named or given by ARN. `S3VAH_REGION` and `S3VAH_ACCOUNT_ID` set the region and account in ARNs; the defaults are `us-east-1` and `000000000000`.
- **Pagination:** the SDK paginators work for `ListVectorBuckets`, `ListIndexes` and `ListVectors`.

### Limits

| | |
|---|---|
| Vector bucket and index names | 3–63 characters: `a-z`, `0-9`, `-` and `.`, starting and ending with a letter or digit |
| Vector keys | 1–1,024 characters |
| Dimension | 1–4,096 |
| Per call | `PutVectors` 500 vectors; `GetVectors` 100 keys; `DeleteVectors` 500 keys; `ListVectors` 1,000 per page; `QueryVectors` `topK` up to 10,000 (returned 100 per page) |
| Metadata per vector | 50 keys; 100 KB in total; 40 KB filterable; each filterable string up to 2 KB; up to 10 non-filterable keys per index |

### Behaviour where AWS's documentation leaves it open

- **Distances:** cosine `distance` is 1 − cosine similarity (0 = same direction). Euclidean is the plain, not squared, distance. Results come closest first.
- **Missing keys:** `GetVectors` leaves out keys that don't exist, and `DeleteVectors` ignores them.
- **Non-filterable keys:** filtering on one is a `ValidationException`, not an empty result.
- **Deleting a bucket that still has indexes** is a `ConflictException`.
- **Rejected vectors:** NaN, infinity, a wrong dimension, values outside float32's range, and all-zero vectors in a cosine index are all `ValidationException`s.
- **Precision:** vectors are stored and returned exactly as the float32 values you sent.

### Not supported

These are answered with a clear `ValidationException`:

- full-text and hybrid search (`queryTexts`, `textSearchConfigurations`);
- KMS encryption;
- bucket policies, tags, and index-mode or metadata-configuration updates.

## Errors

The SDK raises `botocore.exceptions.ClientError`. Its code is the S3 Vectors one:

```python
import boto3
from botocore.exceptions import ClientError

s3v = boto3.client("s3vectors", endpoint_url="http://localhost:8000", region_name="us-east-1",
                   aws_access_key_id="any", aws_secret_access_key="any")
try:
    s3v.get_index(vectorBucketName="no-such-bucket", indexName="nope")
except ClientError as e:
    assert e.response["Error"]["Code"] == "NotFoundException"
    assert e.response["ResponseMetadata"]["HTTPStatusCode"] == 404
```

| Code | Status | When |
|---|---|---|
| `ValidationException` | 400 | bad input, a limit exceeded, or an unsupported feature |
| `NotFoundException` | 404 | an unknown vector bucket or index |
| `ConflictException` | 409 | the bucket or index already exists, or a bucket still has indexes |
| `ServiceUnavailableException` | 503 | a write lost to concurrent writers for 30 seconds; the SDK retries it |
| `InternalServerException` | 500 | the stored data is unreadable |

## Storage

- **Layout:** vector buckets live under `s3vectors/` in the server's MinIO bucket, on the same segmented storage as the text API. See [Operations](operations.md#storage-layout).
- **Concurrency:** many API processes can serve the same vector buckets at once; see [Operations](operations.md#many-processes-one-bucket).
- **Turning it off:** `S3VAH_S3VECTORS_API=false` removes the routes.
