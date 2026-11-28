"""The S3 Vectors API, driven with the real AWS SDK (boto3) over HTTP.

A uvicorn server runs in a thread with the S3 Vectors routes over a fake
MinIO client, so this exercises the actual wire protocol -- botocore's
request serialisation, error parsing, timestamps -- without MinIO.
Server-side validation that botocore would pre-empt client-side is tested
with plain HTTP requests.
"""

import socket
import threading
import time
from datetime import datetime

import boto3
import httpx
import numpy as np
import pytest
import uvicorn
from botocore.config import Config
from botocore.exceptions import ClientError
from fastapi import FastAPI

from api import s3vectors as s3v_api
from tests.fakes import FakeEmbedder, FakeMinioClient
from vectorstore.core import VectorStore
from vectorstore.s3vectors import S3VectorsService


@pytest.fixture(scope="module")
def server():
    client = FakeMinioClient()
    s3v_api.configure(S3VectorsService(client, compaction="inline"))
    app = FastAPI()
    app.include_router(s3v_api.router)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not srv.started and time.time() < deadline:
        time.sleep(0.02)
    yield f"http://127.0.0.1:{port}", client
    srv.should_exit = True
    thread.join(5)
    s3v_api.configure(None)


@pytest.fixture(scope="module")
def s3v(server):
    url, _ = server
    return boto3.client(
        "s3vectors",
        endpoint_url=url,
        region_name="us-east-1",
        aws_access_key_id="any",
        aws_secret_access_key="any",
        config=Config(retries={"max_attempts": 1}),
    )


@pytest.fixture
def raw(server):
    url, _ = server
    with httpx.Client(base_url=url) as c:
        yield c


_n = iter(range(10**6))


def _bucket(s3v) -> str:
    name = f"bucket-{next(_n)}"
    s3v.create_vector_bucket(vectorBucketName=name)
    return name


def _index(s3v, bucket, dim=3, metric="cosine", **kw) -> str:
    name = f"idx-{next(_n)}"
    s3v.create_index(vectorBucketName=bucket, indexName=name, dataType="float32", dimension=dim, distanceMetric=metric, **kw)
    return name


def _code(exc_info) -> str:
    return exc_info.value.response["Error"]["Code"]


def _put(s3v, bucket, index, vectors):
    s3v.put_vectors(
        vectorBucketName=bucket,
        indexName=index,
        vectors=[{"key": k, "data": {"float32": v}, **({"metadata": m} if m is not None else {})} for k, v, m in vectors],
    )


# -- vector buckets ---------------------------------------------------------------


def test_vector_bucket_lifecycle(s3v):
    name = _bucket(s3v)
    got = s3v.get_vector_bucket(vectorBucketName=name)["vectorBucket"]
    assert got["vectorBucketName"] == name
    assert got["vectorBucketArn"] == f"arn:aws:s3vectors:us-east-1:000000000000:bucket/{name}"
    assert isinstance(got["creationTime"], datetime)
    assert got["encryptionConfiguration"] == {"sseType": "AES256"}
    assert s3v.get_vector_bucket(vectorBucketArn=got["vectorBucketArn"])["vectorBucket"]["vectorBucketName"] == name
    with pytest.raises(ClientError) as e:
        s3v.create_vector_bucket(vectorBucketName=name)
    assert _code(e) == "ConflictException"
    s3v.delete_vector_bucket(vectorBucketName=name)
    with pytest.raises(ClientError) as e:
        s3v.get_vector_bucket(vectorBucketName=name)
    assert _code(e) == "NotFoundException" and e.value.response["ResponseMetadata"]["HTTPStatusCode"] == 404
    with pytest.raises(ClientError) as e:
        s3v.delete_vector_bucket(vectorBucketName=name)
    assert _code(e) == "NotFoundException"


def test_list_vector_buckets_with_prefix_and_pages(s3v):
    names = [f"page-test-{i}" for i in range(5)]
    for n in names:
        s3v.create_vector_bucket(vectorBucketName=n)
    seen, token = [], None
    while True:
        kw = {"prefix": "page-test-", "maxResults": 2, **({"nextToken": token} if token else {})}
        resp = s3v.list_vector_buckets(**kw)
        assert len(resp["vectorBuckets"]) <= 2
        seen += [b["vectorBucketName"] for b in resp["vectorBuckets"]]
        token = resp.get("nextToken")
        if not token:
            break
    assert seen == names
    paginated = [b["vectorBucketName"] for p in s3v.get_paginator("list_vector_buckets").paginate(prefix="page-test-") for b in p["vectorBuckets"]]
    assert paginated == names


def test_deleting_a_bucket_with_indexes_is_a_conflict(s3v):
    bucket = _bucket(s3v)
    index = _index(s3v, bucket)
    with pytest.raises(ClientError) as e:
        s3v.delete_vector_bucket(vectorBucketName=bucket)
    assert _code(e) == "ConflictException"
    assert s3v.get_vector_bucket(vectorBucketName=bucket)  # still there, still usable
    s3v.delete_index(vectorBucketName=bucket, indexName=index)
    s3v.delete_vector_bucket(vectorBucketName=bucket)


def test_kms_encryption_is_rejected(s3v):
    with pytest.raises(ClientError) as e:
        s3v.create_vector_bucket(
            vectorBucketName="kms-bucket",
            encryptionConfiguration={"sseType": "aws:kms", "kmsKeyArn": "arn:aws:kms:us-east-1:123456789012:key/x"},
        )
    assert _code(e) == "ValidationException"


# -- indexes ----------------------------------------------------------------------


def test_index_lifecycle(s3v):
    bucket = _bucket(s3v)
    name = _index(s3v, bucket, dim=4, metric="euclidean", metadataConfiguration={"nonFilterableMetadataKeys": ["body"]})
    idx = s3v.get_index(vectorBucketName=bucket, indexName=name)["index"]
    assert (idx["dimension"], idx["distanceMetric"], idx["dataType"]) == (4, "euclidean", "float32")
    assert idx["indexArn"] == f"arn:aws:s3vectors:us-east-1:000000000000:bucket/{bucket}/index/{name}"
    assert idx["metadataConfiguration"] == {"nonFilterableMetadataKeys": ["body"]}
    assert isinstance(idx["creationTime"], datetime)
    assert s3v.get_index(indexArn=idx["indexArn"])["index"]["indexName"] == name
    with pytest.raises(ClientError) as e:
        _index_named(s3v, bucket, name)
    assert _code(e) == "ConflictException"
    s3v.delete_index(indexArn=idx["indexArn"])
    with pytest.raises(ClientError) as e:
        s3v.get_index(vectorBucketName=bucket, indexName=name)
    assert _code(e) == "NotFoundException"


def _index_named(s3v, bucket, name):
    s3v.create_index(vectorBucketName=bucket, indexName=name, dataType="float32", dimension=3, distanceMetric="cosine")


def test_list_indexes_pages_and_prefix(s3v):
    bucket = _bucket(s3v)
    for n in ["alpha.one", "alpha.two", "beta"]:
        _index_named(s3v, bucket, n)
    got = [i["indexName"] for p in s3v.get_paginator("list_indexes").paginate(vectorBucketName=bucket, PaginationConfig={"PageSize": 1}) for i in p["indexes"]]
    assert got == ["alpha.one", "alpha.two", "beta"]
    assert [i["indexName"] for i in s3v.list_indexes(vectorBucketName=bucket, prefix="alpha")["indexes"]] == ["alpha.one", "alpha.two"]


def test_index_in_missing_bucket_is_not_found(s3v):
    with pytest.raises(ClientError) as e:
        _index_named(s3v, "no-such-bucket", "idx")
    assert _code(e) == "NotFoundException"


def test_full_text_search_config_is_rejected(s3v):
    bucket = _bucket(s3v)
    with pytest.raises(ClientError) as e:
        _index(s3v, bucket, metadataConfiguration={"textSearchConfigurations": [{"metadataKey": "body"}]})
    assert _code(e) == "ValidationException"


@pytest.mark.parametrize("name", ["UPPER", "-start", "end-", "has_underscore", "a" * 64])
def test_bad_names_are_rejected(raw, name):
    r = raw.post("/CreateVectorBucket", json={"vectorBucketName": name})
    assert r.status_code == 400 and r.headers["x-amzn-ErrorType"] == "ValidationException"


# -- vectors ------------------------------------------------------------------------


def test_put_get_list_delete_round_trip_exact_floats(s3v):
    bucket = _bucket(s3v)
    index = _index(s3v, bucket)
    data = {"a": [0.1, 0.2, 0.3], "b": [1e-30, -2.5, 3.0e20], "c": [1 / 3, 2 / 3, 1.0]}
    _put(s3v, bucket, index, [(k, v, {"k": k, "n": i, "tags": ["x", k]}) for i, (k, v) in enumerate(data.items())])
    got = s3v.get_vectors(vectorBucketName=bucket, indexName=index, keys=["c", "missing", "a"], returnData=True, returnMetadata=True)["vectors"]
    assert [v["key"] for v in got] == ["c", "a"]  # missing keys omitted, request order kept
    for v in got:
        assert v["data"]["float32"] == [float(np.float32(x)) for x in data[v["key"]]]  # exactly the float32s stored
    assert got[1]["metadata"] == {"k": "a", "n": 0, "tags": ["x", "a"]}
    bare = s3v.get_vectors(vectorBucketName=bucket, indexName=index, keys=["a"])["vectors"]
    assert bare == [{"key": "a"}]
    listed = s3v.list_vectors(vectorBucketName=bucket, indexName=index, returnMetadata=True)["vectors"]
    assert [v["key"] for v in listed] == ["a", "b", "c"] and "data" not in listed[0]
    s3v.delete_vectors(vectorBucketName=bucket, indexName=index, keys=["b", "never-existed"])
    assert [v["key"] for v in s3v.list_vectors(vectorBucketName=bucket, indexName=index)["vectors"]] == ["a", "c"]


def test_put_overwrites_vector_and_metadata(s3v):
    bucket = _bucket(s3v)
    index = _index(s3v, bucket)
    _put(s3v, bucket, index, [("k", [1, 0, 0], {"v": 1, "old": True})])
    _put(s3v, bucket, index, [("k", [0, 1, 0], {"v": 2})])
    got = s3v.get_vectors(vectorBucketName=bucket, indexName=index, keys=["k"], returnData=True, returnMetadata=True)["vectors"][0]
    assert got["data"]["float32"] == [0.0, 1.0, 0.0] and got["metadata"] == {"v": 2}
    q = s3v.query_vectors(vectorBucketName=bucket, indexName=index, queryVector={"float32": [0, 1, 0]}, topK=5, returnDistance=True)
    assert [(v["key"], round(v["distance"], 6)) for v in q["vectors"]] == [("k", 0.0)]


def test_keys_may_contain_any_characters(s3v):
    bucket = _bucket(s3v)
    index = _index(s3v, bucket)
    keys = ["a/b/c", "spaces and ünïcode ✓", "x" * 1024, "$dollar", "dots..."]
    _put(s3v, bucket, index, [(k, [1, 2, 3], None) for k in keys])
    got = s3v.get_vectors(vectorBucketName=bucket, indexName=index, keys=keys)["vectors"]
    assert [v["key"] for v in got] == keys


def test_list_vectors_pages_and_parallel_segments(s3v):
    bucket = _bucket(s3v)
    index = _index(s3v, bucket)
    keys = [f"k{i:03d}" for i in range(57)]
    for i in range(0, len(keys), 20):
        _put(s3v, bucket, index, [(k, [1, i + 1, 2], None) for k in keys[i: i + 20]])
    pages = list(s3v.get_paginator("list_vectors").paginate(vectorBucketName=bucket, indexName=index, PaginationConfig={"PageSize": 10}))
    assert len(pages) == 6 and [v["key"] for p in pages for v in p["vectors"]] == keys
    parts = []
    for seg in range(4):
        got = [v["key"] for p in s3v.get_paginator("list_vectors").paginate(
            vectorBucketName=bucket, indexName=index, segmentCount=4, segmentIndex=seg, PaginationConfig={"PageSize": 7}
        ) for v in p["vectors"]]
        assert got == sorted(got)
        parts.append(set(got))
    assert set().union(*parts) == set(keys) and sum(map(len, parts)) == len(keys)  # disjoint, complete


def test_list_vectors_segment_args_are_validated(s3v, raw):
    bucket = _bucket(s3v)
    index = _index(s3v, bucket)
    for body in ({"segmentCount": 2}, {"segmentCount": 2, "segmentIndex": 2}, {"segmentCount": 17, "segmentIndex": 0}):
        r = raw.post("/ListVectors", json={"vectorBucketName": bucket, "indexName": index, **body})
        assert (r.status_code, r.headers["x-amzn-ErrorType"]) == (400, "ValidationException"), body
    first = s3v.list_vectors(vectorBucketName=bucket, indexName=index, segmentCount=2, segmentIndex=0)
    assert first["vectors"] == []


# -- queries ------------------------------------------------------------------------


def _random_corpus(s3v, metric, n=60, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    bucket = _bucket(s3v)
    index = _index(s3v, bucket, dim=dim, metric=metric)
    vecs = rng.normal(size=(n, dim)).astype(np.float32)
    _put(s3v, bucket, index, [(f"v{i}", vecs[i].tolist(), {"i": i, "even": i % 2 == 0, "name": f"n{i}"}) for i in range(n)])
    return bucket, index, vecs


@pytest.mark.parametrize("metric", ["cosine", "euclidean"])
def test_query_distances_and_order_match_numpy(s3v, metric):
    bucket, index, vecs = _random_corpus(s3v, metric)
    q = np.random.default_rng(1).normal(size=8).astype(np.float32)
    resp = s3v.query_vectors(vectorBucketName=bucket, indexName=index, queryVector={"float32": q.tolist()}, topK=10, returnDistance=True)
    assert resp["distanceMetric"] == metric
    if metric == "cosine":
        expected = 1 - (vecs @ q) / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(q))
    else:
        expected = np.linalg.norm(vecs - q, axis=1)
    order = np.argsort(expected)[:10]
    assert [v["key"] for v in resp["vectors"]] == [f"v{i}" for i in order]
    np.testing.assert_allclose([v["distance"] for v in resp["vectors"]], expected[order], rtol=1e-4, atol=1e-5)
    assert all("metadata" not in v for v in resp["vectors"])


def test_query_with_filters(s3v):
    bucket, index, vecs = _random_corpus(s3v, "cosine")
    q = {"float32": vecs[0].tolist()}

    def keys(flt, k=100):
        r = s3v.query_vectors(vectorBucketName=bucket, indexName=index, queryVector=q, topK=k, filter=flt, returnMetadata=True)
        return {v["metadata"]["i"] for v in r["vectors"]}

    assert keys({"even": True}) == set(range(0, 60, 2))
    assert keys({"i": {"$gte": 10, "$lt": 13}}) == {10, 11, 12}
    assert keys({"$or": [{"i": 3}, {"name": {"$in": ["n5", "n7"]}}]}) == {3, 5, 7}
    assert keys({"name": {"$startsWith": "n5"}}) == {5} | set(range(50, 60))
    assert keys({"i": {"$exists": True}, "even": False}, k=5) and len(keys({"even": False}, k=5)) == 5
    assert keys({"name": "nobody"}) == set()


def test_filter_on_non_filterable_key_is_rejected(s3v):
    bucket = _bucket(s3v)
    index = _index(s3v, bucket, metadataConfiguration={"nonFilterableMetadataKeys": ["body"]})
    _put(s3v, bucket, index, [("a", [1, 2, 3], {"body": "long text " * 100, "lang": "en"})])
    got = s3v.get_vectors(vectorBucketName=bucket, indexName=index, keys=["a"], returnMetadata=True)["vectors"][0]
    assert got["metadata"]["body"].startswith("long text")  # stored and returned
    with pytest.raises(ClientError) as e:
        s3v.query_vectors(vectorBucketName=bucket, indexName=index, queryVector={"float32": [1, 2, 3]}, topK=1, filter={"body": "x"})
    assert _code(e) == "ValidationException"
    r = s3v.query_vectors(vectorBucketName=bucket, indexName=index, queryVector={"float32": [1, 2, 3]}, topK=1, filter={"lang": "en"})
    assert [v["key"] for v in r["vectors"]] == ["a"]


def test_query_pages_beyond_100(s3v):
    bucket, index, vecs = _random_corpus(s3v, "euclidean", n=250, dim=4)
    q = {"float32": vecs[3].tolist()}
    pages = list(s3v.get_paginator("query_vectors").paginate(vectorBucketName=bucket, indexName=index, queryVector=q, topK=230, returnDistance=True)) \
        if s3v.can_paginate("query_vectors") else None
    if pages is None:  # older botocore without a paginator: follow tokens by hand
        pages, token = [], None
        while True:
            r = s3v.query_vectors(vectorBucketName=bucket, indexName=index, queryVector=q, topK=230, returnDistance=True, **({"nextToken": token} if token else {}))
            pages.append(r)
            token = r.get("nextToken")
            if not token:
                break
    assert [len(p["vectors"]) for p in pages] == [100, 100, 30]
    dists = [v["distance"] for p in pages for v in p["vectors"]]
    assert dists == sorted(dists) and len({v["key"] for p in pages for v in p["vectors"]}) == 230
    token = pages[0]["nextToken"]
    with pytest.raises(ClientError) as e:  # a token is only good for the query it came from
        s3v.query_vectors(vectorBucketName=bucket, indexName=index, queryVector={"float32": [0, 0, 0, 1]}, topK=230, nextToken=token)
    assert _code(e) == "ValidationException"


@pytest.mark.parametrize(
    "path,body,why",
    [
        ("/PutVectors", {"vectors": [{"key": "a", "data": {"float32": [1, 2]}}]}, "wrong dimension"),
        ("/PutVectors", {"vectors": [{"key": "a", "data": {"float32": [0, 0, 0]}}]}, "zero vector, cosine"),
        ("/PutVectors", {"vectors": [{"key": "a", "data": {"float32": [1, "x", 3]}}]}, "not a number"),
        ("/PutVectors", {"vectors": [{"key": "a", "data": {"float32": [1, 2, 3]}, "metadata": {"n": {"nested": 1}}}]}, "nested metadata"),
        ("/PutVectors", {"vectors": [{"key": "a", "data": {"float32": [1, 2, 3]}, "metadata": {f"k{i}": i for i in range(51)}}]}, "51 keys"),
        ("/PutVectors", {"vectors": [{"key": "a", "data": {"float32": [1, 2, 3]}, "metadata": {"s": "x" * 2049}}]}, "long filterable string"),
        ("/PutVectors", {"vectors": []}, "empty batch"),
        ("/QueryVectors", {"topK": 1, "queryVector": {"float32": [1, 2]}}, "query dimension"),
        ("/QueryVectors", {"topK": 1, "queryVector": {"float32": [0, 0, 0]}}, "zero query, cosine"),
        ("/QueryVectors", {"topK": 10001, "queryVector": {"float32": [1, 2, 3]}}, "topK too big"),
        ("/QueryVectors", {"topK": 1, "queryTexts": [{"text": "hi", "metadataKey": "body"}]}, "full-text search"),
        ("/QueryVectors", {"topK": 1, "queryVector": {"float32": [1, 2, 3]}, "filter": {"a": {"$in": []}}}, "empty $in"),
        ("/QueryVectors", {"topK": 1, "queryVector": {"float32": [1, 2, 3]}, "filter": {"a": {"$regex": "x"}}}, "unknown operator"),
        ("/QueryVectors", {"topK": 1, "queryVector": {"float32": [1, 2, 3]}, "filter": {"a": {"$in": list(range(101))}}}, "101 constraints"),
        ("/GetVectors", {"keys": [f"k{i}" for i in range(101)]}, "too many keys"),
    ],
)
def test_invalid_requests_are_validation_errors(s3v, raw, path, body, why):
    bucket = _bucket(s3v)
    index = _index(s3v, bucket)
    r = raw.post(path, json={"vectorBucketName": bucket, "indexName": index, **body})
    assert (r.status_code, r.headers.get("x-amzn-ErrorType")) == (400, "ValidationException"), (why, r.text)
    assert r.json()["message"]


def test_nan_is_rejected(s3v, raw):
    bucket = _bucket(s3v)
    index = _index(s3v, bucket)
    vectors = '[{"key": "a", "data": {"float32": [1, NaN, 3]}}]'
    body = f'{{"vectorBucketName": "{bucket}", "indexName": "{index}", "vectors": {vectors}}}'
    r = raw.post("/PutVectors", content=body, headers={"content-type": "application/json"})
    assert r.status_code == 400 and r.headers["x-amzn-ErrorType"] == "ValidationException"


def test_malformed_bodies(raw):
    for content in (b"{not json", b"[1, 2]"):
        r = raw.post("/GetVectorBucket", content=content, headers={"content-type": "application/json"})
        assert r.status_code == 400 and r.headers["x-amzn-ErrorType"] == "ValidationException"


def test_missing_index_and_bucket_are_not_found(s3v):
    bucket = _bucket(s3v)
    for call in (
        lambda: s3v.put_vectors(vectorBucketName=bucket, indexName="nope", vectors=[{"key": "a", "data": {"float32": [1.0]}}]),
        lambda: s3v.query_vectors(vectorBucketName="no-bucket", indexName="nope", queryVector={"float32": [1.0]}, topK=1),
        lambda: s3v.list_vectors(vectorBucketName=bucket, indexName="nope"),
    ):
        with pytest.raises(ClientError) as e:
            call()
        assert _code(e) == "NotFoundException"


def test_unsupported_operations_say_so(s3v):
    bucket = _bucket(s3v)
    with pytest.raises(ClientError) as e:
        s3v.put_vector_bucket_policy(vectorBucketName=bucket, policy="{}")
    assert _code(e) == "ValidationException" and "not supported" in e.value.response["Error"]["Message"]


# -- storage -------------------------------------------------------------------------


def test_data_is_shared_by_every_server_on_the_bucket(server, s3v):
    _, client = server
    bucket = _bucket(s3v)
    index = _index(s3v, bucket, dim=2, metric="euclidean")
    _put(s3v, bucket, index, [("p", [3.0, 4.0], {"t": "x"})])
    other = S3VectorsService(client, compaction="off")  # another process on the same bucket
    got = other.query_vectors({"vectorBucketName": bucket, "indexName": index, "queryVector": {"float32": [0, 0]}, "topK": 1, "returnDistance": True})
    assert got["vectors"] == [{"key": "p", "distance": 5.0}]


def test_s3vectors_data_is_invisible_to_the_text_api(server, s3v):
    _, client = server
    bucket = _bucket(s3v)
    _index(s3v, bucket)
    text_store = VectorStore(client, FakeEmbedder())
    assert text_store.list_indexes() == ["default"]


@pytest.mark.parametrize("delete_first", [True, False])
def test_bucket_delete_racing_index_create_never_strands_an_index(server, delete_first):
    """Whichever wins, no index is left inside a deleted bucket."""
    from vectorstore.s3vectors import S3VectorsError

    _, client = server
    svc = S3VectorsService(client, compaction="off")
    name = f"race-{'delete' if delete_first else 'create'}-first"
    svc.create_vector_bucket({"vectorBucketName": name})
    store = svc._store(name)
    real = store.create_index

    def create(*a, **kw):
        if delete_first:  # the bucket delete lands between CreateIndex's check and its index
            svc.delete_vector_bucket({"vectorBucketName": name})
        real(*a, **kw)

    store.create_index = create
    req = {"vectorBucketName": name, "indexName": "idx", "dataType": "float32", "dimension": 2, "distanceMetric": "cosine"}
    if delete_first:
        with pytest.raises(S3VectorsError) as e:
            svc.create_index(req)
        assert e.value.code == "NotFoundException"
        assert store.list_indexes() == []  # the index was undone
    else:
        svc.create_index(req)
        with pytest.raises(S3VectorsError) as e:
            svc.delete_vector_bucket({"vectorBucketName": name})
        assert e.value.code == "ConflictException"
        assert svc.get_vector_bucket({"vectorBucketName": name})  # still usable


def test_vectors_survive_compaction_and_a_cold_reload_bit_exact(server, s3v):
    _, client = server
    bucket = _bucket(s3v)
    index = _index(s3v, bucket, dim=5, metric="cosine")
    rng = np.random.default_rng(7)
    vecs = {f"k{i}": rng.normal(size=5).astype(np.float32) for i in range(40)}
    for k, v in vecs.items():  # 40 single writes: several background-free (inline) merges
        _put(s3v, bucket, index, [(k, v.tolist(), {"k": k})])
    s3v.delete_vectors(vectorBucketName=bucket, indexName=index, keys=["k3"])
    cold = S3VectorsService(client, compaction="off")
    store = cold._store(bucket)
    assert store.index_stats(index)["segments"] < 10  # merges happened
    store.compact(index)
    got = cold.get_vectors({"vectorBucketName": bucket, "indexName": index, "keys": list(vecs), "returnData": True})
    assert len(got["vectors"]) == 39
    for v in got["vectors"]:
        assert np.array_equal(np.array(v["data"]["float32"], dtype=np.float32), vecs[v["key"]])
    q = cold.query_vectors({"vectorBucketName": bucket, "indexName": index, "queryVector": {"float32": vecs["k9"].tolist()}, "topK": 1, "returnDistance": True})
    assert q["vectors"][0]["key"] == "k9" and abs(q["vectors"][0]["distance"]) < 1e-6
