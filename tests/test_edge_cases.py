"""Edge cases, across the stack: boundaries, odd inputs, and failure modes.

Everything here runs on fakes (no MinIO, no model): the text HTTP API is
exercised through FastAPI's test client over a VectorStore on a fake
client, the S3 Vectors layer through its service object.
"""

import gc
import json
import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.fakes import FakeEmbedder, FakeMinioClient, _s3_error
from vectorstore import core
from vectorstore.core import (
    DEFAULT_INDEX,
    MAX_DOC_ID_LENGTH,
    MAX_PAGE_SIZE,
    MAX_TOP_K,
    IndexCorruptError,
    IndexNotFoundError,
    VectorStore,
    manifest_key,
)
from vectorstore.filters import (
    MAX_FILTER_DEPTH,
    matches,
    validate_filter,
    validate_metadata,
)
from vectorstore.s3vectors import S3VectorsError, S3VectorsService


@pytest.fixture
def store():
    return VectorStore(FakeMinioClient(), FakeEmbedder(), compaction="inline")


@pytest.fixture
def http(store, monkeypatch):
    """The text API over `store` (a fake), via FastAPI's test client."""
    import api.main

    monkeypatch.setattr(core, "_store", store)
    with TestClient(api.main.app) as client:
        yield client


# -- doc ids and texts -----------------------------------------------------------


@pytest.mark.parametrize(
    "doc_id",
    ["ü-ñí-código", "with space", "x" * MAX_DOC_ID_LENGTH, "a?b#c&d=e", "%2F-literal", "  ", "\x00nul", "emoji-🦀", "."],
)
def test_unusual_doc_ids_round_trip_through_storage_and_http(store, http, doc_id):
    store.add_document(doc_id, "some text", {"k": 1})
    fresh = VectorStore(store._client, FakeEmbedder())
    assert fresh.get_document(doc_id)["text"] == "some text"
    from urllib.parse import quote

    r = http.get(f"/document/{quote(doc_id, safe='')}")
    if doc_id == ".":
        return  # "/document/." is normalised to "/document/" by URL rules; storage still works
    if "%" in doc_id:
        # httpx (under the test client) re-normalises "%25" escapes; a real
        # server reached with the raw path serves these ids fine (checked
        # with curl --path-as-is against uvicorn).
        return
    assert r.status_code == 200, r.text
    assert r.json()["doc_id"] == doc_id


@pytest.mark.parametrize("bad", ["", "a/b", "x" * (MAX_DOC_ID_LENGTH + 1), None, 7])
def test_bad_doc_ids_are_rejected(store, bad):
    with pytest.raises(ValueError):
        store.add_document(bad, "t")


@pytest.mark.parametrize("text", ["", " ", "nul \x00 inside", "x" * 200_000, "🦀" * 1000])
def test_unusual_texts_round_trip_exactly(store, text):
    store.add_document("t", text)
    assert VectorStore(store._client, FakeEmbedder()).get_document("t")["text"] == text
    assert store.list_documents()[0]["text_preview"] == text[:64]


@pytest.mark.parametrize(
    "doc",
    [("ok", "lone \ud800 surrogate", None), ("bad\udfff", "t", None), ("ok", "t", {"k": "\ud800"}), ("ok", "t", {"\ud800": 1})],
)
def test_invalid_unicode_is_rejected_up_front(store, doc):
    # JSON can carry lone surrogates; UTF-8 (and so the tokenizer and the
    # stored JSON) can't. They must be a 400, not a 500 from deep inside.
    with pytest.raises(ValueError, match="Unicode"):
        store.add_documents([doc])


def test_invalid_unicode_over_http_is_400(http):
    body = '{"doc_id": "x", "text": "\\ud800"}'
    assert http.post("/upload", content=body, headers={"content-type": "application/json"}).status_code == 400


def test_s3v_invalid_unicode_is_a_validation_error(svc):
    t = _mk(svc)
    for vec in ({"key": "\ud800", "data": {"float32": [1, 2, 3]}}, {"key": "k", "data": {"float32": [1, 2, 3]}, "metadata": {"m": "\udfff"}}):
        assert _err(svc.put_vectors, {**t, "vectors": [vec]}) == "ValidationException"


def test_text_must_be_a_string(store):
    with pytest.raises(ValueError):
        store.add_document("t", b"bytes")


# -- metadata ----------------------------------------------------------------------


def test_metadata_types_survive_storage_exactly(store):
    md = {"int": 2**53 + 1, "neg_zero": -0.0, "float": 0.1, "big": 1e308, "t": True, "f": False, "s": "ü", "empty": [], "mixed": [1, "a", True]}
    store.add_document("m", "t", md)
    got = VectorStore(store._client, FakeEmbedder()).get_document("m")["metadata"]
    assert got == md
    assert type(got["int"]) is int and got["int"] == 2**53 + 1  # no float rounding
    assert type(got["t"]) is bool


@pytest.mark.parametrize(
    "md",
    [
        {"x": float("nan")},
        {"x": float("inf")},
        {"x": {"nested": 1}},
        {"x": [[1]]},
        {"x": None},
        {"$x": 1},
        {"": 1},
        {"k" * 65: 1},
        {f"k{i}": i for i in range(33)},
        {"x": list(range(65))},
        {"x": "y" * 5000},
        [1, 2],
        "string",
    ],
)
def test_bad_metadata_is_rejected(md):
    with pytest.raises(ValueError):
        validate_metadata(md)


def test_none_and_empty_metadata_are_the_same(store):
    store.add_documents([("a", "t", None), ("b", "t", {})])
    assert store.get_document("a")["metadata"] == store.get_document("b")["metadata"] == {}


# -- filter semantics ----------------------------------------------------------------


@pytest.mark.parametrize(
    "flt,md,expected",
    [
        ({"flag": 1}, {"flag": True}, False),  # booleans are not numbers
        ({"flag": True}, {"flag": 1}, False),
        ({"n": 1}, {"n": 1.0}, True),  # 1 == 1.0
        ({"n": {"$gt": 0}}, {"n": True}, False),  # range ops skip booleans
        ({"n": {"$gt": 0}}, {"n": "5"}, False),  # and strings
        ({"tags": "a"}, {"tags": []}, False),
        ({"tags": {"$ne": "a"}}, {"tags": []}, True),
        ({"tags": {"$exists": True}}, {"tags": []}, True),
        ({"missing": {"$ne": 1}}, {}, True),
        ({"missing": {"$nin": [1]}}, {}, True),
        ({"missing": {"$exists": False}}, {}, True),
        ({"missing": {"$in": [1]}}, {}, False),
        ({"tags": {"$nin": ["a"]}}, {"tags": ["b", "a"]}, False),  # any element hits
        ({"n": {"$gte": 1, "$lte": 1}}, {"n": 1}, True),
        ({"n": {"$gt": 1, "$lt": 1}}, {"n": 1}, False),
        ({"$and": [{"a": 1}, {"$or": [{"b": 2}, {"c": 3}]}]}, {"a": 1, "c": 3}, True),
        ({"$or": [{"a": 2}]}, {"a": 1}, False),
        ({"a": 1, "b": 2}, {"a": 1}, False),  # several fields: AND
        ({"s": {"$startsWith": ""}}, {"s": ""}, True),
        ({"s": {"$in": ["x", 1]}}, {"s": 1}, True),
    ],
)
def test_filter_semantics(flt, md, expected):
    validate_filter(flt)
    assert matches(flt, md) is expected


@pytest.mark.parametrize(
    "flt",
    [
        {"$and": []},
        {"$or": {}},
        {"$nor": [{"a": 1}]},
        {"a": {}},
        {"a": [1, 2]},
        {"a": {"$gt": "x"}},
        {"a": {"$gt": float("nan")}},
        {"a": {"$in": "x"}},
        {"a": {"$exists": 1}},
        {"a": {"$eq": {"b": 1}}},
        [],
        "a",
    ],
)
def test_malformed_filters_are_rejected(flt):
    with pytest.raises(ValueError):
        validate_filter(flt)


def test_filter_depth_limit():
    flt = {"a": 1}
    for _ in range(MAX_FILTER_DEPTH):
        flt = {"$and": [flt]}
    validate_filter(flt)
    with pytest.raises(ValueError):
        validate_filter({"$and": [flt]})


# -- search -------------------------------------------------------------------------


def test_search_top_k_boundaries(store):
    store.add_documents([(f"d{i}", f"text {i}") for i in range(5)])
    assert len(store.search("text 1", top_k=MAX_TOP_K)) == 5
    for bad in (0, -1, MAX_TOP_K + 1):
        with pytest.raises(ValueError):
            store.search("x", top_k=bad)


def test_filtered_search_returns_the_best_matching_not_the_best_overall(store):
    store.add_documents([(f"d{i}", f"text {i}", {"even": i % 2 == 0}) for i in range(40)])
    hits = store.search_with_scores("text 7", top_k=3, filter={"even": True})
    assert len(hits) == 3 and all(h["metadata"]["even"] for h in hits)
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_search_with_identical_vectors_returns_each_once(store):
    store.add_documents([(f"dup{i}", "same text") for i in range(10)])
    hits = store.search("same text", top_k=10)
    assert sorted(hits) == [f"dup{i}" for i in range(10)]


def test_scores_are_within_cosine_range(store):
    store.add_documents([(f"d{i}", f"text {i}") for i in range(30)])
    for h in store.search_with_scores("anything", top_k=30):
        assert -1.0 - 1e-5 <= h["score"] <= 1.0 + 1e-5


# -- listing and paging ---------------------------------------------------------------


def test_paging_edges(store):
    store.add_documents([(f"d{i:02d}", "t") for i in range(12)])
    assert store.list_documents(skip=12) == []
    assert store.list_documents(skip=10**9) == []
    assert store.list_documents(limit=0) == []
    assert len(store.list_documents(limit=MAX_PAGE_SIZE)) == 12
    pages = [d["doc_id"] for s in range(0, 12, 5) for d in store.list_documents(skip=s, limit=5)]
    assert pages == sorted(pages) and len(set(pages)) == 12


def test_list_order_is_by_code_point(store):
    ids = ["b", "B", "a", "ä", "_", "10", "9"]
    store.add_documents([(d, "t") for d in ids])
    assert [d["doc_id"] for d in store.list_documents()] == sorted(ids)


# -- bulk uploads ----------------------------------------------------------------------


def test_bulk_upload_of_many_docs_in_one_batch(store):
    n = 5000
    assert store.add_documents([(f"d{i}", f"text {i}") for i in range(n)]) == n
    assert store.count_documents() == n
    assert store.index_stats()["segments"] == 1


def test_bulk_upload_duplicates_count_once_and_last_wins(store):
    assert store.add_documents([("x", "first"), ("y", "y"), ("x", "last")]) == 2
    assert store.get_document("x")["text"] == "last"


def test_empty_bulk_upload_writes_nothing(store):
    store.count_documents()
    before = dict(store._client.objects["vectors"])
    assert store.add_documents([]) == 0
    assert store._client.objects["vectors"] == before


# -- concurrency inside one process ------------------------------------------------------


def test_threads_reading_and_writing_one_store(store):
    errors = []

    def writer(w):
        try:
            for i in range(30):
                store.add_document(f"w{w}-{i}", f"text {w} {i}", {"w": w})
                if i % 3 == 0:
                    store.delete_document(f"w{w}-{i}")
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    def reader():
        try:
            for _ in range(60):
                store.search("text", top_k=5, filter={"w": {"$in": [0, 1]}})
                store.list_documents(limit=10)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(w,)) for w in range(4)] + [threading.Thread(target=reader) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    expected = {f"w{w}-{i}" for w in range(4) for i in range(30) if i % 3}
    assert {d["doc_id"] for d in VectorStore(store._client, FakeEmbedder()).list_documents(limit=1000)} == expected


def test_background_compaction_racing_writes_in_one_process():
    client = FakeMinioClient()
    store = VectorStore(client, FakeEmbedder(), compaction="background")
    threads = [threading.Thread(target=lambda w=w: [store.add_document(f"{w}-{i}", "t") for i in range(40)]) for w in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert store.wait_for_compaction(timeout=30)
    fresh = VectorStore(client, FakeEmbedder())
    assert fresh.count_documents() == 160
    referenced = {e["key"] for e in json.loads(client.objects["vectors"][manifest_key(DEFAULT_INDEX)])["segments"]}
    assert {k for k in client.objects["vectors"] if k.startswith("segments/")} == referenced


# -- storage failures -------------------------------------------------------------------


@pytest.mark.parametrize("damage", ["truncate", "garbage", "wrong-magic", "bad-json-header"])
def test_corrupt_segment_is_a_server_error_not_bad_request(store, http, damage):
    store.add_document("x", "x")
    seg = next(k for k in store._client.objects["vectors"] if k.endswith(".seg"))
    blob = store._client.objects["vectors"][seg]
    store._client.objects["vectors"][seg] = {
        "truncate": blob[: len(blob) // 2],
        "garbage": b"\x00" * 64,
        "wrong-magic": b"NOTASEG!!!" + blob[10:],
        "bad-json-header": blob[:18] + b"{{{{" + blob[22:],
    }[damage]
    fresh = VectorStore(store._client, FakeEmbedder())
    with pytest.raises(IndexCorruptError):
        fresh.count_documents()
    core._store = fresh
    r = http.get("/status")
    assert r.status_code == 500 and "unreadable" in r.json()["detail"]


def test_corrupt_manifest_is_a_server_error(store, http):
    store.add_document("x", "x")
    store._client.objects["vectors"][manifest_key(DEFAULT_INDEX)] = b'{"format": 99}'
    core._store = VectorStore(store._client, FakeEmbedder())
    assert http.get("/documents").status_code == 500


def test_object_store_outage_propagates_and_recovers(store):
    store.add_document("x", "x")
    real_stat = store._client.stat_object
    store._client.stat_object = lambda b, k: (_ for _ in ()).throw(ConnectionError("down"))
    with pytest.raises(ConnectionError):
        store.search("x", top_k=1)
    store._client.stat_object = real_stat
    assert store.search("x", top_k=1) == ["x"]


def test_server_side_error_on_put_is_not_mistaken_for_a_conflict(store):
    store.add_document("x", "x")
    real = store._client._put_object

    def slow_down(bucket, key, data, headers):
        if key.endswith(".manifest.json"):
            raise _s3_error("SlowDown", bucket, key)
        return real(bucket, key, data, headers)

    store._client._put_object = slow_down
    with pytest.raises(Exception) as e:
        store.add_document("y", "y")
    assert "SlowDown" in str(e.value)
    store._client._put_object = real
    assert store.get_document("y") is None


# -- indexes ------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["a", "a" * 63, "0", "a-b", "9lives"])
def test_valid_index_names(store, name):
    store.create_index(name)
    assert name in store.list_indexes()


@pytest.mark.parametrize("name", ["", "a" * 64, "-a", "a-", "A", "a_b", "a.b", "a b", "ä", "default"])
def test_invalid_or_reserved_index_names(store, name):
    with pytest.raises((ValueError, core.IndexAlreadyExistsError)):
        store.create_index(name)


def test_operations_on_a_missing_index(store):
    for op in (
        lambda: store.add_document("x", "t", index="nope"),
        lambda: store.search("x", index="nope"),
        lambda: store.delete_document("x", index="nope"),
        lambda: store.compact(index="nope"),
        lambda: store.index_stats(index="nope"),
        lambda: store.delete_index("nope"),
    ):
        with pytest.raises(IndexNotFoundError):
            op()
    assert "nope" not in store.list_indexes()


def test_http_status_codes_for_index_errors(http):
    assert http.get("/indexes/nope/status").status_code == 404
    assert http.post("/indexes", json={"name": "default"}).status_code == 409
    assert http.post("/indexes", json={"name": "Bad"}).status_code == 400
    assert http.delete("/indexes/default").status_code == 400
    assert http.post("/search", json={"query": "x", "top_k": 0}).status_code == 422
    assert http.get("/documents", params={"limit": MAX_PAGE_SIZE + 1}).status_code == 422
    assert http.post("/upload", json={"doc_id": "a/b", "text": "t"}).status_code in (400, 404)
    assert http.post("/upload", json={"doc_id": "x", "text": 5}).status_code == 422


# -- the GC pause used by rebuilds ---------------------------------------------------------


def test_gc_pause_restores_state_even_on_error():
    assert gc.isenabled()
    with pytest.raises(RuntimeError), core._gc_paused():
        assert not gc.isenabled()
        with core._gc_paused():  # nested
            pass
        assert not gc.isenabled()
        raise RuntimeError
    assert gc.isenabled()
    gc.disable()
    try:
        with core._gc_paused():
            pass
        assert not gc.isenabled()  # was off before: stays off
    finally:
        gc.enable()


# -- S3 Vectors: boundaries and isolation ----------------------------------------------------


@pytest.fixture
def svc():
    return S3VectorsService(FakeMinioClient(), compaction="inline")


def _mk(svc, bucket="bkt", index="idx", dim=3, metric="cosine", **cfg):
    svc.create_vector_bucket({"vectorBucketName": bucket})
    svc.create_index({"vectorBucketName": bucket, "indexName": index, "dataType": "float32", "dimension": dim, "distanceMetric": metric, **cfg})
    return {"vectorBucketName": bucket, "indexName": index}


def _err(fn, *a) -> str:
    with pytest.raises(S3VectorsError) as e:
        fn(*a)
    return e.value.code


def test_s3v_batch_and_size_limits(svc):
    t = _mk(svc)
    v = lambda k: {"key": k, "data": {"float32": [1, 2, 3]}}
    svc.put_vectors({**t, "vectors": [v(f"k{i}") for i in range(500)]})
    assert _err(svc.put_vectors, {**t, "vectors": [v(f"k{i}") for i in range(501)]}) == "ValidationException"
    svc.put_vectors({**t, "vectors": [v("x" * 1024)]})
    assert _err(svc.put_vectors, {**t, "vectors": [v("x" * 1025)]}) == "ValidationException"
    assert len(svc.get_vectors({**t, "keys": [f"k{i}" for i in range(100)]})["vectors"]) == 100
    svc.delete_vectors({**t, "keys": [f"k{i}" for i in range(500)]})
    assert len(svc.list_vectors({**t, "maxResults": 1000})["vectors"]) == 1


def test_s3v_dimension_bounds(svc):
    svc.create_vector_bucket({"vectorBucketName": "dims"})
    base = {"vectorBucketName": "dims", "dataType": "float32", "distanceMetric": "euclidean"}
    svc.create_index({**base, "indexName": "one", "dimension": 1})
    svc.create_index({**base, "indexName": "max", "dimension": 4096})
    svc.put_vectors({"vectorBucketName": "dims", "indexName": "max", "vectors": [{"key": "k", "data": {"float32": [0.5] * 4096}}]})
    for bad in (0, 4097, 2.5, "3", True):
        assert _err(svc.create_index, {**base, "indexName": "bad", "dimension": bad}) == "ValidationException"
    q = svc.query_vectors({"vectorBucketName": "dims", "indexName": "one", "queryVector": {"float32": [1.0]}, "topK": 10})
    assert q["vectors"] == []  # empty index


def test_s3v_metadata_limits_exactly_at_the_edge(svc):
    t = _mk(svc, metadataConfiguration={"nonFilterableMetadataKeys": ["big"]})
    ok = {f"k{i}": i for i in range(49)} | {"big": "x" * 90_000}  # 50 keys; big is non-filterable
    svc.put_vectors({**t, "vectors": [{"key": "a", "data": {"float32": [1, 2, 3]}, "metadata": ok}]})
    too_much_filterable = {"a": "x" * 2048, "b": ["y" * 2000] * 20}
    assert _err(svc.put_vectors, {**t, "vectors": [{"key": "b", "data": {"float32": [1, 2, 3]}, "metadata": too_much_filterable}]}) == "ValidationException"
    assert _err(svc.put_vectors, {**t, "vectors": [{"key": "c", "data": {"float32": [1, 2, 3]}, "metadata": {"big": "x" * 110_000}}]}) == "ValidationException"
    assert _err(svc.put_vectors, {**t, "vectors": [{"key": "d", "data": {"float32": [1, 2, 3]}, "metadata": {"k" * 64: 1}}]}) == "ValidationException"


def test_s3v_same_index_name_in_two_buckets_is_isolated(svc):
    a, b = _mk(svc, "one"), _mk(svc, "two")
    svc.put_vectors({**a, "vectors": [{"key": "k", "data": {"float32": [1, 0, 0]}}]})
    assert svc.get_vectors({**b, "keys": ["k"]})["vectors"] == []
    assert svc.list_indexes({"vectorBucketName": "two"})["indexes"][0]["indexName"] == "idx"


def test_s3v_bucket_named_like_internal_prefixes(svc):
    for name in ("buckets", "indexes", "segments", "s3vectors"):
        t = _mk(svc, name, "x.json")
        svc.put_vectors({**t, "vectors": [{"key": "k", "data": {"float32": [1, 2, 3]}}]})
    names = [b["vectorBucketName"] for b in svc.list_vector_buckets({})["vectorBuckets"]]
    assert names == ["buckets", "indexes", "s3vectors", "segments"]
    for name in names:
        assert len(svc.get_vectors({"vectorBucketName": name, "indexName": "x.json", "keys": ["k"]})["vectors"]) == 1


def test_s3v_many_buckets_share_threads(svc):
    before = threading.active_count()
    for i in range(30):
        t = _mk(svc, f"b{i:02d}")
        svc.put_vectors({**t, "vectors": [{"key": "k", "data": {"float32": [1, 2, 3]}}]})
    assert threading.active_count() - before <= core.DEFAULT_IO_WORKERS + 2


def test_s3v_arn_mismatches_are_rejected(svc):
    _mk(svc)
    arn = svc.index_arn("bkt", "idx")
    assert _err(svc.get_index, {"indexArn": arn, "indexName": "other"}) == "ValidationException"
    assert _err(svc.get_index, {"indexArn": svc.bucket_arn("bkt")}) == "ValidationException"
    assert _err(svc.get_vector_bucket, {"vectorBucketArn": arn}) == "ValidationException"
    assert _err(svc.get_vector_bucket, {"vectorBucketArn": "arn:aws:s3:::bkt"}) == "ValidationException"
    assert _err(svc.get_index, {}) == "ValidationException"
    assert svc.get_index({"indexArn": arn})["index"]["indexName"] == "idx"


def test_s3v_tokens_are_validated(svc):
    t = _mk(svc)
    svc.put_vectors({**t, "vectors": [{"key": f"k{i}", "data": {"float32": [1, i + 1, 2]}} for i in range(5)]})
    for token in ("not-base64!!", "aGVsbG8=", "WzFd"):  # junk, non-JSON, a JSON list
        assert _err(svc.list_vectors, {**t, "nextToken": token}) == "ValidationException"
    page = svc.list_vectors({**t, "maxResults": 2, "segmentCount": 1, "segmentIndex": 0})
    assert _err(svc.list_vectors, {**t, "nextToken": page["nextToken"], "segmentCount": 2, "segmentIndex": 1}) == "ValidationException"


def test_s3v_type_confusion_in_requests(svc):
    t = _mk(svc)
    for bad in (
        {"vectors": "nope"},
        {"vectors": [None]},
        {"vectors": [{"key": 5, "data": {"float32": [1, 2, 3]}}]},
        {"vectors": [{"key": "k", "data": [1, 2, 3]}]},
        {"vectors": [{"key": "k", "data": {"float32": [True, 2, 3]}}]},
        {"vectors": [{"key": "k", "data": {"float32": []}}]},
    ):
        assert _err(svc.put_vectors, {**t, **bad}) == "ValidationException", bad
    for bad in ({"returnData": "yes", "keys": ["k"]}, {"keys": "k"}, {"keys": [1]}, {"keys": []}):
        assert _err(svc.get_vectors, {**t, **bad}) == "ValidationException", bad
    assert _err(svc.query_vectors, {**t, "topK": "5", "queryVector": {"float32": [1, 2, 3]}}) == "ValidationException"
    assert _err(svc.query_vectors, {**t, "topK": 1, "queryVector": {"float32": [1, 2, 3]}, "queryMode": "FAST"}) == "ValidationException"


def test_s3v_query_ties_and_more_top_k_than_vectors(svc):
    t = _mk(svc, metric="euclidean")
    svc.put_vectors({**t, "vectors": [{"key": f"same{i}", "data": {"float32": [1, 1, 1]}} for i in range(4)]})
    r = svc.query_vectors({**t, "queryVector": {"float32": [1, 1, 1]}, "topK": 10000, "returnDistance": True})
    assert sorted(v["key"] for v in r["vectors"]) == [f"same{i}" for i in range(4)]
    assert all(v["distance"] == 0.0 for v in r["vectors"]) and "nextToken" not in r


def test_s3v_cosine_is_scale_invariant(svc):
    t = _mk(svc)
    svc.put_vectors({**t, "vectors": [{"key": "small", "data": {"float32": [1e-20, 2e-20, 3e-20]}}, {"key": "big", "data": {"float32": [1e20, 2e20, 3e20]}}]})
    r = svc.query_vectors({**t, "queryVector": {"float32": [2, 4, 6]}, "topK": 2, "returnDistance": True})
    assert {v["key"] for v in r["vectors"]} == {"small", "big"}
    assert all(abs(v["distance"]) < 1e-6 for v in r["vectors"])


def test_s3v_vectors_overflowing_float32_are_rejected(svc):
    t = _mk(svc)
    # Finite as JSON numbers, but beyond float32: they'd become infinity.
    assert _err(svc.put_vectors, {**t, "vectors": [{"key": "k", "data": {"float32": [1e39, 1, 1]}}]}) == "ValidationException"


def test_s3v_list_vectors_is_stable_while_writes_happen(svc):
    t = _mk(svc)
    svc.put_vectors({**t, "vectors": [{"key": f"k{i:02d}", "data": {"float32": [1, i + 1, 2]}} for i in range(20)]})
    first = svc.list_vectors({**t, "maxResults": 5})
    svc.put_vectors({**t, "vectors": [{"key": "k00a", "data": {"float32": [1, 1, 1]}}]})  # sorts before the cursor
    svc.delete_vectors({**t, "keys": ["k06"]})
    seen = [v["key"] for v in first["vectors"]]
    token = first["nextToken"]
    while token:
        page = svc.list_vectors({**t, "maxResults": 5, "nextToken": token})
        seen += [v["key"] for v in page["vectors"]]
        token = page.get("nextToken")
    # The cursor is a key, so nothing after it is skipped or repeated.
    assert seen == sorted(set(seen)) and "k19" in seen and "k06" not in seen


def test_s3v_index_info_does_not_load_the_index(svc):
    t = _mk(svc)
    svc.put_vectors({**t, "vectors": [{"key": "k", "data": {"float32": [1, 2, 3]}}]})
    cold = S3VectorsService(svc._client, compaction="off")
    before = {k: v for k, v in svc._client.get_counts.items() if k.endswith(".seg")}
    cold.get_index(t)
    cold.list_indexes({"vectorBucketName": "bkt"})
    after = {k: v for k, v in svc._client.get_counts.items() if k.endswith(".seg")}
    assert before == after  # only manifests were read


def test_np_float32_round_trip_of_awkward_values(svc):
    t = _mk(svc, metric="euclidean")
    values = [np.nextafter(np.float32(1), np.float32(2)), np.float32(1.17549435e-38), np.float32(-3.4028235e38)]
    svc.put_vectors({**t, "vectors": [{"key": "k", "data": {"float32": [float(v) for v in values]}}]})
    got = svc.get_vectors({**t, "keys": ["k"], "returnData": True})["vectors"][0]["data"]["float32"]
    assert [np.float32(x) for x in got] == values


# -- CLI, in-process over a fake store ------------------------------------------------------


@pytest.fixture
def cli(store, monkeypatch, capsys):
    from s3_vectors_at_home import cli as cli_mod

    monkeypatch.setattr(core, "_store", store)

    def run(*argv):
        monkeypatch.setattr("sys.argv", ["s3-vectors-cli", *argv])
        code = 0
        try:
            cli_mod.main()
        except SystemExit as e:
            code = e.code or 0
        out = capsys.readouterr()
        return code, out.out, out.err

    return run


def test_cli_rejects_bad_json_arguments_cleanly(cli):
    for argv in (
        ["upload", "--doc-id", "a", "--text", "t", "--metadata", "{not json"],
        ["upload", "--doc-id", "a", "--text", "t", "--metadata", '{"x": {"nested": 1}}'],
        ["search", "--query", "q", "--filter", "{"],
        ["search", "--query", "q", "--filter", '{"a": {"$regex": "x"}}'],
    ):
        code, _, err = cli(*argv)
        assert code == 1 and err.startswith("Error:") and "Traceback" not in err, argv


@pytest.mark.parametrize(
    "content,why",
    [
        ("{not json", "invalid JSON"),
        ('{"doc_id": "a", "text": "t"}', "an object, not a list"),
        ('[{"doc_id": "a"}]', "missing text"),
        ('[{"doc_id": "a", "text": "t"}, {"doc_id": "b/c", "text": "t"}]', "one bad id"),
        (b"\xff\xfe not utf-8", "not UTF-8"),
    ],
)
def test_cli_bulk_upload_bad_files_upload_nothing(cli, store, tmp_path, content, why):
    f = tmp_path / "docs.json"
    f.write_bytes(content if isinstance(content, bytes) else content.encode())
    code, _, err = cli("bulk-upload", "--file", str(f))
    assert code == 1 and err.startswith("Error:") and "Traceback" not in err, why
    assert store.count_documents() == 0


def test_cli_missing_file_is_a_clean_error(cli, tmp_path):
    code, _, err = cli("bulk-upload", "--file", str(tmp_path / "nope.json"))
    assert code == 1 and err.startswith("Error:") and "nope.json" in err


def test_cli_corrupt_storage_is_a_clean_error(cli, store):
    store.add_document("x", "x")
    store._client.objects["vectors"][manifest_key(DEFAULT_INDEX)] = b"garbage"
    core._store = VectorStore(store._client, FakeEmbedder())
    code, _, err = cli("list")
    assert code == 1 and err.startswith("Error:") and "unreadable" in err


def test_cli_round_trip_and_exit_codes(cli):
    assert cli("upload", "--doc-id", "a", "--text", "hello", "--metadata", '{"k": [1, 2]}')[0] == 0
    code, out, _ = cli("get", "--doc-id", "a")
    assert code == 0 and json.loads(out)["metadata"] == {"k": [1, 2]}
    assert cli("get", "--doc-id", "missing")[0] == 1
    assert cli("delete", "--doc-id", "missing")[0] == 1
    assert cli("list", "--index", "nope")[0] == 1
    code, out, _ = cli("search", "--query", "hello", "--scores")
    assert code == 0 and out.split()[1] == "a"
