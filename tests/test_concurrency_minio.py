"""Multi-process writers against a real MinIO.

Each worker process builds its own VectorStore over the same bucket (so
nothing but the object store coordinates them -- exactly the situation of
several uvicorn workers, or the API plus a CLI run) and hammers it with
adds and deletes. Afterwards the index must reflect every acknowledged
operation exactly: no lost updates, no resurrected deletes.

Uses the fake embedder so the workers start in well under a second and
contend hard, instead of spending their time loading a model.
"""

import multiprocessing as mp
import os
import uuid

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("MINIO_ENDPOINT"), reason="needs a real MinIO (MINIO_ENDPOINT)"
)

# Overridable for heavier local stress runs, e.g. CAS_WORKERS=16.
WORKERS = int(os.environ.get("CAS_WORKERS", "6"))
DOCS_PER_WORKER = int(os.environ.get("CAS_DOCS_PER_WORKER", "25"))
READERS = 2


def _store(bucket):
    from minio import Minio
    from tests.fakes import FakeEmbedder
    from vectorstore.core import VectorStore

    client = Minio(
        os.environ["MINIO_ENDPOINT"],
        access_key=os.environ["MINIO_ACCESS_KEY"],
        secret_key=os.environ["MINIO_SECRET_KEY"],
        secure=os.environ.get("MINIO_SECURE", "false").lower() == "true",
    )
    return VectorStore(client, FakeEmbedder(), bucket=bucket)


def _worker(bucket, worker, start, mode, index="default"):
    store = _store(bucket)
    start.wait()
    ids = [f"w{worker}-d{i}" for i in range(DOCS_PER_WORKER)]
    if mode == "single":
        for doc_id in ids:
            store.add_document(doc_id, f"text {doc_id}", {"worker": worker}, index=index)
    else:  # "batches": several small bulk writes
        for i in range(0, len(ids), 5):
            store.add_documents(((d, f"text {d}", {"worker": worker}) for d in ids[i: i + 5]), index=index)
    # Delete every third doc this worker wrote.
    for doc_id in ids[::3]:
        assert store.delete_document(doc_id, index=index)


def _reader(bucket, start, writers_done, index="default"):
    # Reads racing the writers (including the very first index creation)
    # must never fail or see a malformed index.
    store = _store(bucket)
    start.wait()
    while not writers_done.is_set():
        store.search("text w0-d1", top_k=5, index=index, filter={"worker": {"$in": [0, 1]}})
        store.list_documents(limit=50, index=index)
        store.count_documents(index=index)


def _expected():
    return {
        f"w{w}-d{i}"
        for w in range(WORKERS)
        for i in range(DOCS_PER_WORKER)
        if i % 3 != 0
    }


@pytest.mark.parametrize("mode,index", [("single", "default"), ("batches", "default"), ("single", "named")])
def test_concurrent_writer_processes_lose_no_updates(mode, index):
    bucket = f"cas-test-{uuid.uuid4().hex[:10]}"
    if index != "default":
        _store(bucket).create_index(index)
    ctx = mp.get_context("spawn")
    start, writers_done = ctx.Event(), ctx.Event()
    procs = [ctx.Process(target=_worker, args=(bucket, w, start, mode, index)) for w in range(WORKERS)]
    readers = [ctx.Process(target=_reader, args=(bucket, start, writers_done, index)) for _ in range(READERS)]
    for p in procs + readers:
        p.start()
    start.set()
    for p in procs:
        p.join(timeout=240)
    writers_done.set()
    for p in readers:
        p.join(timeout=60)
    assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]
    assert all(p.exitcode == 0 for p in readers), [p.exitcode for p in readers]

    # A fresh store (empty cache) sees exactly the surviving docs.
    store = _store(bucket)
    expected = _expected()
    listed = {d["doc_id"] for d in store.list_documents(limit=1000, index=index)}
    assert listed == expected, (
        f"missing from index: {sorted(expected - listed)[:10]}, "
        f"unexpected: {sorted(listed - expected)[:10]}"
    )
    assert store.count_documents(index=index) == len(expected)
    for doc_id in sorted(expected)[:10]:
        assert store.search(f"text {doc_id}", top_k=1, index=index) == [doc_id]
        worker = int(doc_id[1: doc_id.index("-")])
        assert store.get_document(doc_id, index=index)["metadata"] == {"worker": worker}
    _assert_no_orphan_segments(store, bucket, index)


def _assert_no_orphan_segments(store, bucket, index):
    # Every writer and background merge has finished (processes joined), so
    # every segment in the bucket must be one the manifest references:
    # failed appends and lost merges clean up after themselves.
    import json

    from vectorstore.core import manifest_key

    manifest = json.loads(store._get_bytes(manifest_key(index)))
    referenced = {e["key"] for e in manifest["segments"]}
    stored = {o.object_name for o in store._client.list_objects(bucket, prefix=f"segments/{index}/", recursive=True)}
    assert stored == referenced, f"orphans: {sorted(stored - referenced)[:5]}, missing: {sorted(referenced - stored)[:5]}"
    assert len(referenced) <= 64


def test_conditional_put_beyond_multipart_threshold_is_enforced():
    # minio-py's public put_object switches to multipart above 5 MiB; the
    # index PUT must stay a single conditional request at any size. 12 MiB
    # is a ~8k-document index at 384 dims.
    from vectorstore.core import _WriteConflict

    store = _store(f"cas-big-{uuid.uuid4().hex[:10]}")
    big = os.urandom(12 * 1024 * 1024)
    etag = store._put_conditional("big.bin", big, if_match=None)
    with pytest.raises(_WriteConflict):
        store._put_conditional("big.bin", big, if_match=None)
    with pytest.raises(_WriteConflict):
        store._put_conditional("big.bin", big, if_match="0" * 32)
    new_etag = store._put_conditional("big.bin", big[::-1], if_match=etag)
    assert new_etag != etag
    with pytest.raises(_WriteConflict):
        store._put_conditional("big.bin", big, if_match=etag)  # etag now stale


def test_probe_rejects_real_minio_when_conditions_are_dropped(monkeypatch):
    # The startup probe is the guard against a minio-py change that stops
    # sending our If-Match/If-None-Match headers: simulate exactly that
    # against the real server and check the store refuses to start.
    from minio import Minio
    from vectorstore import core

    real = Minio._put_object

    def strip_conditions(self, bucket, key, data, headers=None, query_params=None):
        headers = {k: v for k, v in (headers or {}).items() if k not in ("If-Match", "If-None-Match")}
        return real(self, bucket, key, data, headers, query_params)

    monkeypatch.setattr(Minio, "_put_object", strip_conditions)
    with pytest.raises(core.ConditionalWritesUnsupportedError):
        _store(f"cas-probe-{uuid.uuid4().hex[:10]}")


SHARED_DOCS = 5
ROUNDS = 12


def _same_doc_worker(bucket, worker, start):
    store = _store(bucket)
    start.wait()
    for r in range(ROUNDS):
        for i in range(SHARED_DOCS):
            doc_id = f"shared-{i}"
            if (worker + r + i) % 4 == 0:
                store.delete_document(doc_id)
            else:
                store.add_document(doc_id, f"{doc_id} by w{worker} r{r}", {"by": worker, "round": r})


def test_concurrent_writers_on_same_doc_ids_stay_consistent():
    # Every process overwrites and deletes the same few doc_ids. Whatever
    # the final winner, each surviving doc's text, metadata and vector must
    # all come from the same write.
    bucket = f"cas-same-{uuid.uuid4().hex[:10]}"
    ctx = mp.get_context("spawn")
    start = ctx.Event()
    procs = [ctx.Process(target=_same_doc_worker, args=(bucket, w, start)) for w in range(WORKERS)]
    for p in procs:
        p.start()
    start.set()
    for p in procs:
        p.join(timeout=240)
    assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]

    store = _store(bucket)
    for d in store.list_documents(limit=100):
        doc = store.get_document(d["doc_id"])
        by, rnd = doc["metadata"]["by"], doc["metadata"]["round"]
        assert doc["text"] == f"{d['doc_id']} by w{by} r{rnd}"  # text and metadata from one write
        top = store.search_with_scores(doc["text"], top_k=1)[0]
        assert top["doc_id"] == d["doc_id"] and top["score"] > 0.9999  # ...and so is the vector
