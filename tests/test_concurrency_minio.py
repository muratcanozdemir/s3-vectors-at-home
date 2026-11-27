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


def _worker(bucket, worker, start, mode):
    store = _store(bucket)
    start.wait()
    ids = [f"w{worker}-d{i}" for i in range(DOCS_PER_WORKER)]
    if mode == "single":
        for doc_id in ids:
            store.add_document(doc_id, f"text {doc_id}")
    else:  # "batches": several small bulk writes
        for i in range(0, len(ids), 5):
            store.add_documents((d, f"text {d}") for d in ids[i: i + 5])
    # Delete every third doc this worker wrote.
    for doc_id in ids[::3]:
        assert store.delete_document(doc_id)


def _reader(bucket, start, writers_done):
    # Reads racing the writers (including the very first index creation)
    # must never fail or see a malformed index.
    store = _store(bucket)
    start.wait()
    while not writers_done.is_set():
        store.search("text w0-d1", top_k=5)
        store.list_documents(limit=50)
        store.count_documents()


def _expected():
    return {
        f"w{w}-d{i}"
        for w in range(WORKERS)
        for i in range(DOCS_PER_WORKER)
        if i % 3 != 0
    }


@pytest.mark.parametrize("mode", ["single", "batches"])
def test_concurrent_writer_processes_lose_no_updates(mode):
    bucket = f"cas-test-{uuid.uuid4().hex[:10]}"
    ctx = mp.get_context("spawn")
    start, writers_done = ctx.Event(), ctx.Event()
    procs = [ctx.Process(target=_worker, args=(bucket, w, start, mode)) for w in range(WORKERS)]
    readers = [ctx.Process(target=_reader, args=(bucket, start, writers_done)) for _ in range(READERS)]
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
    listed = {d["doc_id"] for d in store.list_documents(limit=1000)}
    assert listed == expected, (
        f"missing from index: {sorted(expected - listed)[:10]}, "
        f"unexpected: {sorted(listed - expected)[:10]}"
    )
    assert store.count_documents() == len(expected)
    for doc_id in sorted(expected)[:10]:
        assert store.search(f"text {doc_id}", top_k=1) == [doc_id]


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
