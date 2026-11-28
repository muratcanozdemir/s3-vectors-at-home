"""Randomised checks against simple models: what the store returns must
match a dict plus numpy brute force, whatever the interleaving of writes,
merges and injected write races."""

import random

import numpy as np
import pytest

from tests.fakes import FakeMinioClient, _s3_error
from vectorstore import core
from vectorstore.s3vectors import S3VectorsService


def _flaky_manifests(client, rng, rate=0.15):
    """Make manifest PUTs randomly fail as conflicts, half of them after
    landing (a lost response) -- both must be invisible to callers."""
    real = client._put_object

    def put(bucket, key, data, headers):
        if key.endswith(".manifest.json") and rng.random() < rate:
            if rng.random() < 0.5:
                raise _s3_error("PreconditionFailed", bucket, key)
            real(bucket, key, data, headers)
            raise _s3_error("PreconditionFailed", bucket, key)
        return real(bucket, key, data, headers)

    client._put_object = put
    return real


def _brute_force(model, q, metric, k, flt=None):
    keys = [key for key, (_, md) in model.items() if flt is None or flt(md)]
    if not keys:
        return []
    m = np.array([model[key][0] for key in keys], dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if metric == "cosine":
        d = 1 - (m @ q) / (np.linalg.norm(m, axis=1) * np.linalg.norm(q))
    else:
        d = np.linalg.norm(m - q, axis=1)
    order = np.argsort(d, kind="stable")[:k]
    return [(keys[i], float(d[i])) for i in order]


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("metric", ["cosine", "euclidean"])
def test_s3vectors_random_operations_match_numpy(seed, metric, monkeypatch):
    monkeypatch.setattr(core, "INDEX_WRITE_BACKOFF_BASE_S", 0)
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    client = FakeMinioClient()
    services = [S3VectorsService(client, compaction=rng.choice(["inline", "off"])) for _ in range(3)]
    t = {"vectorBucketName": "prop", "indexName": "idx"}
    services[0].create_vector_bucket({"vectorBucketName": "prop"})
    services[0].create_index({**t, "dataType": "float32", "dimension": 6, "distanceMetric": metric})
    real = _flaky_manifests(client, rng)
    model: dict[str, tuple[np.ndarray, dict]] = {}
    keys = [f"k{i}" for i in range(25)]
    for step in range(120):
        svc = rng.choice(services)
        op = rng.random()
        if op < 0.45:
            batch = {}
            for key in rng.sample(keys, rng.randint(1, 5)):
                vec = nrng.normal(size=6).astype(np.float32)
                batch[key] = (vec, {"step": step, "odd": step % 2 == 1, "tag": rng.choice(["a", "b"])})
            svc.put_vectors({**t, "vectors": [{"key": k, "data": {"float32": v.tolist()}, "metadata": md} for k, (v, md) in batch.items()]})
            model.update(batch)
        elif op < 0.65:
            gone = rng.sample(keys, rng.randint(1, 4))
            svc.delete_vectors({**t, "keys": gone})
            for k in gone:
                model.pop(k, None)
        elif op < 0.72:
            svc._store("prop").compact("idx")
        else:
            q = nrng.normal(size=6).astype(np.float32)
            use_filter = rng.random() < 0.5
            req = {**t, "queryVector": {"float32": q.tolist()}, "topK": rng.randint(1, 30), "returnDistance": True}
            if use_filter:
                req["filter"] = {"tag": "a"}
            got = svc.query_vectors(req)["vectors"]
            want = _brute_force(model, q, metric, req["topK"], (lambda md: md["tag"] == "a") if use_filter else None)
            assert [g["key"] for g in got] == [w[0] for w in want] or _ties(got, want), (step, got, want)
            np.testing.assert_allclose([g["distance"] for g in got], [w[1] for w in want], rtol=1e-4, atol=1e-5)
    client._put_object = real
    for svc in [*services, S3VectorsService(client, compaction="off")]:
        listed = {v["key"]: v for p in _all_pages(svc, t) for v in p}
        assert set(listed) == set(model)
        for key, v in listed.items():
            assert np.array_equal(np.array(v["data"]["float32"], dtype=np.float32), model[key][0])
            assert v["metadata"] == model[key][1]


def _ties(got, want) -> bool:
    """Same distances in the same order: keys may differ only among exact ties."""
    return len(got) == len(want) and all(abs(g["distance"] - w[1]) < 1e-6 for g, w in zip(got, want))


def _all_pages(svc, t):
    token = None
    while True:
        r = svc.list_vectors({**t, "returnData": True, "returnMetadata": True, "maxResults": 7, **({"nextToken": token} if token else {})})
        yield r["vectors"]
        token = r.get("nextToken")
        if not token:
            return


@pytest.mark.parametrize("seed", range(200))
def test_merge_plan_invariants(seed):
    rng = random.Random(seed)
    n = rng.randint(0, 80)
    segments = [{"bytes": rng.choice([rng.randint(1, 5000), rng.randint(1, 10**6), core.MAX_SEGMENT_BYTES])} for _ in range(n)]
    plan = core._plan_merge(segments)
    if plan is None:
        return
    i, j = plan
    assert j == n and 0 <= i < n - 1  # always a run of the newest segments, of at least 2
    forced = n > core.MAX_SEGMENTS
    assert j - i >= (2 if forced else core.MIN_MERGE)
    total = sum(s["bytes"] for s in segments[i:])
    assert total <= core.MAX_SEGMENT_BYTES or j - i == 2  # capped, except the smallest possible merge


@pytest.mark.parametrize("seed", range(30))
def test_segment_codec_round_trip(seed):
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    dim = rng.randint(1, 16)
    alphabet = ["a", "é", "🦀", "\x00", "\n", '"', "\\", " "]
    records = []
    for i in range(rng.randint(0, 40)):
        doc_id = "".join(rng.choices(alphabet, k=rng.randint(1, 8))) + str(i)
        seq = rng.choice([None, rng.randint(1, 10**12)])
        if rng.random() < 0.3:
            records.append((seq, doc_id, None, None, None))
        else:
            md = {"n": rng.randint(-(2**60), 2**60), "f": rng.random(), "l": ["x", True, 1.5]}
            block = nrng.normal(size=(1, dim)).astype(np.float32)
            records.append((seq, doc_id, "".join(rng.choices(alphabet, k=rng.randint(0, 30))), md, (block, 0)))
    blob = core._encode_segment(dim, records)
    back = core._decode_segment(blob, "k", dim, 7)
    assert len(back) == len(records)
    for orig, got in zip(records, back):
        assert got[0] == (orig[0] if orig[0] is not None else 7)
        assert got[1:4] == orig[1:4]
        if orig[2] is None:
            assert got[4] is None
        else:
            block, row = got[4]
            assert np.array_equal(block[row], orig[4][0][0])
