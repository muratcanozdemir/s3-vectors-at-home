"""Regenerate tests/data/embedding_reference.npz.

The reference vectors come from sentence-transformers (PyTorch), which is
what produced every vector stored by earlier versions. The ONNX embedder
must reproduce them, or existing data would stop matching new queries;
tests/test_embedding.py checks it against these on every run. They only
need regenerating if MODEL_REVISION changes.

    uv run --with sentence-transformers==6.0.0 python scripts/make_embedding_fixture.py
"""

import json
import pathlib

import numpy as np
import sentence_transformers
import torch
from sentence_transformers import SentenceTransformer

from vectorstore.embedding import MODEL_ID, MODEL_REVISION

long_words = " ".join(f"word{i} lorem ipsum dolor" for i in range(400))  # far past 256 tokens
mid_words = " ".join(f"token{i}" for i in range(70))  # between 128 and 256 tokens
TEXTS = [
    "hello world",
    "",
    "   ",
    "a",
    "The quick brown fox jumps over the lazy dog.",
    "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG.",
    "Vector search with FAISS and MinIO",
    "Äpfel und Birnen schmecken im Herbst am besten.",
    "東京は日本の首都です。",
    "emoji 🚀🔥 and symbols ∑∫√ ≠ ≤",
    "numbers 3.14159 2718 1e-9 -42 0x1F",
    "line one\nline two\n\ttabbed line three",
    "punctuation!!! ??? ... --- ;;; ''' \"\"\"",
    "https://example.com/path?query=1&b=2#frag",
    "def f(x):\n    return x * 2  # code",
    mid_words,
    long_words,
    "short",
    "a slightly longer sentence about databases, storage and retrieval systems",
    " ".join(["repeat"] * 300),
]


def main():
    model = SentenceTransformer(MODEL_ID, revision=MODEL_REVISION, device="cpu")
    vectors = model.encode(TEXTS, normalize_embeddings=True, batch_size=8)
    tokens = [len(model.tokenizer(t, truncation=False)["input_ids"]) for t in TEXTS]
    out = pathlib.Path(__file__).resolve().parent.parent / "tests" / "data" / "embedding_reference.npz"
    meta = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "max_seq_length": model.max_seq_length,
        "sentence_transformers": sentence_transformers.__version__,
        "torch": torch.__version__,
        "untruncated_token_counts": tokens,
    }
    np.savez_compressed(out, vectors=np.asarray(vectors, dtype=np.float32), texts=json.dumps(TEXTS), meta=json.dumps(meta))
    print(f"wrote {out} ({len(TEXTS)} texts)", json.dumps(meta))


if __name__ == "__main__":
    main()
