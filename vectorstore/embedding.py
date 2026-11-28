"""Text embeddings with ONNX Runtime -- no PyTorch.

Reproduces sentence-transformers' all-MiniLM-L6-v2 pipeline exactly
(tokenize -> BERT -> mean pooling over real tokens -> L2 normalise), so
vectors match the ones earlier versions stored with sentence-transformers
(checked against reference vectors in tests/test_embedding.py). Compared
with sentence-transformers on PyTorch it starts ~9x faster, uses about half
the memory, embeds a query ~5x faster and a bulk upload ~25% faster, and
drops PyTorch (~700 MB) from the install.

The model files are pinned to an exact Hugging Face revision, so a change
upstream can't silently change the vectors.
"""

import os

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME = MODEL_ID.split("/", 1)[1]
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
# sentence-transformers' max_seq_length for this model. The repo's
# tokenizer.json defaults to truncating *and padding* to 128 tokens; both
# are overridden below to match sentence-transformers.
MAX_SEQ_LENGTH = 256
DEFAULT_BATCH_SIZE = 32


class OnnxEmbedder:
    """Drop-in for the subset of SentenceTransformer that VectorStore uses.

    Thread-safe: ONNX Runtime sessions and `tokenizers` encoders may be
    used from several threads at once (FastAPI runs sync endpoints on a
    thread pool).
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        max_seq_length: int = MAX_SEQ_LENGTH,
        batch_size: int = DEFAULT_BATCH_SIZE,
        threads: int | None = None,
    ):
        tok = Tokenizer.from_file(tokenizer_path)
        tok.enable_truncation(max_length=max_seq_length)
        pad_id = tok.token_to_id("[PAD]")
        tok.enable_padding(pad_id=0 if pad_id is None else pad_id, pad_token="[PAD]")  # pad to batch's longest
        self._tok = tok
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            opts.intra_op_num_threads = threads
        # Idle ORT threads would otherwise spin between calls, stealing CPU
        # from FAISS and request handling; costs nothing measurable per call.
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self._session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        self._input_names = {i.name for i in self._session.get_inputs()}
        self._batch_size = batch_size
        self._dim = int(self._session.get_outputs()[0].shape[-1])

    @classmethod
    def from_hub(cls, model_id: str = MODEL_ID, revision: str = MODEL_REVISION, **kwargs) -> "OnnxEmbedder":
        """Download (or reuse from the Hugging Face cache) the pinned model files.

        S3VAH_EMBED_THREADS caps ONNX Runtime's threads per process; set it to
        roughly cores / workers when running several API workers.
        """
        if "threads" not in kwargs and os.environ.get("S3VAH_EMBED_THREADS"):
            kwargs["threads"] = int(os.environ["S3VAH_EMBED_THREADS"])
        model = hf_hub_download(model_id, "onnx/model.onnx", revision=revision)
        tokenizer = hf_hub_download(model_id, "tokenizer.json", revision=revision)
        return cls(model, tokenizer, **kwargs)

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim

    def encode(self, texts, normalize_embeddings: bool = True, batch_size: int | None = None) -> np.ndarray:
        """Embed `texts` -> float32 array of shape (len(texts), dim), in input order.

        Vectors are always unit length: like the sentence-transformers model
        (whose pipeline ends in a Normalize layer), normalising is part of the
        model. `normalize_embeddings` is accepted for compatibility.
        """
        texts = list(texts)
        out = np.empty((len(texts), self._dim), dtype=np.float32)
        if not texts:
            return out
        batch_size = batch_size or self._batch_size
        # Batch similar lengths together so little compute goes on padding.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        for start in range(0, len(order), batch_size):
            idx = order[start: start + batch_size]
            out[idx] = self._encode_batch([texts[i] for i in idx])
        return out

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        encodings = self._tok.encode_batch(texts)
        ids = np.array([e.ids for e in encodings], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.array([e.type_ids for e in encodings], dtype=np.int64)
        hidden = self._session.run(["last_hidden_state"], feed)[0]
        # Mean over real tokens only (padding masked out), as
        # sentence-transformers' Pooling(mean) does, then L2-normalise.
        m = mask[..., None].astype(np.float32)
        pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return pooled / np.clip(norms, 1e-12, None)
