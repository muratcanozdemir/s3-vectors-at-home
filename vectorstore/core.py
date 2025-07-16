import os
import io
import json
import numpy as np
import faiss
from minio import Minio
from sentence_transformers import SentenceTransformer

MINIO_ENDPOINT = os.environ["MINIO_ENDPOINT"]
MINIO_ACCESS_KEY = os.environ["MINIO_ACCESS_KEY"]
MINIO_SECRET_KEY = os.environ["MINIO_SECRET_KEY"]
BUCKET = "vectors"
INDEX_OBJ = "faiss.index"

client = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=False)
embedder = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')

def _ensure_bucket():
    if not client.bucket_exists(BUCKET):
        client.make_bucket(BUCKET)

_ensure_bucket()

def embed(text: str) -> np.ndarray:
    return np.array(embedder.encode([text], normalize_embeddings=True), dtype=np.float32)

def add_document(doc_id: str, text: str):
    vec = embed(text)[0]
    vec_bytes = vec.tobytes()
    client.put_object(BUCKET, f"{doc_id}.npy", data=io.BytesIO(vec_bytes), length=len(vec_bytes))
    meta = {"doc_id": doc_id, "text": text}
    meta_bytes = json.dumps(meta).encode()
    client.put_object(BUCKET, f"{doc_id}.meta.json", data=io.BytesIO(meta_bytes), length=len(meta_bytes))
    _update_index(doc_id, vec)

def _load_all_vectors_and_ids():
    vectors, ids = [], []
    for obj in client.list_objects(BUCKET):
        if obj.object_name.endswith('.npy'):
            vec = np.frombuffer(client.get_object(BUCKET, obj.object_name).read(), dtype=np.float32)
            vectors.append(vec)
            ids.append(obj.object_name.removesuffix('.npy'))
    return (np.vstack(vectors) if vectors else np.empty((0, embedder.get_sentence_embedding_dimension()), dtype=np.float32)), ids

def _upload_index(index, ids):
    # Save index to bytes, upload to MinIO
    buf = io.BytesIO()
    faiss.write_index(index, buf)
    buf.seek(0)
    client.put_object(BUCKET, INDEX_OBJ, buf, length=buf.getbuffer().nbytes)
    # Also persist mapping of ids
    ids_bytes = json.dumps(ids).encode()
    client.put_object(BUCKET, "index.ids.json", data=io.BytesIO(ids_bytes), length=len(ids_bytes))

def _download_index():
    try:
        idx_obj = client.get_object(BUCKET, INDEX_OBJ)
        idx_bytes = idx_obj.read()
        index = faiss.read_index(io.BytesIO(idx_bytes))
        ids = json.loads(client.get_object(BUCKET, "index.ids.json").read())
        return index, ids
    except Exception:
        return None, []

def _update_index(new_id=None, new_vec=None):
    # Load existing index and ids, append if new doc, else rebuild
    index, ids = _download_index()
    if index and new_id and new_vec is not None:
        index.add(np.expand_dims(new_vec, axis=0))
        ids.append(new_id)
        _upload_index(index, ids)
    else:
        vectors, ids = _load_all_vectors_and_ids()
        if len(vectors):
            index = faiss.IndexFlatL2(vectors.shape[1])
            index.add(vectors)
            _upload_index(index, ids)

def search_vectors(query: str, top_k: int = 5):
    index, ids = _download_index()
    if not index:
        _update_index()
        index, ids = _download_index()
    qv = embed(query)
    D, Ind = index.search(qv, top_k)
    return [ids[i] for i in Ind[0] if i < len(ids)]


def get_document(doc_id: str):
    """Fetch document metadata and text by doc_id"""
    try:
        meta_obj = client.get_object(BUCKET, f"{doc_id}.meta.json")
        meta = json.loads(meta_obj.read())
        return meta
    except Exception:
        return None

def list_documents(skip: int = 0, limit: int = 100):
    """List documents (returns doc_id and possibly first N characters of text)"""
    doc_ids = []
    for obj in client.list_objects(BUCKET):
        if obj.object_name.endswith('.meta.json'):
            doc_ids.append(obj.object_name.removesuffix('.meta.json'))
    doc_ids = sorted(doc_ids)  # for consistent paging
    selected = doc_ids[skip: skip + limit]
    # Optionally, fetch summary/meta for each
    docs = []
    for doc_id in selected:
        meta = get_document(doc_id)
        if meta:
            docs.append({"doc_id": doc_id, "text_preview": meta.get("text", "")[:64]})
    return docs

def delete_document(doc_id: str):
    """Delete both vector and metadata, then rebuild index."""
    removed = False
    for suffix in (".npy", ".meta.json"):
        obj_name = f"{doc_id}{suffix}"
        try:
            client.remove_object(BUCKET, obj_name)
            removed = True
        except Exception:
            pass  # Already gone
    if removed:
        # Must fully reload all vectors & rebuild index
        vectors, ids = _load_all_vectors_and_ids()
        if len(vectors):
            index = faiss.IndexFlatL2(vectors.shape[1])
            index.add(vectors)
            _upload_index(index, ids)
        else:
            # If empty, remove index objects from MinIO
            for index_file in (INDEX_OBJ, "index.ids.json"):
                try:
                    client.remove_object(BUCKET, index_file)
                except Exception:
                    pass
        return True
    return False

def count_documents():
    """Return count of docs in bucket."""
    count = 0
    for obj in client.list_objects(BUCKET):
        if obj.object_name.endswith('.meta.json'):
            count += 1
    return count

def embedding_model_name():
    """Return the embedding model name (for status endpoint)."""
    return getattr(embedder, "model_name", "unknown")