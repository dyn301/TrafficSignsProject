import faiss
import numpy as np


def build_faiss_index(embeddings: np.ndarray, index_path: str = None):
    """Build a FAISS index from normalized embeddings and optionally save index to disk."""
    # embeddings assumed normalized (L2 norm = 1)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product = cosine if normalized

    index.add(embeddings)

    if index_path is not None:
        faiss.write_index(index, index_path)

    return index


def load_faiss_index(index_path: str):
    """Load a saved FAISS index from disk."""
    return faiss.read_index(index_path)


def search_index(index, query_embedding: np.ndarray, top_k=5):
    """Search FAISS index with a single normalized query embedding."""
    q = np.asarray(query_embedding, dtype=np.float32)
    if q.ndim == 1:
        q = q.reshape(1, -1)
    if q.shape[1] != index.d:
        raise ValueError('Query dimension does not match index dimension')

    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    distances, indices = index.search(q, top_k)
    return distances[0], indices[0]
