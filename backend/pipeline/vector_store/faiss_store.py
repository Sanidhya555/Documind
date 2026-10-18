import os
import faiss
import numpy as np
from typing import List

class FAISSStore:
    def __init__(self, dimension, embedder):
        self.dimension = dimension
        self.embedder = embedder
        self.index = faiss.IndexFlatIP(dimension)
        self.text_chunks = []

    def _normalize(self, v: np.ndarray):
        faiss.normalize_L2(v)

    def add(self, embeddings: np.ndarray, chunks: List[dict]):
        embeddings = np.array(embeddings).astype("float32")
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        self._normalize(embeddings)
        self.index.add(embeddings)
        self.text_chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        if self.index.ntotal == 0:
            return []
        q = query_embedding.astype("float32")
        self._normalize(q)
        scores, indices = self.index.search(q, min(top_k, self.index.ntotal))
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if 0 <= idx < len(self.text_chunks):
                results.append({"chunk": self.text_chunks[idx], "score": float(score)})
        return results

    def save(self, index_path: str, chunks_path: str):
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        os.makedirs(os.path.dirname(chunks_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        np.save(chunks_path, np.array(self.text_chunks, dtype=object))

    def load(self, index_path: str, chunks_path: str):
        self.index = faiss.read_index(index_path)
        self.text_chunks = list(np.load(chunks_path, allow_pickle=True))

    def rebuild_index(self, chunks: List[dict]):
        self.index = faiss.IndexFlatIP(self.dimension)
        self.text_chunks = chunks
        if not chunks:
            return
        embeddings = self.embedder.embed([c["text"] for c in chunks])
        embeddings = np.array(embeddings).astype("float32")
        self._normalize(embeddings)
        self.index.add(embeddings)
