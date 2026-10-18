from typing import List
import numpy as np
from pipeline.vector_store.faiss_store import FAISSStore
from pipeline.embeddings.embedder import Embedder

class VectorRetriever:
    def __init__(self, embedder: Embedder, vector_store: FAISSStore):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> List[dict]:
        q_emb = self.embedder.embed([query])
        return self.vector_store.search(q_emb, top_k=top_k)
