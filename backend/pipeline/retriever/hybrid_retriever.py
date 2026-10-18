"""Hybrid Retriever: BM25 (sparse) + FAISS (dense) fused with Reciprocal Rank Fusion."""
import logging
from typing import List, Tuple, Dict

from pipeline.retriever.bm25_retriever import BM25Retriever
from pipeline.retriever.vector_retriever import VectorRetriever

logger = logging.getLogger("rag_app")


def _chunk_key(chunk: Dict) -> str:
    return f"{chunk.get('filename', '')}::{chunk.get('chunk_id', '')}"


class HybridRetriever:
    """
    Combines dense (FAISS) and sparse (BM25) retrieval via Reciprocal Rank Fusion (RRF).
    RRF score = dense_weight/(k + dense_rank) + sparse_weight/(k + sparse_rank)
    """

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        bm25_retriever: BM25Retriever,
        rrf_k: int = 60,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
    ):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def _normalize_dense(self, raw: list) -> List[Tuple[Dict, float]]:
        """Convert VectorRetriever output (list of dicts) to (chunk, score) tuples."""
        pairs = []
        for r in raw:
            if isinstance(r, dict) and "chunk" in r:
                pairs.append((r["chunk"], r["score"]))
            elif isinstance(r, (list, tuple)) and len(r) == 2:
                pairs.append(tuple(r))
        return pairs

    def _rrf_fuse(
        self,
        dense: List[Tuple[Dict, float]],
        sparse: List[Tuple[Dict, float]],
        top_k: int,
    ) -> List[Tuple[Dict, float]]:
        chunk_store: Dict[str, Dict] = {}
        rrf_scores: Dict[str, float] = {}

        for rank, (chunk, _) in enumerate(dense):
            if not isinstance(chunk, dict):
                continue
            key = _chunk_key(chunk)
            chunk_store[key] = chunk
            rrf_scores[key] = rrf_scores.get(key, 0.0) + self.dense_weight / (self.rrf_k + rank + 1)

        for rank, (chunk, _) in enumerate(sparse):
            if not isinstance(chunk, dict):
                continue
            key = _chunk_key(chunk)
            chunk_store.setdefault(key, chunk)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + self.sparse_weight / (self.rrf_k + rank + 1)

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(chunk_store[k], score) for k, score in ranked]

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        allowed_filenames: set | None = None,
    ) -> List[Tuple[Dict, float]]:
        if allowed_filenames is not None and not allowed_filenames:
            return []
        fetch_k = top_k * 3

        if allowed_filenames is None:
            dense_fetch_k = fetch_k
        else:
            dense_fetch_k = self.vector_retriever.vector_store.index.ntotal
        raw_dense = self.vector_retriever.retrieve(query, top_k=dense_fetch_k)
        dense = self._normalize_dense(raw_dense)
        sparse_fetch_k = fetch_k if allowed_filenames is None else len(self.bm25_retriever.chunks)
        sparse = self.bm25_retriever.retrieve(
            query,
            top_k=sparse_fetch_k,
            allowed_filenames=allowed_filenames,
        )

        if allowed_filenames is not None:
            dense = [
                (chunk, score) for chunk, score in dense
                if chunk.get("filename") in allowed_filenames
            ]

        results = self._rrf_fuse(dense, sparse, top_k)
        logger.debug(f"Hybrid retrieval: {len(dense)} dense + {len(sparse)} sparse → {len(results)} fused")
        return results
