"""Sparse retrieval using BM25Okapi for keyword-based matching."""
import re
import logging
from typing import List, Tuple, Dict

logger = logging.getLogger("rag_app")

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("rank-bm25 not installed. BM25 retrieval disabled. Run: pip install rank-bm25")


class BM25Retriever:
    def __init__(self):
        self.bm25 = None
        self.chunks: List[Dict] = []

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())

    def index(self, chunks: List[Dict]):
        """Rebuild BM25 index from list of chunk metadata dicts."""
        if not BM25_AVAILABLE or not chunks:
            self.chunks = chunks
            return
        self.chunks = chunks
        tokenized = [self._tokenize(c.get("text", "")) for c in chunks]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None
        logger.info(f"BM25 index built with {len(chunks)} chunks")

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        allowed_filenames: set | None = None,
    ) -> List[Tuple[Dict, float]]:
        """Return (chunk_dict, bm25_score) tuples."""
        if not BM25_AVAILABLE or not self.bm25 or not self.chunks:
            return []
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        candidate_indices = range(len(scores))
        if allowed_filenames is not None:
            candidate_indices = [
                i for i, chunk in enumerate(self.chunks)
                if chunk.get("filename") in allowed_filenames
            ]
        top_idx = sorted(
            candidate_indices, key=lambda i: scores[i], reverse=True
        )[:top_k]
        return [
            (self.chunks[i], float(scores[i]))
            for i in top_idx
            if scores[i] > 0
        ]
