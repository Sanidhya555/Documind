"""Cross-encoder reranker for high-precision result reranking."""
import logging
import os
from typing import List, Tuple, Dict

logger = logging.getLogger("rag_app")


class CrossEncoderReranker:
    """
    Uses a cross-encoder model to rerank retrieved chunks.
    Falls back gracefully if sentence-transformers CrossEncoder is unavailable.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = None
        self.available = False
        try:
            if os.getenv("ALLOW_MODEL_DOWNLOADS", "false").lower() != "true":
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name, max_length=512)
            self.available = True
            logger.info(f"Cross-encoder reranker loaded: {model_name}")
        except Exception as e:
            logger.warning(f"Reranker unavailable (will skip): {e}")

    def rerank(
        self,
        query: str,
        results: List[Tuple[Dict, float]],
        top_k: int = 5,
    ) -> List[Tuple[Dict, float]]:
        """Rerank results using cross-encoder scores; return top_k."""
        if not self.available or not results:
            return results[:top_k]

        try:
            pairs = [
                (query, r[0]["text"] if isinstance(r[0], dict) else str(r[0]))
                for r in results
            ]
            scores = self.model.predict(pairs, show_progress_bar=False)
            reranked = sorted(zip(results, scores), key=lambda x: float(x[1]), reverse=True)
            return [(item[0], float(score)) for item, score in reranked[:top_k]]
        except Exception as e:
            logger.warning(f"Reranking failed, using original order: {e}")
            return results[:top_k]
