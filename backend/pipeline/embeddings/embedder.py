"""Sentence-transformer embedder with a deterministic local fallback."""
import hashlib
import logging
import os
import numpy as np
from typing import List

logger = logging.getLogger("rag_app")


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.dimension = 384
        self.model = None
        try:
            if os.getenv("ALLOW_MODEL_DOWNLOADS", "false").lower() != "true":
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except Exception:
            logger.warning("Embedding model unavailable; using deterministic fallback vectors.")
            self.model = None

    def embed(self, texts: List[str]) -> np.ndarray:
        if self.model is None:
            return np.vstack([self._fallback_embed(text) for text in texts]).astype("float32")

        return self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=32,
        )

    def _fallback_embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype="float32")
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec
