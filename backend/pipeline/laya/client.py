"""Laya Client: Connects to Laya Decision Model with timeout, error handling, and safe fallback."""
import logging
import os
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("rag_app")


class LayaClient:
    """
    Dedicated client for Laya decision engine.
    Supports official Convai Innovations Laya Router interface (choice, score, noul).
    Includes timeouts, structural validation, and graceful degradation.
    """

    def __init__(
        self,
        enabled: Optional[bool] = None,
        model_version: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ):
        self.enabled = (
            enabled
            if enabled is not None
            else os.getenv("LAYA_ENABLED", "false").lower() == "true"
        )
        self.model_version = model_version or os.getenv("LAYA_MODEL_VERSION", "convaiinnovations/laya")
        self.timeout = timeout_seconds or float(os.getenv("LAYA_TIMEOUT_SECONDS", "3.0"))
        self.router = None
        self.is_available = False

        if self.enabled:
            self._init_router()
        else:
            logger.info("Laya decision layer is disabled by configuration (LAYA_ENABLED=false).")

    def _init_router(self):
        try:
            from laya import Router
            logger.info(f"Initializing Laya Router checkpoint: {self.model_version}")
            self.router = Router(preload=False)
            self.is_available = True
        except ImportError:
            logger.warning("The 'laya' package is not installed. Laya features will fall back to deterministic mode.")
            self.is_available = False
        except Exception as e:
            logger.warning(f"Failed to initialize Laya Router ({e}). Falling back to deterministic mode.")
            self.is_available = False

    def predict(self, state: str, questions: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Executes typed prediction across Laya primitives ('choice', 'noul', 'score').
        """
        if not self.enabled or not self.is_available or self.router is None:
            return None

        t0 = time.time()
        try:
            # Bound input state length to prevent memory exhaustion
            bounded_state = state[:4000]
            result = self.router.predict(bounded_state, questions)
            duration = round((time.time() - t0) * 1000, 2)
            logger.debug(f"Laya predict completed in {duration}ms for {len(questions)} questions")
            return result
        except Exception as exc:
            logger.warning(f"Laya inference error: {exc}. Using deterministic fallback.")
            return None
