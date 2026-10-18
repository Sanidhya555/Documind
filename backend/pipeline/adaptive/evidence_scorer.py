"""
Evidence sufficiency scorer.

Determines whether retrieved chunks provide enough signal to answer the query,
or whether the system should retry, expand, clarify, or abstain.

Decision logic is deterministic (no LLM call), keeping it fast and auditable.
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Dict

logger = logging.getLogger("rag_app")


class RetrievalDecision(str, Enum):
    ANSWER = "answer"           # Sufficient evidence — proceed to generation
    RETRIEVE_MORE = "retrieve_more"  # Low coverage — fetch more chunks
    EXPAND_QUERY = "expand_query"    # No signal — try alternative phrasings
    CLARIFY = "clarify"         # Ambiguous query — ask the user for clarification
    ABSTAIN = "abstain"         # Evidence absent / unanswerable


@dataclass
class EvidenceReport:
    decision: RetrievalDecision
    confidence: float           # 0.0 – 1.0
    coverage_score: float       # Fraction of query tokens found in top-k context
    top_score: float            # Best retrieval score (RRF-fused)
    reason: str                 # Human-readable explanation


# ── Tunable thresholds (all configurable via env or caller) ──────────────────
_DEFAULT_CFG: Dict[str, float] = {
    "min_top_score": 0.005,     # RRF score below this → no relevant doc found
    "min_coverage": 0.25,       # token coverage below this → too little overlap
    "good_coverage": 0.50,      # above this → definitely answer
    "good_top_score": 0.012,    # above this → strong single hit
    "ambiguity_threshold": 2,   # if query has <N meaningful tokens → clarify
}

# Stop-words excluded from coverage calculation
_STOP = {
    "what", "which", "who", "when", "where", "why", "how", "is", "are", "was",
    "were", "the", "a", "an", "and", "or", "in", "of", "to", "for", "on",
    "at", "by", "from", "with", "about", "that", "this", "it", "do", "did",
    "does", "can", "could", "would", "should", "has", "have", "had", "be",
    "been", "being",
}


def _meaningful_tokens(text: str) -> List[str]:
    return [
        w for w in re.findall(r"[a-z0-9]+", text.lower())
        if len(w) > 2 and w not in _STOP
    ]


class EvidenceScorer:
    """
    Stateless scorer — call `score()` after each retrieval attempt.
    """

    def __init__(self, cfg: Dict[str, float] | None = None):
        self.cfg = {**_DEFAULT_CFG, **(cfg or {})}

    def score(
        self,
        query: str,
        results: List[Tuple[Dict, float]],
        retry_count: int = 0,
        max_retries: int = 2,
    ) -> EvidenceReport:
        """
        Args:
            query:        The original (or expanded) user query.
            results:      List of (chunk_dict, fused_rrf_score) from HybridRetriever.
            retry_count:  How many retrieval retries have already happened.
            max_retries:  Hard cap on retries — prevents drift.

        Returns:
            EvidenceReport with a decision and supporting metrics.
        """
        q_tokens = _meaningful_tokens(query)
        top_score = results[0][1] if results else 0.0

        # ── No results at all ─────────────────────────────────────────────────
        if not results:
            if retry_count < max_retries:
                return EvidenceReport(
                    decision=RetrievalDecision.EXPAND_QUERY,
                    confidence=0.0,
                    coverage_score=0.0,
                    top_score=0.0,
                    reason="No documents retrieved. Will try alternative phrasings.",
                )
            return EvidenceReport(
                decision=RetrievalDecision.ABSTAIN,
                confidence=0.0,
                coverage_score=0.0,
                top_score=0.0,
                reason="No relevant documents found after multiple retrieval attempts.",
            )

        # ── Ambiguous query (too few meaningful tokens) ───────────────────────
        if len(q_tokens) < self.cfg["ambiguity_threshold"]:
            return EvidenceReport(
                decision=RetrievalDecision.CLARIFY,
                confidence=0.2,
                coverage_score=0.0,
                top_score=top_score,
                reason=(
                    f"Query contains only {len(q_tokens)} meaningful token(s). "
                    "Please provide more context."
                ),
            )

        # ── Coverage score ────────────────────────────────────────────────────
        # Build a combined context window from top-3 chunks
        top_texts = [
            r[0].get("text", "") for r in results[:3] if isinstance(r[0], dict)
        ]
        context_tokens = set(_meaningful_tokens(" ".join(top_texts)))
        if q_tokens:
            coverage = len(set(q_tokens) & context_tokens) / len(set(q_tokens))
        else:
            coverage = 0.0

        # ── Decision tree ─────────────────────────────────────────────────────
        cfg = self.cfg

        if top_score < cfg["min_top_score"] and coverage < cfg["min_coverage"]:
            if retry_count < max_retries:
                decision = RetrievalDecision.RETRIEVE_MORE
                reason = (
                    f"Top RRF score {top_score:.4f} below threshold "
                    f"{cfg['min_top_score']} and coverage {coverage:.2f} is low. "
                    "Retrieving more chunks."
                )
            else:
                decision = RetrievalDecision.ABSTAIN
                reason = (
                    "Insufficient evidence after exhausting retrieval retries. "
                    "The indexed documents do not appear to contain the answer."
                )
            confidence = max(0.0, coverage)
        elif (
            top_score >= cfg["good_top_score"]
            or coverage >= cfg["good_coverage"]
        ):
            decision = RetrievalDecision.ANSWER
            confidence = min(1.0, (top_score / cfg["good_top_score"] + coverage) / 2)
            reason = (
                f"Strong evidence: top_score={top_score:.4f}, "
                f"coverage={coverage:.2f}."
            )
        else:
            # Borderline — answer but signal low confidence
            decision = RetrievalDecision.ANSWER
            confidence = min(0.6, (top_score / cfg["good_top_score"] + coverage) / 2)
            reason = (
                f"Marginal evidence: top_score={top_score:.4f}, "
                f"coverage={coverage:.2f}. Answer may be incomplete."
            )

        logger.debug(
            "EvidenceScorer: decision=%s conf=%.2f cov=%.2f top=%.4f retry=%d",
            decision, confidence, coverage, top_score, retry_count,
        )

        return EvidenceReport(
            decision=decision,
            confidence=confidence,
            coverage_score=round(coverage, 4),
            top_score=round(top_score, 4),
            reason=reason,
        )
