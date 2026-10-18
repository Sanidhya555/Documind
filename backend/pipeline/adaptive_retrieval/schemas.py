"""Schemas for Adaptive Retrieval."""
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class RetrievalAction(str, Enum):
    ANSWER = "answer"
    RETRIEVE_MORE = "retrieve_more"
    CLARIFY = "clarify"
    ABSTAIN = "abstain"


class EvidenceAssessment(BaseModel):
    is_sufficient: bool = Field(..., description="Whether evidence is deemed sufficient to answer")
    coverage_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Estimated coverage of query aspects")
    top_relevance_score: float = Field(default=0.0, description="Highest cross-encoder/reranker score")
    avg_relevance_score: float = Field(default=0.0, description="Average relevance of top candidates")
    unique_sources_count: int = Field(default=0, ge=0, description="Number of distinct source documents")
    duplicate_ratio: float = Field(default=0.0, ge=0.0, le=1.0, description="Ratio of duplicate/redundant chunks")
    has_conflict: bool = Field(default=False, description="Whether potential contradictory facts were detected")
    missing_aspects: List[str] = Field(default_factory=list, description="Identified sub-questions or entities lacking evidence")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence in the assessment")


class RetrievalDecision(BaseModel):
    action: RetrievalAction
    reason_code: str
    evidence_score: Optional[float] = None
    policy_version: str = "adaptive_v2"
    should_expand: bool = False
    expansion_queries: List[str] = Field(default_factory=list)
    candidate_pool_multiplier: float = 1.0
    iteration: int = 1
    clarification_prompt: Optional[str] = None
    abstention_reason: Optional[str] = None


class AdaptiveConfig(BaseModel):
    enabled: bool = True
    policy_mode: str = "deterministic"  # "deterministic" | "laya" | "fixed"
    max_retrieval_attempts: int = 2
    min_coverage_threshold: float = 0.55
    min_top_relevance_threshold: float = 0.20
    max_duplicate_ratio: float = 0.70
    # These are retrieval ceilings. The controller trims the returned context
    # dynamically from the score distribution for each query.
    initial_top_k: int = 12
    expanded_top_k: int = 18
    max_query_expansions: int = 2
    strict_abstention: bool = True
