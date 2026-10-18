"""Schemas for Laya Decision Intelligence Layer."""
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class LayaQueryRoute(str, Enum):
    DOCUMENT_RETRIEVAL = "document_retrieval"
    CONVERSATIONAL = "conversational"
    CLARIFICATION = "clarification"
    DIRECT_RESPONSE = "direct_response"


class LayaRoutingDecision(BaseModel):
    route: LayaQueryRoute
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reason: str = ""
    is_fallback: bool = False


class LayaPassageFilterResult(BaseModel):
    is_relevant: bool
    relevance_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    reason: str = ""


class LayaOutputVerificationResult(BaseModel):
    is_supported: bool
    support_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    unsupported_claims: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    is_fallback: bool = False
