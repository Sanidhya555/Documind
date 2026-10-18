"""Decision Policy: Selects operational action (ANSWER, RETRIEVE_MORE, CLARIFY, ABSTAIN)."""
import logging
from abc import ABC, abstractmethod
from typing import Optional
from .schemas import EvidenceAssessment, RetrievalAction, RetrievalDecision

logger = logging.getLogger("rag_app")


class BaseDecisionPolicy(ABC):
    @abstractmethod
    def decide(
        self,
        query: str,
        assessment: EvidenceAssessment,
        iteration: int,
        max_attempts: int,
    ) -> RetrievalDecision:
        pass


class DeterministicDecisionPolicy(BaseDecisionPolicy):
    """
    Deterministic rule-based baseline decision policy.
    Uses multi-signal assessment with explicit threshold checks and budget bounds.
    """

    def __init__(
        self,
        min_relevance_threshold: float = 0.20,
        min_coverage_threshold: float = 0.55,
        strict_abstention: bool = True,
    ):
        self.min_relevance = min_relevance_threshold
        self.min_coverage = min_coverage_threshold
        self.strict_abstention = strict_abstention

    def _is_ambiguous(self, query: str) -> bool:
        clean = query.strip().lower()
        words = clean.split()
        if len(words) <= 2 and words[0] in {"what", "how", "why", "who", "cost", "price", "when"}:
            return True
        if len(words) == 1 and len(words[0]) < 10:
            return True
        return False

    def decide(
        self,
        query: str,
        assessment: EvidenceAssessment,
        iteration: int,
        max_attempts: int,
    ) -> RetrievalDecision:
        # Check for ambiguity
        if self._is_ambiguous(query):
            return RetrievalDecision(
                action=RetrievalAction.CLARIFY,
                reason_code="ambiguous_query",
                evidence_score=assessment.confidence,
                policy_version="deterministic_v2",
                iteration=iteration,
                clarification_prompt=f"Could you please clarify your question about '{query}' with more context or document name?",
            )

        # 1. If evidence is sufficient -> Answer
        if assessment.is_sufficient:
            return RetrievalDecision(
                action=RetrievalAction.ANSWER,
                reason_code="sufficient_evidence",
                evidence_score=assessment.confidence,
                policy_version="deterministic_v2",
                iteration=iteration,
            )

        # 2. If budget permits and evidence can be improved -> Retrieve More
        if iteration < max_attempts:
            # If completely detached out-of-domain with zero overlap on first attempt
            if assessment.top_relevance_score < -5.0 and assessment.coverage_score == 0.0:
                return RetrievalDecision(
                    action=RetrievalAction.ABSTAIN,
                    reason_code="out_of_domain_unanswerable",
                    evidence_score=assessment.confidence,
                    policy_version="deterministic_v2",
                    iteration=iteration,
                    abstention_reason="This question does not appear to relate to any content indexed in the document repository.",
                )

            return RetrievalDecision(
                action=RetrievalAction.RETRIEVE_MORE,
                reason_code="insufficient_coverage_expand",
                evidence_score=assessment.confidence,
                policy_version="deterministic_v2",
                should_expand=True,
                candidate_pool_multiplier=1.6,
                iteration=iteration,
            )

        # 3. Budget exhausted
        if assessment.top_relevance_score >= (self.min_relevance * 0.75) and assessment.coverage_score >= 0.35:
            # Sufficient partial evidence to attempt grounded answer with caveat
            return RetrievalDecision(
                action=RetrievalAction.ANSWER,
                reason_code="best_effort_budget_exhausted",
                evidence_score=assessment.confidence,
                policy_version="deterministic_v2",
                iteration=iteration,
            )

        # 4. Evidence remains insufficient after exhausting retrieval attempts -> Abstain
        return RetrievalDecision(
            action=RetrievalAction.ABSTAIN,
            reason_code="unanswerable_evidence_exhausted",
            evidence_score=assessment.confidence,
            policy_version="deterministic_v2",
            iteration=iteration,
            abstention_reason="I could not find sufficient relevant evidence in the indexed documents to answer this question accurately.",
        )


class LayaDecisionPolicy(BaseDecisionPolicy):
    """
    Laya-backed decision policy using verified decision primitives.
    Always wraps a deterministic fallback in case Laya is disabled, unreachable, or errors.
    """

    def __init__(
        self,
        laya_service=None,
        fallback_policy: Optional[BaseDecisionPolicy] = None,
    ):
        self.laya_service = laya_service
        self.fallback = fallback_policy or DeterministicDecisionPolicy()

    def decide(
        self,
        query: str,
        assessment: EvidenceAssessment,
        iteration: int,
        max_attempts: int,
    ) -> RetrievalDecision:
        if not self.laya_service or not getattr(self.laya_service, "is_enabled", False):
            return self.fallback.decide(query, assessment, iteration, max_attempts)

        try:
            decision = self.laya_service.decide_retrieval_action(
                query=query,
                assessment=assessment,
                iteration=iteration,
                max_attempts=max_attempts,
            )
            if decision and isinstance(decision, RetrievalDecision):
                return decision
        except Exception as e:
            logger.warning(f"Laya decision failed, using deterministic fallback: {e}")

        return self.fallback.decide(query, assessment, iteration, max_attempts)
