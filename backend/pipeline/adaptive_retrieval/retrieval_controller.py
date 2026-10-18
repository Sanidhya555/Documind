"""Adaptive Retrieval Controller: Orchestrates bounded retrieval, evidence assessment, and decision loops."""
import logging
import time
from typing import Dict, List, Optional, Tuple
from .decision_policy import BaseDecisionPolicy, DeterministicDecisionPolicy
from .evidence_assessor import EvidenceAssessor
from .schemas import AdaptiveConfig, EvidenceAssessment, RetrievalAction, RetrievalDecision

logger = logging.getLogger("rag_app")


def _chunk_key(chunk: Dict) -> str:
    if isinstance(chunk, dict):
        return f"{chunk.get('filename', '')}::{chunk.get('chunk_id', 0)}"
    return str(chunk)


class AdaptiveRetrievalController:
    """
    Manages bounded adaptive retrieval:
    - Runs initial hybrid retrieval and reranking
    - Assesses evidence sufficiency
    - Executes additional retrieval actions within a bounded budget
    - Prevents query drift by always preserving the original query
    """

    def __init__(
        self,
        hybrid_retriever,
        reranker,
        query_expander=None,
        assessor: Optional[EvidenceAssessor] = None,
        policy: Optional[BaseDecisionPolicy] = None,
        config: Optional[AdaptiveConfig] = None,
    ):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.query_expander = query_expander
        self.config = config or AdaptiveConfig()
        self.assessor = assessor or EvidenceAssessor(
            min_relevance_threshold=self.config.min_top_relevance_threshold,
            min_coverage_threshold=self.config.min_coverage_threshold,
            duplicate_threshold=self.config.max_duplicate_ratio,
        )
        self.policy = policy or DeterministicDecisionPolicy(
            min_relevance_threshold=self.config.min_top_relevance_threshold,
            min_coverage_threshold=self.config.min_coverage_threshold,
            strict_abstention=self.config.strict_abstention,
        )

    def _retrieve_and_rerank(
        self,
        query: str,
        top_k: int,
        allowed_filenames: Optional[set] = None,
    ) -> List[Tuple[Dict, float]]:
        if allowed_filenames is None:
            raw_results = self.hybrid_retriever.retrieve(query, top_k=top_k * 2)
        else:
            raw_results = self.hybrid_retriever.retrieve(
                query,
                top_k=top_k * 2,
                allowed_filenames=allowed_filenames,
            )
        if self.reranker and getattr(self.reranker, "available", False) and raw_results:
            results = self.reranker.rerank(query, raw_results, top_k=top_k)
        else:
            results = raw_results[:top_k]
        return results

    def _merge_and_deduplicate(
        self,
        existing: List[Tuple[Dict, float]],
        new_items: List[Tuple[Dict, float]],
    ) -> List[Tuple[Dict, float]]:
        seen: Dict[str, Tuple[Dict, float]] = {}
        for chunk, score in existing:
            key = _chunk_key(chunk)
            seen[key] = (chunk, score)

        for chunk, score in new_items:
            key = _chunk_key(chunk)
            if key not in seen or seen[key][1] < score:
                seen[key] = (chunk, score)

        return sorted(seen.values(), key=lambda x: x[1], reverse=True)

    @staticmethod
    def _select_by_score(
        results: List[Tuple[Dict, float]], max_results: int
    ) -> List[Tuple[Dict, float]]:
        """Choose a query-specific context size using the ranked score spread."""
        ranked = results[:max_results]
        if len(ranked) <= 1:
            return ranked

        scores = [float(score) for _, score in ranked]
        best = scores[0]
        score_range = best - scores[-1]
        if score_range <= 1e-9:
            return ranked

        # Keep candidates in the upper 55% of this query's observed score
        # range, with a floor tied to the leading score to handle near-ties.
        margin = max(score_range * 0.55, abs(best) * 0.18)
        threshold = best - margin
        selected = [item for item in ranked if float(item[1]) >= threshold]
        return selected or ranked[:1]

    def execute_retrieval(
        self,
        query: str,
        top_k: Optional[int] = None,
        allowed_filenames: Optional[set] = None,
    ) -> Tuple[List[Tuple[Dict, float]], RetrievalDecision, EvidenceAssessment, List[str]]:
        target_k = top_k or self.config.initial_top_k
        max_attempts = self.config.max_retrieval_attempts if self.config.enabled else 1

        # Iteration 1: Initial Hybrid Retrieval + Reranking
        t0 = time.time()
        candidates = self._retrieve_and_rerank(
            query, top_k=target_k, allowed_filenames=allowed_filenames
        )
        assessment = self.assessor.assess(query, candidates)

        decision = self.policy.decide(
            query=query,
            assessment=assessment,
            iteration=1,
            max_attempts=max_attempts,
        )

        all_expanded_queries: List[str] = []
        iteration = 1

        # Adaptive Loop: Execute additional bounded retrieval if required and enabled
        while decision.action == RetrievalAction.RETRIEVE_MORE and iteration < max_attempts:
            iteration += 1
            logger.info(
                f"Adaptive retrieval triggering iteration {iteration}/{max_attempts} "
                f"for query: '{query[:40]}...'"
            )

            # Generate controlled query expansions without query drift
            expanded = []
            if self.query_expander and decision.should_expand:
                try:
                    exp_raw = self.query_expander.expand(query)
                    expanded = [q for q in exp_raw if q.strip().lower() != query.strip().lower()][:self.config.max_query_expansions]
                    all_expanded_queries.extend(expanded)
                except Exception as e:
                    logger.warning(f"Query expansion failed during adaptive loop: {e}")

            # Retrieve with expanded candidate pool and alternative queries
            fetch_k = int(target_k * decision.candidate_pool_multiplier)
            new_candidates: List[Tuple[Dict, float]] = []

            for exp_q in expanded:
                res = self._retrieve_and_rerank(
                    exp_q, top_k=fetch_k, allowed_filenames=allowed_filenames
                )
                new_candidates.extend(res)

            # If no expansion produced, fetch larger pool on original query
            if not expanded:
                new_candidates = self._retrieve_and_rerank(
                    query, top_k=fetch_k, allowed_filenames=allowed_filenames
                )

            # Deduplicate and combine with existing candidates
            combined = self._merge_and_deduplicate(candidates, new_candidates)

            # Re-rerank combined candidate pool against the ORIGINAL query to prevent drift
            if self.reranker and getattr(self.reranker, "available", False) and combined:
                candidates = self.reranker.rerank(query, combined, top_k=self.config.expanded_top_k)
            else:
                candidates = combined[:self.config.expanded_top_k]

            # Re-assess evidence
            assessment = self.assessor.assess(query, candidates)
            decision = self.policy.decide(
                query=query,
                assessment=assessment,
                iteration=iteration,
                max_attempts=max_attempts,
            )

        final_candidates = self._select_by_score(candidates, target_k)
        decision.expansion_queries = all_expanded_queries
        decision.iteration = iteration

        duration = round((time.time() - t0) * 1000, 2)
        logger.info(
            f"Adaptive retrieval done in {duration}ms | Action: {decision.action.value} | "
            f"Attempts: {iteration} | Chunks: {len(final_candidates)} | Conf: {assessment.confidence}"
        )

        return final_candidates, decision, assessment, all_expanded_queries
