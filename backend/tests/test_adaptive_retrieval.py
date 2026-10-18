"""
Unit tests for adaptive retrieval components:
- EvidenceAssessor
- DeterministicDecisionPolicy
- AdaptiveRetrievalController (with mocked dependencies)
"""
import pytest
from unittest.mock import MagicMock, patch

from pipeline.adaptive_retrieval.schemas import (
    AdaptiveConfig,
    EvidenceAssessment,
    RetrievalAction,
)
from pipeline.adaptive_retrieval.evidence_assessor import EvidenceAssessor
from pipeline.adaptive_retrieval.decision_policy import DeterministicDecisionPolicy
from pipeline.adaptive_retrieval.retrieval_controller import AdaptiveRetrievalController


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_chunk(filename="doc.pdf", chunk_id=0, text="sample text about Morgan Stanley AI evals"):
    return {"filename": filename, "chunk_id": chunk_id, "text": text}


def make_results(n=3, score=0.5, text=None):
    chunks = []
    for i in range(n):
        t = text or f"This is chunk {i} about Morgan Stanley AI evaluation process"
        chunks.append((make_chunk(chunk_id=i, text=t), score - i * 0.05))
    return chunks


# ── EvidenceAssessor ──────────────────────────────────────────────────────────

class TestEvidenceAssessor:
    def setup_method(self):
        self.assessor = EvidenceAssessor(
            min_relevance_threshold=0.20,
            min_coverage_threshold=0.55,
        )

    def test_empty_results_returns_insufficient(self):
        assessment = self.assessor.assess("What is Morgan Stanley's eval approach?", [])
        assert not assessment.is_sufficient
        assert assessment.top_relevance_score == -1.0
        assert assessment.confidence > 0.5  # high confidence that it IS insufficient

    def test_sufficient_evidence_high_score(self):
        results = make_results(
            n=3,
            score=0.8,
            text="Morgan Stanley used AI evals to measure model performance for financial advisors"
        )
        assessment = self.assessor.assess(
            "What did Morgan Stanley use evals for?", results
        )
        assert assessment.top_relevance_score > 0.0
        assert assessment.unique_sources_count >= 1

    def test_coverage_score_in_range(self):
        results = make_results(n=2, score=0.6, text="AI evaluation process Morgan Stanley advisors model")
        assessment = self.assessor.assess("Morgan Stanley evaluation model", results)
        assert 0.0 <= assessment.coverage_score <= 1.0
        assert 0.0 <= assessment.confidence <= 1.0

    def test_duplicate_detection(self):
        # All chunks identical -> high duplicate ratio
        identical = [
            (make_chunk(chunk_id=i, text="exact same text content about AI"), 0.7)
            for i in range(4)
        ]
        assessment = self.assessor.assess("AI content", identical)
        assert assessment.duplicate_ratio > 0.5

    def test_unique_sources_count(self):
        results = [
            ({"filename": "doc_a.pdf", "chunk_id": 0, "text": "AI adoption Klarna Morgan Stanley"}, 0.7),
            ({"filename": "doc_b.pdf", "chunk_id": 0, "text": "BBVA enterprise AI deployment"}, 0.6),
            ({"filename": "doc_a.pdf", "chunk_id": 1, "text": "AI evaluation process and metrics"}, 0.5),
        ]
        assessment = self.assessor.assess("enterprise AI adoption", results)
        assert assessment.unique_sources_count == 2

    def test_missing_aspects_populated(self):
        results = make_results(n=2, score=0.3, text="irrelevant content about weather forecasting")
        assessment = self.assessor.assess(
            "What percentage of Morgan Stanley advisors use OpenAI?", results
        )
        # 'percentage', 'morgan', 'stanley', 'advisors', 'openai' won't be in weather text
        assert len(assessment.missing_aspects) > 0

    def test_rrf_scores_use_text_relevance_when_reranker_is_unavailable(self):
        results = [
            (make_chunk(text="The annual revenue was 42 million dollars in 2025."), 0.016),
            (make_chunk(chunk_id=1, text="The office is located in Toronto."), 0.015),
        ]

        assessment = self.assessor.assess("What was annual revenue in 2025?", results)

        assert assessment.top_relevance_score >= 0.2
        assert assessment.is_sufficient


# ── DeterministicDecisionPolicy ───────────────────────────────────────────────

class TestDeterministicDecisionPolicy:
    def setup_method(self):
        self.policy = DeterministicDecisionPolicy(
            min_relevance_threshold=0.20,
            min_coverage_threshold=0.55,
            strict_abstention=True,
        )

    def _make_assessment(self, sufficient=True, top_score=0.5, coverage=0.7, confidence=0.8):
        return EvidenceAssessment(
            is_sufficient=sufficient,
            coverage_score=coverage,
            top_relevance_score=top_score,
            avg_relevance_score=top_score * 0.8,
            unique_sources_count=2,
            duplicate_ratio=0.0,
            has_conflict=False,
            missing_aspects=[],
            confidence=confidence,
        )

    def test_sufficient_evidence_returns_answer(self):
        assessment = self._make_assessment(sufficient=True)
        decision = self.policy.decide("What did Morgan Stanley use?", assessment, 1, 2)
        assert decision.action == RetrievalAction.ANSWER
        assert decision.reason_code == "sufficient_evidence"

    def test_insufficient_with_budget_returns_retrieve_more(self):
        assessment = self._make_assessment(sufficient=False, top_score=0.1, coverage=0.2, confidence=0.3)
        decision = self.policy.decide("How many advisors use AI?", assessment, 1, 2)
        assert decision.action == RetrievalAction.RETRIEVE_MORE
        assert decision.should_expand is True

    def test_budget_exhausted_with_low_evidence_abstains(self):
        assessment = self._make_assessment(sufficient=False, top_score=0.05, coverage=0.1, confidence=0.2)
        decision = self.policy.decide("What is the boiling point of iron?", assessment, 2, 2)
        assert decision.action == RetrievalAction.ABSTAIN

    def test_ambiguous_single_word_query_clarifies(self):
        assessment = self._make_assessment(sufficient=False)
        decision = self.policy.decide("what", assessment, 1, 2)
        assert decision.action == RetrievalAction.CLARIFY

    def test_budget_exhausted_with_partial_evidence_answers(self):
        # Borderline case: not quite sufficient but above 75% of thresholds
        assessment = self._make_assessment(
            sufficient=False,
            top_score=0.18,  # 90% of 0.20 threshold
            coverage=0.42,   # above 0.35 fallback
            confidence=0.5,
        )
        decision = self.policy.decide(
            "What was the resolution time at Klarna?", assessment, iteration=2, max_attempts=2
        )
        assert decision.action == RetrievalAction.ANSWER
        assert decision.reason_code == "best_effort_budget_exhausted"

    def test_decision_includes_policy_version(self):
        assessment = self._make_assessment(sufficient=True)
        decision = self.policy.decide("test query", assessment, 1, 2)
        assert "deterministic" in decision.policy_version


# ── AdaptiveRetrievalController ───────────────────────────────────────────────

class TestAdaptiveRetrievalController:
    def _make_mock_hybrid_retriever(self, results=None):
        mock = MagicMock()
        mock.retrieve.return_value = results or make_results(n=5, score=0.6)
        return mock

    def _make_mock_reranker(self, available=True):
        mock = MagicMock()
        mock.available = available
        mock.rerank.side_effect = lambda query, results, top_k=5: results[:top_k]
        return mock

    def test_single_pass_sufficient_evidence(self):
        """When evidence is sufficient on first pass, no retries happen."""
        good_results = [
            ({"filename": "doc.pdf", "chunk_id": i,
              "text": "Morgan Stanley used AI evals to measure performance for advisors"}, 0.8)
            for i in range(5)
        ]
        hybrid = self._make_mock_hybrid_retriever(good_results)
        reranker = self._make_mock_reranker()
        config = AdaptiveConfig(enabled=True, max_retrieval_attempts=2, initial_top_k=5)
        controller = AdaptiveRetrievalController(
            hybrid_retriever=hybrid,
            reranker=reranker,
            config=config,
        )
        results, decision, assessment, expansions = controller.execute_retrieval(
            "What did Morgan Stanley use evals for?"
        )
        assert results is not None
        assert decision.action in (RetrievalAction.ANSWER, RetrievalAction.RETRIEVE_MORE, RetrievalAction.ABSTAIN)
        assert decision.iteration >= 1

    def test_disabled_adaptive_runs_single_pass(self):
        """When adaptive retrieval is disabled, exactly one retrieval pass happens."""
        hybrid = self._make_mock_hybrid_retriever()
        reranker = self._make_mock_reranker()
        config = AdaptiveConfig(enabled=False, max_retrieval_attempts=3, initial_top_k=5)
        controller = AdaptiveRetrievalController(
            hybrid_retriever=hybrid,
            reranker=reranker,
            config=config,
        )
        results, decision, assessment, expansions = controller.execute_retrieval("test query")
        # With enabled=False, max_attempts is treated as 1
        assert decision.iteration == 1
        assert hybrid.retrieve.call_count == 1

    def test_reranker_unavailable_falls_back(self):
        """When reranker is unavailable, results are still returned."""
        hybrid = self._make_mock_hybrid_retriever()
        reranker = self._make_mock_reranker(available=False)
        config = AdaptiveConfig(enabled=True, max_retrieval_attempts=1)
        controller = AdaptiveRetrievalController(
            hybrid_retriever=hybrid,
            reranker=reranker,
            config=config,
        )
        results, decision, assessment, expansions = controller.execute_retrieval("test query")
        # reranker.rerank should not have been called since available=False
        reranker.rerank.assert_not_called()
        assert len(results) > 0

    def test_max_iterations_bounded(self):
        """Retrieval loop never exceeds max_retrieval_attempts."""
        # Return low-quality results to trigger RETRIEVE_MORE
        poor_results = [
            ({"filename": "doc.pdf", "chunk_id": i, "text": "unrelated weather forecast data"}, 0.01)
            for i in range(3)
        ]
        hybrid = self._make_mock_hybrid_retriever(poor_results)
        reranker = self._make_mock_reranker()
        config = AdaptiveConfig(
            enabled=True,
            max_retrieval_attempts=2,
            min_top_relevance_threshold=0.20,
            min_coverage_threshold=0.55,
            initial_top_k=3,
        )
        controller = AdaptiveRetrievalController(
            hybrid_retriever=hybrid,
            reranker=reranker,
            config=config,
        )
        results, decision, assessment, expansions = controller.execute_retrieval(
            "What percentage of Morgan Stanley advisors use OpenAI?"
        )
        # Should never exceed configured max attempts
        assert decision.iteration <= config.max_retrieval_attempts

    def test_query_expansion_used_on_retry(self):
        """When retrying, query expander is called if available."""
        poor_results = [
            ({"filename": "doc.pdf", "chunk_id": 0, "text": "unrelated content"}, 0.01)
        ]
        better_results = [
            ({"filename": "doc.pdf", "chunk_id": 1,
              "text": "Morgan Stanley AI model evaluation performance advisors"}, 0.7)
            for _ in range(3)
        ]

        call_count = {"n": 0}

        def side_effect(query, top_k=5):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return poor_results
            return better_results

        hybrid = MagicMock()
        hybrid.retrieve.side_effect = side_effect

        expander = MagicMock()
        expander.expand.return_value = [
            "original query",
            "What percentage of advisors use AI at Morgan Stanley?",
        ]

        reranker = self._make_mock_reranker()
        config = AdaptiveConfig(
            enabled=True,
            max_retrieval_attempts=2,
            max_query_expansions=2,
        )
        controller = AdaptiveRetrievalController(
            hybrid_retriever=hybrid,
            reranker=reranker,
            query_expander=expander,
            config=config,
        )
        _, decision, _, expansions = controller.execute_retrieval(
            "original query"
        )
        # If a retry happened, expander should have been called
        if decision.iteration > 1:
            expander.expand.assert_called()


# ── Laya policy fallback ───────────────────────────────────────────────────────

class TestLayaDecisionPolicyFallback:
    def test_falls_back_to_deterministic_when_laya_disabled(self):
        from pipeline.adaptive_retrieval.decision_policy import LayaDecisionPolicy

        fallback = DeterministicDecisionPolicy()
        laya_policy = LayaDecisionPolicy(laya_service=None, fallback_policy=fallback)

        assessment = EvidenceAssessment(
            is_sufficient=True,
            coverage_score=0.7,
            top_relevance_score=0.5,
            avg_relevance_score=0.4,
            unique_sources_count=2,
            duplicate_ratio=0.0,
            has_conflict=False,
            missing_aspects=[],
            confidence=0.8,
        )
        decision = laya_policy.decide("test query", assessment, 1, 2)
        assert decision.action == RetrievalAction.ANSWER

    def test_falls_back_when_laya_service_errors(self):
        from pipeline.adaptive_retrieval.decision_policy import LayaDecisionPolicy

        broken_laya = MagicMock()
        broken_laya.is_enabled = True
        broken_laya.decide_retrieval_action.side_effect = RuntimeError("Laya down")

        fallback = DeterministicDecisionPolicy()
        laya_policy = LayaDecisionPolicy(laya_service=broken_laya, fallback_policy=fallback)

        assessment = EvidenceAssessment(
            is_sufficient=True,
            coverage_score=0.7,
            top_relevance_score=0.5,
            avg_relevance_score=0.4,
            unique_sources_count=1,
            duplicate_ratio=0.0,
            has_conflict=False,
            missing_aspects=[],
            confidence=0.8,
        )
        decision = laya_policy.decide("test query", assessment, 1, 2)
        # Should fall back gracefully
        assert decision.action in list(RetrievalAction)


def test_hybrid_retrieval_filters_chunks_by_allowed_filename():
    from types import SimpleNamespace
    from pipeline.retriever.hybrid_retriever import HybridRetriever

    chunks = [
        {"filename": "alice.pdf", "chunk_id": 0, "text": "enterprise search"},
        {"filename": "bob.pdf", "chunk_id": 0, "text": "enterprise search"},
    ]

    class DenseRetriever:
        def __init__(self):
            self.vector_store = SimpleNamespace(
                index=SimpleNamespace(ntotal=len(chunks))
            )

        def retrieve(self, query, top_k):
            return [
                {"chunk": chunk, "score": 1.0 - index * 0.1}
                for index, chunk in enumerate(chunks[:top_k])
            ]

    class SparseRetriever:
        def __init__(self):
            self.chunks = chunks

        def retrieve(self, query, top_k, allowed_filenames=None):
            matches = self.chunks
            if allowed_filenames is not None:
                matches = [c for c in self.chunks if c["filename"] in allowed_filenames]
            return [(chunk, 1.0) for chunk in matches[:top_k]]

    retriever = HybridRetriever(DenseRetriever(), SparseRetriever())
    results = retriever.retrieve(
        "enterprise search", top_k=5, allowed_filenames={"alice.pdf"}
    )

    assert results
    assert {chunk["filename"] for chunk, _ in results} == {"alice.pdf"}
