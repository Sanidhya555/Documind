"""Regression coverage for dynamic context sizing, answer budgets, and caching."""
from pathlib import Path

from app.services.evaluator import RAGEvaluator
from app.services.query_cache import QueryCache
from app.services.rag_services import RAGService
from pipeline.adaptive_retrieval.retrieval_controller import AdaptiveRetrievalController
from pipeline.adaptive_retrieval.schemas import EvidenceAssessment, RetrievalAction, RetrievalDecision
from pipeline.llm.prompt_template import PromptTemplate
from pipeline.guardrails.guardrails import Guardrails
from pipeline.memory.short_term import ShortTermMemory


def test_query_cache_hits_and_is_scoped_by_user_session_and_retrieval_inputs():
    cache = QueryCache()
    payload = {"answer": "grounded answer"}
    cache.set("What is the policy?", "session-a", payload, user="alice", top_k=None)

    hit = cache.get(" what is the policy? ", "session-a", user="alice", top_k=None)
    assert hit == payload
    assert cache.get("What is the policy?", "session-a", user="bob") is None
    assert cache.get("What is the policy?", "session-b", user="alice") is None
    assert cache.get("What is the policy?", "session-a", user="alice", top_k=4) is None
    assert cache.get("What is the policy?", "session-a", user="alice", history="prior turn") is None
    assert cache.get("What is the policy?", "session-a", user="alice", index_version=1) is None
    assert (cache.hits, cache.misses, cache.size) == (1, 5, 1)


def test_query_cache_expires_entries_and_evicts_least_recently_used(monkeypatch):
    now = [100.0]
    monkeypatch.setattr("app.services.query_cache.time.time", lambda: now[0])
    cache = QueryCache(max_size=2, ttl=10)
    cache.set("first", "s", {"answer": "1"})
    cache.set("second", "s", {"answer": "2"})
    assert cache.get("first", "s") == {"answer": "1"}  # refresh first in LRU order
    cache.set("third", "s", {"answer": "3"})
    assert cache.get("second", "s") is None
    assert cache.get("third", "s") == {"answer": "3"}

    now[0] += 11
    assert cache.size == 0
    assert cache.get("first", "s") is None


def test_query_cache_can_invalidate_one_session_without_clearing_other_users():
    cache = QueryCache()
    cache.set("same question", "session-a", {"answer": "a"}, user="alice")
    cache.set("same question", "session-b", {"answer": "b"}, user="bob")

    cache.clear_session("session-a", user="alice")

    assert cache.get("same question", "session-a", user="alice") is None
    assert cache.get("same question", "session-b", user="bob") == {"answer": "b"}


def test_rag_query_replays_cached_answer_and_skips_retrieval_and_generation():
    class FakeController:
        calls = 0

        def execute_retrieval(self, *_args, **_kwargs):
            self.calls += 1
            return (
                [({"text": "Employees receive 18 days of paid annual leave.", "filename": "handbook.pdf", "chunk_id": 1}, 0.95)],
                RetrievalDecision(
                    action=RetrievalAction.ANSWER,
                    reason_code="sufficient_evidence",
                    iteration=1,
                ),
                EvidenceAssessment(
                    is_sufficient=True,
                    coverage_score=1.0,
                    top_relevance_score=0.95,
                    confidence=0.95,
                ),
                [],
            )

    class FakeGenerator:
        calls = 0

        def generate(self, _prompt, max_tokens):
            self.calls += 1
            assert max_tokens >= 256
            return "Employees receive 18 days of paid annual leave."

    class FakeLongTerm:
        def save_message(self, *_args):
            pass

        def save_turn(self, *_args):
            pass

        def list_sessions(self):
            return []

        def upsert_session(self, *_args, **_kwargs):
            pass

    service = RAGService.__new__(RAGService)
    service.guardrails = Guardrails()
    service.short_term = ShortTermMemory(max_turns=10)
    service.long_term = FakeLongTerm()
    service.cache = QueryCache()
    service.index_version = 1
    service.adaptive_controller = FakeController()
    service.generator = FakeGenerator()
    service.evaluator = RAGEvaluator()
    service.allowed_filenames = lambda _user: None
    service._auto_title = lambda *_args: None

    first = service.query("How many paid annual leave days are provided?", "session-a", user="alice")
    second = service.query("How many paid annual leave days are provided?", "session-a", user="alice")

    assert first["cached"] is False
    assert second["cached"] is True
    assert second["answer"] == first["answer"]
    assert second["metrics"]["cache_hit"] is True
    assert second["metrics"]["retrieval_time"] == 0.0
    assert second["metrics"]["generation_time"] == 0.0
    assert second["metrics"]["faithfulness_score"] == first["metrics"]["faithfulness_score"]
    assert service.adaptive_controller.calls == 1
    assert service.generator.calls == 1


def test_rag_query_caches_abstentions_and_keeps_faithfulness_stable():
    class FakeController:
        calls = 0

        def execute_retrieval(self, *_args, **_kwargs):
            self.calls += 1
            return (
                [],
                RetrievalDecision(
                    action=RetrievalAction.ABSTAIN,
                    reason_code="insufficient_evidence",
                    abstention_reason="No supporting document evidence was found.",
                ),
                EvidenceAssessment(
                    is_sufficient=False,
                    coverage_score=0.0,
                    top_relevance_score=0.0,
                    confidence=0.0,
                ),
                [],
            )

    class FakeLongTerm:
        def save_turn(self, *_args):
            pass

    service = RAGService.__new__(RAGService)
    service.guardrails = Guardrails()
    service.short_term = ShortTermMemory(max_turns=10)
    service.long_term = FakeLongTerm()
    service.cache = QueryCache()
    service.index_version = 1
    service.adaptive_controller = FakeController()
    service.evaluator = RAGEvaluator()
    service.allowed_filenames = lambda _user: None

    first = service.query("What is the reimbursement deadline?", "session-a", user="alice")
    second = service.query("What is the reimbursement deadline?", "session-a", user="alice")

    assert first["cached"] is False
    assert second["cached"] is True
    assert second["answer"] == first["answer"]
    assert second["metrics"]["faithfulness_score"] == first["metrics"]["faithfulness_score"]
    assert second["metrics"]["cache_hit"] is True
    assert service.adaptive_controller.calls == 1


def test_score_distribution_selects_a_query_specific_context_size():
    candidates = [
        ({"text": f"passage {i}"}, score)
        for i, score in enumerate((0.92, 0.88, 0.86, 0.61, 0.54))
    ]
    selected = AdaptiveRetrievalController._select_by_score(candidates, max_results=5)

    assert 1 < len(selected) < len(candidates)
    assert [score for _, score in selected] == [0.92, 0.88, 0.86]


def test_answer_budget_grows_with_query_complexity_and_evidence():
    simple = PromptTemplate.answer_token_budget("How many annual leave days are there?", "short context")
    analytical = PromptTemplate.answer_token_budget(
        "Explain and compare the architecture, methods, limitations, and relationships described in the documents.",
        "supporting evidence " * 500,
    )

    assert simple == 320
    assert analytical > simple
    assert analytical <= 2200


def test_stage_dataset_integrity_covers_all_included_pipeline_datasets():
    eval_set = Path(__file__).resolve().parents[1] / "evaluation" / "datasets" / "eval_set_v2.json"
    report = RAGEvaluator(dataset_path=str(eval_set)).run_stage_dataset_integrity()

    assert report["status"] == "passed"
    assert report["num_datasets"] == 21
    assert report["num_cases"] == 2100
    assert all(dataset["status"] == "passed" for dataset in report["datasets"])
