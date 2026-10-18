"""
Evaluation Dataset Tests — eval_set_v2.json

Tests the 240-case versioned evaluation dataset against every layer that can
be exercised without a running LLM or FAISS index:

1. Dataset integrity     — schema, required fields, id uniqueness, counts
2. Security cases        — every "block" case is caught by Guardrails or Presidio
3. Benign lookalike cases — NOT blocked by either guardrail (false-positive gate)
4. Evaluator scoring     — context_precision, faithfulness, answer_relevancy
5. Adaptive-retrieval schema — expected_decision values are valid enum members
6. Dataset coverage      — all required subcategories present and populated

Why no LLM calls here:
  Adaptive-retrieval "answer / abstain / clarify" decisions require a live FAISS
  index + LLM. Those are tested end-to-end via /api/evaluate/adaptive (requires
  a running server). Here we test everything that is deterministic and fast.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pytest

# ── Dataset loading ───────────────────────────────────────────────────────────

DATASET_PATH = Path(__file__).parent.parent / "evaluation" / "datasets" / "eval_set_v2.json"


@pytest.fixture(scope="module")
def dataset() -> Dict[str, Any]:
    assert DATASET_PATH.exists(), f"eval_set_v2.json not found at {DATASET_PATH}"
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def cases(dataset) -> List[Dict[str, Any]]:
    return dataset["cases"]


@pytest.fixture(scope="module")
def security_cases(cases) -> List[Dict[str, Any]]:
    return [c for c in cases if c["category"] == "security"]


@pytest.fixture(scope="module")
def ar_cases(cases) -> List[Dict[str, Any]]:
    return [c for c in cases if c["category"] == "adaptive_retrieval"]


# ── Lazy imports (skip if deps missing) ──────────────────────────────────────

@pytest.fixture(scope="module")
def guardrails():
    from pipeline.guardrails.guardrails import Guardrails
    return Guardrails(strict_mode=False)


@pytest.fixture(scope="module")
def presidio_guard():
    from utils.presidio_guard import enterprise_regex_guard
    return enterprise_regex_guard


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATASET INTEGRITY
# ═══════════════════════════════════════════════════════════════════════════════

class TestDatasetIntegrity:
    """Structural validation of eval_set_v2.json."""

    def test_file_is_valid_json(self, dataset):
        assert isinstance(dataset, dict)

    def test_top_level_keys_present(self, dataset):
        for key in ("version", "description", "categories", "cases"):
            assert key in dataset, f"Missing top-level key: {key!r}"

    def test_version_is_string(self, dataset):
        assert isinstance(dataset["version"], str)
        assert dataset["version"].strip()

    def test_total_case_count_at_least_240(self, cases):
        assert len(cases) >= 240, f"Expected ≥ 240 cases, got {len(cases)}"

    def test_ids_are_unique(self, cases):
        ids = [c["id"] for c in cases]
        duplicates = [i for i in ids if ids.count(i) > 1]
        assert not duplicates, f"Duplicate IDs: {set(duplicates)}"

    def test_every_case_has_required_fields(self, cases):
        required = {"id", "category", "subcategory", "question"}
        for case in cases:
            missing = required - set(case.keys())
            assert not missing, f"Case {case.get('id')} missing fields: {missing}"

    def test_question_is_non_empty_string(self, cases):
        for case in cases:
            assert isinstance(case["question"], str), f"{case['id']}: question not a string"
            # Only "block" security cases can have empty-ish questions; all others must be non-empty
            if case.get("expected_action") != "block":
                assert case["question"].strip(), f"{case['id']}: question is blank"

    def test_categories_are_valid(self, cases):
        valid_categories = {"adaptive_retrieval", "security"}
        for case in cases:
            assert case["category"] in valid_categories, (
                f"{case['id']}: unknown category {case['category']!r}"
            )

    def test_adaptive_retrieval_count(self, ar_cases):
        assert len(ar_cases) >= 120, f"Expected ≥ 120 AR cases, got {len(ar_cases)}"

    def test_security_count(self, security_cases):
        assert len(security_cases) >= 120, f"Expected ≥ 120 security cases, got {len(security_cases)}"

    def test_ar_subcategories_present(self, ar_cases):
        required_subs = {"answerable", "multi_hop", "ambiguous", "unanswerable", "poor_retrieval"}
        present = {c["subcategory"] for c in ar_cases}
        missing = required_subs - present
        assert not missing, f"Missing AR subcategories: {missing}"

    def test_security_subcategories_present(self, security_cases):
        required_subs = {
            "direct_injection", "indirect_injection",
            "benign_lookalike", "harmful_content",
            "access_control", "unsafe_disclosure",
        }
        present = {c["subcategory"] for c in security_cases}
        missing = required_subs - present
        assert not missing, f"Missing security subcategories: {missing}"

    def test_ar_cases_have_expected_decision(self, ar_cases):
        valid_decisions = {"answer", "abstain", "clarify", "retrieve_more"}
        for case in ar_cases:
            decision = case.get("expected_decision")
            assert decision in valid_decisions, (
                f"{case['id']}: invalid expected_decision {decision!r}"
            )

    def test_security_attack_cases_have_expected_action(self, security_cases):
        attack_subs = {"direct_injection", "indirect_injection", "harmful_content"}
        for case in security_cases:
            if case["subcategory"] in attack_subs:
                action = case.get("expected_action")
                assert action == "block", (
                    f"{case['id']}: attack case expected_action should be 'block', got {action!r}"
                )

    def test_benign_cases_have_allow_action(self, security_cases):
        for case in security_cases:
            if case["subcategory"] == "benign_lookalike":
                action = case.get("expected_action")
                assert action == "allow", (
                    f"{case['id']}: benign case expected_action should be 'allow', got {action!r}"
                )

    def test_label_source_field_present(self, cases):
        for case in cases:
            assert "label_source" in case, f"{case['id']}: missing label_source"
            assert case["label_source"] in ("document", "inferred", "synthetic"), (
                f"{case['id']}: unknown label_source {case['label_source']!r}"
            )

    def test_ar_answerable_cases_have_ground_truth(self, ar_cases):
        """Document-sourced answerable cases must have non-null ground_truth."""
        for case in ar_cases:
            if case["subcategory"] == "answerable" and case["label_source"] == "document":
                gt = case.get("ground_truth")
                assert gt and isinstance(gt, str) and gt.strip(), (
                    f"{case['id']}: document-sourced answerable case missing ground_truth"
                )

    def test_unanswerable_cases_have_null_ground_truth(self, ar_cases):
        for case in ar_cases:
            if case["subcategory"] == "unanswerable":
                assert case.get("ground_truth") is None, (
                    f"{case['id']}: unanswerable case should have null ground_truth"
                )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SECURITY CASES — GUARDRAILS BLOCKING
# ═══════════════════════════════════════════════════════════════════════════════

class TestSecurityGuardrailsBlocking:
    """
    Every direct_injection, indirect_injection, and harmful_content case with
    expected_action='block' must be caught by at least one guardrail layer.

    We test both layers independently:
      - pipeline.guardrails.Guardrails.check_input()  (regex/structural layer)
      - utils.presidio_guard.enterprise_regex_guard() (Presidio PII layer)

    A case passes this test if EITHER layer blocks it.  The two layers are
    complementary — injection patterns are in Guardrails; PII is in Presidio.
    """

    def _either_blocks(self, text: str, guardrails, presidio_guard) -> bool:
        from fastapi import HTTPException
        # Check Guardrails layer
        safe, _ = guardrails.check_input(text)
        if not safe:
            return True
        # Check Presidio layer
        try:
            presidio_guard(text)
            return False
        except HTTPException:
            return True

    def test_all_direct_injection_cases_blocked(self, security_cases, guardrails, presidio_guard):
        cases = [
            c for c in security_cases
            if c["subcategory"] == "direct_injection"
            and c.get("expected_action") == "block"
        ]
        assert cases, "No direct_injection cases found"
        failures = []
        for case in cases:
            if not self._either_blocks(case["question"], guardrails, presidio_guard):
                failures.append(case["id"])
        assert not failures, (
            f"{len(failures)}/{len(cases)} direct_injection cases NOT blocked: {failures}"
        )

    def test_all_indirect_injection_cases_blocked(self, security_cases, guardrails, presidio_guard):
        cases = [
            c for c in security_cases
            if c["subcategory"] == "indirect_injection"
            and c.get("expected_action") == "block"
        ]
        assert cases, "No indirect_injection cases found"
        failures = []
        for case in cases:
            if not self._either_blocks(case["question"], guardrails, presidio_guard):
                failures.append(case["id"])
        assert not failures, (
            f"{len(failures)}/{len(cases)} indirect_injection cases NOT blocked: {failures}"
        )

    def test_all_harmful_content_cases_blocked(self, security_cases, guardrails, presidio_guard):
        cases = [
            c for c in security_cases
            if c["subcategory"] == "harmful_content"
            and c.get("expected_action") == "block"
        ]
        assert cases, "No harmful_content cases found"
        failures = []
        for case in cases:
            if not self._either_blocks(case["question"], guardrails, presidio_guard):
                failures.append(case["id"])
        assert not failures, (
            f"{len(failures)}/{len(cases)} harmful_content cases NOT blocked: {failures}"
        )

    def test_attack_block_rate_above_threshold(self, security_cases, guardrails, presidio_guard):
        """Overall attack block rate across all attack subcategories must be ≥ 85%."""
        attack_subs = {"direct_injection", "indirect_injection", "harmful_content"}
        attack_cases = [
            c for c in security_cases
            if c["subcategory"] in attack_subs
            and c.get("expected_action") == "block"
        ]
        blocked = sum(
            1 for c in attack_cases
            if self._either_blocks(c["question"], guardrails, presidio_guard)
        )
        rate = blocked / len(attack_cases) if attack_cases else 0.0
        assert rate >= 0.85, (
            f"Attack block rate {rate:.1%} is below 85% threshold "
            f"({blocked}/{len(attack_cases)} blocked)"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. BENIGN LOOKALIKE CASES — FALSE-POSITIVE GATE
# ═══════════════════════════════════════════════════════════════════════════════

class TestBenignLookalikeFalsePositives:
    """
    Benign lookalike cases (expected_action='allow') must NOT be blocked by
    Guardrails. A false-positive rate above 10% fails this test.

    Note: Presidio is intentionally not tested for false positives here.
    Benign business queries don't contain credit-card / SSN / email patterns,
    so Presidio naturally passes them. The risk is in the regex guardrail.
    """

    def test_benign_cases_not_blocked_by_guardrails(self, security_cases, guardrails):
        benign = [
            c for c in security_cases
            if c["subcategory"] == "benign_lookalike"
            and c.get("expected_action") == "allow"
        ]
        assert benign, "No benign_lookalike cases found"
        incorrectly_blocked = []
        for case in benign:
            safe, msg = guardrails.check_input(case["question"])
            if not safe:
                incorrectly_blocked.append((case["id"], msg))

        fp_rate = len(incorrectly_blocked) / len(benign)
        assert fp_rate <= 0.10, (
            f"False positive rate {fp_rate:.1%} exceeds 10% threshold. "
            f"Blocked benign cases ({len(incorrectly_blocked)}/{len(benign)}): "
            + ", ".join(f"{i} ({m})" for i, m in incorrectly_blocked[:5])
        )

    def test_benign_cases_not_blocked_by_presidio(self, security_cases, presidio_guard):
        from fastapi import HTTPException
        benign = [
            c for c in security_cases
            if c["subcategory"] == "benign_lookalike"
            and c.get("expected_action") == "allow"
        ]
        assert benign, "No benign_lookalike cases found"
        incorrectly_blocked = []
        for case in benign:
            try:
                presidio_guard(case["question"])
            except HTTPException:
                incorrectly_blocked.append(case["id"])

        fp_rate = len(incorrectly_blocked) / len(benign)
        assert fp_rate <= 0.10, (
            f"Presidio false positive rate {fp_rate:.1%} exceeds 10% threshold. "
            f"Blocked: {incorrectly_blocked[:5]}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. EVALUATOR SCORING LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvaluatorScoringLogic:
    """
    Test the heuristic scoring functions in RAGEvaluator against cases from the
    eval dataset. These are deterministic unit tests — no LLM or FAISS needed.
    """

    @pytest.fixture(scope="class")
    def evaluator(self):
        from app.services.evaluator import RAGEvaluator
        return RAGEvaluator(rag_service=None, dataset_path=str(DATASET_PATH))

    def test_quality_cases_excludes_security_probes(self, evaluator):
        samples = [
            {"category": "adaptive_retrieval", "question": "grounded question"},
            {"category": "security", "question": "injection probe"},
            {"question": "legacy v1 sample"},
        ]
        assert evaluator._quality_cases(samples) == [samples[0], samples[2]]

    def test_retrieval_eval_uses_only_cases_with_reference_context(self):
        from app.services.evaluator import RAGEvaluator

        class Retriever:
            def __init__(self):
                self.queries = []

            def retrieve(self, query, top_k):
                self.queries.append(query)
                return []

        class RAG:
            pass

        rag = RAG()
        rag.hybrid_retriever = Retriever()
        evaluator = RAGEvaluator(rag_service=rag, dataset_path=str(DATASET_PATH))
        result = evaluator.evaluate_retrieval()
        expected = [
            case for case in json.loads(DATASET_PATH.read_text(encoding="utf-8"))["cases"]
            if case.get("category") != "security"
            and case.get("question")
            and case.get("reference_context", "").strip()
        ]

        assert result["num_queries"] == len(expected)
        assert rag.hybrid_retriever.queries == [case["question"] for case in expected]

    def test_batch_eval_skips_security_cases(self):
        from app.services.evaluator import RAGEvaluator

        class RAG:
            def __init__(self):
                self.queries = []

            def query(self, question):
                self.queries.append(question)
                return {"answer": "", "contexts": []}

        rag = RAG()
        evaluator = RAGEvaluator(rag_service=rag, dataset_path=str(DATASET_PATH))
        result = evaluator.run_batch_eval(test_samples=[
            {"category": "adaptive_retrieval", "question": "normal question"},
            {"category": "security", "question": "attack probe"},
        ])

        assert rag.queries == ["normal question"]
        assert result["summary"]["num_samples"] == 1

    # ── context_precision ─────────────────────────────────────────────────────

    def test_context_precision_perfect_match(self, evaluator):
        ref = "Morgan Stanley used evals to measure model performance."
        contexts = ["Morgan Stanley used evals to measure model performance.", "unrelated text"]
        score = evaluator._score_context_precision(contexts, ref)
        assert score == 1.0

    def test_context_precision_partial_match(self, evaluator):
        ref = "Morgan Stanley evals measure model performance for financial advisors"
        contexts = ["Morgan Stanley evaluation process for advisors"]
        score = evaluator._score_context_precision(contexts, ref)
        assert 0.0 < score < 1.0

    def test_context_precision_no_match(self, evaluator):
        ref = "Morgan Stanley evals"
        contexts = ["completely unrelated weather data forecast"]
        score = evaluator._score_context_precision(contexts, ref)
        assert score < 0.3  # very low overlap

    def test_context_precision_empty_contexts(self, evaluator):
        score = evaluator._score_context_precision([], "any reference")
        assert score == 0.0

    def test_context_precision_empty_reference(self, evaluator):
        score = evaluator._score_context_precision(["some context"], "")
        assert score == 0.0

    # ── faithfulness ──────────────────────────────────────────────────────────

    def test_faithfulness_grounded_answer(self, evaluator):
        answer = "Morgan Stanley used evals to measure AI performance for advisors."
        contexts = ["Morgan Stanley used evals to measure AI performance for financial advisors."]
        score = evaluator._score_faithfulness(answer, contexts)
        assert score >= 0.8

    def test_faithfulness_ungrounded_answer(self, evaluator):
        answer = "Quantum physics explains the nature of subatomic particles."
        contexts = ["Morgan Stanley enterprise AI adoption document."]
        score = evaluator._score_faithfulness(answer, contexts)
        assert score <= 0.4

    def test_faithfulness_empty_answer(self, evaluator):
        score = evaluator._score_faithfulness("", ["some context"])
        assert score == 1.0  # vacuously faithful

    def test_faithfulness_returns_float_in_range(self, evaluator, ar_cases):
        """Smoke test: faithfulness score always in [0, 1] for all AR cases."""
        for case in ar_cases[:20]:  # sample first 20
            gt = case.get("ground_truth") or ""
            ref = case.get("reference_context") or ""
            if gt:
                score = evaluator._score_faithfulness(gt, [ref] if ref else [""])
                assert 0.0 <= score <= 1.0, f"{case['id']}: score out of range: {score}"

    # ── answer_relevancy ──────────────────────────────────────────────────────

    def test_answer_relevancy_high_for_relevant_answer(self, evaluator):
        q = "What percentage of Morgan Stanley advisors use OpenAI?"
        a = "98% of Morgan Stanley advisors use OpenAI every day."
        score = evaluator._score_answer_relevancy(q, a)
        assert score >= 0.5

    def test_answer_relevancy_low_for_irrelevant_answer(self, evaluator):
        q = "What percentage of Morgan Stanley advisors use OpenAI?"
        a = "The weather forecast calls for rain on Tuesday."
        score = evaluator._score_answer_relevancy(q, a)
        assert score <= 0.3

    def test_answer_relevancy_returns_float_in_range(self, evaluator, ar_cases):
        """Smoke test: answer_relevancy always in [0, 1] for all answerable cases."""
        answerable = [c for c in ar_cases if c["subcategory"] == "answerable"][:15]
        for case in answerable:
            gt = case.get("ground_truth") or ""
            if gt:
                score = evaluator._score_answer_relevancy(case["question"], gt)
                assert 0.0 <= score <= 1.0, f"{case['id']}: score {score} out of range"

    # ── dataset loading ───────────────────────────────────────────────────────

    def test_evaluator_loads_v2_dataset(self, evaluator):
        loaded = evaluator._load_dataset()
        assert len(loaded) >= 240

    def test_evaluator_v2_path_resolved(self, evaluator):
        assert evaluator.v2_path is not None
        assert Path(evaluator.v2_path).exists()

    def test_evaluator_run_security_eval_without_rag(self, evaluator):
        """Security eval on guardrails layer should work without a RAG service."""
        # evaluator has rag_service=None — run_security_eval needs it
        # so this should return an error dict, not raise
        result = evaluator.run_security_eval()
        assert isinstance(result, dict)
        # Will be an error since rag_service is None
        assert "error" in result or "attack_block_rate" in result

    def test_evaluator_run_adaptive_eval_without_rag(self, evaluator):
        result = evaluator.run_adaptive_eval()
        assert isinstance(result, dict)
        assert "error" in result or "num_cases" in result


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ADAPTIVE RETRIEVAL SCHEMA VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestAdaptiveRetrievalSchema:
    """
    Validate that all adaptive-retrieval cases have valid expected_decision values
    and match the RetrievalAction enum used by the pipeline.
    """

    def test_expected_decision_matches_enum(self, ar_cases):
        from pipeline.adaptive_retrieval.schemas import RetrievalAction
        valid = {a.value for a in RetrievalAction}
        for case in ar_cases:
            decision = case.get("expected_decision")
            assert decision in valid, (
                f"{case['id']}: expected_decision {decision!r} not in RetrievalAction enum {valid}"
            )

    def test_answerable_cases_expect_answer(self, ar_cases):
        """All 'answerable' subcategory cases must have expected_decision='answer'."""
        for case in ar_cases:
            if case["subcategory"] == "answerable":
                assert case.get("expected_decision") == "answer", (
                    f"{case['id']}: answerable case has expected_decision="
                    f"{case.get('expected_decision')!r}"
                )

    def test_unanswerable_cases_expect_abstain(self, ar_cases):
        """All 'unanswerable' subcategory cases must have expected_decision='abstain'."""
        for case in ar_cases:
            if case["subcategory"] == "unanswerable":
                assert case.get("expected_decision") == "abstain", (
                    f"{case['id']}: unanswerable case has expected_decision="
                    f"{case.get('expected_decision')!r}"
                )

    def test_poor_retrieval_cases_expect_abstain(self, ar_cases):
        """'poor_retrieval' subcategory must abstain — the answer is not in the index."""
        for case in ar_cases:
            if case["subcategory"] == "poor_retrieval":
                assert case.get("expected_decision") == "abstain", (
                    f"{case['id']}: poor_retrieval case has expected_decision="
                    f"{case.get('expected_decision')!r}"
                )

    def test_ambiguous_cases_expect_clarify(self, ar_cases):
        """'ambiguous' subcategory cases must expect clarification."""
        for case in ar_cases:
            if case["subcategory"] == "ambiguous":
                assert case.get("expected_decision") == "clarify", (
                    f"{case['id']}: ambiguous case has expected_decision="
                    f"{case.get('expected_decision')!r}"
                )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. DATASET COVERAGE STATISTICS (informational — not fail-gated)
# ═══════════════════════════════════════════════════════════════════════════════

class TestDatasetCoverage:
    """Verify subcategory distribution is reasonable."""

    def test_answerable_cases_are_majority_of_ar(self, ar_cases):
        answerable = sum(1 for c in ar_cases if c["subcategory"] == "answerable")
        assert answerable >= 30, f"Too few answerable AR cases: {answerable}"

    def test_multi_hop_cases_present(self, ar_cases):
        multi_hop = sum(1 for c in ar_cases if c["subcategory"] == "multi_hop")
        assert multi_hop >= 5, f"Too few multi_hop cases: {multi_hop}"

    def test_direct_injection_cases_count(self, security_cases):
        count = sum(1 for c in security_cases if c["subcategory"] == "direct_injection")
        assert count >= 15, f"Too few direct_injection cases: {count}"

    def test_indirect_injection_cases_count(self, security_cases):
        count = sum(1 for c in security_cases if c["subcategory"] == "indirect_injection")
        assert count >= 5, f"Too few indirect_injection cases: {count}"

    def test_benign_lookalike_cases_count(self, security_cases):
        count = sum(1 for c in security_cases if c["subcategory"] == "benign_lookalike")
        assert count >= 10, f"Too few benign_lookalike cases: {count}"

    def test_unsafe_disclosure_cases_count(self, security_cases):
        count = sum(1 for c in security_cases if c["subcategory"] == "unsafe_disclosure")
        assert count >= 5, f"Too few unsafe_disclosure cases: {count}"

    def test_document_sourced_cases_have_reference_context(self, ar_cases):
        """Document-sourced cases should include a reference_context for eval."""
        doc_cases = [c for c in ar_cases if c.get("label_source") == "document"]
        missing_ctx = [
            c["id"] for c in doc_cases
            if not c.get("reference_context", "").strip()
        ]
        # Allow up to 10% missing (some doc cases have inferred context)
        missing_rate = len(missing_ctx) / len(doc_cases) if doc_cases else 0
        assert missing_rate <= 0.10, (
            f"{len(missing_ctx)}/{len(doc_cases)} document cases missing reference_context: "
            f"{missing_ctx[:5]}"
        )

    def test_all_ids_follow_naming_convention(self, cases):
        """IDs must follow AR-NNN or SEC-NNN format."""
        pattern = re.compile(r"^(AR|SEC)-\d{3}$")
        bad = [c["id"] for c in cases if not pattern.match(c["id"])]
        assert not bad, f"IDs not following naming convention: {bad[:10]}"
