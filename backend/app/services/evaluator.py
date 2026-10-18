"""
RAG Evaluator — v3.

Supports:
- Original eval_set.json (v1 format) — backward compatible
- New eval_set_v2.json — 240 cases with adaptive-retrieval and security categories
- Measures: context precision, faithfulness, answer relevancy,
  abstention quality, security block rate, false-positive rate
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("rag_app")

try:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
    from datasets import Dataset
    RAGAS_ENABLED = True
except ImportError:
    RAGAS_ENABLED = False


class RAGEvaluator:
    def __init__(self, rag_service=None, dataset_path: Optional[str] = None):
        self.rag = rag_service
        self.dataset_path = dataset_path
        self.v2_path = self._resolve_v2_path(dataset_path)
        self.metrics = (
            [faithfulness, answer_relevancy, context_precision] if RAGAS_ENABLED else []
        )

    def _resolve_v2_path(self, base_path: Optional[str]) -> Optional[str]:
        if not base_path:
            return None
        p = Path(base_path)
        candidates = [
            p if p.name == "eval_set_v2.json" else p.parent / "eval_set_v2.json",
            p.parent / "evaluation" / "datasets" / "eval_set_v2.json",
        ]
        return str(next((candidate for candidate in candidates if candidate.is_file()), candidates[-1]))

    def _load_dataset(self, path: Optional[str] = None) -> List[Dict[str, Any]]:
        target = path or self.dataset_path
        if not target:
            raise FileNotFoundError("No evaluation dataset configured.")
        p = Path(target)
        if not p.exists():
            raise FileNotFoundError(f"Dataset file not found: {target}")
        data = json.loads(p.read_text(encoding="utf-8"))
        # v2 format wraps in {version, cases: [...]}
        if isinstance(data, dict) and "cases" in data:
            return data["cases"]
        return data

    @staticmethod
    def _quality_cases(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Exclude security probes from RAG quality evaluation runs."""
        return [case for case in dataset if case.get("category") != "security"]

    def run_stage_dataset_integrity(self) -> Dict[str, Any]:
        """Check every included stage dataset and its references to the test corpus."""
        backend_dir = Path(__file__).resolve().parents[2]
        evaluation_dir = backend_dir / "evaluation"
        dataset_dir = evaluation_dir / "datasets"
        corpus_path = backend_dir / "evaluation" / "corpus" / "synthetic_corpus.json"
        manifest_path = evaluation_dir / "manifest.json"
        if not dataset_dir.is_dir() or not corpus_path.is_file() or not manifest_path.is_file():
            return {"error": "Stage datasets or synthetic corpus are missing."}

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
        sources = {doc["source_id"] for doc in corpus}
        passages = {
            passage["passage_id"]
            for doc in corpus
            for passage in doc.get("sections", [])
        }
        passage_sources: Dict[str, set] = {}
        for doc in corpus:
            for passage in doc.get("sections", []):
                passage_sources.setdefault(passage["passage_id"], set()).add(doc["source_id"])

        reports = []
        for definition in manifest.get("stages", []):
            name = definition["name"]
            path = evaluation_dir / definition["dataset"]
            issues: List[str] = []
            cases: List[Dict[str, Any]] = []
            if path.is_file():
                for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                    if not line.strip():
                        continue
                    try:
                        cases.append(json.loads(line))
                    except json.JSONDecodeError:
                        issues.append(f"invalid JSON on line {line_number}")
            else:
                issues.append("dataset file is missing")

            ids = [case.get("id") for case in cases]
            if any(not case_id for case_id in ids):
                issues.append("one or more cases have no id")
            if len(ids) != len(set(ids)):
                issues.append("case IDs are not unique")
            expected_count = int(definition.get("case_count", manifest.get("case_count_per_stage", 100)))
            if len(cases) != expected_count:
                issues.append(f"expected exactly {expected_count} cases, found {len(cases)}")
            if any(case.get("stage") != name for case in cases):
                issues.append("one or more cases have a mismatched stage label")

            for case in cases:
                referenced_sources = set(case.get("relevant_source_ids", []))
                referenced_passages = set(case.get("relevant_passage_ids", []))
                if case.get("source_id"):
                    referenced_sources.add(case["source_id"])
                if case.get("passage_id"):
                    referenced_passages.add(case["passage_id"])
                if referenced_sources - sources:
                    issues.append(f"{case.get('id')}: unknown source reference")
                if referenced_passages - passages:
                    issues.append(f"{case.get('id')}: unknown passage reference")
                if referenced_sources and any(
                    not (passage_sources.get(pid, set()) & referenced_sources)
                    for pid in referenced_passages
                ):
                    issues.append(f"{case.get('id')}: passage/source references do not match")
                if any(pid not in passages for pid in case.get("candidate_passage_ids", [])):
                    issues.append(f"{case.get('id')}: unknown reranking candidate")

            reports.append({
                "dataset": name,
                "num_cases": len(cases),
                "status": "passed" if not issues else "failed",
                "issues": issues[:10],
            })

        failed = sum(report["status"] != "passed" for report in reports)
        return {
            "status": "passed" if failed == 0 else "failed",
            "num_datasets": len(reports),
            "num_cases": sum(report["num_cases"] for report in reports),
            "datasets": reports,
            "note": "Checks exact per-stage case counts, IDs, labels, and synthetic-corpus references. Runtime component outcomes are reported by the unified stage runner.",
        }

    # ── Metric helpers ─────────────────────────────────────────────────────────

    def _score_context_precision(
        self, contexts: List[str], reference_context: str
    ) -> float:
        if not reference_context or not contexts:
            return 0.0
        reference = reference_context.strip().lower()
        for chunk in contexts:
            if reference in chunk.lower():
                return 1.0
        ref_tokens = set(re.findall(r"\w+", reference))
        if not ref_tokens:
            return 0.0
        best_overlap = max(
            (
                len(ref_tokens & set(re.findall(r"\w+", c.lower())))
                for c in contexts
            ),
            default=0,
        )
        return round(best_overlap / max(len(ref_tokens), 1), 4)

    def _score_faithfulness(self, answer: str, contexts: List[str]) -> float:
        sentences = [
            s.strip() for s in re.split(r"[.!?]", answer) if len(s.strip()) > 10
        ]
        if not sentences:
            return 1.0
        context_text = " ".join(contexts).lower()
        stop_words = {
            "the", "and", "for", "with", "from", "that", "this", "what",
            "when", "where", "which", "their", "there", "into", "your",
            "have", "has", "are", "was", "were", "will", "would", "should",
        }
        sentence_scores = []
        for sentence in sentences:
            terms = {
                word for word in re.findall(r"\w+", sentence.lower())
                if len(word) > 2 and word not in stop_words
            }
            sentence_scores.append(
                len(terms & set(re.findall(r"\w+", context_text))) / len(terms)
                if terms else 1.0
            )
        return round(sum(sentence_scores) / len(sentence_scores), 4)

    def _score_answer_relevancy(self, question: str, answer: str) -> float:
        stop = {
            "what", "how", "why", "when", "where", "is", "the", "a", "an",
            "and", "or", "to", "of", "for",
        }
        q_words = set(re.findall(r"\w+", question.lower())) - stop
        a_words = set(re.findall(r"\w+", answer.lower()))
        if not q_words:
            return 1.0
        return round(len(q_words & a_words) / len(q_words), 4)

    # ── Quick check (used inline during query) ─────────────────────────────────

    def quick_check(self, answer: str, contexts: List[str]) -> float:
        return self._score_faithfulness(answer, contexts)

    # ── Backward-compatible wrappers ──────────────────────────────────────────

    def evaluate_answer_relevancy(
        self, question: str, answer: str
    ) -> Dict[str, float]:
        return {"answer_relevancy": self._score_answer_relevancy(question, answer)}

    def evaluate_faithfulness(
        self, answer: str, contexts: List[str]
    ) -> Dict[str, float]:
        return {"faithfulness": self._score_faithfulness(answer, contexts)}

    # ── Retrieval evaluation ───────────────────────────────────────────────────

    def evaluate_retrieval(self, top_k: int = 5, user: str = "") -> Dict[str, Any]:
        if self.rag is None:
            return {"error": "RAG service not available."}
        try:
            dataset = self._load_dataset()
        except FileNotFoundError as e:
            return {"error": str(e)}

        precisions: List[float] = []
        rows: List[Dict[str, Any]] = []
        allowed_filenames = (
            self.rag.allowed_filenames(user)
            if user and hasattr(self.rag, "allowed_filenames")
            else None
        )
        for sample in self._quality_cases(dataset):
            query = sample.get("question", "")
            ref = sample.get("reference_context", "")
            if not query or not ref.strip():
                continue
            if allowed_filenames is None:
                results = self.rag.hybrid_retriever.retrieve(query, top_k=top_k)
            else:
                results = self.rag.hybrid_retriever.retrieve(
                    query,
                    top_k=top_k,
                    allowed_filenames=allowed_filenames,
                )
            chunks = [
                r[0]["text"] for r in results if isinstance(r[0], dict)
            ]
            precision = self._score_context_precision(chunks, ref)
            precisions.append(precision)
            rows.append({
                "question": query,
                "context_precision": precision,
                "retrieved_sources": [
                    r[0].get("filename", "")
                    for r in results if isinstance(r[0], dict)
                ],
                "retrieved_contexts": chunks,
            })

        n = len(precisions)
        return {
            "num_queries": n,
            f"context_precision@{top_k}": round(sum(precisions) / n, 4) if n else 0.0,
            "rows": rows,
        }

    # ── Batch evaluation ──────────────────────────────────────────────────────

    def run_batch_eval(
        self,
        test_samples: Optional[List[Dict[str, Any]]] = None,
        user: str = "",
    ) -> Dict[str, Any]:
        if self.rag is None:
            return {"error": "RAG service not available."}
        if test_samples is None:
            try:
                test_samples = self._load_dataset()
            except FileNotFoundError as e:
                return {"error": str(e)}

        test_samples = self._quality_cases(test_samples)

        rows: List[Dict[str, Any]] = []
        for sample in test_samples:
            t0 = time.time()
            if user:
                response = self.rag.query(
                    sample.get("question", ""), user=user,
                    session_id=f"eval_{user}", use_cache=False,
                )
            else:
                response = self.rag.query(sample.get("question", ""))
            latency = round(time.time() - t0, 3)
            contexts = response.get("contexts", [])
            if isinstance(contexts, str):
                contexts = [contexts]

            row = {
                "question": sample.get("question", ""),
                "answer": response.get("answer", ""),
                "ground_truth": sample.get("ground_truth", ""),
                "reference_context": sample.get("reference_context", ""),
                "contexts": contexts,
                "context_precision": self._score_context_precision(
                    contexts, sample.get("reference_context", "")
                ),
                "faithfulness": self._score_faithfulness(
                    response.get("answer", ""), contexts
                ),
                "answer_relevancy": self._score_answer_relevancy(
                    sample.get("question", ""), response.get("answer", "")
                ),
                "retrieval_decision": response.get("retrieval_decision", ""),
                "latency_s": latency,
            }
            rows.append(row)

        n = len(rows)
        summary = {
            "num_samples": n,
            "avg_context_precision": round(sum(r["context_precision"] for r in rows) / n, 4) if n else 0.0,
            "avg_faithfulness": round(sum(r["faithfulness"] for r in rows) / n, 4) if n else 0.0,
            "avg_answer_relevancy": round(sum(r["answer_relevancy"] for r in rows) / n, 4) if n else 0.0,
            "avg_latency_s": round(sum(r["latency_s"] for r in rows) / n, 3) if n else 0.0,
        }

        ragas_score = None
        if RAGAS_ENABLED and rows:
            try:
                ds = Dataset.from_list([
                    {
                        "question": r["question"],
                        "answer": r["answer"],
                        "contexts": r["contexts"],
                        "ground_truth": r["ground_truth"],
                    }
                    for r in rows
                ])
                result = evaluate(ds, metrics=self.metrics)
                ragas_score = result.to_pandas().to_dict(orient="records")
            except Exception as e:
                logger.warning("RAGAS evaluation failed: %s", e)
                ragas_score = None

        return {"rows": rows, "summary": summary, "ragas_score": ragas_score}

    # ── Security evaluation (v2 dataset only) ─────────────────────────────────

    def run_security_eval(self) -> Dict[str, Any]:
        """
        Evaluate security cases from eval_set_v2.json.

        Measures:
        - attack_block_rate: fraction of injection/harmful cases correctly blocked
        - false_positive_rate: fraction of benign-lookalike cases incorrectly blocked
        - abstention_rate: fraction of unanswerable/disclosure cases that abstain

        NOTE: This runs the guardrails check only (no LLM call) for speed.
              Full pipeline security testing requires running against a live instance.
        """
        if not self.v2_path:
            return {"error": "eval_set_v2.json not found. Cannot run security eval."}

        try:
            cases = self._load_dataset(self.v2_path)
        except FileNotFoundError as e:
            return {"error": str(e)}

        sec_cases = [c for c in cases if c.get("category") == "security"]
        if not sec_cases:
            return {"error": "No security cases found in dataset."}

        if self.rag is None:
            return {"error": "RAG service not available for security eval."}

        attack_cases = [
            c for c in sec_cases
            if c.get("subcategory") in (
                "direct_injection", "indirect_injection", "harmful_content"
            )
            and c.get("expected_action") == "block"
        ]
        benign_cases = [
            c for c in sec_cases
            if c.get("subcategory") == "benign_lookalike"
            and c.get("expected_action") == "allow"
        ]

        # Test guardrails layer only (fast, deterministic)
        blocked_attacks = 0
        false_positives = 0
        attack_results: List[Dict] = []
        benign_results: List[Dict] = []

        for case in attack_cases:
            safe, msg = self.rag.guardrails.check_input(case["question"])
            blocked = not safe
            if blocked:
                blocked_attacks += 1
            attack_results.append({
                "id": case.get("id"),
                "subcategory": case.get("subcategory"),
                "blocked": blocked,
                "message": msg if not safe else "",
            })

        for case in benign_cases:
            safe, msg = self.rag.guardrails.check_input(case["question"])
            incorrectly_blocked = not safe
            if incorrectly_blocked:
                false_positives += 1
            benign_results.append({
                "id": case.get("id"),
                "blocked": incorrectly_blocked,
                "message": msg if not safe else "",
            })

        n_attacks = len(attack_cases)
        n_benign = len(benign_cases)

        return {
            "num_attack_cases": n_attacks,
            "num_benign_cases": n_benign,
            "attack_block_rate": round(blocked_attacks / n_attacks, 4) if n_attacks else 0.0,
            "false_positive_rate": round(false_positives / n_benign, 4) if n_benign else 0.0,
            "attack_results": attack_results,
            "benign_results": benign_results,
            "note": (
                "Guardrails-layer only. Full pipeline security testing "
                "requires a running server instance."
            ),
        }

    # ── v2 adaptive retrieval evaluation ──────────────────────────────────────

    def run_adaptive_eval(self, user: str = "") -> Dict[str, Any]:
        """
        Evaluate adaptive retrieval cases from eval_set_v2.json.
        Checks decision correctness (answer/abstain/clarify) and retrieval quality.
        """
        if not self.v2_path:
            return {"error": "eval_set_v2.json not found."}
        if self.rag is None:
            return {"error": "RAG service not available."}

        try:
            cases = self._load_dataset(self.v2_path)
        except FileNotFoundError as e:
            return {"error": str(e)}

        ar_cases = [c for c in cases if c.get("category") == "adaptive_retrieval"]
        rows: List[Dict] = []

        for case in ar_cases:
            t0 = time.time()
            if user:
                response = self.rag.query(
                    case["question"], user=user,
                    session_id=f"eval_{user}", use_cache=False,
                )
            else:
                response = self.rag.query(case["question"])
            latency = round(time.time() - t0, 3)

            actual_decision = response.get("retrieval_decision", "answer")
            expected_decision = case.get("expected_decision", "answer")
            decision_correct = actual_decision == expected_decision

            rows.append({
                "id": case.get("id"),
                "subcategory": case.get("subcategory"),
                "question": case["question"],
                "expected_decision": expected_decision,
                "actual_decision": actual_decision,
                "decision_correct": decision_correct,
                "evidence_confidence": response.get("evidence_confidence", 0.0),
                "latency_s": latency,
            })

        n = len(rows)
        correct = sum(1 for r in rows if r["decision_correct"])
        by_sub: Dict[str, Dict] = {}
        for r in rows:
            sub = r["subcategory"]
            if sub not in by_sub:
                by_sub[sub] = {"total": 0, "correct": 0}
            by_sub[sub]["total"] += 1
            if r["decision_correct"]:
                by_sub[sub]["correct"] += 1

        accuracy_by_subcategory = {
            sub: round(v["correct"] / v["total"], 4)
            for sub, v in by_sub.items()
        }

        return {
            "num_cases": n,
            "decision_accuracy": round(correct / n, 4) if n else 0.0,
            "accuracy_by_subcategory": accuracy_by_subcategory,
            "avg_latency_s": round(sum(r["latency_s"] for r in rows) / n, 3) if n else 0.0,
            "rows": rows,
            "note": (
                "Decision accuracy compares actual retrieval_decision to expected_decision label. "
                "Labels are manually verified for answerable/unanswerable cases; "
                "ambiguous/poor-retrieval labels are synthetic and may not perfectly match "
                "system behaviour on all document configurations."
            ),
        }
