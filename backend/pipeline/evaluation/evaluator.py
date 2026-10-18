"""RAG evaluation — RAGAS-inspired metrics: retrieval quality + faithfulness."""
import json
import re
import logging
from typing import List, Dict, Optional

from pipeline.evaluation.retrieval_metrics import recall_at_k, precision_at_k, mrr_at_k, ndcg_at_k

logger = logging.getLogger("rag_app")


class RAGEvaluator:
    def __init__(self, retriever=None, generator=None, dataset_path: Optional[str] = None):
        self.retriever = retriever
        self.generator = generator
        self.dataset_path = dataset_path

    # ─── Retrieval Quality ────────────────────────────────────────────────────

    def evaluate_retrieval(self, top_k: int = 5) -> Dict:
        """Evaluate retrieval quality against a labelled dataset."""
        if not self.dataset_path:
            return {"error": "No evaluation dataset configured. Set dataset_path."}

        try:
            with open(self.dataset_path) as f:
                dataset = json.load(f)
        except FileNotFoundError:
            return {"error": f"Dataset file not found: {self.dataset_path}"}

        r_list, p_list, mrr_list, ndcg_list = [], [], [], []

        for item in dataset:
            query = item["query"]
            keyword = item["relevant_keyword"]
            results = self.retriever.retrieve(query, top_k=top_k)
            chunks = [
                r[0]["text"] if isinstance(r[0], dict) else str(r[0])
                for r in results
            ]
            r_list.append(recall_at_k(chunks, keyword))
            p_list.append(precision_at_k(chunks, keyword))
            mrr_list.append(mrr_at_k(chunks, keyword))
            ndcg_list.append(ndcg_at_k(chunks, keyword))

        n = len(dataset)
        return {
            "num_queries": n,
            f"recall@{top_k}":    round(sum(r_list) / n, 4) if n else 0,
            f"precision@{top_k}": round(sum(p_list) / n, 4) if n else 0,
            f"mrr@{top_k}":       round(sum(mrr_list) / n, 4) if n else 0,
            f"ndcg@{top_k}":      round(sum(ndcg_list) / n, 4) if n else 0,
        }

    # ─── Faithfulness ─────────────────────────────────────────────────────────

    def evaluate_faithfulness(
        self, question: str, answer: str, context_chunks: List[str]
    ) -> Dict:
        """
        Heuristic faithfulness: what fraction of answer sentences can be
        traced back to significant words in the context.
        """
        sentences = [s.strip() for s in re.split(r'[.!?]', answer) if len(s.strip()) > 10]
        if not sentences:
            return {"faithfulness": 1.0, "num_sentences": 0}

        stop_words = {
            "the", "and", "for", "with", "from", "that", "this", "what",
            "when", "where", "which", "their", "there", "into", "your",
            "have", "has", "are", "was", "were", "will", "would", "should",
        }
        context_terms = set(re.findall(r'\w+', " ".join(context_chunks).lower()))
        sentence_scores = []
        for sent in sentences:
            terms = {
                word for word in re.findall(r'\w+', sent.lower())
                if len(word) > 2 and word not in stop_words
            }
            sentence_scores.append(
                len(terms & context_terms) / len(terms) if terms else 1.0
            )

        return {
            "faithfulness": round(sum(sentence_scores) / len(sentence_scores), 4),
            "grounded_sentences": sum(score == 1.0 for score in sentence_scores),
            "total_sentences": len(sentences),
        }

    # ─── Answer Relevance ────────────────────────────────────────────────────

    def evaluate_answer_relevance(self, question: str, answer: str) -> Dict:
        """
        Heuristic answer relevance: do question keywords appear in the answer?
        """
        q_words = set(re.findall(r'\w+', question.lower())) - {"what", "how", "why", "when", "where", "is", "the", "a"}
        a_words = set(re.findall(r'\w+', answer.lower()))
        if not q_words:
            return {"answer_relevance": 1.0}
        overlap = len(q_words & a_words) / len(q_words)
        return {"answer_relevance": round(overlap, 4)}
