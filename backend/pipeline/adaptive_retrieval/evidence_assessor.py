"""Evidence Assessor: Evaluates retrieved passage relevance, coverage, redundancy, and sufficiency."""
import re
from typing import Dict, List, Set, Tuple
from .schemas import EvidenceAssessment


class EvidenceAssessor:
    """
    Deterministic evidence assessor that analyzes retrieved candidate passages
    against the user query without assuming raw similarity scores equal answerability probabilities.
    """

    def __init__(
        self,
        min_relevance_threshold: float = 0.20,
        min_coverage_threshold: float = 0.55,
        duplicate_threshold: float = 0.85,
    ):
        self.min_relevance = min_relevance_threshold
        self.min_coverage = min_coverage_threshold
        self.duplicate_threshold = duplicate_threshold

    def _extract_query_aspects(self, query: str) -> List[str]:
        """Extract key entities, numbers, and nouns as query aspects."""
        # Stopwords to filter out
        stopwords = {
            "what", "when", "where", "which", "who", "whom", "whose", "why",
            "how", "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "do", "does", "did", "have", "has", "had", "can", "could", "should",
            "would", "may", "might", "must", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "up", "about", "into", "through",
            "and", "or", "but", "so", "if", "than", "then", "this", "that"
        }
        tokens = re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", query.lower())
        aspects = [t for t in tokens if t not in stopwords]
        return list(dict.fromkeys(aspects))  # preserve order, unique

    def _token_set(self, text: str) -> Set[str]:
        return set(re.findall(r"\b\w{3,}\b", text.lower()))

    def _compute_jaccard(self, s1: Set[str], s2: Set[str]) -> float:
        if not s1 or not s2:
            return 0.0
        return len(s1 & s2) / len(s1 | s2)

    def _detect_duplicates(self, chunks: List[str]) -> float:
        """Calculate the proportion of redundant/duplicate chunks."""
        if len(chunks) <= 1:
            return 0.0

        token_sets = [self._token_set(c) for c in chunks]
        redundant_count = 0

        for i in range(1, len(token_sets)):
            for j in range(i):
                if self._compute_jaccard(token_sets[i], token_sets[j]) >= self.duplicate_threshold:
                    redundant_count += 1
                    break

        return redundant_count / len(chunks)

    def _detect_conflicts(self, chunks: List[str]) -> bool:
        """Heuristic conflict detection: contradictory negation patterns."""
        combined_text = " ".join(chunks).lower()
        contradiction_markers = [
            r"\b(however|in contrast|contrary to|on the other hand|disputed by|opposing view)\b",
            r"\b(increased\b.*\bdecreased|fell\b.*\brose|grew\b.*\bdeclined)\b",
        ]
        for marker in contradiction_markers:
            if re.search(marker, combined_text):
                return True
        return False

    def assess(self, query: str, ranked_results: List[Tuple[Dict, float]]) -> EvidenceAssessment:
        if not ranked_results:
            return EvidenceAssessment(
                is_sufficient=False,
                coverage_score=0.0,
                top_relevance_score=-1.0,
                avg_relevance_score=-1.0,
                unique_sources_count=0,
                duplicate_ratio=0.0,
                has_conflict=False,
                missing_aspects=self._extract_query_aspects(query),
                confidence=0.9,
            )

        scores = [float(r[1]) for r in ranked_results]

        # Extract texts & filenames
        chunks_text = [
            r[0]["text"] if isinstance(r[0], dict) and "text" in r[0] else str(r[0])
            for r in ranked_results
        ]
        filenames = {
            r[0].get("filename") for r in ranked_results
            if isinstance(r[0], dict) and r[0].get("filename")
        }

        # Coverage analysis
        aspects = self._extract_query_aspects(query)
        combined_context = " ".join(chunks_text).lower()

        # Reciprocal-rank fusion scores are deliberately small (typically < 0.03),
        # so they cannot be compared to the relevance threshold used for
        # cross-encoder scores. When no reranker is available, estimate relevance
        # from the strongest candidate's query-term coverage instead.
        if scores and max(abs(score) for score in scores) < 0.1:
            best_chunk_tokens = set(
                re.findall(r"\b[a-zA-Z0-9_.-]{2,}\b", chunks_text[0].lower())
            )
            query_terms = set(aspects)
            top_relevance = (
                len(query_terms & best_chunk_tokens) / len(query_terms)
                if query_terms
                else 0.0
            )
            relevance_scores = [top_relevance]
        else:
            top_relevance = max(scores) if scores else 0.0
            relevance_scores = scores
        top_k_for_avg = min(3, len(relevance_scores))
        avg_relevance = (
            sum(relevance_scores[:top_k_for_avg]) / top_k_for_avg
            if top_k_for_avg
            else 0.0
        )

        missing = []
        covered = 0
        for aspect in aspects:
            if aspect in combined_context:
                covered += 1
            else:
                missing.append(aspect)

        coverage_score = (covered / len(aspects)) if aspects else 1.0
        duplicate_ratio = self._detect_duplicates(chunks_text)
        has_conflict = self._detect_conflicts(chunks_text)

        # Sufficiency criteria:
        # 1. Top relevance must be above min_relevance
        # 2. Coverage must be above min_coverage (or query has no specific aspects)
        # 3. Not excessive duplicates leaving only 1 unique chunk with low coverage
        is_sufficient = (
            top_relevance >= self.min_relevance
            and coverage_score >= self.min_coverage
            and len(missing) <= max(1, int(len(aspects) * 0.35))
        )

        confidence = round(min(1.0, max(0.1, (top_relevance * 0.6 + coverage_score * 0.4))), 3)

        return EvidenceAssessment(
            is_sufficient=is_sufficient,
            coverage_score=round(coverage_score, 3),
            top_relevance_score=round(top_relevance, 4),
            avg_relevance_score=round(avg_relevance, 4),
            unique_sources_count=len(filenames),
            duplicate_ratio=round(duplicate_ratio, 3),
            has_conflict=has_conflict,
            missing_aspects=missing,
            confidence=confidence,
        )
