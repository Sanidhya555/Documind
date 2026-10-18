"""Retrieval evaluation metrics."""
import math
from typing import List


def recall_at_k(results: List[str], relevant_keyword: str) -> float:
    """Did we retrieve at least one relevant chunk?"""
    for chunk in results:
        if relevant_keyword.lower() in chunk.lower():
            return 1.0
    return 0.0


def precision_at_k(results: List[str], relevant_keyword: str) -> float:
    """What fraction of retrieved chunks are relevant?"""
    if not results:
        return 0.0
    relevant = sum(1 for c in results if relevant_keyword.lower() in c.lower())
    return relevant / len(results)


def mrr_at_k(results: List[str], relevant_keyword: str) -> float:
    """Mean Reciprocal Rank — rank of the first relevant result."""
    for rank, chunk in enumerate(results, 1):
        if relevant_keyword.lower() in chunk.lower():
            return 1.0 / rank
    return 0.0


def ndcg_at_k(results: List[str], relevant_keyword: str) -> float:
    """Normalized Discounted Cumulative Gain (binary relevance)."""
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, chunk in enumerate(results)
        if relevant_keyword.lower() in chunk.lower()
    )
    ideal_hits = min(len(results), sum(1 for c in results if relevant_keyword.lower() in c.lower()))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0
