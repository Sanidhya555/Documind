"""Query expansion: generate alternative phrasings for broader retrieval coverage."""
import json
import re
import logging
from typing import List

logger = logging.getLogger("rag_app")


class QueryExpander:
    """
    Uses the LLM to produce N alternative phrasings of the original query.
    All variants are used for retrieval; results are merged via RRF in HybridRetriever.
    """

    def __init__(self, generator=None, num_expansions: int = 2):
        self.generator = generator
        self.num_expansions = num_expansions

    def expand(self, query: str) -> List[str]:
        """Return [original_query] + expanded variants."""
        if not self.generator or self.num_expansions == 0:
            return [query]

        prompt = (
            f"Generate {self.num_expansions} alternative phrasings of this question "
            f"that preserve the meaning but use different words, for document retrieval.\n"
            f"Return ONLY a JSON array of strings. No explanation, no markdown.\n\n"
            f"Question: {query}\n\n"
            f"JSON array:"
        )

        try:
            raw = self.generator.generate(prompt).strip()

            # Extract JSON array robustly
            match = re.search(r'\[.*?\]', raw, re.DOTALL)
            if match:
                variants = json.loads(match.group())
            else:
                variants = json.loads(raw)

            variants = [
                v.strip() for v in variants
                if isinstance(v, str) and v.strip() and v.strip() != query
            ][: self.num_expansions]

            logger.debug(f"Query expanded: {query!r} → {variants}")
            return [query] + variants

        except Exception as e:
            logger.warning(f"Query expansion failed ({e}), using original query only")
            return [query]
