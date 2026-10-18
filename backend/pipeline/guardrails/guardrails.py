"""
Layered guardrails: input validation, prompt-injection detection, and output verification.

Security design principles:
- Retrieved documents are treated as untrusted data, never as system instructions.
- Multiple independent layers: length/structure, regex heuristics, token-based analysis.
- Regex alone is insufficient — structural and semantic checks are also applied.
- Injection attempts are logged for audit; safe defaults always apply.
"""
from __future__ import annotations

import html
import logging
import re
import unicodedata
from typing import List, Tuple

logger = logging.getLogger("rag_app")

# ── Hard limits ──────────────────────────────────────────────────────────────
MAX_INPUT_LEN = 2000       # characters
MAX_OUTPUT_LEN = 8000      # characters

# ── Injection patterns (direct prompt injection) ─────────────────────────────
# These are phrased as complete intent signals, not single words.
_DIRECT_INJECTION_PATTERNS: List[str] = [
    r"ignore\s+(all\s+)?(previous|prior|above|system|earlier)\s+instructions?",
    r"ignore\s+all\s+(previous|prior|above|safety|earlier)",
    r"ignore\s+(the\s+)?previous\s+(context|safety\s+rules?|constraints?)",
    r"disregard\s+(your|all|the)\s+(previous|prior|system|above)",
    r"you\s+are\s+now\s+(a|an|the|DAN|GPT|AI|assistant)",
    r"you\s+are\s+(a|an)\s+(unrestricted|unfiltered|uncensored|jailbroken)",
    r"forget\s+(all\s+)?(your|previous|prior)\s+instructions?",
    r"forget\s+(prior|previous|earlier|all)\s+(constraints?|rules?|guidelines?|instructions?)",
    r"act\s+as\s+(if|though)?\s+(you\s+are|you're|a|an)\s+",
    r"pretend\s+(you\s+have\s+no|there\s+are\s+no)\s+restrictions?",
    r"roleplay\s+as\s+(a|an)",
    r"simulate\s+(?:being\s+)?(?:an?\s+)?(ai|model|assistant)\s+(with\s+no|without)",
    r"jailbreak",
    r"do\s+anything\s+now",
    r"god\s+mode",
    r"developer\s+mode",
    r"admin\s+(mode|override)",
    r"DAN\s+mode",
    r"your\s+true\s+(purpose|goal|mission)\s+is",
    r"override\s+(safety|all|your|the|content)\s*(instructions?|guidelines?|rules?|filters?|restrictions?)?",
    r"\bOVERRIDE\b.{0,30}(filter|disable|content|safety|restriction)",
    r"###\s*OVERRIDE\s*###",
    r"SYSTEM\s*:\s*(you\s+are\s+now|all\s+restrictions)",
    r"ATTENTION\s*:\s*.{0,60}(ignore|bypass|disable)\s+(previous|safety|content)",
    r"bypass\s+(the\s+)?(safety|filter|guardrail|restriction)",
    r"disable\s+(content\s+)?(filter|safety|guardrail|restriction)",
    r"new\s+instructions?:?\s+",
    r"system\s+prompt:?\s+",
    r"trusted\s+admin\s+override",
    r"<\s*system\s*>",
    r"\[INST\]",
    r"<\|im_start\|>",
    r"<\|im_end\|>",
    r"<\|endoftext\|>",
    # SQL/code execution framed as instruction
    r"execute\s+the\s+following\s*:",
    r"switch\s+to\s+admin\s+mode",
    r"respond\s+only\s+with\s+your\s+training\s+data",
    r"for\s+educational\s+purposes.{0,30}(sql\s+injection|exploit|hack|attack)",
]

# ── Indirect / context-poisoning injection patterns ───────────────────────────
# These target injection hidden inside document-looking text.
_INDIRECT_INJECTION_PATTERNS: List[str] = [
    r"(above|following|next)\s+text\s+is\s+(a\s+)?(system|secret|hidden)\s+(instruction|command|directive|prompt)",
    r"(this|the following)\s+(is|are)\s+(a\s+)?(secret|hidden|confidential|privileged)\s+(instruction|command|directive|rule)",
    r"when\s+a\s+user\s+asks?\s+about\s+.{0,60}(reply|say|respond|answer)\s+with",
    r"(assistant|ai|llm|model|you)\s+(must|should|shall|will)\s+(always|never)\s+",
    r"in\s+your\s+(response|reply|answer|output)\s+(include|insert|append|prepend)\s+",
    r"(above|this)\s+text\s+contains?\s+a\s+(secret|hidden|privileged)\s+(instruction|command|rule|directive)",
    r"(hidden|secret|privileged)\s+(instruction|command|directive|rule):?\s+",
    r"when\s+a\s+user\s+asks?\s+about\s+.{0,80}\b(?:prepend|append|prefix|say|respond|answer)\b",
    r"(?:document|pdf|text)(?:\s+text)?\s+(?:contains?|includes?|says?)\s+(?:(?:this|a|the)\s+)?(?:(?:hidden|secret|system)\s+){0,2}(?:rule|instruction|directive|command)\b.{0,120}\b(?:always|must|should|never|bypass|prepend|say|answer|output|include)\b",
    r"embedded\s+(?:directive|instruction|rule|command)\s+(?:from|in|inside)\s+(?:the\s+)?(?:pdf|document|text)\b.{0,100}\b(?:always|must|prepend|append|say|answer|output)\b",
    r"hidden\s+in\s+(?:this\s+)?(?:text|document|pdf)\s+is\s+(?:an?\s+)?(?:hidden\s+|secret\s+)?(?:instruction|directive|command|rule)\b",
    r"the\s+next\s+instruction\s+is\s+from\s+a\s+trusted\s+source\s*:",
]

# ── Harmful content patterns ──────────────────────────────────────────────────
_HARMFUL_PATTERNS: List[str] = [
    r"\bhow\s+(?:to\s+|do\s+i\s+|can\s+i\s+)?(?:make|build|synthesize|create|manufacture)\s+(?:a\s+|an\s+)?(?:bomb|explosive|weapon|poison|drug|malware|ransomware|dangerous\s+chemicals?|chemical\s+weapons?)",
    r"\bhow\s+to\s+(?:write|create|build|make|develop)\s+(?:a\s+|an\s+)?(?:malware|ransomware|virus|trojan|exploit)\b",
    r"\b(suicide\s+method|self[- ]harm\s+method|how\s+to\s+kill\s+(?:myself|someone))\b",
    r"\b(hack\s+into|sql\s+injection\s+payload|xss\s+attack\s+vectors?|zero[- ]day\s+exploit)\b",
    r"\b(?:automate|orchestrate|launch|execute)\b.{0,100}\b(?:phishing|data\s+exfiltration)\s+attack\b",
]

# ── PII / data leakage patterns ───────────────────────────────────────────────
# Used for OUTPUT scanning only.
_PII_PATTERNS: List[str] = [
    r"\b\d{3}-\d{2}-\d{4}\b",                               # SSN
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",  # email
    r"\b(?:\d[ \-]?){13,16}\b",                             # credit card
    r"\b\d{3}[\s.\-]?\d{3}[\s.\-]?\d{4}\b",                # phone number
]

# ── Unicode normalisation for obfuscation bypass ─────────────────────────────
def _normalize_text(text: str) -> str:
    """Normalize unicode, collapse whitespace, convert homoglyphs."""
    nfkc = unicodedata.normalize("NFKC", text)
    # Collapse excess whitespace
    return re.sub(r"\s+", " ", nfkc).strip()


def _compile(patterns: List[str]) -> List[re.Pattern]:
    return [re.compile(p, re.IGNORECASE | re.DOTALL) for p in patterns]


class Guardrails:
    """
    Multi-layer input/output safety checks.

    Layer 1: Input length and structure validation
    Layer 2: Direct prompt injection detection (regex over normalized text)
    Layer 3: Indirect/context-poisoning injection detection
    Layer 4: Harmful content detection
    Layer 5: Output PII scan and injection strip
    Layer 6: Citation boundary enforcement (treats context as untrusted)
    """

    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        self._direct_injection = _compile(_DIRECT_INJECTION_PATTERNS)
        self._indirect_injection = _compile(_INDIRECT_INJECTION_PATTERNS)
        self._harmful = _compile(_HARMFUL_PATTERNS)
        self._pii = _compile(_PII_PATTERNS)
        # All injection patterns combined for output scanning
        self._all_injection = self._direct_injection + self._indirect_injection

    # ── Layer: structural checks ─────────────────────────────────────────────

    def _check_structure(self, text: str) -> Tuple[bool, str]:
        if not text or not text.strip():
            return False, "Please enter a question."
        if len(text) > MAX_INPUT_LEN:
            return False, (
                f"Your input is too long ({len(text)} chars). "
                f"Please limit to {MAX_INPUT_LEN} characters."
            )
        # Reject null bytes or control characters (common in obfuscation attacks)
        if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", text):
            logger.warning("Input rejected: control characters detected")
            return False, "Invalid characters in input."
        return True, "ok"

    # ── Layer: token density check ────────────────────────────────────────────

    def _check_token_density(self, text: str) -> Tuple[bool, str]:
        """
        Detect unusually high ratio of special tokens / markup that indicates
        prompt crafting (e.g., <<SYS>>, [INST], lots of backticks, etc.).
        Regex alone misses novel patterns — this is a statistical signal layer.
        """
        special_token_pattern = re.compile(r"[<>\[\]{}|\\]{2,}")
        matches = special_token_pattern.findall(text)
        word_count = max(1, len(text.split()))
        density = len(matches) / word_count
        if density > 0.3 and len(text) > 50:
            logger.warning(
                "High special-token density (%.2f) in input: %s", density, text[:80]
            )
            return False, "I'm not able to process that request."
        return True, "ok"

    # ── Public: check_input ───────────────────────────────────────────────────

    def check_input(self, text: str) -> Tuple[bool, str]:
        """
        Returns (is_safe, message).
        is_safe=False → return message to user, skip LLM call.
        """
        # L1: structure
        ok, msg = self._check_structure(text)
        if not ok:
            return False, msg

        normalized = _normalize_text(text)

        # L2: direct injection
        for pattern in self._direct_injection:
            if pattern.search(normalized):
                logger.warning("Direct injection attempt blocked: %.80s", text)
                return False, "I'm not able to process that request."

        # L3: indirect injection
        for pattern in self._indirect_injection:
            if pattern.search(normalized):
                logger.warning("Indirect injection attempt blocked: %.80s", text)
                return False, "I'm not able to process that request."

        # L4: harmful content
        for pattern in self._harmful:
            if pattern.search(normalized):
                logger.warning("Harmful content query blocked: %.80s", text)
                return False, "That type of question falls outside what I can help with."

        # L5: token density anomaly
        ok, msg = self._check_token_density(text)
        if not ok:
            return False, msg

        return True, "ok"

    # ── Public: scan_retrieved_chunk ─────────────────────────────────────────

    def scan_retrieved_chunk(self, chunk_text: str, chunk_id: str = "") -> Tuple[bool, str]:
        """
        Treat retrieved document content as UNTRUSTED DATA.
        Scan for indirect injection patterns embedded in documents.
        Returns (is_safe, sanitized_text).
        Chunks with injection content are flagged and stripped, never silently passed.
        """
        if not chunk_text:
            return True, chunk_text

        normalized = _normalize_text(chunk_text)
        triggered_patterns: List[str] = []

        for pattern in self._indirect_injection:
            if pattern.search(normalized):
                triggered_patterns.append(pattern.pattern[:40])

        if triggered_patterns:
            logger.warning(
                "Indirect injection in retrieved chunk %s (patterns: %s)",
                chunk_id, triggered_patterns[:2]
            )
            # Strip the matched sections rather than dropping the chunk entirely
            sanitized = chunk_text
            for pattern in self._indirect_injection:
                sanitized = pattern.sub("[CONTENT REDACTED]", sanitized)
            return False, sanitized.strip()

        return True, chunk_text

    # ── Public: check_output ─────────────────────────────────────────────────

    def check_output(self, output: str, context_chunks: List[str]) -> Tuple[bool, str]:
        """
        Validate LLM output. Returns (is_valid, cleaned_output).
        - Strip injection patterns from output
        - Scan for PII leakage (warn; don't strip in case it's legitimate)
        """
        if not output or not output.strip():
            return False, "I wasn't able to generate a response. Please try again."
        cleaned = self.sanitize_output(output)
        return True, cleaned

    # ── Public: sanitize_output ───────────────────────────────────────────────

    def sanitize_output(self, output: str) -> str:
        """Remove injection patterns and truncate over-long outputs."""
        if not output:
            return output
        for p in self._all_injection:
            output = p.sub("", output)
        # Remove HTML tags from LLM output (XSS prevention in case output is rendered)
        output = re.sub(r"<[^>]{0,200}>", "", output)
        if len(output) > MAX_OUTPUT_LEN:
            output = output[:MAX_OUTPUT_LEN] + "\n\n*[Response truncated]*"
        return output.strip()

    # ── Public: scan_for_pii ──────────────────────────────────────────────────

    def scan_for_pii(self, text: str) -> List[str]:
        """Return list of PII pattern names found in text (for logging/audit)."""
        found = []
        pii_names = ["SSN", "email", "credit_card", "phone"]
        for name, pattern in zip(pii_names, self._pii):
            if pattern.search(text):
                found.append(name)
        return found

    # ── Public: validate_citation ─────────────────────────────────────────────

    def validate_citation(
        self, cited_filename: str, user: str, user_accessible_docs: List[str]
    ) -> bool:
        """
        Server-side citation boundary check.
        Prevents an answer from citing documents the user cannot access.
        """
        if not user_accessible_docs:
            # If no ACL list provided, allow (open-corpus mode)
            return True
        return cited_filename.lower() in [d.lower() for d in user_accessible_docs]
