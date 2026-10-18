"""
Layer 1 — Presidio Regex Guardrail.

Screens incoming text for high-risk enterprise PII/sensitive data using
Microsoft Presidio's regex-based recognizers.  This runs *before* the RAG
pipeline and the vector store so that sensitive data is never ingested or
retrieved.

Design decisions
────────────────
• **Regex-only, no heavy ML.**  The AnalyzerEngine is built with a custom
  RecognizerRegistry that contains only pattern-based recognizers.  The spaCy
  en_core_web_sm model is used purely for tokenisation (required by Presidio's
  engine), not for NER inference.  Runtime per call is typically < 5 ms.

• **Globally cached singleton.**  The engine is expensive to construct once
  but cheap to call.  It is initialised at import time and reused for the
  lifetime of the process.

• **Regex fallback.** If Presidio or its NLP model is unavailable, a standard
  library fallback still checks common email, card, IP, crypto, SSN, and phone
  patterns. Set PRESIDIO_STRICT=true to require the full Presidio engine.

• **Audit logging.**  Every blocked request is logged with the entity types
  found so that security events are visible in the server log.

Entities screened
─────────────────
  CREDIT_CARD   — Visa/MC/Amex/Discover card numbers
  EMAIL_ADDRESS — RFC-5321 email addresses
  IP_ADDRESS    — IPv4 addresses (also catches some IPv6 patterns)
  CRYPTO        — Bitcoin and Ethereum wallet addresses
  US_SSN        — US Social Security Numbers (NNN-NN-NNNN)
  PHONE_NUMBER  — US / international phone numbers

Usage
─────
  from utils.presidio_guard import enterprise_regex_guard

  # In a FastAPI route:
  enterprise_regex_guard(request.question)   # raises HTTP 400 on violation
"""
from __future__ import annotations

import logging
import os
import re
from typing import List

from fastapi import HTTPException

logger = logging.getLogger("rag_app")

# ── Constants ─────────────────────────────────────────────────────────────────

#: Minimum Presidio score to count as a violation.  0.4 excludes the very-weak
#: (0.05) SSN sub-patterns while still catching medium/high confidence hits.
_SCORE_THRESHOLD: float = float(os.getenv("PRESIDIO_SCORE_THRESHOLD", "0.4"))

#: When True, fail hard if presidio-analyzer is not installed.
_STRICT: bool = os.getenv("PRESIDIO_STRICT", "false").lower() == "true"

#: Entity types to screen.  Add / remove entries here; no other code changes needed.
_TARGET_ENTITIES: List[str] = [
    "CREDIT_CARD",
    "EMAIL_ADDRESS",
    "IP_ADDRESS",
    "CRYPTO",
    "US_SSN",
    "PHONE_NUMBER",
]

# ── Engine construction ───────────────────────────────────────────────────────

_engine = None          # lazy singleton
_presidio_available = False
_engine_initialized = False


def _build_engine():
    """
    Build a Presidio AnalyzerEngine backed by pattern-only recognizers.

    A custom PatternRecognizer is used for US_SSN because Presidio's built-in
    UsSsnRecognizer has a very low base score (0.05) and relies on context
    word boosting that requires NER tagging.  Our replacement uses a single
    high-confidence pattern that matches the standard NNN-NN-NNNN format
    directly, returning score 0.85 without NER.
    """
    from presidio_analyzer import (
        AnalyzerEngine,
        RecognizerRegistry,
        PatternRecognizer,
        Pattern,
    )
    from presidio_analyzer.predefined_recognizers import (
        CreditCardRecognizer,
        EmailRecognizer,
        IpRecognizer,
        CryptoRecognizer,
        PhoneRecognizer,
    )
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    # ── US SSN — custom high-confidence pattern ───────────────────────────
    # Presidio's built-in UsSsnRecognizer scores 0.05 without NER context
    # boosting; it fires fine via direct pattern match but needs a good score.
    ssn_recognizer = PatternRecognizer(
        supported_entity="US_SSN",
        patterns=[
            Pattern(
                name="SSN NNN-NN-NNNN",
                regex=r"\b\d{3}[- ]\d{2}[- ]\d{4}\b",
                score=0.85,
            ),
        ],
        context=["ssn", "social", "security", "tax", "identification"],
    )

    registry = RecognizerRegistry()
    registry.add_recognizer(ssn_recognizer)
    registry.add_recognizer(CreditCardRecognizer())
    registry.add_recognizer(EmailRecognizer())
    registry.add_recognizer(IpRecognizer())
    registry.add_recognizer(CryptoRecognizer())
    registry.add_recognizer(PhoneRecognizer())

    nlp_provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }
    )
    nlp_engine = nlp_provider.create_engine()
    return AnalyzerEngine(
        registry=registry,
        nlp_engine=nlp_engine,
        supported_languages=["en"],
    )


def _get_engine():
    """Return the cached engine, building it on first call."""
    global _engine, _presidio_available, _engine_initialized
    if _engine_initialized:
        return _engine
    try:
        _engine = _build_engine()
        _presidio_available = True
        logger.info(
            "Presidio AnalyzerEngine initialised (regex-only, entities: %s)",
            _TARGET_ENTITIES,
        )
    except ImportError:
        msg = (
            "presidio-analyzer is not installed. "
            "Run: pip install presidio-analyzer && python -m spacy download en_core_web_sm"
        )
        if _STRICT:
            raise RuntimeError(msg)
        logger.warning("%s  — using the standard-library regex fallback.", msg)
        _presidio_available = False
    except Exception as exc:
        msg = f"Presidio engine failed to initialise: {exc}"
        if _STRICT:
            raise RuntimeError(msg)
        logger.warning("%s  — using the standard-library regex fallback.", msg)
        _presidio_available = False
    _engine_initialized = True
    return _engine


# Eagerly initialise at import time so the first request is not slower.
_get_engine()


# ── Public API ────────────────────────────────────────────────────────────────

def enterprise_regex_guard(text: str) -> str:
    """
    Screen *text* for high-risk enterprise PII using Presidio regex patterns.

    Parameters
    ----------
    text : str
        The incoming payload — typically a query question or a document excerpt.

    Returns
    -------
    str
        The original *text* unmodified if no violations are found.

    Raises
    ------
    fastapi.HTTPException
        HTTP 400 with a structured error body if any target entity is detected.

        Response body::

            {
                "error": "Security violation: sensitive data detected in input.",
                "violated_entities": ["CREDIT_CARD", "EMAIL_ADDRESS"]
            }

    Notes
    -----
    * If Presidio is unavailable, a standard-library regex fallback still
      checks common high-risk PII patterns. ``PRESIDIO_STRICT=true`` requires
      the full Presidio engine and raises during initialization if unavailable.
    * The ``text`` is **never** modified; the function either returns it as-is
      or raises.  Redaction is intentionally outside the scope of this guard —
      the caller decides what to do with blocked requests.
    """
    if not text or not text.strip():
        return text

    # Keep basic PII protection available in lightweight/offline environments.
    violated_entities = _fallback_entities(text)
    engine = _get_engine()

    if engine is not None:
        try:
            results = engine.analyze(
                text=text,
                language="en",
                entities=_TARGET_ENTITIES,
                score_threshold=_SCORE_THRESHOLD,
            )
            violated_entities.update(result.entity_type for result in results)
        except Exception as exc:
            # Preserve regex screening when Presidio's NLP engine is unavailable.
            logger.warning("Presidio analysis failed; using regex fallback: %s", exc)

    if not violated_entities:
        return text

    violated = sorted(violated_entities)

    logger.warning(
        "PII guard blocked input — entities detected: %s | input_length=%d",
        violated,
        len(text),
    )

    raise HTTPException(
        status_code=400,
        detail={
            "error": "Security violation: sensitive data detected in input.",
            "violated_entities": violated,
        },
    )


def _fallback_entities(text: str) -> set[str]:
    """Detect common high-risk PII with stdlib regexes when Presidio is absent."""
    entities: set[str] = set()

    patterns = {
        "EMAIL_ADDRESS": r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        "IP_ADDRESS": (
            r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b"
        ),
        "CRYPTO": (
            r"\b(?:0x[a-fA-F0-9]{40}|[13][a-km-zA-HJ-NP-Z1-9]{25,34})\b"
        ),
        "US_SSN": r"\b(?!000|666|9\d{2})\d{3}[ -](?!00)\d{2}[ -](?!0000)\d{4}\b",
        "PHONE_NUMBER": (
            r"(?<!\d)(?:\+?1[ .-]?)?(?:\(\d{3}\)|\d{3})"
            r"[ .-]?\d{3}[ .-]?\d{4}(?!\d)"
        ),
    }
    for entity, pattern in patterns.items():
        if re.search(pattern, text):
            entities.add(entity)

    for match in re.finditer(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)", text):
        digits = re.sub(r"\D", "", match.group())
        if _passes_luhn(digits):
            entities.add("CREDIT_CARD")
            break

    return entities


def _passes_luhn(number: str) -> bool:
    """Return whether a digit string passes the Luhn checksum."""
    if not 13 <= len(number) <= 19:
        return False
    total = 0
    for offset, char in enumerate(reversed(number)):
        digit = int(char)
        if offset % 2:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0
