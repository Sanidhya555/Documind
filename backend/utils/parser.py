"""Shared asynchronous Docling parser for uploaded enterprise documents."""
from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any


class DocumentParsingError(ValueError):
    """Raised when Docling cannot convert a document into Markdown."""


_converter: Any = None
_converter_lock = threading.Lock()
_conversion_lock = threading.Lock()


def _get_converter() -> Any:
    """Create the heavyweight converter once and reuse it across requests."""
    global _converter
    if _converter is None:
        with _converter_lock:
            if _converter is None:
                from docling.document_converter import DocumentConverter

                _converter = DocumentConverter()
    return _converter


def _convert_to_markdown(path: Path, converter: Any) -> str:
    with _conversion_lock:
        result = converter.convert(str(path))
        document = getattr(result, "document", None)
        if document is None:
            raise DocumentParsingError("Docling did not return a converted document.")
        markdown = document.export_to_markdown()
    if not isinstance(markdown, str):
        raise DocumentParsingError("Docling returned an invalid Markdown document.")
    return markdown


async def extract_document_content(
    file_path: str, *, converter: Any = None
) -> str:
    """Convert a local document to structured Markdown without blocking the loop.

    `converter` is an optional injection point for offline evaluation. Production
    requests share a lazily initialized DocumentConverter instance.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Document not found: {path.name}")

    try:
        active_converter = converter if converter is not None else _get_converter()
        return await asyncio.to_thread(_convert_to_markdown, path, active_converter)
    except DocumentParsingError:
        raise
    except Exception as exc:
        raise DocumentParsingError(
            f"Unable to read or convert '{path.name}' with Docling."
        ) from exc
