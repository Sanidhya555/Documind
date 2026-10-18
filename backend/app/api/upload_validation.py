"""Small, side-effect-free checks for an uploaded document's name and size."""
from pathlib import Path
from typing import Collection

from fastapi import HTTPException


def validate_upload_metadata(
    filename: str,
    size_bytes: int,
    allowed_extensions: Collection[str],
    max_upload_bytes: int,
) -> str:
    extension = Path(filename or "").suffix.lower()
    allowed = {value.lower() for value in allowed_extensions}
    if extension not in allowed:
        allowed_types = ", ".join(sorted(item.upper() for item in allowed))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{extension}'. Allowed formats: {allowed_types}.",
        )
    if size_bytes > max_upload_bytes:
        raise HTTPException(status_code=400, detail="File size exceeds 10 MB limit.")
    return extension
