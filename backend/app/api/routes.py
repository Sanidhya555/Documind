"""FastAPI routes — all endpoints under /api prefix.

Backward-compatible with v2 API contracts.
New in v3:
  - /api/query response includes retrieval_decision and evidence_confidence fields
  - /api/metrics includes adaptive retrieval and Laya config flags
  - Per-user access control enforced server-side via security helpers
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordRequestForm
from app.core.config import settings
from app.core.logger import setup_logger
from app.core.security import (
    create_access_token,
    require_session_owner,
    verify_token,
)
from app.api.schemas import QueryRequest, SessionRequest, TitleRequest
from app.api.upload_validation import validate_upload_metadata
from app.services.rag_services import RAGService
from utils.parser import DocumentParsingError, extract_document_content
from utils.presidio_guard import enterprise_regex_guard

router = APIRouter()
rag = RAGService()
logger = setup_logger()


# ─── Health ───────────────────────────────────────────────────────────────────

@router.get("/health")
def health():
    return {
        "status": "healthy",
        "documents_indexed": len(rag.list_documents()),
        "adaptive_retrieval": settings.ADAPTIVE_RETRIEVAL_ENABLED,
    }


# ─── Auth ─────────────────────────────────────────────────────────────────────

@router.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    if (
        form_data.username == settings.ADMIN_USERNAME
        and form_data.password == settings.ADMIN_PASSWORD
    ):
        token = create_access_token({"sub": form_data.username})
        return {"access_token": token, "token_type": "bearer"}
    logger.warning("Failed login attempt: %s", form_data.username)
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials",
    )


# ─── Documents ────────────────────────────────────────────────────────────────

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    user: str = Depends(verify_token),
):
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)
    extension = validate_upload_metadata(
        file.filename or "",
        size,
        settings.ALLOWED_UPLOAD_EXTENSIONS,
        settings.MAX_UPLOAD_BYTES,
    )

    # Sanitize filename — no path traversal
    filename = os.path.basename(file.filename or "upload")
    if not filename or filename in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid filename.")

    # ── Layer 1: Presidio PII guard on filename ───────────────────────────────
    # Filenames occasionally encode sensitive data (e.g. "SSN_123-45-6789.pdf").
    enterprise_regex_guard(filename)

    existing = {d["filename"].lower() for d in rag.list_documents(user=user)}
    if filename.lower() in existing:
        raise HTTPException(status_code=400, detail="Document already indexed.")

    raw_dir = rag.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    save_path = raw_dir / filename
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".docling-", suffix=extension,
            dir=raw_dir, delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            file.file.seek(0)
            shutil.copyfileobj(file.file, temporary_file)

        markdown = await extract_document_content(str(temporary_path))
        if not markdown.strip():
            raise HTTPException(
                status_code=422,
                detail="Docling could not extract readable content from this file.",
            )

        os.replace(temporary_path, save_path)
        temporary_path = None
        result = rag.add_document(
            str(save_path), extracted_content=markdown, owner=user
        )
    except DocumentParsingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    logger.info("User %s uploaded %s", user, filename)
    return result


@router.get("/documents")
def list_documents(user: str = Depends(verify_token)):
    return rag.list_documents(user=user)


@router.post("/documents/{filename}/reindex")
async def reindex_document(filename: str, user: str = Depends(verify_token)):
    safe_filename = os.path.basename(filename)
    if not safe_filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    return await rag.reindex_document(safe_filename, user=user)


@router.delete("/documents/{filename}")
def delete_document(filename: str, user: str = Depends(verify_token)):
    # Sanitize path parameter
    safe_filename = os.path.basename(filename)
    if not safe_filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    logger.info("User %s deleting %s", user, safe_filename)
    return rag.delete_document(safe_filename, user=user)


# ─── Query ────────────────────────────────────────────────────────────────────

@router.post("/query")
async def ask_question(
    request: QueryRequest,
    user: str = Depends(verify_token),
):
    # ── Layer 1: Presidio PII guard ───────────────────────────────────────────
    # Screen the question for sensitive data before it reaches the RAG pipeline
    # or is stored in conversation memory.
    enterprise_regex_guard(request.question)

    # Enforce session ownership
    require_session_owner(request.session_id, user)

    if request.stream:
        async def event_stream():
            try:
                async for token in rag.query_stream(
                    request.question, request.session_id, request.top_k, user=user
                ):
                    yield f"data: {json.dumps({'token': token})}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return rag.query(
        request.question,
        session_id=request.session_id,
        top_k=request.top_k,
        use_cache=request.use_cache,
        user=user,
    )


# ─── Sessions ─────────────────────────────────────────────────────────────────

@router.get("/sessions")
def list_sessions(user: str = Depends(verify_token)):
    return rag.list_sessions(user=user)


@router.post("/sessions/new")
def new_session(body: SessionRequest, user: str = Depends(verify_token)):
    require_session_owner(body.session_id, user)
    return rag.new_session(body.session_id)


@router.get("/sessions/{session_id}/messages")
def get_session_messages(
    session_id: str,
    user: str = Depends(verify_token),
):
    require_session_owner(session_id, user)
    return rag.get_session_messages(session_id)


@router.put("/sessions/{session_id}/title")
def update_session_title(
    session_id: str,
    body: TitleRequest,
    user: str = Depends(verify_token),
):
    require_session_owner(session_id, user)
    rag.long_term.update_session_title(session_id, body.title)
    return {"status": "updated"}


@router.delete("/sessions/{session_id}")
def delete_session(
    session_id: str,
    user: str = Depends(verify_token),
):
    require_session_owner(session_id, user)
    return rag.delete_session(session_id, user=user)


# ─── Evaluation ───────────────────────────────────────────────────────────────

@router.post("/evaluate/retrieval")
def evaluate_retrieval(
    top_k: int = 5,
    user: str = Depends(verify_token),
):
    return rag.evaluator.evaluate_retrieval(top_k=top_k, user=user)


@router.post("/evaluate/batch")
def evaluate_batch(user: str = Depends(verify_token)):
    return rag.evaluator.run_batch_eval(user=user)


@router.post("/evaluate/answer_relevancy")
def evaluate_answer_relevancy(
    question: str,
    answer: str,
    user: str = Depends(verify_token),
):
    return rag.evaluator.evaluate_answer_relevancy(question, answer)


@router.post("/evaluate/security")
def evaluate_security(user: str = Depends(verify_token)):
    """Run security guardrails evaluation against eval_set_v2.json."""
    return rag.evaluator.run_security_eval()


@router.post("/evaluate/adaptive")
def evaluate_adaptive(user: str = Depends(verify_token)):
    """Run adaptive retrieval decision accuracy evaluation against eval_set_v2.json."""
    return rag.evaluator.run_adaptive_eval(user=user)


def _run_backend_test_suite() -> dict:
    """Execute the repository's stage and security tests and capture a summary."""
    backend_dir = Path(__file__).resolve().parents[2]
    started = time.monotonic()
    stage_counts = {}
    with tempfile.TemporaryDirectory(prefix="rag-eval-") as temp_dir:
        report_path = Path(temp_dir) / "junit.xml"
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    "--tb=short",
                    f"--junitxml={report_path}",
                ],
                cwd=backend_dir,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            output = "\n".join(
                item.decode(errors="replace") if isinstance(item, bytes) else (item or "")
                for item in (exc.stdout, exc.stderr)
            )
            return {
                "status": "timeout",
                "duration_seconds": round(time.monotonic() - started, 2),
                "test_counts": {},
                "stages": {},
                "output": output[-12000:],
            }

        if report_path.exists():
            try:
                xml_root = ET.parse(report_path).getroot()
                for case in xml_root.iter("testcase"):
                    filename = Path(case.get("file") or case.get("classname", "tests")).name
                    stage = stage_counts.setdefault(filename, {"passed": 0, "failed": 0, "errors": 0, "skipped": 0})
                    if case.find("failure") is not None:
                        stage["failed"] += 1
                    elif case.find("error") is not None:
                        stage["errors"] += 1
                    elif case.find("skipped") is not None:
                        stage["skipped"] += 1
                    else:
                        stage["passed"] += 1
            except (ET.ParseError, OSError) as exc:
                logger.warning("Could not read pytest stage report: %s", exc)

    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    counts = {key: 0 for key in ("passed", "failed", "errors", "skipped", "xfailed", "xpassed")}
    for count, label in re.findall(
        r"(\d+)\s+(passed|failed|errors?|skipped|xfailed|xpassed)", output
    ):
        normalized = "errors" if label in ("error", "errors") else label
        counts[normalized] += int(count)

    return {
        "status": "passed" if completed.returncode == 0 else "failed",
        "exit_code": completed.returncode,
        "duration_seconds": round(time.monotonic() - started, 2),
        "test_counts": counts,
        "stages": stage_counts,
        "output": output[-12000:],
    }


@router.post("/evaluate/all")
def evaluate_all(user: str = Depends(verify_token)):
    """Run the complete offline, 100-case-per-stage evaluation dashboard suite."""
    started = time.monotonic()
    from evaluation.runners.run_all import run_all_stage_evaluations

    try:
        stage_results = run_all_stage_evaluations()
    except Exception as exc:
        logger.exception("Unified stage evaluation failed")
        stage_results = {"status": "failed", "error": str(exc), "stages": []}
    dataset_integrity = rag.evaluator.run_stage_dataset_integrity()
    suites = {
        "stage_evaluations": stage_results,
        "dataset_integrity": dataset_integrity,
    }
    status_value = "passed" if all(
        result.get("status") == "passed" for result in suites.values()
    ) else "failed"
    return {
        "status": status_value,
        "duration_seconds": round(time.monotonic() - started, 2),
        "suites": suites,
    }


@router.post("/tests/run")
def run_all_tests(user: str = Depends(verify_token)):
    """Run the complete backend pytest suite and return its summary and output."""
    return _run_backend_test_suite()


# ─── Metrics ──────────────────────────────────────────────────────────────────

@router.get("/metrics")
def metrics(user: str = Depends(verify_token)):
    return rag.get_metrics(user=user)
