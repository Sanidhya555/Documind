"""
Integration tests for the FastAPI routes.

These tests use TestClient with mocked RAGService to avoid needing
a running LLM or FAISS index. They verify:
- Auth flow (login, token validation)
- Input validation (empty question, oversized payload)
- Query endpoint response contract
- Session management endpoints
- Document endpoints (mock)
- Metrics endpoint
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


def _build_mock_rag():
    """Build a fully-configured mock RAGService."""
    mock_rag = MagicMock()

    mock_rag.list_documents.return_value = [
        {"filename": "test.pdf", "uploaded_at": "2024-01-01T00:00:00", "chunks": 10}
    ]
    mock_rag.raw_dir = MagicMock()
    mock_rag.raw_dir.mkdir = MagicMock()
    mock_rag.add_document.return_value = {
        "status": "indexed", "filename": "test.pdf", "chunks_added": 5
    }
    mock_rag.delete_document.return_value = {
        "status": "deleted", "filename": "test.pdf"
    }
    mock_rag.reindex_document = AsyncMock(return_value={
        "status": "reindexed", "filename": "test.pdf", "chunks_added": 1,
        "chunks_replaced": 1,
    })
    mock_rag.query.return_value = {
        "answer": "Morgan Stanley used evals to measure AI performance.",
        "sources": [{"filename": "doc.pdf", "chunk_id": 0, "score": 0.85}],
        "expanded_queries": ["alternative phrasing"],
        "contexts": ["Morgan Stanley context text"],
        "retrieval_decision": "answer",
        "evidence_confidence": 0.82,
        "latency_ms": 150.0,
        "cached": False,
        "metrics": {
            "total_latency": 0.15,
            "retrieval_time": 0.05,
            "generation_time": 0.10,
            "faithfulness_score": 0.9,
            "adaptive_iterations": 1,
            "evidence_coverage": 0.75,
        },
    }
    mock_rag.list_sessions.return_value = [
        {
            "session_id": "sess_default",
            "title": "Test Session",
            "created_at": "2024-01-01T00:00:00",
            "last_active": "2024-01-01T00:01:00",
            "summary": "",
        }
    ]
    mock_rag.new_session.return_value = {"session_id": "sess_new", "status": "created"}
    mock_rag.get_session_messages.return_value = [
        {"role": "user", "content": "Hello", "timestamp": "2024-01-01T00:00:00"},
    ]
    mock_rag.delete_session.return_value = {"status": "deleted", "session_id": "sess_default"}
    mock_rag.long_term = MagicMock()
    mock_rag.long_term.update_session_title = MagicMock()
    mock_rag.get_metrics.return_value = {
        "documents_indexed": 1,
        "total_chunks": 10,
        "vector_dimension": 384,
        "bm25_chunks_indexed": 10,
        "reranker_available": False,
        "active_sessions": 1,
        "query_cache_size": 0,
        "adaptive_retrieval_enabled": True,
        "adaptive_policy": "deterministic",
        "laya_enabled": False,
    }
    mock_rag.evaluator = MagicMock()
    mock_rag.evaluator.evaluate_retrieval.return_value = {
        "num_queries": 5, "context_precision@5": 0.72,
    }
    mock_rag.evaluator.run_batch_eval.return_value = {
        "rows": [],
        "summary": {
            "num_samples": 0,
            "avg_context_precision": 0.0,
            "avg_faithfulness": 0.0,
            "avg_answer_relevancy": 0.0,
        },
        "ragas_score": None,
    }
    mock_rag.evaluator.run_security_eval.return_value = {
        "num_attack_cases": 10, "attack_block_rate": 0.9, "false_positive_rate": 0.05,
    }
    mock_rag.evaluator.run_adaptive_eval.return_value = {
        "num_cases": 10, "decision_accuracy": 0.85,
    }
    mock_rag.evaluator.run_stage_dataset_integrity.return_value = {
        "status": "passed", "num_datasets": 12, "num_cases": 1200, "datasets": [],
    }
    mock_rag.evaluator.evaluate_answer_relevancy.return_value = {"answer_relevancy": 0.75}
    mock_rag.guardrails = MagicMock()
    return mock_rag


# ── App import with patched RAGService ────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """
    Create a TestClient with a mocked RAG service before importing the routes.

    The routes module creates its service at import time, so patching
    `routes.rag` after import still initializes the full ML stack and provider
    client. Replace the service factory first so API contract tests stay local
    and independent of model downloads, provider credentials, and indexes.
    """
    # First, ensure a clean import path
    import sys

    # Stub heavy pipeline dependencies so routes.py can import without
    # loading FAISS, torch, or sentence_transformers
    heavy_mocks = [
        "faiss", "torch", "sentence_transformers",
        "pipeline.embeddings.embedder",
        "pipeline.vector_store.faiss_store",
        "pipeline.retriever.vector_retriever",
        "pipeline.retriever.bm25_retriever",
        "pipeline.retriever.hybrid_retriever",
        "pipeline.reranker.cross_encoder_reranker",
        "pipeline.adaptive_retrieval.retrieval_controller",
        "pipeline.adaptive_retrieval.decision_policy",
        "pipeline.adaptive_retrieval.evidence_assessor",
        "pipeline.llm.generator",
        "pipeline.memory.long_term",
        "pipeline.memory.short_term",
        "pipeline.guardrails.guardrails",
        "pipeline.query_expansion.query_expander",
        "pipeline.ingestion.ingestion_pipeline",
        "pipeline.chunking.recursive_chunker",
        "app.services.evaluator",
    ]

    stubs = {}
    for mod in heavy_mocks:
        if mod not in sys.modules:
            stubs[mod] = MagicMock()
            sys.modules[mod] = stubs[mod]

    # Prevent routes.py's module-level service from loading models or clients.
    import importlib
    import app.services.rag_services as rag_services_module
    real_rag_service = rag_services_module.RAGService
    rag_services_module.RAGService = lambda: _build_mock_rag()
    try:
        import app.api.routes as routes_module
        importlib.reload(routes_module)
    finally:
        rag_services_module.RAGService = real_rag_service
    routes_module.rag = _build_mock_rag()

    from app.main import app
    with TestClient(app) as c:
        yield c

    # Clean up injected stubs
    for mod in stubs:
        sys.modules.pop(mod, None)


def get_token(client) -> str:
    """Helper: get a valid JWT token."""
    resp = client.post(
        "/api/login",
        data={"username": "admin", "password": "admin123"},
    )
    assert resp.status_code == 200
    return resp.json()["access_token"]


# ── Auth ──────────────────────────────────────────────────────────────────────

class TestAuth:
    def test_login_success(self, client):
        resp = client.post(
            "/api/login",
            data={"username": "admin", "password": "admin123"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_login_wrong_password(self, client):
        resp = client.post(
            "/api/login",
            data={"username": "admin", "password": "wrongpass"},
        )
        assert resp.status_code == 401

    def test_login_wrong_user(self, client):
        resp = client.post(
            "/api/login",
            data={"username": "notexist", "password": "admin123"},
        )
        assert resp.status_code == 401

    def test_protected_endpoint_without_token(self, client):
        resp = client.get("/api/documents")
        assert resp.status_code == 401

    def test_protected_endpoint_invalid_token(self, client):
        resp = client.get(
            "/api/documents",
            headers={"Authorization": "Bearer invalidtoken"},
        )
        assert resp.status_code == 401


# ── Health ────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"


# ── Documents ─────────────────────────────────────────────────────────────────

class TestDocuments:
    def test_list_documents(self, client):
        token = get_token(client)
        resp = client.get(
            "/api/documents",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_reindex_document_endpoint(self, client):
        import app.api.routes as routes_module

        routes_module.rag.reindex_document.return_value = {
            "status": "reindexed", "filename": "paper.pdf", "chunks_added": 12,
            "chunks_replaced": 10,
        }
        token = get_token(client)

        resp = client.post(
            "/api/documents/paper.pdf/reindex",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert resp.status_code == 200
        assert resp.json()["chunks_added"] == 12
        routes_module.rag.reindex_document.assert_called_once_with("paper.pdf", user="admin")

    def test_upload_unsupported_format(self, client):
        token = get_token(client)
        resp = client.post(
            "/api/upload",
            headers={"Authorization": f"Bearer {token}"},
            files={"file": ("malware.exe", b"binary content", "application/octet-stream")},
        )
        assert resp.status_code == 400
        assert "Unsupported" in resp.json()["detail"]

    def test_upload_oversized_file(self, client):
        token = get_token(client)
        big_content = b"x" * (11 * 1024 * 1024)  # 11 MB
        # The middleware raises HTTPException(413); TestClient may surface it
        # as an HTTP response OR raise the exception depending on raise_server_exceptions.
        try:
            resp = client.post(
                "/api/upload",
                headers={"Authorization": f"Bearer {token}"},
                files={"file": ("big.pdf", big_content, "application/pdf")},
            )
            assert resp.status_code in (400, 413)
        except Exception as exc:
            # TestClient raises the HTTPException directly when raise_server_exceptions=True
            assert "413" in str(exc) or "Payload too large" in str(exc) or "400" in str(exc)


# ── Query ─────────────────────────────────────────────────────────────────────

class TestQuery:
    def test_query_returns_expected_fields(self, client):
        token = get_token(client)
        resp = client.post(
            "/api/query",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "question": "What did Morgan Stanley use evals for?",
                "session_id": "default",
                "top_k": 5,
                "stream": False,
                "use_cache": False,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "sources" in data
        assert "retrieval_decision" in data
        assert "evidence_confidence" in data
        assert "metrics" in data

    def test_query_empty_question_rejected(self, client):
        token = get_token(client)
        resp = client.post(
            "/api/query",
            headers={"Authorization": f"Bearer {token}"},
            json={"question": "", "session_id": "default"},
        )
        assert resp.status_code == 422  # Pydantic min_length validation

    def test_query_top_k_out_of_range(self, client):
        token = get_token(client)
        resp = client.post(
            "/api/query",
            headers={"Authorization": f"Bearer {token}"},
            json={"question": "test", "session_id": "default", "top_k": 50},
        )
        assert resp.status_code == 422  # Pydantic ge/le validation


# ── Sessions ──────────────────────────────────────────────────────────────────

class TestSessions:
    def test_list_sessions(self, client):
        token = get_token(client)
        resp = client.get(
            "/api/sessions",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_create_session(self, client):
        token = get_token(client)
        resp = client.post(
            "/api/sessions/new",
            headers={"Authorization": f"Bearer {token}"},
            json={"session_id": "sess_new"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data

    def test_get_session_messages(self, client):
        token = get_token(client)
        resp = client.get(
            "/api/sessions/sess_default/messages",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_delete_session(self, client):
        token = get_token(client)
        resp = client.delete(
            "/api/sessions/sess_default",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200


# ── Metrics ───────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_metrics_returns_expected_fields(self, client):
        token = get_token(client)
        resp = client.get(
            "/api/metrics",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "documents_indexed" in data
        assert "total_chunks" in data
        assert "adaptive_retrieval_enabled" in data
        assert "laya_enabled" in data


# ── Evaluation ────────────────────────────────────────────────────────────────

class TestEvaluation:
    def test_full_evaluation_dashboard_runs_every_suite(self, client, monkeypatch):
        import app.api.routes as routes_module
        monkeypatch.setattr(
            "evaluation.runners.run_all.run_all_stage_evaluations",
            lambda: {"status": "passed", "stage_count": 21, "total_cases": 2100, "passed": 2100, "failed": 0, "stages": []},
        )
        monkeypatch.setattr(
            routes_module.rag.evaluator,
            "run_stage_dataset_integrity",
            lambda: {"status": "passed", "num_datasets": 21, "num_cases": 2100, "datasets": []},
        )
        token = get_token(client)

        response = client.post(
            "/api/evaluate/all",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "passed"
        assert set(data["suites"]) == {"stage_evaluations", "dataset_integrity"}

    def test_retrieval_eval_endpoint(self, client):
        token = get_token(client)
        resp = client.post(
            "/api/evaluate/retrieval?top_k=5",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200

    def test_batch_eval_endpoint(self, client):
        token = get_token(client)
        resp = client.post(
            "/api/evaluate/batch",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200

    def test_security_eval_endpoint(self, client):
        token = get_token(client)
        resp = client.post(
            "/api/evaluate/security",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200

    def test_adaptive_eval_endpoint(self, client):
        token = get_token(client)
        resp = client.post(
            "/api/evaluate/adaptive",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200

    def test_run_all_tests_executes_fixed_pytest_suite(self, client, monkeypatch):
        import subprocess
        import app.api.routes as routes_module

        mocked_run = MagicMock(
            return_value=subprocess.CompletedProcess(
                args=["python", "-m", "pytest"],
                returncode=0,
                stdout="12 passed, 1 skipped in 1.2s",
                stderr="",
            )
        )
        monkeypatch.setattr(routes_module.subprocess, "run", mocked_run)
        token = get_token(client)

        resp = client.post(
            "/api/tests/run",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "passed"
        assert data["test_counts"]["passed"] == 12
        assert data["test_counts"]["skipped"] == 1
        command = mocked_run.call_args.args[0]
        assert command[:3] == [routes_module.sys.executable, "-m", "pytest"]
        assert "tests" in command
