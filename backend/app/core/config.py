"""Application configuration — all settings read from environment variables."""
from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # ── Auth ──────────────────────────────────────────────────────────────────
    SECRET_KEY: str = os.getenv("SECRET_KEY", "change-me-INSECURE-default")
    ADMIN_USERNAME: str = os.getenv("ADMIN_USERNAME", "admin")
    ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD", "admin123")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "120"))

    # ── LLM ───────────────────────────────────────────────────────────────────
    USE_OLLAMA: bool = os.getenv("USE_OLLAMA", "true").lower() == "true"
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # ── Embeddings ────────────────────────────────────────────────────────────
    ALLOW_MODEL_DOWNLOADS: bool = os.getenv("ALLOW_MODEL_DOWNLOADS", "false").lower() == "true"

    # ── Adaptive retrieval ────────────────────────────────────────────────────
    ADAPTIVE_RETRIEVAL_ENABLED: bool = os.getenv("ADAPTIVE_RETRIEVAL_ENABLED", "true").lower() == "true"
    ADAPTIVE_MAX_ATTEMPTS: int = int(os.getenv("ADAPTIVE_MAX_ATTEMPTS", "2"))
    ADAPTIVE_POLICY: str = os.getenv("ADAPTIVE_POLICY", "deterministic")  # "deterministic" | "laya"

    # ── Laya (optional decision layer) ───────────────────────────────────────
    LAYA_ENABLED: bool = os.getenv("LAYA_ENABLED", "false").lower() == "true"
    LAYA_MODEL_VERSION: str = os.getenv("LAYA_MODEL_VERSION", "convaiinnovations/laya")
    LAYA_TIMEOUT_SECONDS: float = float(os.getenv("LAYA_TIMEOUT_SECONDS", "3.0"))

    # ── Security ──────────────────────────────────────────────────────────────
    GUARDRAILS_STRICT_MODE: bool = os.getenv("GUARDRAILS_STRICT_MODE", "false").lower() == "true"
    # File extensions accepted by the Docling upload parser.
    ALLOWED_UPLOAD_EXTENSIONS: list = [
        ".pdf", ".docx", ".xlsx", ".pptx",
        ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp",
    ]
    MAX_UPLOAD_BYTES: int = 10 * 1024 * 1024  # 10 MB

    # Per-user document access control:
    # "open"   — all authenticated users can access all documents (default)
    # "user"   — each user can only access their own uploads
    DOC_ACCESS_MODE: str = os.getenv("DOC_ACCESS_MODE", "open")

    # ── CORS ──────────────────────────────────────────────────────────────────
    # Comma-separated allowed origins.  Use "*" only in development.
    ALLOWED_ORIGINS: list = [
        o.strip()
        for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")
        if o.strip()
    ]


settings = Settings()
