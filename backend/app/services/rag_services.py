"""
RAG Service — central orchestrator (v3 — Adaptive Retrieval + Hardened Security).

Changes vs v2:
- AdaptiveRetrievalController replaces bare _retrieve_rerank / _multi_query_retrieve.
- Guardrails now scans each retrieved chunk before it enters the prompt context
  (treats documents as untrusted data, never as system instructions).
- Evidence assessment result is included in query response for transparency.
- Laya decision layer is wired in when LAYA_ENABLED=true and the package is present;
  deterministic fallback is always available.
- Per-user session and document access helpers called from routes layer.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional, Tuple
from urllib.parse import quote

from fastapi import HTTPException

from app.core.config import settings
from app.core.logger import setup_logger
from app.services.evaluator import RAGEvaluator
from app.services.query_cache import QueryCache
from pipeline.adaptive_retrieval.retrieval_controller import AdaptiveRetrievalController
from pipeline.adaptive_retrieval.schemas import (
    AdaptiveConfig,
    RetrievalAction,
    RetrievalDecision,
)
from pipeline.adaptive_retrieval.decision_policy import (
    DeterministicDecisionPolicy,
    LayaDecisionPolicy,
)
from pipeline.chunking.recursive_chunker import RecursiveChunker
from pipeline.embeddings.embedder import Embedder
from pipeline.guardrails.guardrails import Guardrails
from pipeline.llm.generator import OllamaGenerator, OpenAIGenerator
from pipeline.llm.prompt_template import PromptTemplate
from pipeline.memory.long_term import LongTermMemory
from pipeline.memory.short_term import ShortTermMemory
from pipeline.query_expansion.query_expander import QueryExpander
from pipeline.reranker.cross_encoder_reranker import CrossEncoderReranker
from pipeline.retriever.bm25_retriever import BM25Retriever
from pipeline.retriever.hybrid_retriever import HybridRetriever
from pipeline.retriever.vector_retriever import VectorRetriever
from pipeline.vector_store.faiss_store import FAISSStore
from utils.parser import DocumentParsingError, extract_document_content

logger = setup_logger()


# ─── RAG Service ──────────────────────────────────────────────────────────────

class RAGService:

    def __init__(self):
        self.base_dir = Path(__file__).resolve().parents[2]
        self.data_dir = self.base_dir / "data"
        self.raw_dir = self.data_dir / "raw"
        self.embeddings_dir = self.data_dir / "embeddings"
        self.index_path = self.embeddings_dir / "faiss.index"
        self.chunks_path = self.embeddings_dir / "chunks.npy"
        self.memory_path = self.data_dir / "memory.db"

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

        # Core retrieval stack
        self.embedder = Embedder()
        self.vector_store = FAISSStore(dimension=384, embedder=self.embedder)
        self.vector_retriever = VectorRetriever(self.embedder, self.vector_store)
        self.bm25_retriever = BM25Retriever()
        self.hybrid_retriever = HybridRetriever(
            self.vector_retriever, self.bm25_retriever
        )
        self.reranker = CrossEncoderReranker()

        # LLM
        if settings.USE_OLLAMA:
            logger.info("LLM backend: Ollama (%s)", settings.OLLAMA_MODEL)
            self.generator = OllamaGenerator(model=settings.OLLAMA_MODEL)
        else:
            logger.info("LLM backend: OpenAI (%s)", settings.OPENAI_MODEL)
            self.generator = OpenAIGenerator()

        # Query expansion
        self.query_expander = QueryExpander(generator=self.generator, num_expansions=2)

        # Guardrails (layered)
        self.guardrails = Guardrails(strict_mode=settings.GUARDRAILS_STRICT_MODE)

        # Memory
        self.short_term = ShortTermMemory(max_turns=10)
        self.long_term = LongTermMemory(db_path=str(self.memory_path))

        # Cache
        self.cache = QueryCache(max_size=200, ttl=600)
        self.index_version = 0

        # Adaptive retrieval controller
        adaptive_cfg = AdaptiveConfig(
            enabled=settings.ADAPTIVE_RETRIEVAL_ENABLED,
            max_retrieval_attempts=settings.ADAPTIVE_MAX_ATTEMPTS,
            policy_mode=settings.ADAPTIVE_POLICY,
        )
        policy = self._build_policy(adaptive_cfg)
        self.adaptive_controller = AdaptiveRetrievalController(
            hybrid_retriever=self.hybrid_retriever,
            reranker=self.reranker,
            query_expander=self.query_expander,
            config=adaptive_cfg,
            policy=policy,
        )

        # Evaluator
        dataset_path = str(self.base_dir / "evaluation" / "datasets" / "eval_set_v2.json")
        self.evaluator = RAGEvaluator(
            rag_service=self,
            dataset_path=dataset_path,
        )

        # Load persisted FAISS index
        if self.index_path.exists() and self.chunks_path.exists():
            self.vector_store.load(
                index_path=str(self.index_path),
                chunks_path=str(self.chunks_path),
            )
            self.bm25_retriever.index(self.vector_store.text_chunks)
            logger.info(
                "Loaded index: %d vectors", self.vector_store.index.ntotal
            )
        else:
            logger.info("No index found — starting fresh")

    def _build_policy(self, cfg: AdaptiveConfig):
        """Build the decision policy, optionally wrapping Laya."""
        deterministic = DeterministicDecisionPolicy(
            min_relevance_threshold=cfg.min_top_relevance_threshold,
            min_coverage_threshold=cfg.min_coverage_threshold,
            strict_abstention=cfg.strict_abstention,
        )
        if cfg.policy_mode == "laya" and settings.LAYA_ENABLED:
            try:
                from pipeline.laya.client import LayaClient
                laya_client = LayaClient(
                    enabled=True,
                    model_version=settings.LAYA_MODEL_VERSION,
                    timeout_seconds=settings.LAYA_TIMEOUT_SECONDS,
                )
                return LayaDecisionPolicy(
                    laya_service=laya_client,
                    fallback_policy=deterministic,
                )
            except Exception as e:
                logger.warning(
                    "Could not initialise Laya policy (%s); using deterministic fallback", e
                )
        return deterministic

    # ─── Internal helpers ────────────────────────────────────────────────────

    def _sanitize_chunks_for_context(
        self, results: List[Tuple[Dict, float]]
    ) -> List[Tuple[Dict, float]]:
        """
        Scan each retrieved chunk for indirect injection before it enters the prompt.
        Chunks with injections are sanitized in-place; a warning is logged.
        Retrieved documents are treated as UNTRUSTED DATA.
        """
        sanitized: List[Tuple[Dict, float]] = []
        for chunk, score in results:
            if not isinstance(chunk, dict):
                continue
            text = chunk.get("text", "")
            chunk_id = f"{chunk.get('filename','?')}::{chunk.get('chunk_id','?')}"
            _safe, clean_text = self.guardrails.scan_retrieved_chunk(text, chunk_id)
            # Always include the chunk (sanitized), never silently promote injection
            sanitized_chunk = {**chunk, "text": clean_text}
            sanitized.append((sanitized_chunk, score))
        return sanitized

    def _build_context(
        self, results: List[Tuple[Dict, float]]
    ) -> Tuple[str, List[Dict]]:
        chunks = [r[0]["text"] for r in results if isinstance(r[0], dict)]
        context = "\n\n---\n\n".join(chunks) if chunks else "No relevant documents found."
        sources = [
            {
                "filename": r[0].get("filename", ""),
                "chunk_id": r[0].get("chunk_id", 0),
                "score": round(r[1], 4),
            }
            for r in results
            if isinstance(r[0], dict)
        ]
        return context, sources

    def _auto_title(self, session_id: str, question: str, answer: str):
        sessions = self.long_term.list_sessions()
        session = next(
            (s for s in sessions if s["session_id"] == session_id), None
        )
        if session and session.get("title") == "New Chat":
            prompt = PromptTemplate.build_summary([
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ])
            try:
                title = self.generator.generate(prompt, max_tokens=32).strip()[:60]
                self.long_term.update_session_title(session_id, title)
            except Exception:
                pass

    def _fallback_answer(
        self, results: List[Tuple[Dict, float]], error: Exception
    ) -> str:
        logger.warning(
            "LLM generation unavailable; returning fallback: %s", error
        )
        snippets = [
            r[0].get("text", "").strip()
            for r in results
            if isinstance(r[0], dict) and r[0].get("text")
        ][:3]
        if not snippets:
            return (
                "The language model backend is unavailable, and no relevant "
                "document context was found for this question."
            )
        excerpt_text = "\n\n".join(
            f"Context {i + 1}: {s[:700]}" for i, s in enumerate(snippets)
        )
        return (
            "The language model backend is unavailable. "
            f"Relevant document context:\n\n{excerpt_text}"
        )

    # ─── Public query API ────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        session_id: str = "default",
        top_k: Optional[int] = None,
        use_cache: bool = True,
        user: str = "",
    ) -> Dict:
        t0 = time.time()

        # L1: Input guardrails
        safe, reason = self.guardrails.check_input(question)
        if not safe:
            return {
                "answer": reason,
                "sources": [],
                "latency_ms": 0,
                "cached": False,
                "retrieval_decision": "blocked_by_guardrails",
                "metrics": {
                    "total_latency": 0.0,
                    "retrieval_time": 0.0,
                    "generation_time": 0.0,
                    "faithfulness_score": None,
                    "cache_hit": False,
                },
            }

        history = self.short_term.format_for_prompt(session_id)
        cache_history = history
        history_messages = self.short_term.get(session_id)
        reusable_history = list(history_messages)
        while (
            len(reusable_history) >= 2
            and reusable_history[-2].get("role") == "user"
            and reusable_history[-2].get("content", "").strip().lower() == question.strip().lower()
            and reusable_history[-1].get("role") == "assistant"
        ):
            reusable_history = reusable_history[:-2]
        cache_history = "\n".join(
            f"{'User' if item['role'] == 'user' else 'Assistant'}: {item['content']}"
            for item in reusable_history
        )

        # L2: Cache lookup
        if use_cache:
            cached = self.cache.get(
                question,
                session_id,
                user,
                top_k=top_k,
                history=cache_history,
                index_version=self.index_version,
            )
            if cached:
                logger.info("Cache hit for session=%s", session_id)
                self.short_term.add(session_id, "user", question)
                self.short_term.add(session_id, "assistant", cached.get("answer", ""))
                self.long_term.save_turn(session_id, question, cached.get("answer", ""))
                elapsed = time.time() - t0
                cached_metrics = {
                    **cached.get("metrics", {}),
                    "total_latency": round(elapsed, 3),
                    "retrieval_time": 0.0,
                    "generation_time": 0.0,
                    "cache_hit": True,
                }
                return {
                    **cached,
                    "metrics": cached_metrics,
                    "latency_ms": round(elapsed * 1000, 2),
                    "cached": True,
                }

        # Access filtering is required for retrieval, but a user-scoped cache hit
        # can safely return before loading the document list.
        allowed_filenames = self.allowed_filenames(user)

        # L3: Adaptive retrieval (with bounded retries + evidence assessment)
        retrieval_start = time.time()
        results, decision, assessment, expansion_queries = (
            self.adaptive_controller.execute_retrieval(
                question,
                top_k=top_k,
                allowed_filenames=allowed_filenames,
            )
        )
        retrieval_latency = time.time() - retrieval_start

        # L4: Scan retrieved chunks for indirect injection (untrusted data)
        results = self._sanitize_chunks_for_context(results)
        context, sources = self._build_context(results)

        # L5: Handle abstention / clarification
        if decision.action == RetrievalAction.ABSTAIN:
            abstention_msg = (
                decision.abstention_reason
                or "I could not find sufficient relevant information in the indexed documents to answer this question."
            )
            self.short_term.add(session_id, "user", question)
            self.short_term.add(session_id, "assistant", abstention_msg)
            self.long_term.save_turn(session_id, question, abstention_msg)
            metrics = {
                "total_latency": round(time.time() - t0, 3),
                "retrieval_time": round(retrieval_latency, 3),
                "generation_time": 0.0,
                "faithfulness_score": self.evaluator.quick_check(abstention_msg, []),
                "cache_hit": False,
                "adaptive_iterations": decision.iteration,
            }
            response_payload = {
                "answer": abstention_msg,
                "sources": [],
                "expanded_queries": expansion_queries,
                "contexts": [],
                "retrieval_decision": decision.action.value,
                "evidence_confidence": assessment.confidence,
                "metrics": metrics,
            }
            if use_cache:
                self.cache.set(
                    question, session_id, response_payload, user,
                    top_k=top_k, history=cache_history,
                    index_version=self.index_version,
                )
            return {
                **response_payload,
                "latency_ms": round((time.time() - t0) * 1000, 2),
                "cached": False,
            }

        if decision.action == RetrievalAction.CLARIFY:
            clarification = decision.clarification_prompt or "Could you please clarify your question?"
            self.short_term.add(session_id, "user", question)
            self.short_term.add(session_id, "assistant", clarification)
            self.long_term.save_turn(session_id, question, clarification)
            metrics = {
                "total_latency": round(time.time() - t0, 3),
                "retrieval_time": round(retrieval_latency, 3),
                "generation_time": 0.0,
                "faithfulness_score": self.evaluator.quick_check(clarification, []),
                "cache_hit": False,
                "adaptive_iterations": decision.iteration,
            }
            response_payload = {
                "answer": clarification,
                "sources": [],
                "expanded_queries": [],
                "contexts": [],
                "retrieval_decision": decision.action.value,
                "evidence_confidence": assessment.confidence,
                "metrics": metrics,
            }
            if use_cache:
                self.cache.set(
                    question, session_id, response_payload, user,
                    top_k=top_k, history=cache_history,
                    index_version=self.index_version,
                )
            return {
                **response_payload,
                "latency_ms": round((time.time() - t0) * 1000, 2),
                "cached": False,
            }

        # L6: Conversation history
        # L7: Prompt → generate
        generation_start = time.time()
        prompt = PromptTemplate.build(context, question, history)
        try:
            token_budget = PromptTemplate.answer_token_budget(question, context)
            answer = self.generator.generate(prompt, max_tokens=token_budget)
        except Exception as exc:
            answer = self._fallback_answer(results, exc)
        generation_latency = time.time() - generation_start

        # L8: Output guardrails (strip injection from LLM output)
        _, answer = self.guardrails.check_output(
            answer, [r[0].get("text", "") for r in results if isinstance(r[0], dict)]
        )

        # L9: PII audit log (no blocking; just audit)
        pii_found = self.guardrails.scan_for_pii(answer)
        if pii_found:
            logger.warning(
                "PII patterns found in answer for session=%s: %s", session_id, pii_found
            )

        # L10: Memory
        self.short_term.add(session_id, "user", question)
        self.short_term.add(session_id, "assistant", answer)
        self.long_term.save_turn(session_id, question, answer)

        # L11: Cache + auto-title
        metrics = {
            "total_latency": round(time.time() - t0, 3),
            "retrieval_time": round(retrieval_latency, 3),
            "generation_time": round(generation_latency, 3),
            "faithfulness_score": self.evaluator.quick_check(
                answer,
                [r[0]["text"] for r in results if isinstance(r[0], dict)],
            ),
            "adaptive_iterations": decision.iteration,
            "evidence_coverage": assessment.coverage_score,
            "context_chunks": len(results),
            "cache_hit": False,
        }

        response_payload = {
            "answer": answer,
            "sources": sources,
            "expanded_queries": expansion_queries,
            "contexts": [
                r[0]["text"] for r in results if isinstance(r[0], dict)
            ],
            "retrieval_decision": decision.action.value,
            "evidence_confidence": assessment.confidence,
            "metrics": metrics,
        }
        if use_cache:
            self.cache.set(
                question,
                session_id,
                response_payload,
                user,
                top_k=top_k,
                history=cache_history,
                index_version=self.index_version,
            )
        self._auto_title(session_id, question, answer)

        latency = round((time.time() - t0) * 1000, 2)
        logger.info(
            "Query done in %sms | session=%s | decision=%s | iter=%d",
            latency, session_id, decision.action.value, decision.iteration,
        )

        return {
            **response_payload,
            "latency_ms": latency,
            "cached": False,
        }

    async def query_stream(
        self,
        question: str,
        session_id: str = "default",
        top_k: int = 5,
        user: str = "",
    ) -> AsyncIterator[str]:
        """Streaming query — yields tokens. Non-streaming metadata is omitted."""
        safe, reason = self.guardrails.check_input(question)
        if not safe:
            yield reason
            return

        results, decision, assessment, _ = self.adaptive_controller.execute_retrieval(
            question,
            top_k=top_k,
            allowed_filenames=self.allowed_filenames(user),
        )

        if decision.action in (RetrievalAction.ABSTAIN, RetrievalAction.CLARIFY):
            msg = (
                decision.abstention_reason
                or decision.clarification_prompt
                or "Could not find sufficient information."
            )
            yield msg
            return

        results = self._sanitize_chunks_for_context(results)
        context, _ = self._build_context(results)
        history = self.short_term.format_for_prompt(session_id)
        prompt = PromptTemplate.build(context, question, history)

        full_tokens: List[str] = []
        try:
            async for token in self.generator.stream(prompt):
                full_tokens.append(token)
                yield token
        except Exception as exc:
            fallback = self._fallback_answer(results, exc)
            full_tokens.append(fallback)
            yield fallback

        answer = "".join(full_tokens)
        answer = self.guardrails.sanitize_output(answer)

        self.short_term.add(session_id, "user", question)
        self.short_term.add(session_id, "assistant", answer)
        self.long_term.save_message(session_id, "user", question)
        self.long_term.save_message(session_id, "assistant", answer)
        self._auto_title(session_id, question, answer)

    # ─── Document management ─────────────────────────────────────────────────

    def add_document(
        self, file_path: str, extracted_content: str, owner: str = ""
    ) -> Dict:
        chunker = RecursiveChunker(chunk_size=400, overlap=60)

        if not extracted_content.strip():
            raise HTTPException(
                status_code=422, detail="No readable content was extracted from the document."
            )
        chunks = chunker.chunk(extracted_content)

        filename = os.path.basename(file_path)
        timestamp = datetime.utcnow().isoformat()

        meta_chunks = [
            {
                "chunk_id": i,
                "text": chunk,
                "filename": filename,
                "uploaded_at": timestamp,
                "owner": owner,
            }
            for i, chunk in enumerate(chunks)
        ]

        embeddings = self.embedder.embed([c["text"] for c in meta_chunks])
        self.vector_store.add(embeddings, meta_chunks)
        self.vector_store.save(
            index_path=str(self.index_path),
            chunks_path=str(self.chunks_path),
        )
        self.bm25_retriever.index(self.vector_store.text_chunks)
        self.index_version += 1
        self.cache.clear()

        logger.info("Indexed: %s (%d chunks) owner=%s", filename, len(meta_chunks), owner)
        return {
            "status": "indexed",
            "filename": filename,
            "chunks_added": len(meta_chunks),
        }

    async def reindex_document(self, filename: str, user: str = "") -> Dict:
        """Re-extract and replace one document's chunks and embeddings."""
        filename = os.path.basename(filename.strip())
        matched = [
            chunk for chunk in self.vector_store.text_chunks
            if isinstance(chunk, dict)
            and chunk.get("filename", "").strip().lower() == filename.lower()
        ]
        if not matched:
            raise HTTPException(status_code=404, detail=f"Document '{filename}' not found")

        owners = {chunk.get("owner", "") for chunk in matched}
        if settings.DOC_ACCESS_MODE == "user" and user not in owners:
            raise HTTPException(
                status_code=403,
                detail="You do not have permission to reindex this document.",
            )

        file_path = self.raw_dir / filename
        if not file_path.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"The original file for '{filename}' is unavailable; upload it again.",
            )

        try:
            text = await extract_document_content(str(file_path))
        except DocumentParsingError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if not text or not text.strip():
            raise HTTPException(
                status_code=422,
                detail="No readable text was found in this document. If it is a scanned PDF, run OCR and upload the searchable copy.",
            )

        chunks = RecursiveChunker(chunk_size=400, overlap=60).chunk(text)
        owner = next(iter(owners), user)
        timestamp = datetime.utcnow().isoformat()
        replacements = [
            {
                "chunk_id": i,
                "text": chunk,
                "filename": filename,
                "uploaded_at": timestamp,
                "owner": owner,
            }
            for i, chunk in enumerate(chunks)
        ]
        retained = [
            chunk for chunk in self.vector_store.text_chunks
            if not (
                isinstance(chunk, dict)
                and chunk.get("filename", "").strip().lower() == filename.lower()
            )
        ]
        rebuilt_chunks = retained + replacements
        self.vector_store.rebuild_index(rebuilt_chunks)
        self.vector_store.save(
            index_path=str(self.index_path),
            chunks_path=str(self.chunks_path),
        )
        self.bm25_retriever.index(rebuilt_chunks)
        self.index_version += 1
        self.cache.clear()

        logger.info("Reindexed: %s (%d chunks) owner=%s", filename, len(replacements), owner)
        return {
            "status": "reindexed",
            "filename": filename,
            "chunks_added": len(replacements),
            "chunks_replaced": len(matched),
        }

    def list_documents(self, user: str = "") -> List[Dict]:
        docs: Dict[str, Dict] = {}
        for chunk in self.vector_store.text_chunks:
            if not isinstance(chunk, dict):
                continue
            fn = chunk["filename"]
            # In 'user' ACL mode, filter to documents owned by this user
            if settings.DOC_ACCESS_MODE == "user":
                if not user:
                    continue
                if chunk.get("owner", "") not in (user, ""):
                    continue
            if fn not in docs:
                docs[fn] = {
                    "filename": fn,
                    "uploaded_at": chunk.get("uploaded_at", ""),
                    "chunks": 0,
                }
            docs[fn]["chunks"] += 1
        return list(docs.values())

    def allowed_filenames(self, user: str = "") -> Optional[set]:
        """Return an access-scoped filename set, or None when access is open."""
        if settings.DOC_ACCESS_MODE != "user":
            return None
        if not user:
            return set()
        return {doc["filename"] for doc in self.list_documents(user=user)}

    def delete_document(self, filename: str, user: str = "") -> Dict:
        filename = filename.strip()
        matched = [
            c for c in self.vector_store.text_chunks
            if isinstance(c, dict)
            and c["filename"].strip().lower() == filename.lower()
        ]
        if not matched:
            raise HTTPException(
                status_code=404, detail=f"Document '{filename}' not found"
            )

        # ACL check: in 'user' mode only the owner can delete
        if settings.DOC_ACCESS_MODE == "user":
            owners = {c.get("owner", "") for c in matched}
            if user not in owners:
                raise HTTPException(
                    status_code=403,
                    detail="You do not have permission to delete this document.",
                )

        filtered = [
            c for c in self.vector_store.text_chunks
            if isinstance(c, dict)
            and c["filename"].strip().lower() != filename.lower()
        ]
        self.vector_store.rebuild_index(filtered)
        self.vector_store.save(
            index_path=str(self.index_path),
            chunks_path=str(self.chunks_path),
        )
        self.bm25_retriever.index(filtered)
        self.index_version += 1
        self.cache.clear()
        logger.info("Deleted: %s by user=%s", filename, user)
        return {"status": "deleted", "filename": filename}

    # ─── Session management ──────────────────────────────────────────────────

    def new_session(self, session_id: str) -> Dict:
        self.short_term.clear(session_id)
        self.long_term.upsert_session(session_id)
        return {"session_id": session_id, "status": "created"}

    def list_sessions(self, user: str = "") -> List[Dict]:
        sessions = self.long_term.list_sessions()
        if settings.DOC_ACCESS_MODE != "user":
            return sessions
        if not user:
            return []
        prefix = f"sess_{quote(user, safe='')}_"
        return [session for session in sessions if session["session_id"].startswith(prefix)]

    def get_session_messages(self, session_id: str) -> List[Dict]:
        return self.long_term.get_messages(session_id)

    def delete_session(self, session_id: str, user: str = "") -> Dict:
        self.long_term.delete_session(session_id)
        self.short_term.clear(session_id)
        self.cache.clear_session(session_id, user=user or None)
        return {"status": "deleted", "session_id": session_id}

    # ─── System metrics ──────────────────────────────────────────────────────

    def get_metrics(self, user: str = "") -> Dict:
        docs = self.list_documents(user=user)
        filenames = {doc["filename"] for doc in docs}
        chunks = self.vector_store.text_chunks
        if settings.DOC_ACCESS_MODE == "user":
            chunks = [
                chunk for chunk in chunks
                if isinstance(chunk, dict) and chunk.get("filename") in filenames
            ]
        return {
            "documents_indexed": len(docs),
            "total_chunks": len(chunks),
            "vector_dimension": self.vector_store.dimension,
            "bm25_chunks_indexed": len(chunks) if settings.DOC_ACCESS_MODE == "user" else len(self.bm25_retriever.chunks),
            "reranker_available": self.reranker.available,
            "active_sessions": len(self.list_sessions(user=user)),
            "query_cache_size": self.cache.size,
            "query_cache_hits": self.cache.hits,
            "query_cache_misses": self.cache.misses,
            "query_cache_enabled": True,
            "adaptive_retrieval_enabled": settings.ADAPTIVE_RETRIEVAL_ENABLED,
            "adaptive_policy": settings.ADAPTIVE_POLICY,
            "laya_enabled": settings.LAYA_ENABLED,
        }
