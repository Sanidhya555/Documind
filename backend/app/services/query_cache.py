"""Small process-local cache for complete RAG responses."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from copy import deepcopy
from collections import OrderedDict
from typing import Dict, Optional, Tuple


class QueryCache:
    """Bounded TTL/LRU cache scoped by user, session, history, and index version."""

    def __init__(self, max_size: int = 200, ttl: int = 600):
        self._store: OrderedDict[str, Tuple[Dict, float]] = OrderedDict()
        self._scopes: Dict[str, Tuple[str, str]] = {}
        self.max_size = max(1, int(max_size))
        self.ttl = max(1, int(ttl))
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(
        question: str,
        session_id: str,
        user: str,
        top_k: Optional[int],
        history: str,
        index_version: int,
    ) -> str:
        parts = [
            user,
            session_id,
            question.strip().lower(),
            str(top_k if top_k is not None else "dynamic"),
            history,
            str(index_version),
        ]
        encoded = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _purge_expired(self, now: float) -> None:
        expired = [key for key, (_, inserted_at) in self._store.items() if now - inserted_at >= self.ttl]
        for key in expired:
            self._store.pop(key, None)
            self._scopes.pop(key, None)

    def get(
        self,
        question: str,
        session_id: str = "",
        user: str = "",
        top_k: Optional[int] = None,
        history: str = "",
        index_version: int = 0,
    ) -> Optional[Dict]:
        key = self._key(question, session_id, user, top_k, history, index_version)
        now = time.time()
        with self._lock:
            entry = self._store.get(key)
            if entry and now - entry[1] < self.ttl:
                self._store.move_to_end(key)
                self.hits += 1
                return deepcopy(entry[0])
            self._store.pop(key, None)
            self._scopes.pop(key, None)
            self.misses += 1
            return None

    def set(
        self,
        question: str,
        session_id: str,
        payload: Dict,
        user: str = "",
        top_k: Optional[int] = None,
        history: str = "",
        index_version: int = 0,
    ) -> None:
        key = self._key(question, session_id, user, top_k, history, index_version)
        now = time.time()
        with self._lock:
            self._purge_expired(now)
            self._store[key] = (deepcopy(payload), now)
            self._scopes[key] = (session_id, user)
            self._store.move_to_end(key)
            while len(self._store) > self.max_size:
                evicted, _ = self._store.popitem(last=False)
                self._scopes.pop(evicted, None)

    def clear_session(self, session_id: str, user: Optional[str] = None) -> None:
        """Remove responses for one session without evicting other users' cache entries."""
        with self._lock:
            keys = [
                key for key, (cached_session, cached_user) in self._scopes.items()
                if cached_session == session_id and (user is None or cached_user == user)
            ]
            for key in keys:
                self._store.pop(key, None)
                self._scopes.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._scopes.clear()

    @property
    def size(self) -> int:
        with self._lock:
            self._purge_expired(time.time())
            return len(self._store)
