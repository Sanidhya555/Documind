"""Short-term in-memory conversation buffer per session."""
from collections import deque
from typing import List, Dict
import threading


class ShortTermMemory:
    """Stores the last N conversation turns in memory per session_id."""

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self._store: Dict[str, deque] = {}
        self._lock = threading.Lock()

    def add(self, session_id: str, role: str, content: str):
        with self._lock:
            if session_id not in self._store:
                self._store[session_id] = deque(maxlen=self.max_turns * 2)
            self._store[session_id].append({"role": role, "content": content})

    def get(self, session_id: str) -> List[Dict]:
        with self._lock:
            return list(self._store.get(session_id, []))

    def clear(self, session_id: str):
        with self._lock:
            self._store.pop(session_id, None)

    def format_for_prompt(self, session_id: str) -> str:
        """Return formatted conversation history for inclusion in prompt."""
        history = self.get(session_id)
        if not history:
            return ""
        lines = []
        for msg in history:
            prefix = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{prefix}: {msg['content']}")
        return "\n".join(lines)
