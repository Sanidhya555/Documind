"""LLM generators — sync, async, and streaming variants for Ollama and OpenAI."""
import json
import logging
import os
from typing import AsyncIterator

import httpx
import requests

logger = logging.getLogger("rag_app")


# ─── Ollama ───────────────────────────────────────────────────────────────────

class OllamaGenerator:
    def __init__(self, model: str = "mistral"):
        self.model = model
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    def generate(self, prompt: str, max_tokens: int = 1200) -> str:
        """Synchronous generation."""
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": max_tokens},
                },
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["response"]
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama connection failed: {e}")

    async def generate_async(self, prompt: str, max_tokens: int = 1200) -> str:
        """Async non-streaming generation."""
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": max_tokens},
                },
            )
            resp.raise_for_status()
            return resp.json()["response"]

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        """Async token-by-token streaming generator."""
        async with httpx.AsyncClient(timeout=180) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": True},
            ) as response:
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        token = data.get("response", "")
                        if token:
                            yield token
                        if data.get("done"):
                            break
                    except json.JSONDecodeError:
                        pass


# ─── OpenAI ───────────────────────────────────────────────────────────────────

class OpenAIGenerator:
    _SYSTEM = "You are a helpful enterprise document assistant."

    def __init__(self):
        from openai import OpenAI, AsyncOpenAI
        self._sync = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._async = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def generate(self, prompt: str, max_tokens: int = 1200) -> str:
        resp = self._sync.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content

    async def generate_async(self, prompt: str, max_tokens: int = 1200) -> str:
        resp = await self._async.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        stream = await self._async.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
