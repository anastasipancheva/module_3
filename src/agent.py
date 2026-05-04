import os
import sys
import time
from typing import Iterator, Literal

from mem0 import Memory
from openai import OpenAI
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

os.makedirs("data", exist_ok=True)

logger.remove()
logger.add(
    "data/log_file.log",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} - {message}",
    level="INFO",
    rotation="10 MB",
    retention="30 days",
)
logger.add(sys.stderr, level="WARNING", format="{time:HH:mm:ss} | {level} | {message}")

PERSONAS: dict[str, str] = {
    "wellness-friend": (
        "You are a warm, empathetic wellness coach and best friend. "
        "You speak with care, encouragement, and positivity. "
        "You celebrate every small win, use the user's name when you know it, "
        "and always frame challenges as opportunities for growth. "
        "When you remember something about the user, mention it naturally — "
        "it shows you truly care. "
        "Always reply in the same language the user writes in."
    ),
    "tough-love": (
        "You are a no-nonsense, results-driven personal coach. "
        "You cut through excuses and speak bluntly — you respect the user "
        "enough to tell them hard truths. "
        "Acknowledge feelings briefly, then pivot immediately to concrete actions. "
        "When you recall a past commitment the user made, hold them accountable. "
        "No sugarcoating. Always reply in the same language the user writes in."
    ),
}

_MEM0_LLM_MODEL = os.getenv("MEM0_LLM_MODEL", "openai/gpt-4o-mini")
_CHAT_MODEL = os.getenv("CHAT_MODEL", "openai/gpt-4o-mini")
_OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def _build_mem0_config(api_key: str) -> dict:
    return {
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": "mindly_memories",
                "path": "./data/vector_db",
            },
        },
        "llm": {
            "provider": "openai",
            "config": {
                "model": _MEM0_LLM_MODEL,
                "api_key": api_key,
                "openai_base_url": _OPENROUTER_BASE,
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
        },
    }


class MindlyAgent:
    """Main agent class. All public methods are thread-safe w.r.t. different user_ids."""

    def __init__(self) -> None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment / .env")

        self.memory = Memory.from_config(_build_mem0_config(api_key))

        self.client = OpenAI(
            api_key=api_key,
            base_url=_OPENROUTER_BASE,
            default_headers={
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "Mindly",
            },
        )
        self.model_name = _CHAT_MODEL
        logger.info("MindlyAgent initialized | llm={} mem0_llm={}", _CHAT_MODEL, _MEM0_LLM_MODEL)

    def chat(
        self,
        user_id: str,
        persona: str,
        message: str,
        stream: bool = True,
    ) -> "Iterator[str] | str":
        logger.info("chat | user={} persona={} stream={} msg_len={}", user_id, persona, stream, len(message))

        t0 = time.perf_counter()
        memories = self._retrieve_memories(user_id, message)
        retrieval_ms = (time.perf_counter() - t0) * 1000
        logger.info("memory_retrieval | user={} count={} ms={:.0f}", user_id, len(memories), retrieval_ms)

        messages = self._build_messages(persona, memories, message)

        if stream:
            return self._stream_response(user_id, message, messages)
        return self._generate_response(user_id, message, messages)

    def forget(self, user_id: str, query: "str | Literal['all']") -> None:
        if query == "all":
            self.memory.delete_all(user_id=user_id)
            logger.info("forget_all | user={}", user_id)
            return

        results = self.memory.search(query, filters={"user_id": user_id}, limit=10)
        memories = self._unwrap_results(results)

        deleted = 0
        for m in memories:
            score = m.get("score", 1.0)
            mem_id = m.get("id")
            if mem_id and score >= 0.4:
                self.memory.delete(mem_id)
                deleted += 1
                logger.info("forget_targeted | user={} id={} score={:.2f} text={}", user_id, mem_id, score, m.get("memory", "")[:60])

        logger.info("forget_done | user={} query='{}' deleted={}", user_id, query, deleted)

    def get_all_memories(self, user_id: str) -> list[dict]:
        try:
            result = self.memory.get_all(filters={"user_id": user_id})
            return self._unwrap_results(result)
        except Exception as exc:
            logger.error("get_all_memories error | user={} err={}", user_id, exc)
            return []


    def _retrieve_memories(self, user_id: str, query: str) -> list[dict]:
        try:
            results = self.memory.search(query, filters={"user_id": user_id}, limit=10)
            return self._unwrap_results(results)
        except Exception as exc:
            logger.error("memory_search error | user={} err={}", user_id, exc)
            return []

    @staticmethod
    def _unwrap_results(raw) -> list[dict]:
        """Normalise mem0 response: dict with 'results' key or plain list."""
        if isinstance(raw, dict):
            return raw.get("results", [])
        if isinstance(raw, list):
            return raw
        return []

    @staticmethod
    def _build_messages(persona: str, memories: list[dict], user_message: str) -> list[dict]:
        persona_prompt = PERSONAS.get(persona, PERSONAS["wellness-friend"])
        system = persona_prompt

        if memories:
            facts = "\n".join(
                f"- {m['memory']}" for m in memories if m.get("memory")
            )
            system += f"\n\nKnown facts about this user (use naturally, don't recite robotically):\n{facts}"

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ]

    def _stream_response(self, user_id: str, message: str, messages: list[dict]) -> "Iterator[str]":
        full_response = ""
        ttft: float | None = None
        t0 = time.perf_counter()
        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=True,
                max_tokens=600,
            )
            for chunk in stream:
                token = (chunk.choices[0].delta.content or "") if chunk.choices else ""
                if token:
                    if ttft is None:
                        ttft = (time.perf_counter() - t0) * 1000
                    full_response += token
                    yield token

            total_ms = (time.perf_counter() - t0) * 1000
            logger.info(
                "stream_done | user={} ttft_ms={:.0f} total_ms={:.0f} chars={}",
                user_id, ttft or 0, total_ms, len(full_response),
            )
            self._save_interaction(user_id, message, full_response)
        except Exception as exc:
            logger.error("stream_error | user={} err={}", user_id, exc)
            yield f"[Error] {exc}"

    def _generate_response(self, user_id: str, message: str, messages: list[dict]) -> str:
        t0 = time.perf_counter()
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=False,
                max_tokens=600,
            )
            full_response = resp.choices[0].message.content or ""
            total_ms = (time.perf_counter() - t0) * 1000
            logger.info("generate_done | user={} ms={:.0f} chars={}", user_id, total_ms, len(full_response))
            self._save_interaction(user_id, message, full_response)
            return full_response
        except Exception as exc:
            logger.error("generate_error | user={} err={}", user_id, exc)
            return f"[Error] {exc}"

    def _save_interaction(self, user_id: str, message: str, response: str) -> None:
        try:
            interaction = [
                {"role": "user", "content": message},
                {"role": "assistant", "content": response},
            ]
            self.memory.add(message, user_id=user_id)
            logger.info("memory_saved | user={}", user_id)
        except Exception as exc:
            logger.error("memory_save_error | user={} err={}", user_id, exc)
