"""Conversation memory manager with periodic summarization."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


def _estimate_tokens(text: str) -> int:
    # Simple and fast approximation suitable for runtime thresholds.
    return max(1, len((text or "").strip()) // 4)


@dataclass
class MemoryMessage:
    role: str
    content: str
    created_at: float = field(default_factory=time.time)
    meta: dict[str, Any] = field(default_factory=dict)


class ConversationMemory:
    def __init__(
        self,
        max_recent_messages: int = 12,
        summarize_after_messages: int = 16,
        summarize_after_tokens: int = 900,
    ):
        self._raw_messages: deque[MemoryMessage] = deque()
        self._summary_chunks: list[str] = []
        self.max_recent_messages = max(6, max_recent_messages)
        self.summarize_after_messages = max(10, summarize_after_messages)
        self.summarize_after_tokens = max(350, summarize_after_tokens)

    def add_message(self, role: str, content: str, meta: dict[str, Any] | None = None) -> None:
        text = (content or "").strip()
        if not text:
            return
        self._raw_messages.append(MemoryMessage(role=role, content=text, meta=meta or {}))

    def clear(self) -> None:
        self._raw_messages.clear()
        self._summary_chunks.clear()

    def raw_message_count(self) -> int:
        return len(self._raw_messages)

    def estimated_tokens(self) -> int:
        return sum(_estimate_tokens(msg.content) for msg in self._raw_messages)

    async def summarize_if_needed(self, summarizer: Callable[[list[MemoryMessage]], Awaitable[str]]) -> bool:
        if not self._should_summarize():
            return False

        keep_recent = self.max_recent_messages
        if len(self._raw_messages) <= keep_recent + 2:
            return False

        to_summarize = list(self._raw_messages)[:-keep_recent]
        if not to_summarize:
            return False

        summary_text = (await summarizer(to_summarize)).strip()
        if summary_text:
            self._summary_chunks.append(summary_text)

        # Preserve only newest tail in raw memory.
        tail = list(self._raw_messages)[-keep_recent:]
        self._raw_messages = deque(tail)
        return True

    def _should_summarize(self) -> bool:
        if len(self._raw_messages) >= self.summarize_after_messages:
            return True
        if self.estimated_tokens() >= self.summarize_after_tokens:
            return True
        return False

    def build_messages_for_llm(self, system_prompt: str, max_recent: int = 8) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

        if self._summary_chunks:
            memory_block = "\n\n".join(self._summary_chunks[-3:])
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "ملخص الحوار السابق (للحفاظ على النية والنبرة والكيانات المهمة):\n"
                        f"{memory_block}"
                    ),
                }
            )

        recent = list(self._raw_messages)[-max(4, max_recent):]
        messages.extend({"role": msg.role, "content": msg.content} for msg in recent)
        return messages

    def export_state(self) -> dict[str, Any]:
        return {
            "raw_messages": [
                {"role": msg.role, "content": msg.content, "created_at": msg.created_at}
                for msg in self._raw_messages
            ],
            "summaries": list(self._summary_chunks),
            "estimated_tokens": self.estimated_tokens(),
        }
