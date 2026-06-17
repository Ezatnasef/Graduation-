"""Modular API-driven voice pipeline orchestration."""

from __future__ import annotations

import time
from typing import Any, Awaitable, Callable, Optional


class VoicePipeline:
    """Coordinates stages: Input -> STT -> Processing -> LLM -> TTS -> Output."""

    def __init__(
        self,
        stt_fn: Callable[..., Awaitable[dict[str, Any]]],
        processing_fn: Callable[[str], Awaitable[dict[str, Any]]],
        llm_fn: Callable[[str, dict[str, Any]], Awaitable[str]],
        tts_fn: Callable[..., Awaitable[tuple[bytes, str]]],
    ):
        self._stt_fn = stt_fn
        self._processing_fn = processing_fn
        self._llm_fn = llm_fn
        self._tts_fn = tts_fn

    async def run_audio_turn(
        self,
        audio_bytes: bytes,
        mime_type: str,
        tts_dialect: str,
        tts_gender: str,
        language: str = "ar",
        include_tts: bool = True,
    ) -> dict[str, Any]:
        started = time.perf_counter()

        stt_result = await self._stt_fn(audio_bytes=audio_bytes, mime_type=mime_type, language=language)
        text = (stt_result.get("text") or "").strip()
        if not text:
            raise RuntimeError("STT produced empty transcription")

        processing = await self._processing_fn(text)
        response_text = await self._llm_fn(text, processing)

        output: dict[str, Any] = {
            "text": text,
            "analysis": processing,
            "response_text": response_text,
            "stt": stt_result,
        }

        if include_tts:
            tts_audio, tts_format = await self._tts_fn(
                text=response_text,
                dialect=tts_dialect,
                gender=tts_gender,
                emotion=(processing.get("voice_emotion") or "neutral"),
            )
            output["tts"] = {
                "audio_bytes": tts_audio,
                "audio_format": tts_format,
            }

        output["latency_ms"] = int((time.perf_counter() - started) * 1000)
        return output

    async def run_text_turn(
        self,
        text: str,
        tts_dialect: str,
        tts_gender: str,
        include_tts: bool = False,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        cleaned = (text or "").strip()
        if not cleaned:
            raise RuntimeError("text is required")

        processing = await self._processing_fn(cleaned)
        response_text = await self._llm_fn(cleaned, processing)

        output: dict[str, Any] = {
            "text": cleaned,
            "analysis": processing,
            "response_text": response_text,
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }

        if include_tts:
            tts_audio, tts_format = await self._tts_fn(
                text=response_text,
                dialect=tts_dialect,
                gender=tts_gender,
                emotion=(processing.get("voice_emotion") or "neutral"),
            )
            output["tts"] = {
                "audio_bytes": tts_audio,
                "audio_format": tts_format,
            }

        return output
