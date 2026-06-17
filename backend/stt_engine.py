"""Server-side STT engine with provider fallback.

Provider order in auto mode:
1) EgypTalk NeMo (local/HF)
2) Faster-Whisper (local)
3) Whisper API (remote)
4) Browser fallback signal
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
import uuid
import gc
import aiohttp
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

try:
    import torch
except Exception:
    torch = None

logger = logging.getLogger("servia-voice")

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_DIR = BACKEND_DIR.parent

WHISPER_API_URL = os.getenv("STT_WHISPER_API_URL", "https://api.openai.com/v1/audio/transcriptions").strip()
WHISPER_API_KEY = os.getenv("STT_WHISPER_API_KEY", os.getenv("OPENAI_API_KEY", "")).strip()
WHISPER_MODEL = os.getenv("STT_WHISPER_MODEL", "gpt-4o-mini-transcribe").strip() or "gpt-4o-mini-transcribe"

STT_PROVIDER = os.getenv("STT_PROVIDER", "auto").strip().lower() or "auto"
STT_EGYPTALK_MODEL_ID = os.getenv("STT_EGYPTALK_MODEL_ID", "NAMAA-Space/EgypTalk-ASR-v2").strip() or "NAMAA-Space/EgypTalk-ASR-v2"
STT_EGYPTALK_LOCAL_MODEL = os.getenv("STT_EGYPTALK_LOCAL_MODEL", "").strip()

STT_AUTO_USE_EGYPTALK = os.getenv("STT_AUTO_USE_EGYPTALK", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

default_egyptalk_dir = PROJECT_ROOT_DIR / "models" / "EgypTalk-ASR-v2"
if not default_egyptalk_dir.exists():
    default_egyptalk_dir = PROJECT_ROOT_DIR / "EgypTalk-ASR-v2"
if not STT_EGYPTALK_LOCAL_MODEL and default_egyptalk_dir.exists():
    STT_EGYPTALK_LOCAL_MODEL = str(default_egyptalk_dir)

STT_FASTER_WHISPER_MODEL = os.getenv("STT_FASTER_WHISPER_MODEL", "small").strip() or "small"
STT_FASTER_WHISPER_DEVICE = os.getenv("STT_FASTER_WHISPER_DEVICE", "auto").strip() or "auto"
STT_FASTER_WHISPER_COMPUTE_TYPE = os.getenv("STT_FASTER_WHISPER_COMPUTE_TYPE", "int8").strip() or "int8"
USE_GPU = os.getenv("USE_GPU", "1").strip().lower() in {"1", "true", "yes", "on"}
CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", "0").strip()
STT_DEVICE = os.getenv("STT_DEVICE", STT_FASTER_WHISPER_DEVICE).strip().lower() or "auto"
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda" if USE_GPU else "cpu").strip().lower() or "cpu"
EGYPTALK_DEVICE = os.getenv("EGYPTALK_DEVICE", "cpu").strip().lower() or "cpu"


def _torch_cuda_available() -> bool:
    return bool(torch and getattr(torch.cuda, "is_available", lambda: False)())


def _cleanup_inference_memory() -> None:
    gc.collect()
    if torch and _torch_cuda_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _get_vram_usage_percent() -> float:
    try:
        if not torch or not _torch_cuda_available():
            return 0.0
        props = torch.cuda.get_device_properties(0)
        total = float(getattr(props, "total_memory", 0) or 0)
        used = float(torch.cuda.memory_allocated(0) or 0)
        if total <= 0:
            return 0.0
        return (used / total) * 100.0
    except Exception:
        return 0.0


def _is_cuda_oom_error(error: BaseException) -> bool:
    message = str(error).lower()
    return "cuda out of memory" in message or ("cuda" in message and "oom" in message)


def _resolve_faster_whisper_device() -> str:
    if STT_DEVICE in {"cuda", "cpu"}:
        return "cuda" if STT_DEVICE == "cuda" and _torch_cuda_available() and USE_GPU else "cpu"
    if STT_FASTER_WHISPER_DEVICE in {"cuda", "cpu"}:
        return "cuda" if STT_FASTER_WHISPER_DEVICE == "cuda" and _torch_cuda_available() and USE_GPU else "cpu"
    return "cuda" if USE_GPU and _torch_cuda_available() else "cpu"


def _resolve_faster_whisper_compute_type(device: str) -> str:
    requested = STT_FASTER_WHISPER_COMPUTE_TYPE
    if requested and requested not in {"auto", "int8"}:
        return requested
    if device == "cuda":
        return "int8_float16"
    return "int8"

try:
    STT_MAX_LATENCY_MS = max(600, int(os.getenv("STT_MAX_LATENCY_MS", "15000")))
except ValueError:
    STT_MAX_LATENCY_MS = 15000

try:
    STT_MIN_CONFIDENCE = float(os.getenv("STT_MIN_CONFIDENCE", "0.30"))
except ValueError:
    STT_MIN_CONFIDENCE = 0.30

try:
    STT_WHISPER_TIMEOUT_SECONDS = float(os.getenv("STT_WHISPER_TIMEOUT_SECONDS", "12"))
except ValueError:
    STT_WHISPER_TIMEOUT_SECONDS = 12.0

try:
    STT_LOCAL_TIMEOUT_SECONDS = float(os.getenv("STT_LOCAL_TIMEOUT_SECONDS", "30"))
except ValueError:
    STT_LOCAL_TIMEOUT_SECONDS = 30.0

try:
    STT_EGYPTALK_TIMEOUT_SECONDS = float(os.getenv("STT_EGYPTALK_TIMEOUT_SECONDS", "10"))
except ValueError:
    STT_EGYPTALK_TIMEOUT_SECONDS = 10.0


@dataclass
class STTResult:
    text: str
    provider: str
    confidence: float
    latency_ms: int
    segments: list[dict[str, Any]]
    language: str
    fallback_used: bool


class STTEngine:
    def __init__(self):
        self._egyptalk_model = None
        self._egyptalk_error: Optional[str] = None
        self._faster_whisper_model = None
        self._faster_whisper_error: Optional[str] = None
        self._chunk_buffers: dict[str, bytearray] = {}

    def get_status(self) -> dict[str, Any]:
        return {
            "requested_provider": STT_PROVIDER,
            "auto_use_egyptalk": STT_AUTO_USE_EGYPTALK,
            "use_gpu": USE_GPU,
            "cuda_visible_devices": CUDA_VISIBLE_DEVICES,
            "stt_device": _resolve_faster_whisper_device(),
            "whisper_device": WHISPER_DEVICE,
            "egyptalk_device": "cuda" if EGYPTALK_DEVICE == "cuda" and _torch_cuda_available() else "cpu",
            "whisper_api_configured": bool(WHISPER_API_KEY),
            "whisper_model": WHISPER_MODEL,
            "egyptalk_model_id": STT_EGYPTALK_MODEL_ID,
            "egyptalk_local_model": STT_EGYPTALK_LOCAL_MODEL,
            "egyptalk_loaded": self._egyptalk_model is not None,
            "egyptalk_error": self._egyptalk_error,
            "faster_whisper_loaded": self._faster_whisper_model is not None,
            "faster_whisper_error": self._faster_whisper_error,
            "faster_whisper_model": STT_FASTER_WHISPER_MODEL,
            "faster_whisper_compute_type": _resolve_faster_whisper_compute_type(_resolve_faster_whisper_device()),
            "timeouts_seconds": {
                "local": STT_LOCAL_TIMEOUT_SECONDS,
                "egyptalk": STT_EGYPTALK_TIMEOUT_SECONDS,
                "whisper_api": STT_WHISPER_TIMEOUT_SECONDS,
            },
            "max_latency_ms": STT_MAX_LATENCY_MS,
            "min_confidence": STT_MIN_CONFIDENCE,
        }

    def _resolve_provider_order(self) -> list[str]:
        requested = STT_PROVIDER
        if requested == "egyptalk":
            return ["egyptalk", "faster_whisper", "whisper_api", "browser"]
        if requested == "whisper_api":
            return ["whisper_api", "faster_whisper", "browser"]
        if requested == "faster_whisper":
            return ["faster_whisper", "whisper_api", "browser"]
        if requested == "browser":
            return ["browser"]

        # Auto mode is latency-first for real-time voice UX.
        ordered: list[str] = ["faster_whisper"]

        if WHISPER_API_KEY:
            ordered.append("whisper_api")

        if STT_AUTO_USE_EGYPTALK:
            ordered.append("egyptalk")

        ordered.append("browser")

        deduped: list[str] = []
        for provider in ordered:
            if provider not in deduped:
                deduped.append(provider)
        return deduped

    async def transcribe(
        self,
        audio_bytes: bytes,
        mime_type: str = "audio/wav",
        language: str = "ar",
        session_id: Optional[str] = None,
    ) -> STTResult:
        providers = self._resolve_provider_order()
        errors: list[str] = []

        for index, provider in enumerate(providers):
            is_last = index == (len(providers) - 1)
            try:
                if provider == "whisper_api":
                    result = await self._transcribe_whisper_api(audio_bytes, mime_type, language)
                elif provider == "egyptalk":
                    result = await self._transcribe_egyptalk(audio_bytes, mime_type, language)
                elif provider == "faster_whisper":
                    result = await self._transcribe_faster_whisper(audio_bytes, mime_type, language)
                else:
                    raise RuntimeError("Browser STT fallback required")
            except Exception as exc:
                errors.append(f"{provider}: {exc}")
                continue

            acceptable = self._is_result_acceptable(result, is_last=is_last)
            if acceptable:
                result.fallback_used = provider != providers[0]
                return result

            errors.append(
                f"{provider}: below thresholds (confidence={result.confidence:.2f}, latency={result.latency_ms}ms)"
            )

        raise RuntimeError("STT providers failed. " + " | ".join(errors[:4]))

    def append_audio_chunk(self, session_id: str, chunk: bytes) -> int:
        if not session_id:
            raise ValueError("session_id is required")
        buf = self._chunk_buffers.setdefault(session_id, bytearray())
        buf.extend(chunk)
        return len(buf)

    def pop_chunked_audio(self, session_id: str) -> bytes:
        buf = self._chunk_buffers.pop(session_id, bytearray())
        return bytes(buf)

    def _is_result_acceptable(self, result: STTResult, is_last: bool) -> bool:
        if not result.text.strip():
            return False
        if is_last:
            return True
        if result.latency_ms > STT_MAX_LATENCY_MS:
            return False
        if result.confidence < STT_MIN_CONFIDENCE:
            return False
        return True

    async def _transcribe_whisper_api(self, audio_bytes: bytes, mime_type: str, language: str) -> STTResult:
        if not WHISPER_API_KEY:
            raise RuntimeError("Missing STT_WHISPER_API_KEY / OPENAI_API_KEY")

        started = time.perf_counter()
        response_data = await asyncio.wait_for(
            self._call_whisper_api_async(audio_bytes, mime_type, language),
            timeout=STT_WHISPER_TIMEOUT_SECONDS,
        )
        latency_ms = int((time.perf_counter() - started) * 1000)

        text = (response_data.get("text") or "").strip()
        segments = response_data.get("segments") or []
        language_out = response_data.get("language") or language or "ar"

        confidence = 0.7
        if segments:
            probs = []
            for seg in segments:
                no_speech_prob = seg.get("no_speech_prob")
                if isinstance(no_speech_prob, (int, float)):
                    probs.append(max(0.0, min(1.0, 1.0 - float(no_speech_prob))))
            if probs:
                confidence = sum(probs) / len(probs)

        return STTResult(
            text=text,
            provider="whisper_api",
            confidence=float(confidence),
            latency_ms=latency_ms,
            segments=[
                {
                    "start": float(seg.get("start", 0.0)),
                    "end": float(seg.get("end", 0.0)),
                    "text": (seg.get("text") or "").strip(),
                }
                for seg in segments
                if (seg.get("text") or "").strip()
            ],
            language=str(language_out),
            fallback_used=False,
        )

    def _call_whisper_api_sync(self, audio_bytes: bytes, mime_type: str, language: str) -> dict[str, Any]:
        boundary = f"----ServiaSTT{uuid.uuid4().hex}"
        filename = "speech.wav"
        if "webm" in (mime_type or "").lower():
            filename = "speech.webm"
        elif "mp3" in (mime_type or "").lower() or "mpeg" in (mime_type or "").lower():
            filename = "speech.mp3"

        lines: list[bytes] = []

        def add_field(name: str, value: str):
            lines.append(f"--{boundary}".encode("utf-8"))
            lines.append(f'Content-Disposition: form-data; name="{name}"'.encode("utf-8"))
            lines.append(b"")
            lines.append(value.encode("utf-8"))

        add_field("model", WHISPER_MODEL)
        add_field("language", language or "ar")

        lines.append(f"--{boundary}".encode("utf-8"))
        lines.append(
            f'Content-Disposition: form-data; name="file"; filename="{filename}"'.encode("utf-8")
        )
        lines.append(f"Content-Type: {mime_type or 'audio/wav'}".encode("utf-8"))
        lines.append(b"")
        lines.append(audio_bytes)
        lines.append(f"--{boundary}--".encode("utf-8"))
        body = b"\r\n".join(lines) + b"\r\n"

        # Synchronous whisper API calls are disabled in favor of async aiohttp implementation.
        raise RuntimeError("Synchronous whisper API call disabled; use async path")

    async def _call_whisper_api_async(self, audio_bytes: bytes, mime_type: str, language: str) -> dict[str, Any]:
        boundary = f"----ServiaSTT{uuid.uuid4().hex}"
        filename = "speech.wav"
        if "webm" in (mime_type or "").lower():
            filename = "speech.webm"
        elif "mp3" in (mime_type or "").lower() or "mpeg" in (mime_type or "").lower():
            filename = "speech.mp3"

        parts: list[tuple[str, tuple[str, bytes, str]]] = []
        # aiohttp accepts tuples (name, (filename, data, content_type)) in data for multipart
        parts.append(("model", (None, WHISPER_MODEL.encode("utf-8"), "text/plain")))
        parts.append(("language", (None, (language or "ar").encode("utf-8"), "text/plain")))
        parts.append(("file", (filename, audio_bytes, mime_type or "audio/wav")))

        timeout = aiohttp.ClientTimeout(total=max(2.0, min(30.0, STT_WHISPER_TIMEOUT_SECONDS)))
        async with aiohttp.ClientSession(timeout=timeout) as session:
            data = aiohttp.FormData()
            data.add_field("model", WHISPER_MODEL)
            data.add_field("language", language or "ar")
            data.add_field("file", audio_bytes, filename=filename, content_type=mime_type or "audio/wav")
            async with session.post(WHISPER_API_URL, data=data, headers={"Authorization": f"Bearer {WHISPER_API_KEY}"}) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    raise RuntimeError(f"Whisper API HTTP {resp.status}: {text[:300]}")
                return json.loads(text)

    async def _transcribe_faster_whisper(self, audio_bytes: bytes, mime_type: str, language: str) -> STTResult:
        started = time.perf_counter()
        result = await asyncio.wait_for(
            asyncio.to_thread(self._transcribe_faster_whisper_sync, audio_bytes, mime_type, language),
            timeout=STT_LOCAL_TIMEOUT_SECONDS,
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        result.latency_ms = latency_ms
        return result

    def _load_faster_whisper_model(self):
        if self._faster_whisper_model is not None:
            return self._faster_whisper_model

        try:
            from faster_whisper import WhisperModel
        except Exception as exc:
            self._faster_whisper_error = str(exc)
            raise RuntimeError(f"faster-whisper import failed: {exc}") from exc

        device = _resolve_faster_whisper_device()
        # VRAM guard: if GPU is nearly full, prefer CPU to avoid OOM
        try:
            vram_pct = _get_vram_usage_percent()
            if device == "cuda" and vram_pct > 90.0:
                logger.warning("VRAM usage %.1f%% >90%%, loading faster-whisper on CPU instead", vram_pct)
                device = "cpu"
        except Exception:
            pass

        compute_type = _resolve_faster_whisper_compute_type(device)
        logger.info("Faster-Whisper loading on %s with compute_type=%s", device, compute_type)

        try:
            self._faster_whisper_model = WhisperModel(
                STT_FASTER_WHISPER_MODEL,
                device=device,
                compute_type=compute_type,
            )
            self._faster_whisper_error = None
            logger.debug("Faster-Whisper successfully loaded with compute_type=%s", compute_type)
            return self._faster_whisper_model
        except Exception as load_error:
            error_msg = str(load_error).lower()
            # If int8_float16 is not supported, retry with float16, then int8
            if "int8_float16" in compute_type and ("int8_float16" in error_msg or "not support" in error_msg or "efficient" in error_msg):
                logger.warning("int8_float16 not supported on %s; retrying with float16", device)
                try:
                    self._faster_whisper_model = WhisperModel(
                        STT_FASTER_WHISPER_MODEL,
                        device=device,
                        compute_type="float16",
                    )
                    self._faster_whisper_error = None
                    logger.info("Faster-Whisper loaded with fallback compute_type=float16")
                    return self._faster_whisper_model
                except Exception as float16_error:
                    logger.warning("float16 also failed; retrying with int8")
                    try:
                        self._faster_whisper_model = WhisperModel(
                            STT_FASTER_WHISPER_MODEL,
                            device=device,
                            compute_type="int8",
                        )
                        self._faster_whisper_error = None
                        logger.info("Faster-Whisper loaded with fallback compute_type=int8")
                        return self._faster_whisper_model
                    except Exception as int8_error:
                        self._faster_whisper_error = f"All compute types failed: {int8_error}"
                        raise RuntimeError(self._faster_whisper_error) from int8_error
            else:
                self._faster_whisper_error = str(load_error)
                raise

    def _transcribe_faster_whisper_sync(self, audio_bytes: bytes, mime_type: str, language: str) -> STTResult:
        model = self._load_faster_whisper_model()

        suffix = ".wav"
        mime = (mime_type or "").lower()
        if "webm" in mime:
            suffix = ".webm"
        elif "mpeg" in mime or "mp3" in mime:
            suffix = ".mp3"

        temp_path = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(audio_bytes)
                temp_path = temp_file.name

            try:
                # vram check before transcribe
                try:
                    before_vram = _get_vram_usage_percent()
                except Exception:
                    before_vram = 0.0

                if getattr(model, "device", None) == "cuda" and before_vram > 90.0:
                    logger.warning("VRAM before transcribe %.1f%% >90%%; rebuilding model on CPU to avoid OOM", before_vram)
                    self._faster_whisper_model = None
                    from faster_whisper import WhisperModel as _WhisperModel2

                    model = _WhisperModel2(
                        STT_FASTER_WHISPER_MODEL,
                        device="cpu",
                        compute_type="int8",
                    )

                segments_iter, info = model.transcribe(
                    temp_path,
                    language=language or "ar",
                    beam_size=3,
                    vad_filter=True,
                )
            except Exception as exc:
                if _is_cuda_oom_error(exc) and _torch_cuda_available():
                    _cleanup_inference_memory()
                    logger.warning("Faster-Whisper CUDA OOM; retrying on CPU")
                    self._faster_whisper_model = None
                    device = "cpu"
                    self._faster_whisper_model = WhisperModel(
                        STT_FASTER_WHISPER_MODEL,
                        device=device,
                        compute_type="int8",
                    )
                    segments_iter, info = self._faster_whisper_model.transcribe(
                        temp_path,
                        language=language or "ar",
                        beam_size=3,
                        vad_filter=True,
                    )
                else:
                    raise

            segments: list[dict[str, Any]] = []
            transcript_parts: list[str] = []
            conf_scores: list[float] = []

            for seg in segments_iter:
                text = (seg.text or "").strip()
                if not text:
                    continue
                segments.append(
                    {
                        "start": float(getattr(seg, "start", 0.0) or 0.0),
                        "end": float(getattr(seg, "end", 0.0) or 0.0),
                        "text": text,
                    }
                )
                transcript_parts.append(text)
                avg_logprob = getattr(seg, "avg_logprob", None)
                if isinstance(avg_logprob, (int, float)):
                    conf_scores.append(max(0.0, min(1.0, pow(2.718281828, float(avg_logprob)))))

            confidence = (sum(conf_scores) / len(conf_scores)) if conf_scores else 0.65
            text = " ".join(transcript_parts).strip()
            language_out = str(getattr(info, "language", language or "ar"))

            return STTResult(
                text=text,
                provider="faster_whisper",
                confidence=float(confidence),
                latency_ms=0,
                segments=segments,
                language=language_out,
                fallback_used=False,
            )
        finally:
            if temp_path:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            _cleanup_inference_memory()

    async def _transcribe_egyptalk(self, audio_bytes: bytes, mime_type: str, language: str) -> STTResult:
        started = time.perf_counter()
        result = await asyncio.wait_for(
            asyncio.to_thread(self._transcribe_egyptalk_sync, audio_bytes, mime_type, language),
            timeout=STT_EGYPTALK_TIMEOUT_SECONDS,
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        result.latency_ms = latency_ms
        return result

    def _load_egyptalk_model(self):
        if self._egyptalk_model is not None:
            return self._egyptalk_model

        try:
            from nemo.collections.asr.models import ASRModel  # type: ignore[import-not-found]
        except Exception as exc:
            self._egyptalk_error = str(exc)
            raise RuntimeError(f"NeMo ASR import failed: {exc}") from exc

        local_model = STT_EGYPTALK_LOCAL_MODEL
        model = None

        if local_model:
            local_path = Path(local_model)
            if local_path.is_file() and local_path.suffix.lower() == ".nemo":
                model = ASRModel.restore_from(str(local_path))
            elif local_path.is_dir():
                nemo_candidates = sorted(local_path.rglob("*.nemo"))
                if nemo_candidates:
                    model = ASRModel.restore_from(str(nemo_candidates[0]))

        if model is None:
            model = ASRModel.from_pretrained(STT_EGYPTALK_MODEL_ID)

        if EGYPTALK_DEVICE == "cuda" and _torch_cuda_available():
            try:
                model = model.to("cuda")
            except Exception as exc:
                logger.warning("EgypTalk CUDA move failed, keeping model on CPU: %s", exc)

        self._egyptalk_model = model
        self._egyptalk_error = None
        return model

    def _transcribe_egyptalk_sync(self, audio_bytes: bytes, mime_type: str, language: str) -> STTResult:
        model = self._load_egyptalk_model()

        suffix = ".wav"
        mime = (mime_type or "").lower()
        if "webm" in mime:
            suffix = ".webm"
        elif "mpeg" in mime or "mp3" in mime:
            suffix = ".mp3"

        temp_path = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(audio_bytes)
                temp_path = temp_file.name

            output = model.transcribe([temp_path], batch_size=1)

            text = ""
            if isinstance(output, list) and output:
                first_item = output[0]
                if isinstance(first_item, str):
                    text = first_item.strip()
                elif hasattr(first_item, "text"):
                    text = str(getattr(first_item, "text") or "").strip()
                elif isinstance(first_item, dict):
                    text = str(first_item.get("text") or first_item.get("pred_text") or "").strip()
            elif isinstance(output, str):
                text = output.strip()

            return STTResult(
                text=text,
                provider="egyptalk",
                confidence=0.72 if text else 0.0,
                latency_ms=0,
                segments=[],
                language=language or "ar",
                fallback_used=False,
            )
        finally:
            if temp_path:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass


STT_ENGINE = STTEngine()
