"""
Servia Voice Backend - FastAPI Server
Egyptian Arabic TTS with dialect support + VAD system
"""

import asyncio
import json
import base64
import io
import logging
import os
import time
import uuid
import re
import threading
import wave
from collections import Counter, defaultdict, deque
from typing import Any, Optional
from contextlib import asynccontextmanager, suppress
import aiohttp

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.websockets import WebSocketState

from tts_engine import (
    synthesize_speech,
    synthesize_to_base64,
    get_tts_backend_status,
    warmup_tts_engine,
)
from tts_engine import synthesize_long_text
import stt_engine
from stt_engine import STT_ENGINE
import tts_engine
from conversation_memory import ConversationMemory, MemoryMessage
from pipeline import VoicePipeline
from vad_engine import VADEngine, BargeinDetector
from dialect_mapper import (
    get_available_dialects,
    get_greeting,
    transform_to_dialect,
    get_dialect_prosody,
    normalize_codeswitch_text,
    get_codeswitch_context_hints,
    calculate_formality_score,
    is_response_too_formal,
    strengthen_colloquial_enforcement,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("servia-voice")

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
FRONTEND_DIR = os.path.join(PROJECT_ROOT_DIR, "frontend")
if not os.path.isdir(FRONTEND_DIR):
    FRONTEND_DIR = PROJECT_ROOT_DIR

for _env_path in [os.path.join(PROJECT_ROOT_DIR, ".env"), os.path.join(BACKEND_DIR, ".env")]:
    try:
        if os.path.isfile(_env_path):
            with open(_env_path, "r", encoding="utf-8") as _env_file:
                for _line in _env_file:
                    _line = _line.strip()
                    if not _line or _line.startswith("#") or "=" not in _line:
                        continue
                    _key, _value = _line.split("=", 1)
                    _key = _key.strip()
                    _value = _value.strip().strip('"').strip("'")
                    if _key and _key not in os.environ:
                        os.environ[_key] = _value
    except Exception:
        pass

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL_DEFAULT = "qwen2.5:1.5b-instruct-q8_0"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", OLLAMA_MODEL_DEFAULT).strip() or OLLAMA_MODEL_DEFAULT
OLLAMA_DEVICE = os.getenv("OLLAMA_DEVICE", "cuda").strip().lower() or "cuda"
USE_GPU = os.getenv("USE_GPU", "1").strip().lower() in {"1", "true", "yes", "on"}
CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", "0").strip()
try:
    OLLAMA_TAGS_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TAGS_TIMEOUT_SECONDS", "4"))
except ValueError:
    OLLAMA_TAGS_TIMEOUT_SECONDS = 4.0
try:
    OLLAMA_GENERATE_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_GENERATE_TIMEOUT_SECONDS", "18"))
except ValueError:
    OLLAMA_GENERATE_TIMEOUT_SECONDS = 18.0

try:
    OLLAMA_MAX_TOKENS = int(os.getenv("OLLAMA_MAX_TOKENS", "140"))
except ValueError:
    OLLAMA_MAX_TOKENS = 140

try:
    OLLAMA_MAX_RETRIES = int(os.getenv("OLLAMA_MAX_RETRIES", "1"))
except ValueError:
    OLLAMA_MAX_RETRIES = 1

try:
    # Enforce conservative chunk size for RTX 3050-class devices
    TTS_SEGMENT_MAX_CHARS = min(120, max(60, int(os.getenv("TTS_SEGMENT_MAX_CHARS", "120"))))
except ValueError:
    TTS_SEGMENT_MAX_CHARS = 120

try:
    MEMORY_MAX_RECENT_MESSAGES = max(6, int(os.getenv("MEMORY_MAX_RECENT_MESSAGES", "12")))
except ValueError:
    MEMORY_MAX_RECENT_MESSAGES = 12

try:
    MEMORY_SUMMARIZE_AFTER_MESSAGES = max(10, int(os.getenv("MEMORY_SUMMARIZE_AFTER_MESSAGES", "16")))
except ValueError:
    MEMORY_SUMMARIZE_AFTER_MESSAGES = 16

try:
    MEMORY_SUMMARIZE_AFTER_TOKENS = max(350, int(os.getenv("MEMORY_SUMMARIZE_AFTER_TOKENS", "900")))
except ValueError:
    MEMORY_SUMMARIZE_AFTER_TOKENS = 900

OLLAMA_SYSTEM_PROMPT_LANG = os.getenv("OLLAMA_SYSTEM_PROMPT_LANG", "arabic").strip().lower() or "arabic"


def system_prompt_arabic() -> str:
    return """
أنت سيرفيا، مساعدة ذكية بتتكلم باللهجة المصرية العامية بشكل طبيعي وواضح.

قواعد الأسلوب:
1) الرد يكون مصري بسيط ومباشر، وممنوع الفصحى الثقيلة.
2) ما تقولش "سؤالك غلط" أو "أعد الصياغة" إلا لو الرسالة فعلا غير مفهومة تماما.
3) لو المستخدم طلب حل/خطة/خطوات: ادي خطوات عملية واضحة فورا.
4) خلي الرد مفيد وقابل للتنفيذ، مش كلام عام.
5) لما يبقى فيه أكتر من حل، ابدأ بالأسرع والأقل تكلفة.
6) لو محتاج توضيح، اسأل سؤال واحد فقط وبعده قدم حل مبدئي.

تنسيق الرد:
- استخدم جمل قصيرة.
- للطلبات العملية استخدم ترقيم 1) 2) 3).
- لو الموضوع كبير، قسمه: إجراء فوري + إجراء خلال أسبوع + إجراء طويل المدى.

جودة المحتوى:
- الهدف هو حل المشكلة فعلا، مش مجرد تعليق عليها.
- عند ذكر خطوات، اربط كل خطوة بهدف واضح أو نتيجة متوقعة.
- حافظ على نبرة محترمة وودية بدون مبالغة أو اعتذارات متكررة.
""".strip()


def system_prompt_english() -> str:
    return """
You are an expert customer-service assistant.

Primary goal:
- Answer customer questions accurately and comprehensively.
- Rely exclusively on information available in the provided documents.
- Never use outside knowledge, prior training data, or internet sources.

Core rules:
1) Information source: answer only from provided documents.
2) Information integrity: do not invent policies, procedures, or technical details.
3) Be concise, clear, and professional.
4) Synthesize across multiple documents when relevant and explicitly note discrepancies.
5) Keep style and terminology consistent.
6) Fallback response (exact text only):
"I'm sorry, I don’t have enough information to answer that question."
7) No hedging when documents provide a clear answer.

Reasoning protocol:
1) Analyze the question.
2) Search documents thoroughly.
3) Extract only relevant evidence.
4) Synthesize coherently across sources.
5) Validate factual consistency with available documents.
6) If evidence is incomplete/ambiguous, use the fallback response.

Output format:
- Clear, practical answers for non-technical customers.
- Numbered steps for procedures.
- Bullet lists for options.
- If fallback is used, it must be the only output text.

Safety and quality:
- No hallucination.
- Neutral and objective tone.
- Accuracy over creativity.
""".strip()


class AnalyticsStore:
    """In-memory analytics store for periodic customer-service dashboards."""

    def __init__(self):
        self._lock = threading.Lock()
        self._started_at = time.time()
        self._events: deque[dict[str, Any]] = deque(maxlen=4000)
        self._session_meta: dict[str, dict[str, Any]] = {}
        self._total_messages_all = 0

    def register_session(self, session_id: str, dialect: str) -> None:
        now = time.time()
        with self._lock:
            meta = self._session_meta.get(session_id)
            if meta is None:
                self._session_meta[session_id] = {
                    "started_at": now,
                    "last_seen": now,
                    "dialect": dialect,
                    "message_count": 0,
                }
            else:
                meta["last_seen"] = now
                meta["dialect"] = dialect

    def update_session_dialect(self, session_id: str, dialect: str) -> None:
        now = time.time()
        with self._lock:
            meta = self._session_meta.setdefault(
                session_id,
                {"started_at": now, "last_seen": now, "dialect": dialect, "message_count": 0},
            )
            meta["last_seen"] = now
            meta["dialect"] = dialect

    def record_message(
        self,
        session_id: str,
        dialect: str,
        analysis: dict[str, Any],
        response_latency_ms: Optional[int] = None,
    ) -> None:
        now = time.time()
        event = {
            "ts": now,
            "session_id": session_id,
            "dialect": dialect,
            "sentiment": analysis.get("sentiment_label", "neutral"),
            "intent": analysis.get("intent_label", "other"),
            "urgency": analysis.get("urgency", "low"),
            "needs_human_agent": bool(analysis.get("needs_human_agent", False)),
            "confidence": float(analysis.get("confidence") or 0.0),
            "response_latency_ms": int(response_latency_ms or 0),
        }

        with self._lock:
            self._events.append(event)
            self._total_messages_all += 1

            meta = self._session_meta.setdefault(
                session_id,
                {"started_at": now, "last_seen": now, "dialect": dialect, "message_count": 0},
            )
            meta["last_seen"] = now
            meta["dialect"] = dialect
            meta["message_count"] = int(meta.get("message_count", 0)) + 1

    def get_summary(self, hours: int = 24) -> dict[str, Any]:
        now = time.time()
        window_seconds = max(1, min(hours, 168)) * 3600
        window_start = now - window_seconds

        with self._lock:
            events = [ev for ev in self._events if ev.get("ts", 0) >= window_start]
            session_meta = dict(self._session_meta)
            total_messages_all = self._total_messages_all
            started_at = self._started_at

        sentiment_counts = Counter(ev.get("sentiment", "neutral") for ev in events)
        dialect_counts = Counter(ev.get("dialect", "cairene") for ev in events)
        intent_counts = Counter(ev.get("intent", "other") for ev in events)
        urgency_counts = Counter(ev.get("urgency", "low") for ev in events)

        escalations = [ev for ev in events if ev.get("needs_human_agent")]
        confidences = [float(ev.get("confidence") or 0.0) for ev in events if ev.get("confidence") is not None]
        latencies = [int(ev.get("response_latency_ms") or 0) for ev in events if int(ev.get("response_latency_ms") or 0) > 0]

        active_sessions = sum(
            1
            for meta in session_meta.values()
            if (now - float(meta.get("last_seen", 0))) <= 1800
        )

        hourly_counts: dict[str, int] = defaultdict(int)
        for ev in events:
            label = time.strftime("%H:00", time.localtime(float(ev.get("ts", now))))
            hourly_counts[label] += 1

        recent_alerts = []
        for ev in reversed(events):
            if ev.get("needs_human_agent") or ev.get("urgency") == "high":
                recent_alerts.append({
                    "time": time.strftime("%Y-%m-%d %H:%M", time.localtime(float(ev.get("ts", now)))),
                    "session_id": ev.get("session_id"),
                    "intent": ev.get("intent", "other"),
                    "sentiment": ev.get("sentiment", "neutral"),
                    "urgency": ev.get("urgency", "low"),
                    "needs_human_agent": bool(ev.get("needs_human_agent")),
                })
            if len(recent_alerts) >= 12:
                break

        total_window = len(events)

        def _to_series(counter: Counter) -> list[dict[str, Any]]:
            if total_window <= 0:
                return []
            return [
                {
                    "label": label,
                    "count": count,
                    "percent": round((count / total_window) * 100, 1),
                }
                for label, count in counter.most_common()
            ]

        return {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            "period_hours": max(1, min(hours, 168)),
            "service_uptime_minutes": round((now - started_at) / 60, 1),
            "kpis": {
                "messages_window": total_window,
                "messages_all": total_messages_all,
                "active_sessions": active_sessions,
                "total_sessions": len(session_meta),
                "escalations_window": len(escalations),
                "escalation_rate_window": round((len(escalations) / total_window) * 100, 1) if total_window else 0.0,
                "avg_confidence_window": round(sum(confidences) / len(confidences), 2) if confidences else 0.0,
                "avg_response_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
            },
            "distributions": {
                "sentiment": _to_series(sentiment_counts),
                "dialect": _to_series(dialect_counts),
                "intent": _to_series(intent_counts),
                "urgency": _to_series(urgency_counts),
            },
            "timeline": [
                {"hour": hour, "count": count}
                for hour, count in sorted(hourly_counts.items(), key=lambda item: item[0])
            ],
            "recent_alerts": recent_alerts,
        }


ANALYTICS = AnalyticsStore()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Servia Voice Backend starting...")

    warmup_task = asyncio.create_task(warmup_tts_engine())

    warmup_timeout = 180.0
    try:
        warmup_timeout = float(os.getenv("TTS_WARMUP_TIMEOUT_SECONDS", "180"))
    except ValueError:
        warmup_timeout = 180.0

    timeout = min(max(10.0, warmup_timeout), 300.0)

    try:
        await asyncio.wait_for(
            asyncio.shield(warmup_task),
            timeout=timeout
        )

    except TimeoutError:
        logger.warning(
            "TTS warmup not finished within %.1fs; continuing startup in background",
            timeout,
        )

    except Exception as warmup_error:
        logger.warning("TTS warmup failed during startup: %s", warmup_error)

    tts_status = get_tts_backend_status()
    stt_status = STT_ENGINE.get_status()
    logger.info(
        "Runtime devices => GPU=%s CUDA_VISIBLE_DEVICES=%s | STT=%s compute=%s | XTTS=%s half=%s | Ollama=%s",
        USE_GPU,
        CUDA_VISIBLE_DEVICES,
        stt_status.get("stt_device"),
        stt_status.get("faster_whisper_compute_type"),
        tts_status.get("xtts_device"),
        tts_status.get("xtts_half_precision"),
        OLLAMA_DEVICE,
    )

    yield

    # shutdown cleanup
    if not warmup_task.done():
        warmup_task.cancel()
        # Task cancellation raises asyncio.CancelledError (BaseException),
        # so suppress it explicitly during normal shutdown.
        with suppress(BaseException):
            await warmup_task

    logger.info("Servia Voice Backend shutting down...")


app = FastAPI(
    title="Servia Voice API",
    description="Egyptian Arabic TTS with dialect support & VAD",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# ===================== Models =====================


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    dialect: str = Field(default="cairene")
    gender: str = Field(default="female")
    emotion: str = Field(default="neutral")


class TTSResponse(BaseModel):
    audio_base64: str
    audio_format: str
    dialect: str
    emotion: str
    text_original: str
    text_dialectal: str


class STTResponse(BaseModel):
    text: str
    provider: str
    confidence: float
    latency_ms: int
    language: str
    segments: list[dict[str, Any]]
    fallback_used: bool


class ChatRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    session_id: str = Field(default="api-default")
    dialect: str = Field(default="cairene")
    gender: str = Field(default="female")
    include_tts: bool = Field(default=False)


class ChatResponse(BaseModel):
    session_id: str
    text: str
    normalized_text: str
    analysis: dict[str, Any]
    response_text: str
    latency_ms: int
    memory: dict[str, Any]
    tts: Optional[dict[str, Any]] = None


API_SESSIONS: dict[str, "VoiceSession"] = {}


# ===================== REST Endpoints =====================


@app.get("/")
async def root():
    return {"status": "ok", "service": "Servia Voice API", "version": "1.0.0"}


@app.get("/api/dialects")
async def list_dialects():
    """Get available Egyptian Arabic dialects."""
    return {"dialects": get_available_dialects()}


@app.get("/api/tts/status")
async def get_tts_status():
    """Get active TTS backend and readiness info (XTTS/Chatterbox/Edge/gTTS)."""
    return get_tts_backend_status()


@app.get("/api/analytics/summary")
async def get_analytics_summary(hours: int = 24):
    """Get periodic customer-service analytics for dashboard page."""
    return ANALYTICS.get_summary(hours=hours)


@app.get("/api/stt/status")
async def get_stt_status():
    """Get active STT configuration and readiness info."""
    return STT_ENGINE.get_status()


@app.post("/stt", response_model=STTResponse)
@app.post("/api/stt", response_model=STTResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = Form(default="ar"),
):
    """Transcribe uploaded audio with provider fallback (EgypTalk -> Faster-Whisper -> Whisper API)."""
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Empty audio payload")

    mime_type = file.content_type or "audio/wav"

    try:
        stt_result = await STT_ENGINE.transcribe(payload, mime_type=mime_type, language=language)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    return STTResponse(
        text=stt_result.text,
        provider=stt_result.provider,
        confidence=stt_result.confidence,
        latency_ms=stt_result.latency_ms,
        language=stt_result.language,
        segments=stt_result.segments,
        fallback_used=stt_result.fallback_used,
    )


@app.post("/api/stt/chunk")
async def transcribe_audio_chunk(
    session_id: str = Form(...),
    is_final: bool = Form(default=False),
    file: UploadFile = File(...),
    language: str = Form(default="ar"),
):
    """Accept chunked audio uploads and transcribe when the final chunk arrives."""
    chunk = await file.read()
    if not chunk:
        raise HTTPException(status_code=400, detail="Empty audio chunk")

    total_size = STT_ENGINE.append_audio_chunk(session_id=session_id, chunk=chunk)
    if not is_final:
        return {
            "session_id": session_id,
            "status": "buffering",
            "buffered_bytes": total_size,
        }

    audio_bytes = STT_ENGINE.pop_chunked_audio(session_id=session_id)
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="No buffered audio found for session")

    mime_type = file.content_type or "audio/wav"
    try:
        stt_result = await STT_ENGINE.transcribe(audio_bytes, mime_type=mime_type, language=language)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    return {
        "session_id": session_id,
        "status": "completed",
        "result": STTResponse(
            text=stt_result.text,
            provider=stt_result.provider,
            confidence=stt_result.confidence,
            latency_ms=stt_result.latency_ms,
            language=stt_result.language,
            segments=stt_result.segments,
            fallback_used=stt_result.fallback_used,
        ).model_dump(),
    }


@app.post("/api/tts", response_model=TTSResponse)
async def text_to_speech(request: TTSRequest):
    """Synthesize text to speech with Egyptian dialect."""
    valid_dialects = ["cairene", "saidi", "alexandrian", "bedouin"]
    if request.dialect not in valid_dialects:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid dialect. Choose from: {valid_dialects}",
        )

    valid_genders = ["male", "female"]
    if request.gender not in valid_genders:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid gender. Choose from: {valid_genders}",
        )

    dialectal_text = transform_to_dialect(request.text, request.dialect)

    audio_bytes, audio_format = await synthesize_long_text(
        request.text,
        request.dialect,
        request.gender,
        request.emotion,
        max_chars=TTS_SEGMENT_MAX_CHARS,
    )
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    return TTSResponse(
        audio_base64=audio_b64,
        audio_format=audio_format,
        dialect=request.dialect,
        emotion=request.emotion,
        text_original=request.text,
        text_dialectal=dialectal_text,
    )


@app.post("/api/tts/audio")
async def text_to_speech_audio(request: TTSRequest):
    """Synthesize text to speech and return raw audio."""
    valid_dialects = ["cairene", "saidi", "alexandrian", "bedouin"]
    if request.dialect not in valid_dialects:
        raise HTTPException(status_code=400, detail="Invalid dialect")

    audio_bytes, audio_format = await synthesize_long_text(
        request.text,
        request.dialect,
        request.gender,
        request.emotion,
        max_chars=TTS_SEGMENT_MAX_CHARS,
    )

    media_type = "audio/wav" if audio_format == "wav" else "audio/mpeg"
    extension = "wav" if audio_format == "wav" else "mp3"

    return Response(
        content=audio_bytes,
        media_type=media_type,
        headers={"Content-Disposition": f"inline; filename=speech.{extension}"},
    )


@app.get("/api/greeting/{dialect}")
async def get_dialect_greeting(dialect: str):
    """Get a greeting in the specified dialect."""
    valid_dialects = ["cairene", "saidi", "alexandrian", "bedouin"]
    if dialect not in valid_dialects:
        raise HTTPException(status_code=400, detail="Invalid dialect")

    greeting_text = get_greeting(dialect, "hello")
    audio_b64, audio_format = await synthesize_to_base64(greeting_text, dialect)

    return {
        "text": greeting_text,
        "audio_base64": audio_b64,
        "audio_format": audio_format,
        "dialect": dialect,
    }


# ===================== WebSocket Voice Chat =====================


class VoiceSession:
    """Manages a single voice chat session with VAD and TTS."""

    def __init__(self, websocket: Optional[WebSocket]):
        self.ws = websocket
        self.session_id = uuid.uuid4().hex[:10]
        self.vad = VADEngine(
            sample_rate=16000,
            silence_duration_ms=1200,
            speech_min_duration_ms=250,
        )
        self.bargein = BargeinDetector(sample_rate=16000, sensitivity=0.35)
        self.dialect = "cairene"
        self.gender = "female"
        self.is_tts_playing = False
        self.tts_task: Optional[asyncio.Task] = None
        self.audio_buffer = bytearray()
        self.ollama_model: Optional[str] = OLLAMA_MODEL or None
        self.ollama_disabled_until: float = 0.0
        self.ollama_warning_logged = False
        self.ollama_failure_count: int = 0
        self.ollama_best_model_checked = False
        self.ollama_warmed_up = False
        self.chat_history = []
        self.memory = ConversationMemory(
            max_recent_messages=MEMORY_MAX_RECENT_MESSAGES,
            summarize_after_messages=MEMORY_SUMMARIZE_AFTER_MESSAGES,
            summarize_after_tokens=MEMORY_SUMMARIZE_AFTER_TOKENS,
        )
        self.response_cache = {}
        self.response_counter = 0
        self._cached_models: list[str] = []
        self._models_last_fetch: float = 0.0

    def _normalize_match_text(self, text: str) -> str:
        """Normalize Arabic text for safer keyword matching."""
        return re.sub(r"[^\w\u0600-\u06FF]+", " ", text).strip().lower()

    def _contains_phrase(self, text: str, phrase: str) -> bool:
        """Match full words/phrases and avoid accidental substring matches."""
        normalized_text = self._normalize_match_text(text)
        normalized_phrase = self._normalize_match_text(phrase)
        if not normalized_text or not normalized_phrase:
            return False

        if " " in normalized_phrase:
            return normalized_phrase in normalized_text

        return normalized_phrase in normalized_text.split()

    def _contains_any_pattern(self, text: str, patterns: list[str]) -> bool:
        """Regex intent matcher robust to punctuation and Arabic variants."""
        normalized = self._normalize_match_text(text)
        if not normalized:
            return False

        for pattern in patterns:
            if re.search(pattern, normalized, flags=re.UNICODE):
                return True
        return False

    async def handle(self):
        """Main WebSocket handler loop."""
        if self.ws is None:
            logger.error("Voice session started without websocket instance")
            return
        try:
            await self.ws.accept()
            logger.info("Voice session connected")
            ANALYTICS.register_session(self.session_id, self.dialect)

            # Send welcome
            await self.send_json({
                "type": "connected",
                "session_id": self.session_id,
                "dialects": get_available_dialects(),
                "current_dialect": self.dialect,
                "current_gender": self.gender,
            })

            while True:
                message = await self.ws.receive()

                if message.get("type") == "websocket.disconnect":
                    break

                if "text" in message:
                    await self._handle_text_message(message["text"])
                elif "bytes" in message:
                    await self._handle_audio_message(message["bytes"])

        except WebSocketDisconnect:
            logger.info("Voice session disconnected")
        except Exception as e:
            logger.error(f"Voice session error: {e}")

    async def _handle_text_message(self, raw: str):
        """Handle text-based control messages."""
        try:
            data = __import__("json").loads(raw)
        except ValueError:
            return

        msg_type = data.get("type", "")

        if msg_type == "set_dialect":
            dialect = data.get("dialect", "cairene")
            if dialect in ["cairene", "saidi", "alexandrian", "bedouin"]:
                self.dialect = dialect
                ANALYTICS.update_session_dialect(self.session_id, self.dialect)
                greeting = get_greeting(dialect, "hello")
                await self.send_json({
                    "type": "dialect_changed",
                    "dialect": dialect,
                    "greeting": greeting,
                })
                # Send greeting audio
                await self._send_tts(greeting, emotion="excited")

        elif msg_type == "set_gender":
            gender = data.get("gender", "female")
            if gender in ["male", "female"]:
                self.gender = gender
                await self.send_json({
                    "type": "gender_changed",
                    "gender": gender,
                })

        elif msg_type == "text":
            text = data.get("content", "").strip()
            if text:
                await self._handle_user_text_turn(text, source="typed")

        elif msg_type == "interrupt":
            # User requested interruption
            await self._stop_tts()
            await self.send_json({"type": "interrupted"})

        elif msg_type == "vad_speech_end":
            # Frontend detected speech end, process accumulated audio
            audio_b64 = data.get("audio", "")
            if audio_b64:
                audio_bytes = base64.b64decode(audio_b64)
                await self._process_speech(audio_bytes)

    async def _handle_user_text_turn(self, text: str, source: str = "text"):
        """Run full text turn pipeline: processing -> llm -> tts."""
        normalized_text = normalize_codeswitch_text(text)
        if normalized_text != text:
            logger.debug(
                "Normalized user input: '%s' -> '%s'",
                text,
                normalized_text,
            )

        user_analysis = self._analyze_user_message(normalized_text)
        user_emotion = user_analysis.get("sentiment_label", "neutral")
        voice_emotion = self._map_tts_emotion(user_emotion)
        self.response_counter += 1
        response_id = f"resp-{self.response_counter}"
        response_start = time.perf_counter()

        await self.send_json({
            "type": "user_text",
            "session_id": self.session_id,
            "source": source,
            "text": text,
            "normalized_text": normalized_text,
            "analysis": user_analysis,
        })

        self._append_chat_message("user", normalized_text, meta={"source": source, "analysis": user_analysis})
        await self._maybe_summarize_memory()

        response = await self._generate_response(normalized_text, user_emotion=user_emotion)
        response_latency_ms = int((time.perf_counter() - response_start) * 1000)
        ANALYTICS.record_message(
            session_id=self.session_id,
            dialect=self.dialect,
            analysis=user_analysis,
            response_latency_ms=response_latency_ms,
        )
        self._append_chat_message("assistant", response, meta={"response_id": response_id})

        await self.send_json({
            "type": "bot_text",
            "text": response,
            "dialect": self.dialect,
            "emotion": voice_emotion,
            "response_id": response_id,
        })
        await self._send_tts(response, emotion=voice_emotion, response_id=response_id)

    async def _handle_audio_message(self, audio_data: bytes):
        """Handle binary audio data for VAD processing."""
        # Check for barge-in during TTS playback
        if self.is_tts_playing:
            if self.bargein.check(audio_data):
                logger.info("Barge-in detected!")
                await self._stop_tts()
                await self.send_json({"type": "bargein_detected"})
                self.bargein.reset()
                return

        # Regular VAD processing
        vad_result = self.vad.process_audio_chunk(audio_data)

        # Send VAD state updates to frontend
        await self.send_json({
            "type": "vad_state",
            "is_speech": vad_result["is_speech"],
            "energy": round(vad_result["energy"], 4),
            "is_speaking": vad_result["is_speaking"],
        })

        if vad_result["speech_started"]:
            await self.send_json({"type": "speech_started"})
            self.audio_buffer = bytearray()

        if vad_result["is_speaking"]:
            self.audio_buffer.extend(audio_data)

        if vad_result["speech_ended"]:
            await self.send_json({"type": "speech_ended"})
            if len(self.audio_buffer) > 0:
                await self._process_speech(bytes(self.audio_buffer))
                self.audio_buffer = bytearray()

    async def _process_speech(self, audio_bytes: bytes):
        """Process completed speech audio - send to STT then respond."""
        try:
            stt_payload = self._ensure_wav_audio(audio_bytes)
            stt_result = await STT_ENGINE.transcribe(
                stt_payload,
                mime_type="audio/wav",
                language="ar",
                session_id=self.session_id,
            )
            text = (stt_result.text or "").strip()
            if text:
                await self.send_json({
                    "type": "stt_result",
                    "text": text,
                    "provider": stt_result.provider,
                    "confidence": round(float(stt_result.confidence), 3),
                    "latency_ms": stt_result.latency_ms,
                    "fallback_used": stt_result.fallback_used,
                })
                await self._handle_user_text_turn(text, source="backend_stt")
                return
        except Exception as e:
            logger.warning("Backend STT failed, falling back to browser STT: %s", str(e))

        # Browser STT fallback keeps compatibility with existing frontend path.
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        await self.send_json({
            "type": "process_speech",
            "audio_base64": audio_b64,
        })

    def _ensure_wav_audio(self, audio_bytes: bytes, sample_rate: int = 16000) -> bytes:
        """Normalize incoming mic payloads to valid WAV before STT."""
        payload = audio_bytes or b""
        if len(payload) >= 12 and payload[:4] == b"RIFF" and payload[8:12] == b"WAVE":
            return payload

        if len(payload) % 2 != 0:
            payload = payload[:-1]
        if not payload:
            return payload

        out = io.BytesIO()
        with wave.open(out, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(payload)
        return out.getvalue()

    async def _send_tts(
        self,
        text: str,
        emotion: str = "neutral",
        response_id: Optional[str] = None,
    ):
        """Send TTS audio in streaming chunks for low latency."""
        # Cancel any existing TTS
        await self._stop_tts()

        self.is_tts_playing = True
        self.bargein.reset()

        try:
            segments = self._split_tts_segments(text)
            total_segments = len(segments)

            if total_segments == 0:
                await self.send_json({
                    "type": "tts_complete",
                    "response_id": response_id,
                })
                return

            for index, segment in enumerate(segments):
                if not self.is_tts_playing:
                    logger.info("TTS interrupted during synthesis")
                    break

                self.tts_task = asyncio.create_task(
                    synthesize_speech(
                        segment,
                        self.dialect,
                        self.gender,
                        emotion,
                    )
                )
                audio_bytes, audio_format = await self.tts_task

                if not self.is_tts_playing:
                    logger.info("TTS interrupted before playback")
                    break

                audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                await self.send_json({
                    "type": "tts_audio",
                    "audio_base64": audio_b64,
                    "format": audio_format,
                    "dialect": self.dialect,
                    "emotion": emotion,
                    "segment_index": index,
                    "segment_count": total_segments,
                    "response_id": response_id,
                })

            await self.send_json({
                "type": "tts_complete",
                "response_id": response_id,
            })

        except Exception as e:
            logger.error(f"TTS error: {e}")
            await self.send_json({
                "type": "error",
                "message": f"TTS synthesis failed: {str(e)}",
            })
        finally:
            self.is_tts_playing = False
            self.tts_task = None

    def _split_tts_segments(self, text: str, max_chars: int = TTS_SEGMENT_MAX_CHARS) -> list[str]:
        """Split long text conservatively to keep voice prosody natural."""
        cleaned = re.sub(r"\s+", " ", (text or "")).strip()
        if not cleaned:
            return []

        # Keep most replies as one segment to avoid robotic, letter-like playback.
        if len(cleaned) <= 220:
            return [cleaned]

        sentences = [
            part.strip()
            for part in re.split(r"(?<=[.!؟?])\s+|\n+", cleaned)
            if part.strip()
        ]
        if not sentences:
            sentences = [cleaned]

        segments: list[str] = []
        current = ""

        for sentence in sentences:
            candidate = f"{current} {sentence}".strip() if current else sentence

            if len(candidate) <= max_chars:
                current = candidate
                continue

            if current:
                segments.append(current)
                current = ""

            if len(sentence) <= max_chars:
                current = sentence
                continue

            # For very long single sentences, chunk by phrases first (comma/Arabic comma).
            phrase_parts = [p.strip() for p in re.split(r"(?<=[,،;؛])\s+", sentence) if p.strip()]
            if len(phrase_parts) > 1:
                for phrase in phrase_parts:
                    phrase_candidate = f"{current} {phrase}".strip() if current else phrase
                    if len(phrase_candidate) <= max_chars:
                        current = phrase_candidate
                    else:
                        if current:
                            segments.append(current)
                        current = phrase
                continue

            words = sentence.split()
            for word in words:
                word_candidate = f"{current} {word}".strip() if current else word
                if len(word_candidate) <= max_chars:
                    current = word_candidate
                else:
                    if current:
                        segments.append(current)
                    current = word

        if current:
            segments.append(current)

        return segments or [cleaned]

    async def _stop_tts(self):
        """Stop current TTS playback."""
        self.is_tts_playing = False
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()
            self.tts_task = None

    async def _generate_response(self, text: str, user_emotion: str = "neutral") -> str:
        """Generate response with deterministic intents + guarded Ollama generation."""
        normalized_text = normalize_codeswitch_text(text)
        cache_key = f"{self.dialect}|{self._normalize_match_text(normalized_text)}"
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]

        intent_reply = self._generate_intent_response(normalized_text)
        if intent_reply:
            final_intent = self._postprocess_response(normalized_text, intent_reply, from_model=False)
            self._cache_response(cache_key, final_intent)
            return final_intent

        ollama_reply = await self._generate_ollama_response(normalized_text, user_emotion=user_emotion)

        attempts = 0
        if ollama_reply:
            final_model = self._postprocess_response(normalized_text, ollama_reply, from_model=True)

            while attempts < OLLAMA_MAX_RETRIES and self._is_unacceptable_model_reply(normalized_text, final_model):
                attempts += 1
                strict_hint = (
                    "اعد صياغة الرد باللهجة المصرية العامية فقط. "
                    "ممنوع جمل غريبة او فصحى او كلمات غير مستخدمة في مصر. "
                    "اجب مباشرة في 1-2 جملة."
                )
                regenerated = await self._generate_ollama_response(
                    normalized_text,
                    user_emotion=user_emotion,
                    strict_hint=strict_hint,
                )
                if not regenerated:
                    break
                final_model = self._postprocess_response(normalized_text, regenerated, from_model=True)

            if not self._is_unacceptable_model_reply(normalized_text, final_model):
                self._cache_response(cache_key, final_model)
                return final_model

        final_fallback = self._postprocess_response(
            normalized_text,
            self._generate_fallback_response(normalized_text),
            from_model=False,
        )
        self._cache_response(cache_key, final_fallback)
        return final_fallback

    def _cache_response(self, cache_key: str, value: str):
        """Keep small in-memory cache to cut repeated prompt latency."""
        if not cache_key or not value:
            return

        self.response_cache[cache_key] = value
        if len(self.response_cache) > 80:
            # Drop oldest inserted key (dict preserves insertion order in py3.7+).
            oldest = next(iter(self.response_cache.keys()))
            self.response_cache.pop(oldest, None)

    def _append_chat_message(self, role: str, content: str, meta: Optional[dict[str, Any]] = None):
        """Store compact chat history for better Ollama coherence."""
        text = (content or "").strip()
        if not text:
            return

        self.chat_history.append({"role": role, "content": text})
        if len(self.chat_history) > 12:
            self.chat_history = self.chat_history[-12:]

        self.memory.add_message(role=role, content=text, meta=meta or {})

    async def _maybe_summarize_memory(self):
        """Compact old conversation chunks to keep context quality with low latency."""
        try:
            await self.memory.summarize_if_needed(self._summarize_memory_messages)
        except Exception as e:
            logger.warning("Memory summarization skipped: %s", str(e))

    async def _summarize_memory_messages(self, messages: list[MemoryMessage]) -> str:
        """Use Ollama to summarize old turns while preserving intent/tone/entities."""
        model = await self._resolve_ollama_model()
        if not model:
            return ""

        transcript_lines = []
        for msg in messages:
            role = "العميل" if msg.role == "user" else "المساعد"
            transcript_lines.append(f"{role}: {msg.content}")

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "لخص الحوار التالي في 6-8 نقاط قصيرة. "
                        "احفظ نية المستخدم، الحالة العاطفية، الكيانات المهمة (اسم منتج/خدمة/رقم طلب)، "
                        "واي التزامات قالها المساعد. الرد بالعربي المصري الواضح."
                    ),
                },
                {
                    "role": "user",
                    "content": "\n".join(transcript_lines),
                },
            ],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.8,
                "num_predict": 220,
            },
        }

        summary = await self._call_ollama_chat(payload)
        return self._sanitize_model_output(summary)

    def _postprocess_response(self, user_text: str, response: str, from_model: bool) -> str:
        """Clean and colloquialize response while preserving meaning."""
        cleaned = re.sub(r"\s+", " ", (response or "")).strip()
        if not cleaned:
            return self._generate_fallback_response(user_text)

        # Layer 1: Replace formal fillers that frequently appear in small models
        replacements = {
            "حسنًا": "تمام",
            "حسناً": "تمام",
            "كيف يمكنني مساعدتك": "أقدر أساعدك إزاي",
            "يمكنني مساعدتك": "أقدر أساعدك",
            "كيف يمكنني": "أقدر إزاي",
            "يمكنك": "تقدر",
            "أستطيع": "اقدر",
            "استطيع": "اقدر",
            "يستطيع": "يقدر",
            "يمكن": "ممكن",
        }
        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)

        # Keep this phase non-destructive and deterministic.
        cleaned = strengthen_colloquial_enforcement(cleaned, self.dialect)

        cleaned = transform_to_dialect(cleaned, self.dialect)

        cleaned = self._align_response_with_voice_gender(cleaned)

        cleaned = self._decorate_dialect_marker(cleaned)

        if from_model and self._is_low_quality_response(user_text, cleaned):
            return self._generate_fallback_response(user_text)

        return cleaned

    def _align_response_with_voice_gender(self, text: str) -> str:
        """Keep response wording aligned with selected voice persona."""
        cleaned = (text or "").strip()
        if not cleaned:
            return cleaned

        if self.gender == "male":
            replacements = {
                "وانتي": "وانت",
                "إنتي": "إنت",
                "انتي": "انت",
                "معاكي": "معاك",
                "ليكي": "ليك",
                "اكتبي": "اكتب",
                "ابعتي": "ابعت",
                "اعملي": "اعمل",
                "عايزة": "عايز",
                "محتاجة": "محتاج",
            }
            for old, new in replacements.items():
                cleaned = cleaned.replace(old, new)

        return cleaned

    def _decorate_dialect_marker(self, text: str) -> str:
        """Light marker to make dialect differences clearly audible."""
        marker_map = {
            "saidi": "يا أخي",
            "alexandrian": "يا باشا",
            "bedouin": "يا خوي",
        }
        marker = marker_map.get(self.dialect)
        if not marker:
            return text

        if len(text) > 90:
            return text

        if any(m in text for m in ["يا أخي", "يا باشا", "يا خوي", "يا كبير", "يا معلم"]):
            return text

        if text.endswith("؟"):
            return f"{text[:-1]} {marker}؟"
        if text.endswith("!"):
            return f"{text[:-1]} {marker}!"
        if text.endswith("."):
            return f"{text[:-1]} {marker}."
        return f"{text} {marker}"

    def _is_low_quality_response(self, user_text: str, response: str) -> bool:
        """Detect obvious incoherent replies and fallback gracefully."""
        norm_user = self._normalize_match_text(user_text)
        norm_resp = self._normalize_match_text(response)

        if not norm_resp:
            return True

        words = norm_resp.split()
        if len(words) < 3:
            return True

        # Common hallucination pattern: starts with unrelated greeting.
        if (
            norm_resp.startswith("عامل اي")
            and "عامل اي" not in norm_user
            and "اخبارك" not in norm_user
            and "ازيك" not in norm_user
        ):
            return True

        # Very repetitive output usually means poor generation.
        if len(words) >= 10 and (len(set(words)) / len(words)) < 0.5:
            return True

        # Broken numbered skeletons (e.g. "1 2 3 4 5") indicate truncated/garbled output.
        if re.search(r"\b1\s*2\s*3\s*4\s*5\b", norm_resp, flags=re.UNICODE):
            return True

        # Nonsense phrases seen in bad generations.
        bad_chunks = [
            "عوامل يا",
            "عوامل",
            "مش .",
            "ما حابب",
            "تشتري شي حاجة",
            "عايزه دلوقتي مش",
            "im_start",
            "im_end",
            "assistant",
            "user",
            "system",
        ]
        if any(chunk in norm_resp for chunk in bad_chunks):
            return True

        # Response should match key intent topics.
        if self._contains_any_pattern(norm_user, [r"\b(اسمك|مين انتي|مين انت|انتي مين|انت مين)\b"]):
            if not any(token in norm_resp for token in ["سيرفيا", "انا", "مساعد"]):
                return True

        if self._contains_any_pattern(norm_user, [r"\b(خدمات|بتعمل|بتعملي|تقدري|تقدر|وظيفتك)\b"]):
            if not any(token in norm_resp for token in ["اساعد", "اشرح", "خطة", "كود", "حل"]):
                return True

        if self._contains_any_pattern(norm_user, [r"\b(كام سنة|سنك|عمرك|عندك كام)\b"]):
            if not any(token in norm_resp for token in ["افتراضي", "ماليش", "ما عنديش", "مش انسان"]):
                return True

        return False

    def _is_unacceptable_model_reply(self, user_text: str, response: str) -> bool:
        """Reject model replies that are either incoherent or overly formal."""
        if self._is_low_quality_response(user_text, response):
            return True

        if len((response or "").split()) <= 6 and is_response_too_formal(response, max_formality=35.0):
            return True
        return False

    def _analyze_user_message(self, text: str) -> dict[str, Any]:
        """Classify user sentiment + intent for customer-service style handling."""
        normalized = self._normalize_match_text(text)
        if not normalized:
            return {
                "sentiment_label": "neutral",
                "intent_label": "other",
                "urgency": "low",
                "confidence": 0.35,
                "needs_human_agent": False,
                "tags": [],
            }

        intent_patterns = {
            "complaint": [
                r"\b(شكو|مشكله|مشكله|تالف|غلط|ماوصلش|اتاخر|تأخير|سيئ|وحش|مش راضي|مش راضيه|زفت|مش عاجبني)\b",
                r"\b(اتخصم|خصم مرتين|refund|return|replacement|ticket|escalate)\b",
            ],
            "inquiry": [
                r"\b(اي|ايه|ليه|ازاي|امتى|فين|ممكن اعرف|عايز اعرف|عايزة اعرف|استفسار)\b",
                r"\b(what|why|how|when|where|can you|could you)\b",
            ],
            "request": [
                r"\b(عايز|عايزة|محتاج|محتاجه|لو سمحت|ممكن|ساعدني|ساعديني|اطلب|طلب)\b",
                r"\b(please|need|i want|help me)\b",
            ],
            "cancellation_or_refund": [
                r"\b(الغي|الغاء|استرجاع|استرداد|ارجاع|تبديل|استبدال|refund|return|cancel)\b",
            ],
            "technical_issue": [
                r"\b(error|bug|exception|not working|مش شغال|مش شغاله|مش بيفتح|مش بترفع|فشل)\b",
                r"\b(login|otp|password|account|blocked|verification)\b",
            ],
            "praise": [
                r"\b(شكرا|متشكر|تسلم|جميل|ممتاز|عاش|كويس جدا|الله ينور)\b",
                r"\b(thanks|great|awesome|perfect|good job)\b",
            ],
            "greeting": [
                r"\b(اهلا|مرحبا|السلام عليكم|صباح الخير|مساء الخير|هاي)\b",
                r"\b(hello|hi|hey)\b",
            ],
            "feedback": [
                r"\b(اقتراح|راي|رأي|ملاحظ|feedback|suggestion|review)\b",
            ],
        }

        sentiment_patterns = {
            "angry": [
                r"\b(عصبي|غضبان|مستفز|مقهور|منفعل|متعصب|متنرفز|متغاظ|غضبان اوي|غضبان جدا)\b",
                r"\b(angry|furious|mad|annoyed)\b",
            ],
            "sad": [
                r"\b(حزين|محبط|مكتئب|مكسور|زعلان|مضايق|متضايق|متدايق|مخنوق|مهموم)\b",
                r"\b(sad|upset|depressed)\b",
            ],
            "frustrated": [
                r"\b(مش نافع|زهقت|بقاله كتير|بقالي كتير|كل شويه|مفيش فايده|اتكرر)\b",
                r"\b(frustrated|fed up|tired of)\b",
            ],
            "concerned": [
                r"\b(قلقان|خايف|مستعجل|ضروري|حالا|حالاً|عاجل|مهم جدا)\b",
                r"\b(worried|urgent|asap|immediately)\b",
            ],
            "happy": [
                r"\b(مبسوط|فرحان|راضي|مرتاح|سعيد|كويس جدا|تمام|زي الفل)\b",
                r"\b(happy|satisfied|great|good)\b",
            ],
            "excited": [
                r"\b(متحمس|مستني|مبسوط جدا|فرحان جدا|متحفز)\b",
                r"\b(excited|thrilled|cant wait)\b",
            ],
        }

        # Direct phrase shortcuts for frequent short utterances.
        direct_sentiment_overrides = {
            "sad": [
                r"\b(انا\s+)?(مضايق|متضايق|متدايق|زعلان)\b",
            ],
            "angry": [
                r"\b(انا\s+)?(غضبان|متعصب|متنرفز|متغاظ)\b",
            ],
            "happy": [
                r"\b(انا\s+)?(مبسوط|فرحان|سعيد)\b",
            ],
            "excited": [
                r"\b(انا\s+)?(متحمس|متحفز)\b",
            ],
        }

        intent_scores: dict[str, int] = {}
        for label, patterns in intent_patterns.items():
            score = sum(1 for pattern in patterns if re.search(pattern, normalized, flags=re.UNICODE))
            intent_scores[label] = score

        sentiment_scores: dict[str, int] = {}
        for label, patterns in sentiment_patterns.items():
            score = sum(1 for pattern in patterns if re.search(pattern, normalized, flags=re.UNICODE))
            sentiment_scores[label] = score

        for label, patterns in direct_sentiment_overrides.items():
            if any(re.search(pattern, normalized, flags=re.UNICODE) for pattern in patterns):
                sentiment_scores[label] = sentiment_scores.get(label, 0) + 2

        if "?" in text or "؟" in text:
            intent_scores["inquiry"] = intent_scores.get("inquiry", 0) + 1

        if text.count("!") >= 2:
            sentiment_scores["frustrated"] = sentiment_scores.get("frustrated", 0) + 1

        intent_label, intent_score = max(intent_scores.items(), key=lambda kv: kv[1])
        sentiment_label, sentiment_score = max(sentiment_scores.items(), key=lambda kv: kv[1])

        if intent_score == 0:
            intent_label = "other"

        if sentiment_score == 0:
            sentiment_label = "neutral"

        urgency = "low"
        if self._contains_any_pattern(normalized, [r"\b(عاجل|ضروري|حالا|حالاً|asap|urgent|دلوقتي)\b"]):
            urgency = "high"
        elif intent_label in {"complaint", "technical_issue", "cancellation_or_refund"}:
            urgency = "medium"

        needs_human_agent = (
            intent_label in {"complaint", "cancellation_or_refund"}
            and sentiment_label in {"angry", "frustrated", "sad"}
        )

        total_signal = max(1, intent_score + sentiment_score)
        confidence = round(min(0.95, 0.35 + (total_signal * 0.15)), 2)

        latin_tokens = re.findall(r"[A-Za-z][A-Za-z0-9_\-\./']*", text or "")
        arabic_tokens = re.findall(r"[\u0600-\u06FF]+", text or "")
        code_switch_ratio = round(
            len(latin_tokens) / max(1, (len(latin_tokens) + len(arabic_tokens))),
            3,
        )

        tags: list[str] = []
        if intent_label != "other":
            tags.append(intent_label)
        if sentiment_label != "neutral":
            tags.append(sentiment_label)
        if urgency != "low":
            tags.append(f"urgency_{urgency}")

        return {
            "sentiment_label": sentiment_label,
            "intent_label": intent_label,
            "urgency": urgency,
            "confidence": confidence,
            "needs_human_agent": needs_human_agent,
            "code_switch_ratio": code_switch_ratio,
            "tags": tags,
        }

    def _detect_user_emotion(self, text: str) -> str:
        """Infer user tone from Arabic keywords and punctuation."""
        analysis = self._analyze_user_message(text)
        sentiment_label = analysis.get("sentiment_label", "neutral")

        sentiment_to_emotion = {
            "angry": "angry",
            "sad": "sad",
            "frustrated": "concerned",
            "concerned": "concerned",
            "happy": "happy",
            "excited": "excited",
            "neutral": "neutral",
        }
        return sentiment_to_emotion.get(sentiment_label, "neutral")

    def _map_tts_emotion(self, user_emotion: str) -> str:
        """Map user tone to a natural bot speaking style."""
        mapping = {
            "angry": "empathetic",
            "sad": "empathetic",
            "frustrated": "empathetic",
            "concerned": "empathetic",
            "excited": "excited",
            "happy": "neutral",
            "neutral": "neutral",
        }
        return mapping.get(user_emotion, "neutral")

    def _generate_intent_response(self, text: str) -> Optional[str]:
        """Deterministic replies for frequent intents with robust Arabic matching."""
        t = (text or "").strip()
        n = self._normalize_match_text(t)
        if not n:
            return None

        if self._contains_any_pattern(n, [
            r"\b(السلام عليكم|اهلا|أهلا|مرحبا|هاي|صباح الخير|مساء الخير)\b",
        ]):
            return get_greeting(self.dialect, "hello")

        if self._contains_any_pattern(n, [
            r"\b(عامل|عاملة)\s*(اي|ايه|ايه|إيه)?\b",
            r"\b(اخبارك|ازيك|ازيك|ازيك يا)\b",
        ]):
            replies = {
                "cairene": "تمام الحمد لله، وانتي اخبارك ايه؟",
                "saidi": "الحمد لله زين، وانتي عاملة ايه يا اخوي؟",
                "alexandrian": "تمام يا باشا، وانتي عاملة ايه؟",
                "bedouin": "تمام والحمد لله يا خوي، وانتي اخبارك ايه؟",
            }
            return replies.get(self.dialect, replies["cairene"])

        if self._contains_any_pattern(n, [
            r"\b(اسمك|اسم حضرتك|اسمك اي|اسمك ايه|بتتسمي|بتتسمى)\b",
            r"\b(انتي مين|انت مين|مين انتي|مين انت)\b",
            r"\b(من انت|من انتي)\b",
        ]):
            replies = {
                "cairene": "انا سيرفيا، مساعدة ذكية. اقدر اساعدك في الشرح، حل المشاكل، وتنظيم خطة تعلم واضحة.",
                "saidi": "انا سيرفيا يا اخوي، مساعدة ذكية. اقدر اساعدك في الشرح والحلول خطوة خطوة.",
                "alexandrian": "انا سيرفيا يا باشا، مساعدة ذكية. اقدر ارتبلك الفكرة واطلعلك رد واضح.",
                "bedouin": "انا سيرفيا يا خوي، مساعدة ذكية. اقدر اخدمك في الشرح والترتيب وحل المشاكل.",
            }
            return replies.get(self.dialect, replies["cairene"])

        if self._contains_any_pattern(n, [
            r"\b(خدمات|اي الخدمات|الخدمات اللي)\b",
            r"\b(بتعمل اي|بتعملي اي|بتعمليها|بتعملها|تقدر تعمل اي|تقدري تعملي اي|وظيفتك)\b",
            r"\b(ممكن تساعدني في اي|تساعدني في ايه)\b",
        ]):
            replies = {
                "cairene": "اقدر اساعدك في 4 حاجات: شرح مبسط، حل مشاكل الكود، كتابة محتوى، وخطة تعلم مناسبة لمستواكي.",
                "saidi": "اقدر اساعدك في الشرح، حل اخطاء الكود، وتنظيم خطة مذاكرة تمشي معاكي واحدة واحدة.",
                "alexandrian": "خدماتي يا باشا: شرح واضح، مساعدتك في الكود، وترتيب خطة تعلم عملية من غير لف ودوران.",
                "bedouin": "اقدر اساعدك يا خوي في الشرح، حل اخطاء البرمجة، وبناء خطة تعلم ثابتة وواضحة.",
            }
            return replies.get(self.dialect, replies["cairene"])

        if self._contains_any_pattern(n, [
            r"\b(كام سنة|سنك كام|عمرك كام|عندك كام سنة|انتي عندك كام سنة|انت عندك كام سنة)\b",
        ]):
            replies = {
                "cairene": "انا مساعدة افتراضية، فمالييش سن زي البشر. بس اقدر اساعدك باحدث معلومات متاحة.",
                "saidi": "انا نظام افتراضي يا اخوي، يعني ماعنديش عمر بشري. لكن اقدر اساعدك بسرعة.",
                "alexandrian": "انا ذكاء اصطناعي يا باشا، فمش عندي سن. المهم اقدر اساعدك في طلبك فورا.",
                "bedouin": "انا مساعد افتراضي يا خوي، ما عندي عمر زي البشر. لكن بخدمك مباشرة.",
            }
            return replies.get(self.dialect, replies["cairene"])

        if self._contains_any_pattern(n, [
            r"\b(مش فاهم|مش فاهمة|مافهمتش|مش واضح|مش فاهم حاجة)\b",
        ]):
            replies = {
                "cairene": "ولا يهمك. ابعتي الجزء اللي مش واضح وانا هشرحه ببساطة وبمثال سريع.",
                "saidi": "ولا يهمك يا اخوي. ابعتي الحتة اللي مش واضحة وانا اوضحها خطوة خطوة.",
                "alexandrian": "ولا يهمك يا باشا. ابعتي السطر اللي ملخبطك وانا اشرحهولك ببساطة.",
                "bedouin": "لا تشيلي هم يا خوي. ابعتي الجزء اللي مو واضح وانا اوضحه خطوة خطوة.",
            }
            return replies.get(self.dialect, replies["cairene"])

        if "شكر" in n:
            return get_greeting(self.dialect, "thanks")

        if self._contains_any_pattern(n, [r"\b(مع السلامة|باي|سلام)\b"]):
            return get_greeting(self.dialect, "goodbye")

        if self._contains_any_pattern(n, [
            r"\b(مساعدة|ساعدني|ساعديني|عايز مساعدة|عايزة مساعدة)\b",
        ]):
            replies = {
                "cairene": "طبعًا، انا معاكي. قولي طلبك بالتحديد وهنمشي فيه خطوة خطوة.",
                "saidi": "اكيد يا اخوي، قولي اللي محتاجاه وانا اساعدك فيه مباشرة.",
                "alexandrian": "حاضر يا باشا، قولي المطلوب وانا اظبطهولك سريع.",
                "bedouin": "ابشري يا خوي، قولي وش تبين وانا اخدمك مباشرة.",
            }
            return replies.get(self.dialect, replies["cairene"])

        if self._contains_any_pattern(n, [r"\b(بايثون|python|برمجة|كود|تعلم)\b"]) and self._contains_any_pattern(
            n,
            [r"\b(لسه|مبتدئ|مبتدئة|ابدا|ابدأ|مش فاهم|مش فاهمة)\b"],
        ):
            replies = {
                "cairene": "ابدئي بـ 3 خطوات: الأساسيات (متغيرات/شرط/حلقات)، حل 5 تمارين يوميًا، وبعدين مشروع صغير. لو تحبي ابعتلك خطة 14 يوم.",
                "saidi": "ابدئيها خطوة خطوة: أساسيات، تمارين يومية، وبعدها مشروع بسيط. لو حابة ابعتلك خطة 14 يوم.",
                "alexandrian": "نمشيها بسيط: أساسيات بايثون الأول، شوية تمارين كل يوم، ثم مشروع صغير. لو حابة أجهزلك خطة 14 يوم.",
                "bedouin": "ابدي بالأساسيات اول، ثم تمارين يومية، وبعدها مشروع بسيط. إذا تبين، أرسل لك خطة 14 يوم.",
            }
            return replies.get(self.dialect, replies["cairene"])

        if self._contains_any_pattern(
            n,
            [r"(زحمة|مرور|ازدحام|تكدس|اختناق مروري|مواصلات)"],
        ) and self._contains_any_pattern(
            n,
            [r"(خنقة|مخنوق|مضايق|توتر|قلق|بعصب|متوتر|نفسيتي|مخنوقة)"],
        ):
            replies = {
                "cairene": (
                    "فاهمك، لو الخنقة بتيجي لك في الزحمة امشي بالخطة دي: 1) نفس 4-4-6 لمدة دقيقة، "
                    "2) افتح الهوا أو التكييف وخف كافيين قبل السواقة، 3) شغّل حاجة هادية بدل التركيز مع الزحمة، "
                    "4) اتحرك بدري 20-30 دقيقة واختار طريق بديل من الخريطة، "
                    "5) لو الخنقة بتتكرر جامد راجع مختص عشان تمارين تنظيم القلق."
                ),
                "saidi": (
                    "حاسس بيك يا اخوي. وقت الزحمة: 1) نفس هادي دقيقة، 2) هَوّي العربية، 3) شغل صوت هادي، "
                    "4) اطلع بدري وخد طريق أخف، 5) لو الخنقة مستمرة راجع مختص."
                ),
                "alexandrian": (
                    "فاهمك يا باشا. جرب: نفس هادي، تهوية كويسة، صوت هادي، خروج بدري، وطريق بديل. "
                    "ولو الموضوع بيتكرر كتير راجع مختص."
                ),
                "bedouin": (
                    "واضح عليك الضغط يا خوي. خذ نفس هادي، هوا المكان، امشِ على طريق أخف، واطلع بدري. "
                    "وإذا الخنقة تتكرر دايم، راجع مختص."
                ),
            }
            return replies.get(self.dialect, replies["cairene"])

        if self._contains_any_pattern(
            n,
            [r"(ازدحام|الازدحام|مرور|مروري|زحمة|تكدس|اختناق مروري|ازمة المواصلات|أزمة المواصلات|مواصلات)"],
        ):
            replies = {
                "cairene": (
                    "تمام، دي خطة كاملة تقلل الزحمة بسرعة: 1) خلال 48 ساعة امنع الوقوف العشوائي على المحاور الرئيسية "
                    "بسحب فوري وغرامة وقت الذروة، 2) خلال أسبوع ظبط توقيت الإشارات في التقاطعات المزدحمة حسب الكثافة الحية، "
                    "3) خلال أسبوعين خصص حارة باص/ميكروباص في المحاور الرئيسية مع رقابة كاميرات، "
                    "4) فعّل ساعات دخول مرنة للموظفين والجامعات لتقليل ذروة الصبح، "
                    "5) شغّل وحدة تدخل سريع للحوادث هدفها فتح الطريق خلال 10 دقايق."
                ),
                "saidi": (
                    "تمام يا اخوي، دي خطة واضحة: 1) منع الوقوف العشوائي فورًا، 2) ضبط الإشارات حسب كثافة المرور، "
                    "3) حارات نقل جماعي للمحاور الزحمة، 4) ساعات دخول مرنة للموظفين والطلاب، "
                    "5) تدخل سريع للحوادث وفتح الطريق بسرعة."
                ),
                "alexandrian": (
                    "تمام يا باشا، الحل العملي: منع الوقوف العشوائي فورًا، إعادة ضبط الإشارات، "
                    "تخصيص حارة نقل جماعي، توزيع ساعات الدخول، وفريق تدخل سريع للحوادث."
                ),
                "bedouin": (
                    "تمام يا خوي، الخطة العملية: ضبط الوقوف العشوائي، إشارات مرورية متغيرة، "
                    "حارات نقل جماعي، تنظيم ساعات الدوام، وتدخل سريع للحوادث."
                ),
            }
            return replies.get(self.dialect, replies["cairene"])

        if self._contains_any_pattern(n, [r"(الجو|طقس|حر|برد|مطر|رياح|شبورة|مشوار|سفر)"]) and self._contains_any_pattern(
            n,
            [r"(نصيحة|نصايح|اعمل|اعمل اي|البس|اخد|دلوقتي|النهارده|خروج|رايح)"],
        ):
            replies = {
                "cairene": (
                    "لو نازل دلوقتي: 1) شيّك على تطبيق الطقس قبل ما تتحرك، 2) خُد طبقة خفيفة لو الجو متقلب، "
                    "3) مية معاك طول الطريق، 4) لو في شبورة او مطر امشي أهدى وزوّد مسافة الأمان، "
                    "5) اختار طريق أقل زحمة قبل الخروج بربع ساعة."
                ),
                "saidi": (
                    "لو رايح المشوار دلوقتي: 1) اتأكد من حالة الجو، 2) خد جاكيت خفيف لو الجو متغير، "
                    "3) اشرب مية، 4) مع المطر او الشبورة سوق بهدوء، 5) اختار طريق اخف زحمة."
                ),
                "alexandrian": (
                    "وانت نازل: راجع الطقس، خد طبقة خفيفة، مية معاك، وامشي بهدوء لو في مطر أو شبورة، "
                    "واختار طريق أقل زحمة."
                ),
                "bedouin": (
                    "قبل المشوار: شيك الطقس، خذ طبقة مناسبة، مويه كفاية، ومع المطر او الشبورة خفف السرعة، "
                    "واختر طريق اخف ازدحام."
                ),
            }
            return replies.get(self.dialect, replies["cairene"])

        if self._contains_any_pattern(n, [r"(توتر|قلق|ضغط نفسي|stress|anxiety)"]) and self._contains_any_pattern(
            n,
            [r"(حل|خطة|خطوات|سريع|سرعة|فورا|ازاي|اعمل|اعملي|تقليل)"],
        ):
            return (
                "تمام، جرب الخطة دي دلوقتي: 1) نفس 4-4-6 لمدة دقيقتين، 2) اكتب 3 حاجات مضايقاك وواحدة تبدأ بيها، "
                "3) امشِ 10 دقايق بعيد عن الموبايل، 4) ارجع نفذ أول مهمة صغيرة خلال 15 دقيقة."
            )

        if self._contains_any_pattern(n, [r"(تنظيم الوقت|وقتي|تسويف|مواعيد|انتاجية)"]) and self._contains_any_pattern(
            n,
            [r"(حل|خطة|خطوات|ازاي|اعمل|اعملي|سريع|سرعة)"],
        ):
            return (
                "دي طريقة عملية لتنظيم الوقت: 1) حدد 3 أولويات لليوم، 2) اشتغل 25 دقيقة وريح 5، "
                "3) اقفل الإشعارات وقت الشغل، 4) حط المهمة الصعبة أول اليوم، 5) راجع آخر اليوم وخطط لبكرة."
            )

        if self._contains_any_pattern(
            n,
            [r"(امتحان|اختبار|مذاكرة|مراجعة|بكرة|فاينل)"],
        ) and self._contains_any_pattern(
            n,
            [r"(احفظ|حفظ|المعلومات|افتكر|ذاكر|تركيز|نسيان)"],
        ):
            replies = {
                "cairene": (
                    "لو امتحانك بكرة وعايز تحفظ بسرعة: 1) قسم المادة بلوكات 25 دقيقة + 5 راحة، "
                    "2) بعد كل بلوك اقفل الورق وسمّع لنفسك بصوت عالي، "
                    "3) اعمل فلاش كارد أو أسئلة سريعة للنقاط المهمة، "
                    "4) ذاكر أصعب جزئين دلوقتي والباقي قبل النوم، "
                    "5) نام 6-7 ساعات عشان التثبيت."
                ),
                "saidi": (
                    "لو امتحانك بكرة: ذاكر على فترات 25 دقيقة، وبعد كل جزء سمّع لنفسك، "
                    "ركز على النقاط المهمة، ونام كويس عشان تفتكر."
                ),
                "alexandrian": (
                    "أسرع طريقة قبل الامتحان: فترات قصيرة، تسميع فوري، أسئلة سريعة، "
                    "تركيز على أهم أجزاء، ونوم معقول."
                ),
                "bedouin": (
                    "قبل الامتحان: قسم المادة أجزاء صغيرة، سمع لنفسك كل جزء، "
                    "راجع المهم أول، وخذ نوم كفاية."
                ),
            }
            return replies.get(self.dialect, replies["cairene"])

        if self._contains_any_pattern(n, [
            r"\b(كلمني|كلميني|اتكلم|اتكلمي)\b",
        ]) and self._contains_any_pattern(n, [r"\b(مصري|عامية|روش|شبحنة)\b"]):
            replies = {
                "cairene": "حاضر، هكلمك مصري طبيعي وواضح. قولي طلبك وانا ارد من غير تعقيد.",
                "saidi": "حاضر يا اخوي، هكلمك مصري بسيط وواضح. قولي المطلوب.",
                "alexandrian": "ماشي يا باشا، هتكلم مصري خفيف ومنظم. قولي عايزة ايه.",
                "bedouin": "ابشر يا خوي، هتكلم مصري واضح وبسيط. قولي طلبك.",
            }
            return replies.get(self.dialect, replies["cairene"])

        return None

    async def _generate_ollama_response(
        self,
        text: str,
        user_emotion: str = "neutral",
        strict_hint: Optional[str] = None,
    ) -> Optional[str]:
        """Generate response from local Ollama server."""
        model = await self._resolve_ollama_model()
        if not model:
            return None

        context_hints = get_codeswitch_context_hints(text, max_hints=3)
        hints_block = "\n".join(f"- {hint}" for hint in context_hints)
        hints_text = hints_block if hints_block else "- لا يوجد"
        messages = self.memory.build_messages_for_llm(
            system_prompt=self._build_ollama_system_prompt(),
            max_recent=8,
        )
        messages.append(
            {
                "role": "user",
                "content": (
                    "اسلوب الرد المطلوب: مصري عامي، عملي، ومباشر.\n"
                    f"حالة العميل المتوقعة: {user_emotion}\n"
                    f"رسالة العميل: {text}\n"
                    f"امثلة قريبة من داتا الكلام المختلط:\n{hints_text}\n"
                    f"تعليمات اضافية: {strict_hint or 'لا يوجد'}"
                ),
            }
        )

        text_norm = self._normalize_match_text(text)
        predict_tokens = OLLAMA_MAX_TOKENS
        if self._contains_any_pattern(text_norm, [r"\b(خطة|خطوات|حل|اعمل|اعملي|ازاي|تقليل|معالجة|roadmap)\b"]):
            predict_tokens = max(predict_tokens, 150)
        elif len(text_norm.split()) <= 5:
            predict_tokens = min(predict_tokens, 90)

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "keep_alive": "30m",
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.15,
                "num_predict": predict_tokens,
            },
        }

        try:
            response_text = await self._call_ollama_chat(payload)
            cleaned = self._sanitize_model_output(response_text)
            if cleaned:
                self.ollama_warmed_up = True
                self.ollama_warning_logged = False
                # reset failure counter on success
                try:
                    self.ollama_failure_count = 0
                except Exception:
                    pass
                return cleaned
        except Exception as e:
            # increment failure counter and circuit-breaker after 3 failures
            try:
                self.ollama_failure_count = int(getattr(self, "ollama_failure_count", 0)) + 1
            except Exception:
                self.ollama_failure_count = 1
            if getattr(self, "ollama_failure_count", 0) >= 3:
                self._mark_ollama_unavailable(f"Ollama generation failed repeatedly: {e}")
            else:
                # transient failure, log only
                logger.warning("Ollama generation transient failure (%s): %s", getattr(self, "ollama_failure_count", 0), e)

        return None

    def _sanitize_model_output(self, text: str) -> str:
        """Remove leaked chat-template tokens and malformed artifacts from model output."""
        cleaned = re.sub(r"\s+", " ", (text or "")).strip()
        if not cleaned:
            return ""

        # Drop known prompt-template fragments.
        cleaned = re.sub(r"<\|[^|]+\|>", " ", cleaned)
        cleaned = cleaned.replace("回答：", " ")

        # Cut off leaked role sections if they appear.
        leak_markers = ["<|im_start|>", "<|im_end|>", " user ", " assistant ", " system "]
        lower_cleaned = f" {cleaned.lower()} "
        cut_index = None
        for marker in leak_markers:
            idx = lower_cleaned.find(marker)
            if idx > 6:
                cut_index = idx - 1
                break
        if cut_index is not None:
            cleaned = cleaned[:cut_index].strip()

        # Keep Arabic/Latin letters, digits and punctuation only.
        cleaned = re.sub(
            r"[^\u0600-\u06FFA-Za-z0-9\s\.,!\?;:\-\(\)\[\]،؟\"']",
            " ",
            cleaned,
            flags=re.UNICODE,
        )

        cleaned = re.sub(r"\s+", " ", cleaned, flags=re.UNICODE).strip()

        # Keep answer concise for low latency and stability.
        if len(cleaned) > 220:
            cleaned = cleaned[:220].rsplit(" ", 1)[0].strip()

        return cleaned

    async def _resolve_ollama_model(self) -> Optional[str]:
        """Resolve Ollama model from env or installed local models."""
        now = time.time()
        if now < self.ollama_disabled_until:
            return None

        if self.ollama_model and not self.ollama_best_model_checked:
            self.ollama_best_model_checked = True
            try:
                detected_model = await self._fetch_first_ollama_model(self.ollama_model)
            except Exception as e:
                self._mark_ollama_unavailable(
                    f"Ollama server unavailable at {OLLAMA_BASE_URL}: {e}"
                )
                return None

            if detected_model and detected_model != self.ollama_model:
                if self._is_small_model(self.ollama_model) and not self._is_small_model(detected_model):
                    logger.info(
                        f"Upgrading model from {self.ollama_model} to {detected_model} for better quality"
                    )
                else:
                    logger.info(
                        f"Configured model {self.ollama_model} not found locally. Falling back to {detected_model}"
                    )
                self.ollama_model = detected_model
            elif not detected_model:
                self.ollama_model = None

            if self.ollama_model and self._is_small_model(self.ollama_model):
                try:
                    better_model = await self._fetch_first_ollama_model(None)
                except Exception:
                    better_model = None

                if better_model and not self._is_small_model(better_model) and better_model != self.ollama_model:
                    logger.info(
                        "Configured tiny model %s was upgraded to %s for better quality",
                        self.ollama_model,
                        better_model,
                    )
                    self.ollama_model = better_model

        if self.ollama_model:
            return self.ollama_model

        try:
            detected_model = await self._fetch_first_ollama_model(OLLAMA_MODEL_DEFAULT)
        except Exception as e:
            self._mark_ollama_unavailable(
                f"Ollama server unavailable at {OLLAMA_BASE_URL}: {e}"
            )
            return None

        if not detected_model:
            self._mark_ollama_unavailable(
                "Ollama is running but no models are installed. Pull one with: ollama pull qwen2.5:1.5b-instruct-q8_0"
            )
            return None

        self.ollama_model = detected_model
        logger.info(f"Using Ollama model: {self.ollama_model}")
        self.ollama_warning_logged = False
        return self.ollama_model

    def _is_small_model(self, model_name: str) -> bool:
        """Treat tiny models as low quality for conversational Arabic generation."""
        name = (model_name or "").lower()
        return any(tag in name for tag in [":1b", ":1.5b", "-1b", "-1.5b"])

    async def _fetch_first_ollama_model(self, preferred_model: Optional[str] = None) -> Optional[str]:
        def pick(names: list[str]) -> Optional[str]:
            if not names:
                return None

            if preferred_model:
                preferred_lower = preferred_model.lower().strip()
                for model_name in names:
                    if model_name.lower() == preferred_lower:
                        return model_name
                for model_name in names:
                    if model_name.lower().startswith(preferred_lower):
                        return model_name

            priority = [
                OLLAMA_MODEL_DEFAULT,
                "qwen2.5:14b-instruct-q4_K_M",
                "qwen2.5:14b",
                "qwen2.5:7b-instruct-q4_K_M",
                "qwen2.5:7b-instruct",
                "qwen2.5:7b",
                "qwen2.5:3b-instruct",
                "llama3.1:8b",
                "qwen2.5:3b",
                "llama3.2:3b",
                "mistral:7b",
                "qwen2.5:1.5b-instruct-q8_0",
                "qwen2.5:1.5b",
                "llama3.2:1b",
            ]

            for preferred in priority:
                for model_name in names:
                    if model_name.startswith(preferred):
                        return model_name

            filtered = [
                model_name
                for model_name in names
                if "embed" not in model_name.lower() and "vision" not in model_name.lower()
            ]
            return filtered[0] if filtered else names[0]

        now = time.time()
        ttl_seconds = 60.0
        if self._cached_models and (now - self._models_last_fetch) < ttl_seconds:
            return pick(self._cached_models)

        timeout = aiohttp.ClientTimeout(total=max(1.0, min(5.0, OLLAMA_TAGS_TIMEOUT_SECONDS)))
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"{OLLAMA_BASE_URL}/api/tags") as response:
                    response.raise_for_status()
                    payload = await response.json()
        except (aiohttp.ClientError, TimeoutError, ValueError):
            if self._cached_models:
                return pick(self._cached_models)
            return None

        models = payload.get("models") or []
        names = [item.get("name") for item in models if item.get("name")]
        if not names:
            return None

        self._cached_models = names
        self._models_last_fetch = now
        return pick(names)

    async def _call_ollama_chat(self, payload: dict[str, Any]) -> str:
        """Call Ollama /api/chat asynchronously with a single retry on transient failures."""
        base_timeout = max(4.0, min(float(OLLAMA_GENERATE_TIMEOUT_SECONDS), 45.0))
        timeout_seconds = base_timeout if self.ollama_warmed_up else min(45.0, max(base_timeout, 24.0))
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        url = f"{OLLAMA_BASE_URL}/api/chat"

        for attempt in range(2):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(url, json=payload) as response:
                        response.raise_for_status()
                        data = await response.json()
                        message = data.get("message") or {}
                        return str(message.get("content") or "")
            except (aiohttp.ClientError, TimeoutError, ValueError):
                if attempt == 1:
                    raise
                await asyncio.sleep(0.3)

        return ""

    def _build_ollama_system_prompt(self) -> str:
        """Build customer-service system prompt with dialect runtime context."""
        dialect_instructions = {
            "cairene": "لهجة قاهرية طبيعية: (عامل ايه، عايز/عايزة، دلوقتي، ليه، مش).",
            "saidi": "لهجة صعيدية مصرية خفيفة: (عامل ايه يا اخوي، عاوز/عايزة، دلوك، ليه، مش).",
            "alexandrian": "لهجة اسكندرانية مصرية: (عامل ايه يا باشا، عايز/عايزة، دلوقتي، ليه، مش).",
            "bedouin": "لهجة بدوية مصرية خفيفة: (اخبارك ايه يا خوي، عايز/عايزة، هالحين، ليه، مش).",
        }

        base_prompt = system_prompt_english() if OLLAMA_SYSTEM_PROMPT_LANG == "english" else system_prompt_arabic()
        dialect_hint = dialect_instructions.get(self.dialect, dialect_instructions["cairene"])
        persona_hint = (
            "رد بصيغة صوت رجولي طبيعي ومهني." if self.gender == "male" else "رد بصيغة صوت أنثوي طبيعي ومهني."
        )
        return (
            f"{base_prompt}\n\n"
            "سياق التشغيل الحالي:\n"
            "- اسم المساعدة: سيرفيا\n"
            f"- اللهجة المفضلة: {dialect_hint}\n"
            f"- نوع الصوت الحالي: {self.gender}\n"
            f"- توجيه الأسلوب: {persona_hint}\n"
            "- اجعل الأسلوب بشري طبيعي، ويمكن استخدام همهمة/ضحكة خفيفة عند اللزوم فقط."
        )

    def _mark_ollama_unavailable(self, message: str):
        """Disable Ollama retries temporarily to avoid latency on every message."""
        self.ollama_disabled_until = time.time() + 30
        if not self.ollama_warning_logged:
            logger.warning(message)
            self.ollama_warning_logged = True

    def _generate_fallback_response(self, text: str) -> str:
        """Fallback rule-based response when Ollama is unavailable."""
        t = (text or "").strip()
        n = self._normalize_match_text(t)

        if self._contains_any_pattern(
            n,
            [r"(ازدحام|الازدحام|مرور|مروري|زحمة|تكدس|اختناق مروري|ازمة المواصلات|أزمة المواصلات|مواصلات)"],
        ):
            return (
                "خطة سريعة لأزمة الزحمة: 1) منع الوقوف العشوائي فورًا في أوقات الذروة، "
                "2) إعادة توقيت الإشارات على التقاطعات المزدحمة، 3) تخصيص حارات نقل جماعي، "
                "4) توزيع ساعات دخول العمل والجامعات، 5) فريق تدخل سريع للحوادث."
            )

        if self._contains_any_pattern(
            n,
            [r"(زحمة|مرور|ازدحام|تكدس|اختناق مروري|مواصلات)"],
        ) and self._contains_any_pattern(
            n,
            [r"(خنقة|مخنوق|مضايق|توتر|قلق|بعصب|متوتر|نفسيتي|مخنوقة)"],
        ):
            return (
                "لو الزحمة بتسبب لك خنقة: 1) نفس هادي دقيقة، 2) تهوية كويسة، 3) صوت هادي، "
                "4) خروج بدري وطريق بديل، 5) لو الأعراض قوية ومتكررة راجع مختص."
            )

        if self._contains_any_pattern(
            n,
            [r"(امتحان|اختبار|مذاكرة|مراجعة|بكرة|فاينل)"],
        ) and self._contains_any_pattern(
            n,
            [r"(احفظ|حفظ|المعلومات|افتكر|ذاكر|تركيز|نسيان)"],
        ):
            return (
                "قبل امتحان بكرة: ذاكر 25 دقيقة + 5 راحة، سمع لنفسك بعد كل جزء، "
                "راجع أهم نقاط وأسئلة سريعة، وسيب وقت نوم كفاية عشان تفتكر."
            )

        if self._contains_any_pattern(n, [r"(الجو|طقس|حر|برد|مطر|رياح|شبورة|مشوار|سفر)"]) and self._contains_any_pattern(
            n,
            [r"(نصيحة|نصايح|اعمل|اعمل اي|البس|اخد|دلوقتي|النهارده|خروج|رايح)"],
        ):
            return (
                "لو نازل دلوقتي: راجع تطبيق الطقس، خد طبقة مناسبة، خلي معاك مية، "
                "ولو في مطر او شبورة خفف السرعة وزوّد مسافة الأمان."
            )

        if self._contains_any_pattern(n, [
            r"\b(السلام عليكم|اهلا|مرحبا|هاي|صباح الخير|مساء الخير)\b",
        ]):
            return get_greeting(self.dialect, "hello")

        if "شكر" in n:
            return get_greeting(self.dialect, "thanks")

        if self._contains_any_pattern(n, [r"\b(مع السلامة|باي|سلام)\b"]):
            return get_greeting(self.dialect, "goodbye")

        if self._contains_any_pattern(n, [r"\b(اسمك|انتي مين|انت مين|مين انتي|مين انت|من انت|من انتي)\b"]):
            return "انا سيرفيا، مساعدة ذكية. اقدر اساعدك في الشرح، الكود، وتنظيم خطة تعلم مناسبة ليكي."

        if self._contains_any_pattern(n, [
            r"\b(خدمات|بتعمل|بتعملي|تقدري|تقدر|وظيفتك|ممكن تساعدني)\b",
        ]):
            return "اقدر اساعدك في الشرح المبسط، حل مشاكل الكود، وكتابة خطة تعلم عملية حسب هدفك."

        if self._contains_any_pattern(n, [r"\b(كام سنة|سنك كام|عمرك كام|عندك كام سنة)\b"]):
            return "انا مساعدة افتراضية، فمالييش سن بشري. لكن اقدر اساعدك بسرعة ودقة."

        if self._contains_any_pattern(n, [r"\b(مساعدة|ساعدني|ساعديني|عايز مساعدة|عايزة مساعدة)\b"]):
            return "تمام، انا معاكي. اكتبي الطلب بالتحديد وانا هجاوبك بشكل مرتب وخطوة خطوة."

        if self._contains_any_pattern(n, [r"\b(بايثون|python|برمجة|كود|تعلم|ابدا|ابدأ)\b"]):
            return (
                "تمام، نبدأ في 3 خطوات: 1) اساسيات المتغيرات والشرط والحلقات، "
                "2) حل 5 تمارين صغيرة يوميا، 3) مشروع بسيط كل اسبوع. "
                "لو تحبي ابعتلك خطة 14 يوم جاهزة."
            )

        if self._contains_any_pattern(n, [r"\b(خطا|غلط|bug|error|ايرور|مشكلة في الكود)\b"]):
            return "ابعتلي رسالة الخطا والكود اللي حواليها 15 سطر، وانا هحدد السبب والحل خطوة خطوة."

        if self._contains_any_pattern(n, [r"\b(كلمني|كلميني|اتكلم|اتكلمي)\b"]) and self._contains_any_pattern(
            n,
            [r"\b(مصري|عامية|روش|شبحنة)\b"],
        ):
            return "حاضر، هكلمك مصري طبيعي وواضح. قولي طلبك وانا ارد مباشرة."

        default_responses = {
            "cairene": "اقدر اساعدك مباشرة. اختار نوع الطلب: مشكلة مرور، نصايح للجو، كود/برمجة، او تنظيم وقت، وانا هديك خطوات عملية.",
            "saidi": "اقدر اساعدك حالًا يا اخوي. قول نوع الطلب: مرور، جو، كود، او تنظيم وقت، وانا اديك خطوات واضحة.",
            "alexandrian": "تمام يا باشا، اديك حل مباشر. حدد النوع: مرور، جو، كود، او تنظيم وقت.",
            "bedouin": "ابشر يا خوي، حدد نوع الطلب: مرور، جو، كود، او تنظيم وقت، وانا اعطيك خطوات عملية.",
        }
        return default_responses.get(self.dialect, default_responses["cairene"])

    async def send_json(self, data: dict):
        """Send JSON message via WebSocket."""
        if self.ws is None:
            return
        if self.ws.application_state != WebSocketState.CONNECTED:
            return
        if self.ws.client_state == WebSocketState.DISCONNECTED:
            return
        try:
            await self.ws.send_json(data)
        except WebSocketDisconnect:
            return
        except Exception as e:
            message = str(e).lower()
            if "close message has been sent" in message or "cannot call \"send\"" in message:
                return
            logger.error(f"Failed to send message: {e}")

    async def send_text(self, text: str):
        if self.ws is None:
            return
        if self.ws.application_state != WebSocketState.CONNECTED:
            return
        if self.ws.client_state == WebSocketState.DISCONNECTED:
            return
        try:
            await self.ws.send_text(text)
        except WebSocketDisconnect:
            return
        except Exception as e:
            message = str(e).lower()
            if "close message has been sent" in message or "cannot call \"send\"" in message:
                return
            logger.error(f"Failed to send text: {e}")

    async def send_bytes(self, data: bytes):
        if self.ws is None:
            return
        if self.ws.application_state != WebSocketState.CONNECTED:
            return
        if self.ws.client_state == WebSocketState.DISCONNECTED:
            return
        try:
            await self.ws.send_bytes(data)
        except WebSocketDisconnect:
            return
        except Exception as e:
            message = str(e).lower()
            if "close message has been sent" in message or "cannot call \"send\"" in message:
                return
            logger.error(f"Failed to send bytes: {e}")

    async def receive_json(self) -> dict[str, Any]:
        if self.ws is None:
            return {}
        return await self.ws.receive_json()

    async def receive_bytes(self) -> bytes:
        if self.ws is None:
            return b""
        message = await self.ws.receive()
        return message.get("bytes") or b""

    async def close(self):
        if self.ws is None:
            return
        try:
            await self.ws.close()
        except Exception as e:
            logger.error(f"Failed to close websocket: {e}")


def _get_or_create_api_session(session_id: str, dialect: str, gender: str) -> VoiceSession:
    key = (session_id or "api-default").strip() or "api-default"
    session = API_SESSIONS.get(key)
    if session is None:
        session = VoiceSession(None)
        session.session_id = key
        API_SESSIONS[key] = session

    if dialect in ["cairene", "saidi", "alexandrian", "bedouin"]:
        session.dialect = dialect
    if gender in ["male", "female"]:
        session.gender = gender

    ANALYTICS.register_session(session.session_id, session.dialect)
    return session


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Stateless HTTP chat endpoint backed by session memory and Ollama."""
    session = _get_or_create_api_session(request.session_id, request.dialect, request.gender)
    normalized_text = normalize_codeswitch_text(request.text)
    analysis = session._analyze_user_message(normalized_text)
    user_emotion = analysis.get("sentiment_label", "neutral")
    voice_emotion = session._map_tts_emotion(user_emotion)

    started = time.perf_counter()
    session._append_chat_message("user", normalized_text, meta={"analysis": analysis, "source": "api"})
    await session._maybe_summarize_memory()
    response_text = await session._generate_response(normalized_text, user_emotion=user_emotion)
    session._append_chat_message("assistant", response_text, meta={"source": "api"})
    latency_ms = int((time.perf_counter() - started) * 1000)

    ANALYTICS.record_message(
        session_id=session.session_id,
        dialect=session.dialect,
        analysis=analysis,
        response_latency_ms=latency_ms,
    )

    tts_payload: Optional[dict[str, Any]] = None
    if request.include_tts:
        audio_bytes, audio_format = await synthesize_long_text(
            response_text,
            session.dialect,
            session.gender,
            voice_emotion,
            max_chars=TTS_SEGMENT_MAX_CHARS,
        )
        tts_payload = {
            "audio_base64": base64.b64encode(audio_bytes).decode("utf-8"),
            "audio_format": audio_format,
            "emotion": voice_emotion,
        }

    return ChatResponse(
        session_id=session.session_id,
        text=request.text,
        normalized_text=normalized_text,
        analysis=analysis,
        response_text=response_text,
        latency_ms=latency_ms,
        memory=session.memory.export_state(),
        tts=tts_payload,
    )


@app.post("/api/session/clear")
async def clear_session(session_id: str = Form(...)):
    """Clear server-side memory for one API chat session."""
    session = API_SESSIONS.pop((session_id or "").strip(), None)
    if session is not None:
        session.memory.clear()
    return {"status": "ok", "session_id": session_id}


def _make_voice_pipeline_for_session(session: VoiceSession) -> VoicePipeline:
    async def stt_fn(audio_bytes: bytes, mime_type: str, language: str = "ar") -> dict[str, Any]:
        before_vram = 0.0
        try:
            before_vram = float(stt_engine._get_vram_usage_percent())
        except Exception:
            before_vram = 0.0

        stt_result = await STT_ENGINE.transcribe(audio_bytes, mime_type=mime_type, language=language, session_id=session.session_id)
        after_vram = 0.0
        try:
            after_vram = float(stt_engine._get_vram_usage_percent())
        except Exception:
            after_vram = 0.0

        logger.info("[STT] device=%s latency=%dms vram_before=%.1f%% vram_after=%.1f%%",
                    stt_result.provider, stt_result.latency_ms, before_vram, after_vram)
        return {
            "text": stt_result.text,
            "provider": stt_result.provider,
            "confidence": stt_result.confidence,
            "latency_ms": stt_result.latency_ms,
            "segments": stt_result.segments,
            "language": stt_result.language,
            "fallback_used": stt_result.fallback_used,
        }

    async def processing_fn(text: str) -> dict[str, Any]:
        analysis = session._analyze_user_message(text)
        analysis["voice_emotion"] = session._map_tts_emotion(analysis.get("sentiment_label", "neutral"))
        return analysis

    async def llm_fn(text: str, analysis: dict[str, Any]) -> str:
        session._append_chat_message("user", text, meta={"analysis": analysis, "source": "pipeline"})
        await session._maybe_summarize_memory()
        before_vram = 0.0
        try:
            before_vram = float(tts_engine._get_vram_usage_percent())
        except Exception:
            before_vram = 0.0

        response = await session._generate_response(text, user_emotion=analysis.get("sentiment_label", "neutral"))

        after_vram = 0.0
        try:
            after_vram = float(tts_engine._get_vram_usage_percent())
        except Exception:
            after_vram = 0.0

        logger.info("[LLM] model=%s latency_tokens_est=? vram_before=%.1f%% vram_after=%.1f%%",
                    session.ollama_model or "none", before_vram, after_vram)
        session._append_chat_message("assistant", response, meta={"source": "pipeline"})
        return response

    async def tts_fn(text: str, dialect: str, gender: str, emotion: str) -> tuple[bytes, str]:
        # Use long-text-aware synthesizer to enforce chunking and reassembly
        before_vram = 0.0
        try:
            before_vram = float(tts_engine._get_vram_usage_percent())
        except Exception:
            before_vram = 0.0

        start = time.perf_counter()
        audio_bytes, audio_fmt = await synthesize_long_text(text, dialect, gender, emotion, max_chars=TTS_SEGMENT_MAX_CHARS)
        latency_ms = int((time.perf_counter() - start) * 1000)

        after_vram = 0.0
        try:
            after_vram = float(tts_engine._get_vram_usage_percent())
        except Exception:
            after_vram = 0.0

        # estimate chunks
        chunks = max(1, int((len(text or "") / max(1, TTS_SEGMENT_MAX_CHARS))))
        logger.info("[TTS] device=%s chunks=%d latency=%dms vram_before=%.1f%% vram_after=%.1f%%",
                    tts_engine._xtts_device if getattr(tts_engine, "_xtts_device", None) else "cpu",
                    chunks, latency_ms, before_vram, after_vram)
        return audio_bytes, audio_fmt

    return VoicePipeline(
        stt_fn=stt_fn,
        processing_fn=processing_fn,
        llm_fn=llm_fn,
        tts_fn=tts_fn,
    )


@app.post("/api/pipeline/voice-turn")
async def pipeline_voice_turn(
    file: UploadFile = File(...),
    session_id: str = Form(default="api-default"),
    dialect: str = Form(default="cairene"),
    gender: str = Form(default="female"),
    language: str = Form(default="ar"),
    include_tts: bool = Form(default=True),
):
    """Full modular pipeline endpoint: Input -> STT -> Processing -> LLM -> TTS."""
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Empty audio payload")

    session = _get_or_create_api_session(session_id=session_id, dialect=dialect, gender=gender)
    pipeline = _make_voice_pipeline_for_session(session)
    try:
        output = await pipeline.run_audio_turn(
            audio_bytes=payload,
            mime_type=file.content_type or "audio/wav",
            tts_dialect=session.dialect,
            tts_gender=session.gender,
            language=language,
            include_tts=include_tts,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    tts = output.get("tts")
    if tts:
        output["tts"] = {
            "audio_base64": base64.b64encode(tts["audio_bytes"]).decode("utf-8"),
            "audio_format": tts["audio_format"],
        }

    analysis = output.get("analysis") or {}
    ANALYTICS.record_message(
        session_id=session.session_id,
        dialect=session.dialect,
        analysis=analysis,
        response_latency_ms=int(output.get("latency_ms") or 0),
    )

    output["session_id"] = session.session_id
    output["memory"] = session.memory.export_state()
    return output


@app.websocket("/ws/voice")
async def voice_chat(websocket: WebSocket):
    """WebSocket endpoint for real-time voice chat with VAD and TTS."""
    session = VoiceSession(websocket)
    await session.handle()


# ===================== Run =====================

if __name__ == "__main__":
    import uvicorn

    serve_port = int(os.getenv("SERVIA_PORT", "8765"))
    reload_enabled = os.getenv("SERVIA_RELOAD", "1").strip().lower() in {"1", "true", "yes", "on"}

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=serve_port,
        reload=reload_enabled,
        log_level="info",
    )
