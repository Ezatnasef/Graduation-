"""
Egyptian Arabic TTS Engine

Provider order:
1) Local provider: Nile XTTS
2) Gemini TTS fallback
3) Edge TTS fallback
4) gTTS final fallback
"""

import asyncio
import base64
import gc
import inspect
import io
import logging
import mimetypes
import os
import re
import struct
import time
import unicodedata
import wave
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional, cast

_EARLY_BACKEND_DIR = Path(__file__).resolve().parent
_EARLY_PROJECT_ROOT_DIR = _EARLY_BACKEND_DIR.parent
for _early_env_path in [_EARLY_PROJECT_ROOT_DIR / ".env", _EARLY_BACKEND_DIR / ".env"]:
    try:
        if _early_env_path.exists():
            for _line in _early_env_path.read_text(encoding="utf-8").splitlines():
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

import numpy as np
from gtts import gTTS

try:
    from pydub import AudioSegment
except Exception:
    AudioSegment = None

from dialect_mapper import get_dialect_prosody, transform_to_dialect

try:
    import edge_tts

    EDGE_TTS_AVAILABLE = True
except Exception:
    edge_tts = None
    EDGE_TTS_AVAILABLE = False

GEMINI_IMPORT_ERROR = ""
try:
    import google.genai as genai  # type: ignore[reportMissingImports]
    from google.genai import types as genai_types  # type: ignore[reportMissingImports]

    GEMINI_RUNTIME_AVAILABLE = True
except Exception as e:
    genai = None
    genai_types = None
    GEMINI_RUNTIME_AVAILABLE = False
    GEMINI_IMPORT_ERROR = str(e)

TORCH_IMPORT_ERROR = ""
try:
    import torch
except Exception as e:
    torch = None
    TORCH_IMPORT_ERROR = str(e)


def _patch_transformers_for_xtts() -> None:
    """Expose generation classes where older Coqui XTTS expects them."""
    try:
        import transformers
        from transformers import generation
        from transformers.generation import beam_search
    except Exception:
        return

    aliases = {
        "BeamSearchScorer": beam_search,
        "LogitsProcessorList": generation,
        "StoppingCriteriaList": generation,
    }
    for name, module in aliases.items():
        if not hasattr(transformers, name) and hasattr(module, name):
            setattr(transformers, name, getattr(module, name))


XTTS_IMPORT_ERROR = ""
try:
    _patch_transformers_for_xtts()
    from TTS.tts.configs.xtts_config import XttsConfig  # type: ignore[reportMissingImports]
    from TTS.tts.models.xtts import Xtts  # type: ignore[reportMissingImports]

    XTTS_RUNTIME_AVAILABLE = torch is not None
    if torch is None:
        XTTS_IMPORT_ERROR = f"Torch unavailable: {TORCH_IMPORT_ERROR or 'not installed'}"
except Exception as e:
    XttsConfig = None
    Xtts = None
    XTTS_RUNTIME_AVAILABLE = False
    XTTS_IMPORT_ERROR = str(e)

TRANSFORMERS_IMPORT_ERROR = ""
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    TRANSFORMERS_RUNTIME_AVAILABLE = True
except Exception as e:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    TRANSFORMERS_RUNTIME_AVAILABLE = False
    TRANSFORMERS_IMPORT_ERROR = str(e)

CHATTERBOX_IMPORT_ERROR = ""
try:
    from chatterbox import ChatterboxMultilingualTTS, ChatterboxTTS  # type: ignore[reportMissingImports]

    CHATTERBOX_RUNTIME_AVAILABLE = True
except Exception as e:
    ChatterboxMultilingualTTS = None
    ChatterboxTTS = None
    CHATTERBOX_RUNTIME_AVAILABLE = False
    CHATTERBOX_IMPORT_ERROR = str(e)

logger = logging.getLogger("servia-voice")

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_DIR = BACKEND_DIR.parent
MODELS_DIR = PROJECT_ROOT_DIR / "models"

for _env_path in [PROJECT_ROOT_DIR / ".env", BACKEND_DIR / ".env"]:
    try:
        if _env_path.exists():
            for _line in _env_path.read_text(encoding="utf-8").splitlines():
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

DEFAULT_XTTS_MODEL_DIR = MODELS_DIR / "NileTTS-XTTS"
if not DEFAULT_XTTS_MODEL_DIR.exists():
    DEFAULT_XTTS_MODEL_DIR = PROJECT_ROOT_DIR / "NileTTS-XTTS"

DEFAULT_CHATTERBOX_MODEL_DIR = MODELS_DIR / "chatterbox-egyptian-v0"
if not DEFAULT_CHATTERBOX_MODEL_DIR.exists():
    DEFAULT_CHATTERBOX_MODEL_DIR = PROJECT_ROOT_DIR / "chatterbox-egyptian-v0"

DEFAULT_XTTS_REFERENCE_AUDIO = PROJECT_ROOT_DIR / "artifacts" / "tts_e2e_local.wav"
DEFAULT_XTTS_REFERENCE_CANDIDATES = [
    DEFAULT_XTTS_REFERENCE_AUDIO,
    MODELS_DIR / "sample_06.wav",
    MODELS_DIR / "sample_01 (1).wav",
    PROJECT_ROOT_DIR / "sample_06.wav",
    PROJECT_ROOT_DIR / "sample_01 (1).wav",
]

TTS_PROVIDER = os.getenv("TTS_PROVIDER", "auto").strip().lower() or "auto"
XTTS_MODEL_DIR = Path(
    os.getenv("XTTS_MODEL_DIR", str(DEFAULT_XTTS_MODEL_DIR)).strip() or str(DEFAULT_XTTS_MODEL_DIR)
)
CHATTERBOX_MODEL_DIR = Path(
    os.getenv("CHATTERBOX_MODEL_DIR", str(DEFAULT_CHATTERBOX_MODEL_DIR)).strip() or str(DEFAULT_CHATTERBOX_MODEL_DIR)
)
XTTS_LANGUAGE = os.getenv("XTTS_LANGUAGE", "ar").strip() or "ar"
XTTS_REFERENCE_AUDIO = os.getenv("XTTS_REFERENCE_AUDIO", str(DEFAULT_XTTS_REFERENCE_AUDIO)).strip()
XTTS_REFERENCE_AUDIO_FEMALE = os.getenv("XTTS_REFERENCE_AUDIO_FEMALE", "").strip()
XTTS_REFERENCE_AUDIO_MALE = os.getenv("XTTS_REFERENCE_AUDIO_MALE", "").strip()
GEMINI_TTS_MODEL = os.getenv("GEMINI_TTS_MODEL", "gemini-2.5-pro-preview-tts").strip() or "gemini-2.5-pro-preview-tts"
try:
    GEMINI_TTS_TEMPERATURE = float(os.getenv("GEMINI_TTS_TEMPERATURE", "1.0"))
except ValueError:
    GEMINI_TTS_TEMPERATURE = 1.0
GEMINI_TTS_VOICE = os.getenv("GEMINI_TTS_VOICE", "Zephyr").strip() or "Zephyr"
GEMINI_TTS_VOICE_FEMALE = os.getenv("GEMINI_TTS_VOICE_FEMALE", "").strip()
GEMINI_TTS_VOICE_MALE = os.getenv("GEMINI_TTS_VOICE_MALE", "").strip()
GEMINI_TTS_VOICE_CAIRENE = os.getenv("GEMINI_TTS_VOICE_CAIRENE", "").strip()
GEMINI_TTS_VOICE_SAIDI = os.getenv("GEMINI_TTS_VOICE_SAIDI", "").strip()
GEMINI_TTS_VOICE_ALEXANDRIAN = os.getenv("GEMINI_TTS_VOICE_ALEXANDRIAN", "").strip()
GEMINI_TTS_VOICE_BEDOUIN = os.getenv("GEMINI_TTS_VOICE_BEDOUIN", "").strip()


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _parse_quality_order(value: str) -> list[str]:
    allowed = {"chatterbox", "xtts"}
    parsed: list[str] = []
    for item in (value or "").split(","):
        provider = item.strip().lower()
        if provider in allowed and provider not in parsed:
            parsed.append(provider)

    if not parsed:
        parsed = ["chatterbox", "xtts"]

    return parsed


XTTS_GPT_COND_LEN = _env_int("XTTS_GPT_COND_LEN", 6)
XTTS_MAX_REF_LEN = _env_int("XTTS_MAX_REF_LEN", 12)
XTTS_BASE_TEMPERATURE = _env_float("XTTS_TEMPERATURE", 0.62)
XTTS_TIMEOUT_SECONDS = max(10, _env_int("XTTS_TIMEOUT_SECONDS", 180))
XTTS_INPUT_MAX_CHARS = max(32, _env_int("XTTS_INPUT_MAX_CHARS", 220))
XTTS_EMPTY_CACHE_INTERVAL_SECONDS = max(5, _env_int("XTTS_EMPTY_CACHE_INTERVAL_SECONDS", 20))
CHATTERBOX_TIMEOUT_SECONDS = max(4, _env_int("CHATTERBOX_TIMEOUT_SECONDS", 14))
XTTS_FAILURE_COOLDOWN_SECONDS = max(15, _env_int("XTTS_FAILURE_COOLDOWN_SECONDS", 45))
CHATTERBOX_FAILURE_COOLDOWN_SECONDS = max(15, _env_int("CHATTERBOX_FAILURE_COOLDOWN_SECONDS", 45))
CHATTERBOX_LANGUAGE = os.getenv("CHATTERBOX_LANGUAGE", "ar").strip().lower() or "ar"
CHATTERBOX_ALLOW_REMOTE = os.getenv("CHATTERBOX_ALLOW_REMOTE", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
CHATTERBOX_USE_MULTILINGUAL = os.getenv("CHATTERBOX_USE_MULTILINGUAL", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
TTS_HUMAN_STYLE_ENABLED = os.getenv("TTS_HUMAN_STYLE_ENABLED", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

TTS_DUAL_QUALITY_GRACE_MS = max(0, _env_int("TTS_DUAL_QUALITY_GRACE_MS", 320))
TTS_DUAL_SECONDARY_LAUNCH_MS = max(0, _env_int("TTS_DUAL_SECONDARY_LAUNCH_MS", 1500))
TTS_DUAL_QUALITY_ORDER = _parse_quality_order(
    os.getenv("TTS_DUAL_QUALITY_ORDER", "chatterbox,xtts")
)
TTS_AUTO_LOCAL_ORDER = _parse_quality_order(
    os.getenv("TTS_AUTO_LOCAL_ORDER", "chatterbox,xtts")
)
TTS_AUTO_USE_DUAL = os.getenv("TTS_AUTO_USE_DUAL", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
TTS_CACHE_SIZE = max(0, _env_int("TTS_CACHE_SIZE", 80))
TTS_RETURN_SILENCE_ON_FAILURE = os.getenv("TTS_RETURN_SILENCE_ON_FAILURE", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
TTS_FAILURE_SILENCE_MS = max(250, _env_int("TTS_FAILURE_SILENCE_MS", 650))

USE_GPU = os.getenv("USE_GPU", "1").strip().lower() in {"1", "true", "yes", "on"}
CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", "0").strip()

XTTS_DEVICE = os.getenv("XTTS_DEVICE", "cuda" if USE_GPU else "cpu").strip().lower() or "cpu"
XTTS_HALF_PRECISION = os.getenv("XTTS_HALF_PRECISION", "1").strip().lower() in {"1", "true", "yes", "on"}
XTTS_ENABLE_CHUNKING = os.getenv("XTTS_ENABLE_CHUNKING", "1").strip().lower() in {"1", "true", "yes", "on"}
XTTS_USE_DEEPSPEED = os.getenv("XTTS_USE_DEEPSPEED", "0").strip().lower() in {"1", "true", "yes", "on"}

STT_DEVICE = os.getenv("STT_DEVICE", "cuda" if USE_GPU else "cpu").strip().lower() or "cpu"
EGYPTALK_DEVICE = os.getenv("EGYPTALK_DEVICE", "cpu").strip().lower() or "cpu"
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda" if USE_GPU else "cpu").strip().lower() or "cpu"
OLLAMA_DEVICE = os.getenv("OLLAMA_DEVICE", "cuda" if USE_GPU else "cpu").strip().lower() or "cpu"


VOICE_MAP: Dict[str, str] = {
    "female": "ar-EG-SalmaNeural",
    "male": "ar-EG-ShakirNeural",
}

EMOTION_PROFILES: Dict[str, Dict[str, int]] = {
    "neutral": {"rate": 0, "pitch_hz": 0, "volume": 0},
    "excited": {"rate": 6, "pitch_hz": 3, "volume": 4},
    "angry": {"rate": 3, "pitch_hz": 1, "volume": 3},
    "sad": {"rate": -6, "pitch_hz": -3, "volume": -3},
    "frustrated": {"rate": -1, "pitch_hz": -1, "volume": 1},
    "concerned": {"rate": -2, "pitch_hz": -1, "volume": 0},
    "empathetic": {"rate": -4, "pitch_hz": -2, "volume": 0},
}

XTTS_EMOTION_TEMPERATURE = {
    "neutral": XTTS_BASE_TEMPERATURE,
    "excited": min(1.0, XTTS_BASE_TEMPERATURE + 0.12),
    "angry": min(0.95, XTTS_BASE_TEMPERATURE + 0.08),
    "sad": max(0.45, XTTS_BASE_TEMPERATURE - 0.12),
    "frustrated": max(0.5, XTTS_BASE_TEMPERATURE - 0.06),
    "concerned": max(0.5, XTTS_BASE_TEMPERATURE - 0.08),
    "empathetic": max(0.48, XTTS_BASE_TEMPERATURE - 0.1),
}


_xtts_model = None
_xtts_device = "cpu"
_xtts_sample_rate = 24000
_xtts_model_error: Optional[str] = None
_xtts_conditioning_cache: dict[str, tuple[Any, Any]] = {}
_xtts_lock = Lock()
_xtts_infer_lock = Lock()

_chatterbox_model = None
_chatterbox_device = "cpu"
_chatterbox_sample_rate = 24000
_chatterbox_model_error: Optional[str] = None
_chatterbox_prepared_reference: Optional[str] = None
_chatterbox_lock = Lock()

_local_provider_disabled_until: dict[str, float] = {
    "xtts": 0.0,
    "chatterbox": 0.0,
}

_tts_cache: OrderedDict[tuple, tuple[bytes, str]] = OrderedDict()
_tts_cache_lock = Lock()
_last_cuda_cache_cleanup_at = 0.0


def _cleanup_spoken_text(text: str) -> str:
    spoken = (text or "").strip()
    spoken = re.sub(r"\s+", " ", spoken, flags=re.UNICODE)
    spoken = re.sub(r"\.{3,}", "...", spoken, flags=re.UNICODE)
    spoken = re.sub(r"[!]{2,}", "!", spoken, flags=re.UNICODE)
    spoken = re.sub(r"[?]{2,}", "?", spoken, flags=re.UNICODE)
    spoken = re.sub(r"\s+([؟!.,،])", r"\1", spoken, flags=re.UNICODE)
    return spoken


def _torch_cuda_available() -> bool:
    return bool(torch and getattr(torch.cuda, "is_available", lambda: False)())


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


def _gpu_requested(device_name: str) -> bool:
    return (device_name or "").strip().lower() == "cuda" and _torch_cuda_available()


def _cleanup_inference_memory(force: bool = False) -> None:
    global _last_cuda_cache_cleanup_at

    gc.collect()
    if torch and _torch_cuda_available():
        now = time.time()
        should_empty_cache = force or (now - _last_cuda_cache_cleanup_at) >= float(XTTS_EMPTY_CACHE_INTERVAL_SECONDS)
        if not should_empty_cache:
            return
        try:
            torch.cuda.empty_cache()
            _last_cuda_cache_cleanup_at = now
        except Exception:
            pass


def _is_cuda_oom_error(error: BaseException) -> bool:
    message = str(error).lower()
    return "cuda out of memory" in message or ("cuda" in message and "oom" in message)


def _is_cuda_device_assert_error(error: BaseException) -> bool:
    message = str(error).lower()
    return "device-side assert triggered" in message


def _xtts_autocast_context():
    if not (XTTS_HALF_PRECISION and _xtts_device == "cuda" and torch is not None):
        return nullcontext()

    try:
        amp_module = cast(Any, getattr(torch, "amp", None))
        if amp_module is not None and hasattr(amp_module, "autocast"):
            return amp_module.autocast("cuda", dtype=torch.float16)
        return torch.cuda.amp.autocast(dtype=torch.float16)
    except Exception:
        return nullcontext()


def _sanitize_xtts_text(text: str) -> str:
    """Keep XTTS input within a conservative Arabic-friendly character set."""
    spoken = _cleanup_spoken_text(text)
    spoken = re.sub(r"[^\u0600-\u06FF0-9A-Za-z\s؟!.,،;:()\-]", " ", spoken, flags=re.UNICODE)
    spoken = re.sub(r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF]", " ", spoken, flags=re.UNICODE)
    spoken = re.sub(r"\s+", " ", spoken, flags=re.UNICODE).strip()
    return spoken


def _prepare_xtts_text(text: str, dialect: str, emotion: str, max_chars: int = 220) -> str:
    cleaned = _cleanup_spoken_text(transform_to_dialect(text or "", dialect or "cairene"))
    cleaned = _apply_human_voice_cues(cleaned, emotion)
    cleaned = _sanitize_xtts_text(cleaned)
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rsplit(" ", 1)[0].strip() or cleaned[:max_chars].strip()
    if not cleaned:
        raise RuntimeError("XTTS input is empty after sanitization")
    return cleaned


def _safe_autocast():
    if not (torch and _xtts_device == "cuda" and XTTS_HALF_PRECISION):
        return nullcontext()
    try:
        amp_module = cast(Any, getattr(torch, "amp", None))
        if amp_module is not None and hasattr(amp_module, "autocast"):
            return amp_module.autocast("cuda", dtype=torch.float16)
        return torch.cuda.amp.autocast(dtype=torch.float16)
    except Exception:
        try:
            return torch.cuda.amp.autocast(dtype=torch.float16)
        except Exception:
            return nullcontext()


def _validate_xtts_input_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = re.sub(r"[\u0000-\u001f\u007f-\u009f]", " ", normalized)
    normalized = re.sub(r"[\U00010000-\U0010ffff]", " ", normalized)

    safe_text = _sanitize_xtts_text(normalized)
    if not safe_text:
        raise RuntimeError("XTTS text became empty after sanitization")

    if len(safe_text) > XTTS_INPUT_MAX_CHARS:
        truncated = safe_text[:XTTS_INPUT_MAX_CHARS].strip()
        if " " in truncated:
            truncated = truncated.rsplit(" ", 1)[0].strip() or truncated
        safe_text = truncated

    return safe_text


def _apply_human_voice_cues(text: str, emotion: str) -> str:
    """Inject subtle fillers to make delivery sound more human-like."""
    spoken = _cleanup_spoken_text(text)
    if not TTS_HUMAN_STYLE_ENABLED or len(spoken) < 10:
        return spoken

    e = (emotion or "neutral").strip().lower()
    if e in {"excited", "happy"}:
        return f"ههه، {spoken}"
    if e in {"empathetic", "sad", "concerned"}:
        return f"ممم، {spoken}"
    if e in {"frustrated"}:
        return f"آه، {spoken}"
    return spoken


def _clamp(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(value, max_value))


def _clamp_float(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


def _parse_audio_mime_type(mime_type: str) -> dict[str, int]:
    bits_per_sample = 16
    rate = 24000

    parts = (mime_type or "").split(";")
    for param in parts:
        param = param.strip()
        if param.lower().startswith("rate="):
            try:
                rate = int(param.split("=", 1)[1])
            except (ValueError, IndexError):
                pass
        elif param.lower().startswith("audio/l"):
            try:
                bits_per_sample = int(param.split("L", 1)[1])
            except (ValueError, IndexError):
                pass

    return {"bits_per_sample": bits_per_sample, "rate": rate}


def _convert_raw_audio_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    params = _parse_audio_mime_type(mime_type)
    bits_per_sample = params["bits_per_sample"]
    sample_rate = params["rate"]
    num_channels = 1
    data_size = len(audio_data)
    bytes_per_sample = max(1, bits_per_sample // 8)
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        chunk_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header + audio_data


def _build_tts_cache_key(
    provider: str,
    text: str,
    dialect: str,
    gender: str,
    emotion: str,
) -> tuple:
    return (
        provider,
        (dialect or "cairene").strip().lower(),
        (gender or "female").strip().lower(),
        (emotion or "neutral").strip().lower(),
        _cleanup_spoken_text(text),
    )


def _get_cached_tts(
    provider: str,
    text: str,
    dialect: str,
    gender: str,
    emotion: str,
) -> Optional[tuple[bytes, str]]:
    if TTS_CACHE_SIZE <= 0:
        return None

    key = _build_tts_cache_key(provider, text, dialect, gender, emotion)
    with _tts_cache_lock:
        cached = _tts_cache.get(key)
        if cached is None:
            return None
        _tts_cache.move_to_end(key)
        audio_bytes, audio_format = cached
        return bytes(audio_bytes), audio_format


def _put_cached_tts(
    provider: str,
    text: str,
    dialect: str,
    gender: str,
    emotion: str,
    audio_bytes: bytes,
    audio_format: str,
):
    if TTS_CACHE_SIZE <= 0 or not audio_bytes:
        return

    key = _build_tts_cache_key(provider, text, dialect, gender, emotion)
    with _tts_cache_lock:
        _tts_cache[key] = (bytes(audio_bytes), audio_format)
        _tts_cache.move_to_end(key)
        while len(_tts_cache) > TTS_CACHE_SIZE:
            _tts_cache.popitem(last=False)


def _rate_multiplier_to_percent(rate_value: str) -> int:
    try:
        multiplier = float(rate_value)
    except (TypeError, ValueError):
        return 0
    return int(round((multiplier - 1.0) * 100))


def _build_voice_config(dialect: str, gender: str, emotion: str) -> dict[str, str]:
    normalized_gender = "male" if (gender or "").lower() == "male" else "female"
    normalized_emotion = (emotion or "neutral").lower()

    prosody = get_dialect_prosody(dialect)
    base_rate_percent = _rate_multiplier_to_percent(prosody.get("rate", "1.0"))

    emotion_profile = EMOTION_PROFILES.get(normalized_emotion, EMOTION_PROFILES["neutral"])
    final_rate_percent = _clamp(base_rate_percent + emotion_profile["rate"], -30, 35)
    final_pitch_hz = _clamp(emotion_profile["pitch_hz"], -12, 12)
    final_volume_percent = _clamp(emotion_profile["volume"], -20, 20)

    return {
        "voice": VOICE_MAP[normalized_gender],
        "rate": f"{final_rate_percent:+d}%",
        "pitch": f"{final_pitch_hz:+d}Hz",
        "volume": f"{final_volume_percent:+d}%",
    }


def _is_gemini_ready() -> bool:
    return GEMINI_RUNTIME_AVAILABLE and bool(os.getenv("GEMINI_API_KEY", "").strip())


def _select_gemini_voice(dialect: str, gender: str) -> str:
    d = (dialect or "cairene").strip().lower()
    g = (gender or "female").strip().lower()

    dialect_voice_map = {
        "cairene": GEMINI_TTS_VOICE_CAIRENE,
        "saidi": GEMINI_TTS_VOICE_SAIDI,
        "alexandrian": GEMINI_TTS_VOICE_ALEXANDRIAN,
        "bedouin": GEMINI_TTS_VOICE_BEDOUIN,
    }

    gender_voice = GEMINI_TTS_VOICE_MALE if g == "male" else GEMINI_TTS_VOICE_FEMALE
    return (
        dialect_voice_map.get(d)
        or gender_voice
        or GEMINI_TTS_VOICE
        or "Zephyr"
    )


def _build_gemini_tts_prompt(text: str, dialect: str, emotion: str) -> str:
    dialect_instruction = {
        "cairene": "تكلم بلهجة قاهرية مصرية طبيعية وواضحة.",
        "saidi": "تكلم بلهجة صعيدية مصرية واضحة من غير مبالغة.",
        "alexandrian": "تكلم بلهجة اسكندرانية مصرية طبيعية وواضحة.",
        "bedouin": "تكلم بلهجة بدوية مصرية مفهومة وبسيطة.",
    }

    emotion_instruction = {
        "neutral": "النبرة متزنة وهادية.",
        "excited": "النبرة فيها حيوية وتفاعل.",
        "angry": "النبرة حادة لكن مفهومة ومنضبطة.",
        "sad": "النبرة هادية وتميل للحزن الخفيف.",
        "frustrated": "النبرة متضايقة بشكل واضح لكن بدون صراخ.",
        "concerned": "النبرة قلقة واهتمامها واضح.",
        "empathetic": "النبرة متعاطفة وداعمة.",
    }

    d = (dialect or "cairene").strip().lower()
    e = (emotion or "neutral").strip().lower()
    spoken_text = _cleanup_spoken_text(transform_to_dialect(text, d))
    spoken_text = _apply_human_voice_cues(spoken_text, e)

    return (
        "انت مولد صوت عربي للمصري فقط. "
        f"{dialect_instruction.get(d, dialect_instruction['cairene'])} "
        f"{emotion_instruction.get(e, emotion_instruction['neutral'])} "
        "لا تضيف اي كلمات خارج النص، ولا اي مقدمات. "
        f"النص المطلوب نطقه: {spoken_text}"
    )


def _set_model_arg(config: Any, key: str, value: str):
    model_args = getattr(config, "model_args", None)
    if isinstance(model_args, dict):
        model_args[key] = value
        return

    if model_args is not None and hasattr(model_args, key):
        setattr(model_args, key, value)


def _load_wav_with_stdlib(file_path: Path):
    if torch is None:
        raise RuntimeError("Torch is required to load XTTS reference audio")

    with wave.open(str(file_path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        n_frames = wav_file.getnframes()
        raw = wav_file.readframes(n_frames)

    if sample_width != 2:
        raise RuntimeError("Only 16-bit PCM WAV is supported")

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    return torch.from_numpy(audio).unsqueeze(0), sample_rate


def _patch_torchaudio_load_for_xtts():
    if not XTTS_RUNTIME_AVAILABLE:
        return

    try:
        import torchaudio  # type: ignore[reportMissingImports]
    except Exception:
        return

    if getattr(torchaudio, "_servia_safe_load_patch", False):
        return

    original_load = torchaudio.load

    def _safe_load(audiopath, *args, **kwargs):
        try:
            return original_load(audiopath, *args, **kwargs)
        except Exception as original_error:
            path = Path(str(audiopath))
            if path.suffix.lower() == ".wav":
                try:
                    return _load_wav_with_stdlib(path)
                except Exception:
                    pass
            raise original_error

    torchaudio.load = _safe_load
    setattr(torchaudio, "_servia_safe_load_patch", True)


def _resolve_xtts_checkpoint(model_dir: Path) -> Optional[Path]:
    preferred = [
        model_dir / "checkpoint_25500-001.pth",
        model_dir / "checkpoint_25500.pth",
        model_dir / "model.pth",
    ]
    for candidate in preferred:
        if candidate.exists():
            return candidate

    checkpoints = sorted(model_dir.glob("checkpoint*.pth"))
    return checkpoints[0] if checkpoints else None


def _resolve_xtts_reference_audio(gender: str = "female") -> Optional[Path]:
    g = (gender or "female").strip().lower()

    gender_specific = XTTS_REFERENCE_AUDIO_MALE if g == "male" else XTTS_REFERENCE_AUDIO_FEMALE
    if gender_specific:
        candidate = Path(gender_specific)
        if candidate.exists():
            return candidate

    if XTTS_REFERENCE_AUDIO:
        configured = Path(XTTS_REFERENCE_AUDIO)
        if configured.exists():
            return configured

    for candidate in DEFAULT_XTTS_REFERENCE_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def _xtts_model_files_status(model_dir: Path) -> dict[str, bool]:
    checkpoint = _resolve_xtts_checkpoint(model_dir)
    return {
        "config": (model_dir / "config.json").exists(),
        "checkpoint": checkpoint is not None,
        "vocab": (model_dir / "vocab.json").exists(),
        "dvae": (model_dir / "dvae.pth").exists(),
        "mel_stats": (model_dir / "mel_stats.pth").exists(),
    }


def _is_xtts_ready() -> bool:
    if not XTTS_RUNTIME_AVAILABLE:
        return False
    files = _xtts_model_files_status(XTTS_MODEL_DIR)
    return XTTS_MODEL_DIR.exists() and all(files.values())


def _is_provider_temporarily_disabled(provider: str) -> bool:
    return _local_provider_disabled_until.get(provider, 0.0) > time.time()


def _mark_local_provider_failed(provider: str, reason: str) -> None:
    cooldown_seconds = (
        XTTS_FAILURE_COOLDOWN_SECONDS
        if provider == "xtts"
        else CHATTERBOX_FAILURE_COOLDOWN_SECONDS
    )
    _local_provider_disabled_until[provider] = time.time() + float(cooldown_seconds)
    logger.warning(
        "Temporarily disabling local TTS provider '%s' for %.0fs: %s",
        provider,
        cooldown_seconds,
        reason,
    )


def _chatterbox_model_files_status(model_dir: Path) -> dict[str, bool]:
    single_required = ["s3gen.pt", "ve.pt", "tokenizer.json"]
    multilingual_required = ["cfg_scale_weights.pt", "s3gen.pt", "tokenizer.json", "ve.pt"]
    return {
        "single_compatible": all((model_dir / name).exists() for name in single_required),
        "multilingual_compatible": all((model_dir / name).exists() for name in multilingual_required),
    }


def _is_chatterbox_ready() -> bool:
    if not CHATTERBOX_RUNTIME_AVAILABLE:
        return False

    if _is_provider_temporarily_disabled("chatterbox"):
        return False

    if CHATTERBOX_MODEL_DIR.exists():
        status = _chatterbox_model_files_status(CHATTERBOX_MODEL_DIR)
        if status["single_compatible"] or status["multilingual_compatible"]:
            return True

    return CHATTERBOX_ALLOW_REMOTE


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _patch_xtts_generation_runtime(model: Any) -> None:
    gpt = getattr(model, "gpt", None)
    if gpt is None:
        return

    tokenizer = getattr(model, "tokenizer", None) or getattr(gpt, "tokenizer", None)
    eos_id = _safe_int(getattr(tokenizer, "eos_token_id", None)) if tokenizer is not None else None
    pad_id = _safe_int(getattr(tokenizer, "pad_token_id", None)) if tokenizer is not None else None

    if tokenizer is not None and (pad_id is None or (eos_id is not None and pad_id == eos_id)):
        try:
            if hasattr(tokenizer, "add_special_tokens"):
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                if hasattr(gpt, "resize_token_embeddings") and hasattr(tokenizer, "__len__"):
                    gpt.resize_token_embeddings(len(tokenizer))
        except Exception:
            pass
        pad_id = _safe_int(getattr(tokenizer, "pad_token_id", None))

    if pad_id is None or (eos_id is not None and pad_id == eos_id):
        candidates = [
            _safe_int(getattr(tokenizer, "unk_token_id", None)) if tokenizer is not None else None,
            1,
            2,
            3,
            0,
        ]
        for candidate in candidates:
            if candidate is not None and (eos_id is None or candidate != eos_id):
                pad_id = candidate
                break

    safe_pad_id = int(pad_id if pad_id is not None else 0)
    if eos_id is not None and safe_pad_id == eos_id:
        safe_pad_id = 1 if eos_id != 1 else 0

    safe_token_id = safe_pad_id
    if safe_token_id == 0 or (eos_id is not None and safe_token_id == eos_id):
        safe_token_id = 1 if eos_id != 1 else 2

    if tokenizer is not None:
        try:
            tokenizer.pad_token_id = safe_pad_id
            tokenizer.padding_side = "right"
        except Exception:
            pass

    patch_targets: list[Any] = []
    for target in [gpt, getattr(gpt, "gpt", None), getattr(gpt, "gpt_inference", None)]:
        if target is not None and callable(getattr(target, "generate", None)) and target not in patch_targets:
            patch_targets.append(target)

    def _target_device(target: Any) -> Any:
        if torch is None:
            return None
        try:
            return next(target.parameters()).device
        except Exception:
            return getattr(target, "device", None)

    def _target_vocab_size(target: Any) -> int:
        config = getattr(target, "config", None)
        size = _safe_int(getattr(config, "vocab_size", None)) or 0
        if size <= 0:
            try:
                embeds = target.get_input_embeddings() if hasattr(target, "get_input_embeddings") else None
                weight = getattr(embeds, "weight", None)
                if weight is not None:
                    size = int(weight.shape[0])
            except Exception:
                size = 0
        return size

    for target in patch_targets:
        for cfg in [getattr(target, "config", None), getattr(target, "generation_config", None)]:
            if cfg is None:
                continue
            try:
                setattr(cfg, "pad_token_id", safe_pad_id)
            except Exception:
                pass
            if eos_id is not None:
                try:
                    setattr(cfg, "eos_token_id", eos_id)
                except Exception:
                    pass

        if getattr(target, "_servia_generate_patched", False):
            continue

        original_generate = getattr(target, "generate", None)
        if not callable(original_generate):
            continue

        vocab_size = _target_vocab_size(target)
        target_safe_token_id = safe_token_id
        if vocab_size > 0 and target_safe_token_id >= vocab_size:
            target_safe_token_id = max(1, min(vocab_size - 1, safe_pad_id))
        if target_safe_token_id == 0:
            target_safe_token_id = 1
        restore_token_kwargs = target is not gpt

        def _safe_generate(
            *args: Any,
            _target: Any = target,
            _original_generate: Any = original_generate,
            _vocab_size: int = vocab_size,
            _safe_token_id: int = target_safe_token_id,
            _restore_token_kwargs: bool = restore_token_kwargs,
            **kwargs: Any,
        ):
            call_pad_id = _safe_int(kwargs.pop("pad_token_id", None))
            call_eos_id = _safe_int(kwargs.pop("eos_token_id", None))
            call_bos_id = _safe_int(kwargs.get("bos_token_id", None))
            effective_eos_id = call_eos_id if call_eos_id is not None else eos_id
            effective_pad_id = safe_pad_id
            if effective_eos_id is not None and effective_pad_id == effective_eos_id:
                effective_pad_id = 1 if effective_eos_id != 1 else 0
            if call_pad_id is not None and (effective_eos_id is None or call_pad_id != effective_eos_id):
                effective_pad_id = call_pad_id

            args_list: list[Any] = list(args)
            input_ids = kwargs.get("input_ids", kwargs.get("inputs"))
            input_key = "input_ids" if "input_ids" in kwargs else ("inputs" if "inputs" in kwargs else "")
            input_from_args = False
            if input_ids is None and args_list:
                candidate = args_list[0]
                if hasattr(candidate, "shape"):
                    input_ids = candidate
                    input_from_args = True

            if input_ids is not None and torch is not None:
                try:
                    target_device = _target_device(_target) or getattr(input_ids, "device", None)
                    if target_device is not None:
                        input_ids = input_ids.to(device=target_device, dtype=torch.long)
                    else:
                        input_ids = input_ids.to(dtype=torch.long)

                    unsafe = (input_ids <= 0) | (input_ids >= _vocab_size) if _vocab_size > 0 else input_ids <= 0
                    if unsafe.any():
                        input_ids = torch.where(unsafe, torch.full_like(input_ids, _safe_token_id), input_ids)
                    if input_ids.numel() > 0:
                        flat_input_ids = input_ids.reshape(-1)
                        if int(flat_input_ids[0].item()) == 0:
                            flat_input_ids[0] = _safe_token_id
                        input_ids = flat_input_ids.reshape(input_ids.shape)

                    input_ids = input_ids.contiguous()
                    if input_from_args:
                        args_list[0] = input_ids
                    elif input_key:
                        kwargs[input_key] = input_ids
                    else:
                        kwargs["input_ids"] = input_ids

                    attention_mask = kwargs.get("attention_mask")
                    audio_generation = call_bos_id is not None or call_eos_id is not None
                    if attention_mask is None and audio_generation:
                        kwargs["attention_mask"] = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
                    elif attention_mask is None:
                        kwargs["attention_mask"] = input_ids.ne(int(safe_pad_id)).to(
                            dtype=torch.long,
                            device=input_ids.device,
                        )
                    elif hasattr(attention_mask, "to"):
                        kwargs["attention_mask"] = cast(Any, attention_mask).to(
                            device=input_ids.device,
                            dtype=torch.long,
                        )
                except Exception:
                    try:
                        if kwargs.get("attention_mask") is None:
                            kwargs["attention_mask"] = torch.ones_like(input_ids, dtype=torch.long)
                    except Exception:
                        pass

            if _restore_token_kwargs:
                kwargs["pad_token_id"] = int(effective_pad_id)
                if effective_eos_id is not None:
                    kwargs["eos_token_id"] = int(effective_eos_id)

            try:
                return _original_generate(*args_list, **kwargs)
            except RuntimeError:
                if torch is not None and input_ids is not None:
                    try:
                        safe_input_ids = cast(Any, input_ids)
                        cpu_args = list(args_list)
                        cpu_kwargs = dict(kwargs)
                        if input_from_args and cpu_args:
                            cpu_args[0] = safe_input_ids.to("cpu")
                        elif input_key:
                            cpu_kwargs[input_key] = safe_input_ids.to("cpu")
                        else:
                            cpu_kwargs["input_ids"] = safe_input_ids.to("cpu")
                        attention_mask = cpu_kwargs.get("attention_mask")
                        if hasattr(attention_mask, "to"):
                            cpu_kwargs["attention_mask"] = cast(Any, attention_mask).to("cpu")
                        if hasattr(_target, "to"):
                            _target.to("cpu")
                        if hasattr(torch, "cuda") and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        return _original_generate(*cpu_args, **cpu_kwargs)
                    except Exception:
                        pass
                raise

        target.generate = _safe_generate
        setattr(target, "_servia_generate_patched", True)



def _get_sample_rate_from_config(config: Any) -> int:
    audio_config = getattr(config, "audio", None)
    if isinstance(audio_config, dict):
        return int(audio_config.get("output_sample_rate") or audio_config.get("sample_rate") or 24000)

    if audio_config is not None:
        for attr in ["output_sample_rate", "sample_rate"]:
            value = getattr(audio_config, attr, None)
            if value:
                return int(value)

    return 24000


def _load_xtts_model_sync():
    global _xtts_model, _xtts_device, _xtts_sample_rate, _xtts_model_error

    if _xtts_model is not None:
        return _xtts_model

    with _xtts_lock:
        if _xtts_model is not None:
            return _xtts_model

        if not XTTS_RUNTIME_AVAILABLE:
            _xtts_model_error = f"XTTS runtime unavailable: {XTTS_IMPORT_ERROR or TORCH_IMPORT_ERROR or 'missing deps'}"
            raise RuntimeError(_xtts_model_error)

        model_dir = XTTS_MODEL_DIR
        config_path = model_dir / "config.json"
        checkpoint_path = _resolve_xtts_checkpoint(model_dir)
        vocab_path = model_dir / "vocab.json"
        dvae_path = model_dir / "dvae.pth"
        mel_stats_path = model_dir / "mel_stats.pth"

        missing = [
            str(p)
            for p in [config_path, vocab_path, dvae_path, mel_stats_path]
            if not p.exists()
        ]
        if checkpoint_path is None:
            missing.append("checkpoint*.pth")
        if missing:
            _xtts_model_error = "Missing XTTS files: " + ", ".join(missing)
            raise RuntimeError(_xtts_model_error)

        _patch_torchaudio_load_for_xtts()

        try:
            if XttsConfig is None or Xtts is None:
                raise RuntimeError("XTTS classes are unavailable")
            config = XttsConfig()
            config.load_json(str(config_path))

            _set_model_arg(config, "tokenizer_file", str(vocab_path))
            _set_model_arg(config, "dvae_checkpoint", str(dvae_path))
            _set_model_arg(config, "mel_norm_file", str(mel_stats_path))
            _set_model_arg(config, "xtts_checkpoint", str(checkpoint_path))

            model = Xtts.init_from_config(config)
            model.load_checkpoint(
                config,
                checkpoint_dir=str(model_dir),
                checkpoint_path=str(checkpoint_path),
                vocab_path=str(vocab_path),
                use_deepspeed=XTTS_USE_DEEPSPEED,
            )

            preferred_device = "cuda" if _gpu_requested(XTTS_DEVICE) else "cpu"
            _xtts_device = preferred_device
            if hasattr(model, "to"):
                try:
                    model.to(preferred_device)
                except Exception:
                    _xtts_device = "cpu"
                    model.to("cpu")
            if hasattr(model, "eval"):
                model.eval()

            # Apply half-precision at load time if CUDA + half_precision enabled
            # This ensures warmup and all subsequent operations have matching dtypes
            if XTTS_HALF_PRECISION and _xtts_device == "cuda" and torch is not None:
                try:
                    # Convert entire model to half, then restore norm layers to float32
                    model.half()
                    for m in model.modules():
                        if isinstance(m, torch.nn.LayerNorm) or isinstance(m, torch.nn.GroupNorm):
                            m.float()
                    speaker_encoder = getattr(getattr(model, "hifigan_decoder", None), "speaker_encoder", None)
                    if speaker_encoder is not None and hasattr(speaker_encoder, "float"):
                        # XTTS feeds reference audio to the speaker encoder as float32.
                        # Keep this submodule in float32, then cast its embedding output below.
                        speaker_encoder.float()
                    logger.info("XTTS half-precision applied (norms in float32)")
                except Exception as half_err:
                    logger.warning("XTTS half-precision conversion skipped: %s", half_err)

            _patch_xtts_generation_runtime(model)

            _xtts_sample_rate = _get_sample_rate_from_config(config)
            _xtts_model = model
            _xtts_model_error = None

            logger.info(
                "XTTS loaded from %s on %s (sr=%s, half_precision=%s, chunking=%s)",
                str(model_dir),
                _xtts_device,
                _xtts_sample_rate,
                XTTS_HALF_PRECISION,
                XTTS_ENABLE_CHUNKING,
            )
        except Exception as e:
            _xtts_model_error = str(e)
            raise

    return _xtts_model


def _prepare_chatterbox_reference_if_needed(model: Any, gender: str = "female") -> None:
    global _chatterbox_prepared_reference

    reference_audio = _resolve_xtts_reference_audio(gender)
    if reference_audio is None:
        return

    ref_key = str(reference_audio.resolve())
    if _chatterbox_prepared_reference == ref_key:
        return

    prepare_fn = getattr(model, "prepare_conditionals", None)
    if not callable(prepare_fn):
        return

    try:
        signature = inspect.signature(prepare_fn)
        kwargs: dict[str, Any] = {}
        if "audio_prompt_path" in signature.parameters:
            kwargs["audio_prompt_path"] = str(reference_audio)
        if "language" in signature.parameters:
            kwargs["language"] = CHATTERBOX_LANGUAGE
        if "exaggeration" in signature.parameters:
            kwargs["exaggeration"] = 0.45
        if "cfg_weight" in signature.parameters:
            kwargs["cfg_weight"] = 0.55

        if kwargs:
            prepare_fn(**kwargs)
            _chatterbox_prepared_reference = ref_key
    except Exception:
        # Reference preparation is optional; generation can still proceed.
        return


def _load_chatterbox_model_sync():
    global _chatterbox_model, _chatterbox_device, _chatterbox_sample_rate, _chatterbox_model_error

    if _chatterbox_model is not None:
        return _chatterbox_model

    with _chatterbox_lock:
        if _chatterbox_model is not None:
            return _chatterbox_model

        if not CHATTERBOX_RUNTIME_AVAILABLE:
            _chatterbox_model_error = f"Chatterbox runtime unavailable: {CHATTERBOX_IMPORT_ERROR or 'missing deps'}"
            raise RuntimeError(_chatterbox_model_error)

        preferred_device = "cpu"
        _chatterbox_device = preferred_device
        model = None

        try:
            if CHATTERBOX_MODEL_DIR.exists():
                status = _chatterbox_model_files_status(CHATTERBOX_MODEL_DIR)
                if CHATTERBOX_USE_MULTILINGUAL and status["multilingual_compatible"] and ChatterboxMultilingualTTS is not None:
                    model = ChatterboxMultilingualTTS.from_local(str(CHATTERBOX_MODEL_DIR), preferred_device)
                elif status["single_compatible"] and ChatterboxTTS is not None:
                    model = ChatterboxTTS.from_local(str(CHATTERBOX_MODEL_DIR), preferred_device)
                elif status["multilingual_compatible"] and ChatterboxMultilingualTTS is not None:
                    model = ChatterboxMultilingualTTS.from_local(str(CHATTERBOX_MODEL_DIR), preferred_device)
                else:
                    raise RuntimeError("Chatterbox model directory exists but required files are missing")
            elif CHATTERBOX_ALLOW_REMOTE:
                if CHATTERBOX_USE_MULTILINGUAL and ChatterboxMultilingualTTS is not None:
                    model = ChatterboxMultilingualTTS.from_pretrained(preferred_device)
                elif ChatterboxTTS is not None:
                    model = ChatterboxTTS.from_pretrained(preferred_device)
                else:
                    raise RuntimeError("Chatterbox classes unavailable")
            else:
                raise RuntimeError(
                    "Chatterbox model dir not found. Set CHATTERBOX_MODEL_DIR or enable CHATTERBOX_ALLOW_REMOTE=1"
                )

            if model is None:
                raise RuntimeError("Unable to initialize Chatterbox model")

            _chatterbox_model = model
            _chatterbox_sample_rate = int(getattr(model, "sr", 24000) or 24000)
            _chatterbox_model_error = None
            _prepare_chatterbox_reference_if_needed(model, "female")

            logger.info(
                "Chatterbox loaded from %s on %s (sr=%s)",
                str(CHATTERBOX_MODEL_DIR),
                _chatterbox_device,
                _chatterbox_sample_rate,
            )
        except Exception as e:
            _chatterbox_model_error = str(e)
            raise

    return _chatterbox_model


def _get_xtts_conditioning_sync(model: Any, reference_audio: Path):
    cache_key = f"{reference_audio.resolve()}::{_xtts_device}::fp16={int(XTTS_HALF_PRECISION and _xtts_device == 'cuda')}"
    cached = _xtts_conditioning_cache.get(cache_key)
    if cached is not None:
        return cached

    conditioning_context = _xtts_autocast_context()

    with conditioning_context:
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            audio_path=str(reference_audio),
            gpt_cond_len=max(3, XTTS_GPT_COND_LEN),
            max_ref_length=max(6, XTTS_MAX_REF_LEN),
            sound_norm_refs=False,
        )

    if XTTS_HALF_PRECISION and _xtts_device == "cuda" and torch is not None:
        gpt_cond_latent = gpt_cond_latent.half()
        speaker_embedding = speaker_embedding.half()

    if torch is not None:
        for name, tensor in [("gpt_cond_latent", gpt_cond_latent), ("speaker_embedding", speaker_embedding)]:
            if hasattr(tensor, "to"):
                try:
                    moved = tensor.to(_xtts_device)
                    if name == "gpt_cond_latent":
                        gpt_cond_latent = moved
                    else:
                        speaker_embedding = moved
                except Exception:
                    pass

    _xtts_conditioning_cache[cache_key] = (gpt_cond_latent, speaker_embedding)
    return gpt_cond_latent, speaker_embedding


def _float_wav_to_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    audio = np.asarray(samples, dtype=np.float32).squeeze()
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767.0).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())
    return buffer.getvalue()


def _to_numpy_audio_array(wav: Any) -> np.ndarray:
    """Normalize model outputs (tensor/list/tuple) to a float numpy array."""
    candidate = wav
    if isinstance(candidate, tuple) and candidate:
        candidate = candidate[0]
    dynamic_candidate = cast(Any, candidate)
    if hasattr(dynamic_candidate, "detach"):
        dynamic_candidate = dynamic_candidate.detach().cpu().numpy()
    return np.asarray(dynamic_candidate)


def safe_inference(model: Any, **kwargs: Any) -> Any:
    """Safely call XTTS inference with explicit logging and error handling."""
    inference_fn = getattr(model, "inference", None)
    if not callable(inference_fn):
        raise RuntimeError("XTTS model does not expose inference()")

    safe_kwargs = dict(kwargs)
    # Remove generation-only kwargs that conflict with inference().
    safe_kwargs.pop("pad_token_id", None)
    safe_kwargs.pop("attention_mask", None)

    text = safe_kwargs.get("text", "")
    logger.debug(
        "XTTS inference start: device=%s, text_len=%d, has_gpt_cond=%s, has_speaker=%s",
        _xtts_device,
        len(text) if text else 0,
        "gpt_cond_latent" in safe_kwargs,
        "speaker_embedding" in safe_kwargs,
    )

    try:
        result = inference_fn(**safe_kwargs)
        logger.debug("XTTS inference success")
        return result
    except Exception as e:
        logger.error(
            "XTTS inference failed: %s: %s (device=%s, text_len=%d)",
            type(e).__name__,
            str(e)[:150],
            _xtts_device,
            len(text) if text else 0,
        )
        raise


def _synthesize_sync_xtts(
    text: str,
    dialect: str = "cairene",
    gender: str = "female",
    emotion: str = "neutral",
) -> bytes:
    global _xtts_device
    model = _load_xtts_model_sync()
    reference_audio = _resolve_xtts_reference_audio(gender)
    if reference_audio is None:
        raise RuntimeError(
            "XTTS requires reference audio. Set XTTS_REFERENCE_AUDIO or XTTS_REFERENCE_AUDIO_FEMALE/XTTS_REFERENCE_AUDIO_MALE."
        )

    text_input = _cleanup_spoken_text(transform_to_dialect(text, dialect))
    text_input = _apply_human_voice_cues(text_input, emotion)
    text_input_safe = _validate_xtts_input_text(text_input)
    text_input_plain = _validate_xtts_input_text(_apply_human_voice_cues(text, emotion))
    temperature = XTTS_EMOTION_TEMPERATURE.get((emotion or "neutral").lower(), XTTS_BASE_TEMPERATURE)

    candidates = [
        text_input,
        text_input_safe,
        text_input_plain,
    ]

    output = None
    last_error: Optional[Exception] = None
    cpu_retry_done = False

    with _xtts_infer_lock:
        for candidate in candidates:
            candidate_text = _cleanup_spoken_text(candidate)
            if not candidate_text:
                logger.debug("Skipping empty candidate text")
                continue

            for attempt in range(2):
                try:
                    logger.debug(
                        "XTTS inference attempt: candidate=%d, attempt=%d, text_len=%d, device=%s",
                        candidates.index(candidate),
                        attempt + 1,
                        len(candidate_text),
                        _xtts_device,
                    )
                    gpt_cond_latent, speaker_embedding = _get_xtts_conditioning_sync(model, reference_audio)
                    inference_kwargs = {
                        "text": candidate_text,
                        "language": XTTS_LANGUAGE,
                        "gpt_cond_latent": gpt_cond_latent,
                        "speaker_embedding": speaker_embedding,
                        "temperature": float(temperature),
                        "enable_text_splitting": bool(XTTS_ENABLE_CHUNKING),
                    }

                    with torch.inference_mode() if torch is not None else nullcontext():
                        with _xtts_autocast_context():
                            output = safe_inference(model, **inference_kwargs)

                    logger.info("XTTS inference succeeded on attempt %d", attempt + 1)
                    last_error = None
                    break
                except RuntimeError as e:
                    last_error = e
                    logger.error(
                        "XTTS RuntimeError (attempt %d): %s: %s",
                        attempt + 1,
                        type(e).__name__,
                        str(e)[:200],
                    )
                    if _xtts_device == "cuda" or _is_cuda_oom_error(e) or _is_cuda_device_assert_error(e):
                        _cleanup_inference_memory(force=True)
                        if _xtts_device == "cuda" and not cpu_retry_done:
                            logger.warning("XTTS CUDA failure; switching to CPU retry")
                            _move_xtts_to_cpu(model)
                            _xtts_device = "cpu"
                            _xtts_conditioning_cache.clear()
                            cpu_retry_done = True
                            continue
                    break
                except Exception as e:
                    last_error = e
                    break

            if output is not None:
                break

    if output is None:
        raise RuntimeError(f"XTTS inference failed after retries: {last_error}")

    wav = output.get("wav") if isinstance(output, dict) else output
    wav_array = _to_numpy_audio_array(wav)

    _cleanup_inference_memory(force=False)

    return _float_wav_to_bytes(wav_array, _xtts_sample_rate)


def _move_xtts_to_cpu(model: Any) -> None:
    try:
        if hasattr(model, "to"):
            model.to("cpu")
    except Exception:
        pass
    try:
        if hasattr(model, "cpu"):
            model.cpu()
    except Exception:
        pass


def _synthesize_sync_chatterbox(
    text: str,
    dialect: str = "cairene",
    gender: str = "female",
    emotion: str = "neutral",
) -> bytes:
    model = _load_chatterbox_model_sync()

    text_input = _cleanup_spoken_text(transform_to_dialect(text, dialect))
    text_input = _apply_human_voice_cues(text_input, emotion)
    if not text_input:
        text_input = _cleanup_spoken_text(text)
    if not text_input:
        raise RuntimeError("Empty text for Chatterbox synthesis")

    generate_fn = getattr(model, "generate", None)
    if not callable(generate_fn):
        raise RuntimeError("Chatterbox model does not expose generate()")

    signature = inspect.signature(generate_fn)
    kwargs: dict[str, Any] = {}

    if "exaggeration" in signature.parameters:
        kwargs["exaggeration"] = 0.45
    if "cfg_weight" in signature.parameters:
        kwargs["cfg_weight"] = 0.55
    if "temperature" in signature.parameters:
        kwargs["temperature"] = 0.6
    if "language_id" in signature.parameters:
        kwargs["language_id"] = CHATTERBOX_LANGUAGE

    reference_audio = _resolve_xtts_reference_audio(gender)
    if reference_audio is not None and "audio_prompt_path" in signature.parameters:
        kwargs["audio_prompt_path"] = str(reference_audio)

    wav = generate_fn(text_input, **kwargs)
    wav_array = _to_numpy_audio_array(wav)

    _cleanup_inference_memory()

    return _float_wav_to_bytes(wav_array, _chatterbox_sample_rate)


def _synthesize_sync_gtts(text: str, dialect: str = "cairene", emotion: str = "neutral") -> bytes:
    dialectal_text = _cleanup_spoken_text(transform_to_dialect(text, dialect))
    dialectal_text = _apply_human_voice_cues(dialectal_text, emotion)
    if not dialectal_text:
        dialectal_text = _cleanup_spoken_text(text)
    if not dialectal_text:
        raise RuntimeError("No text available for gTTS synthesis")

    rate = float(get_dialect_prosody(dialect).get("rate", "1.0"))
    if emotion in {"sad", "empathetic", "concerned"}:
        rate = min(rate, 0.9)
    elif emotion in {"excited", "angry"}:
        rate = max(rate, 1.0)

    candidates: list[str] = []
    for candidate in [dialectal_text, _cleanup_spoken_text(text), "مرحبا"]:
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    last_error: Optional[Exception] = None
    for candidate in candidates:
        try:
            tts = gTTS(text=candidate, lang="ar", slow=(rate < 0.9))
            audio_buf = io.BytesIO()
            tts.write_to_fp(audio_buf)
            audio_buf.seek(0)
            return audio_buf.read()
        except Exception as e:
            last_error = e

    raise RuntimeError(f"gTTS synthesis failed: {last_error}")


def _synthesize_sync_gemini(
    text: str,
    dialect: str = "cairene",
    gender: str = "female",
    emotion: str = "neutral",
) -> tuple[bytes, str]:
    if not GEMINI_RUNTIME_AVAILABLE or genai is None or genai_types is None:
        raise RuntimeError(f"Gemini runtime unavailable: {GEMINI_IMPORT_ERROR or 'google-genai missing'}")

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY")

    gemini_client_module = cast(Any, genai)
    gemini_types_module = cast(Any, genai_types)

    client = gemini_client_module.Client(api_key=api_key)
    voice_name = _select_gemini_voice(dialect, gender)
    prompt = _build_gemini_tts_prompt(text, dialect, emotion)

    contents = [
        gemini_types_module.Content(
            role="user",
            parts=[gemini_types_module.Part.from_text(text=prompt)],
        )
    ]

    config = gemini_types_module.GenerateContentConfig(
        temperature=float(_clamp_float(GEMINI_TTS_TEMPERATURE, 0.0, 2.0)),
        response_modalities=["audio"],
        speech_config=gemini_types_module.SpeechConfig(
            voice_config=gemini_types_module.VoiceConfig(
                prebuilt_voice_config=gemini_types_module.PrebuiltVoiceConfig(
                    voice_name=voice_name
                )
            )
        ),
    )

    audio_chunks: list[bytes] = []
    mime_type = "audio/wav"

    for chunk in client.models.generate_content_stream(
        model=GEMINI_TTS_MODEL,
        contents=contents,
        config=config,
    ):
        parts = getattr(chunk, "parts", None) or []
        for part in parts:
            inline_data = getattr(part, "inline_data", None)
            if inline_data is None:
                continue
            data = getattr(inline_data, "data", None)
            if not data:
                continue
            audio_chunks.append(data)
            mt = getattr(inline_data, "mime_type", None)
            if mt:
                mime_type = mt

    if not audio_chunks:
        raise RuntimeError("Gemini returned no audio chunks")

    raw_audio = b"".join(audio_chunks)

    ext = mimetypes.guess_extension(mime_type or "") or ""
    if ext in {".mp3", ".mpeg"} or "mpeg" in (mime_type or "").lower():
        return raw_audio, "mp3"

    if ext == ".wav" or "wav" in (mime_type or "").lower():
        return raw_audio, "wav"

    return _convert_raw_audio_to_wav(raw_audio, mime_type), "wav"


async def _stream_edge_tts(
    text: str,
    dialect: str = "cairene",
    gender: str = "female",
    emotion: str = "neutral",
):
    if not EDGE_TTS_AVAILABLE or edge_tts is None:
        raise RuntimeError("Edge TTS is not available")

    edge_tts_module = cast(Any, edge_tts)

    config = _build_voice_config(dialect, gender, emotion)
    dialectal_text = _cleanup_spoken_text(transform_to_dialect(text, dialect))
    dialectal_text = _apply_human_voice_cues(dialectal_text, emotion)
    communicator = edge_tts_module.Communicate(
        text=dialectal_text,
        voice=config["voice"],
        rate=config["rate"],
        pitch=config["pitch"],
        volume=config["volume"],
    )

    async for chunk in communicator.stream():
        if chunk.get("type") == "audio":
            yield chunk.get("data", b"")


def _is_local_provider_ready(provider: str) -> bool:
    if provider == "xtts":
        return _is_xtts_ready() and not _is_provider_temporarily_disabled("xtts")
    if provider == "chatterbox":
        return _is_chatterbox_ready()
    return False


def _resolve_provider() -> str:
    requested = TTS_PROVIDER
    if requested not in {"auto", "dual", "xtts", "chatterbox", "gemini", "edge", "gtts"}:
        requested = "auto"

    xtts_ready = _is_local_provider_ready("xtts")
    chatterbox_ready = _is_local_provider_ready("chatterbox")
    gemini_ready = _is_gemini_ready()

    if requested == "dual":
        requested = "auto"

    if requested == "xtts":
        if xtts_ready:
            return "xtts"
        if chatterbox_ready:
            return "chatterbox"
        if gemini_ready:
            return "gemini"
        return "edge" if EDGE_TTS_AVAILABLE else "gtts"

    if requested == "chatterbox":
        if chatterbox_ready:
            return "chatterbox"
        if xtts_ready:
            return "xtts"
        if gemini_ready:
            return "gemini"
        return "edge" if EDGE_TTS_AVAILABLE else "gtts"

    if requested == "edge":
        return "edge" if EDGE_TTS_AVAILABLE else "gtts"

    if requested == "gemini":
        return "gemini" if gemini_ready else ("edge" if EDGE_TTS_AVAILABLE else "gtts")

    if requested == "gtts":
        return "gtts"

    if _is_local_provider_ready("xtts") and _xtts_device == "cuda":
        return "xtts"

    for local_provider in TTS_AUTO_LOCAL_ORDER:
        if _is_local_provider_ready(local_provider):
            return local_provider

    if _is_local_provider_ready("xtts"):
        return "xtts"

    if gemini_ready:
        return "gemini"
    if EDGE_TTS_AVAILABLE:
        return "edge"
    return "gtts"


async def _synthesize_local_provider(
    provider: str,
    text: str,
    dialect: str,
    gender: str,
    emotion: str,
) -> tuple[bytes, str]:
    global _xtts_device
    cached = _get_cached_tts(provider, text, dialect, gender, emotion)
    if cached is not None:
        return cached

    if provider == "xtts":
        try:
            audio = await asyncio.wait_for(
                asyncio.to_thread(_synthesize_sync_xtts, text, dialect, gender, emotion),
                timeout=float(XTTS_TIMEOUT_SECONDS),
            )
        except asyncio.TimeoutError as timeout_error:
            retry_timeout = max(float(XTTS_TIMEOUT_SECONDS) + 25.0, float(XTTS_TIMEOUT_SECONDS) * 1.5)
            try:
                audio = await asyncio.wait_for(
                    asyncio.to_thread(_synthesize_sync_xtts, text, dialect, gender, emotion),
                    timeout=retry_timeout,
                )
            except asyncio.TimeoutError as retry_error:
                raise RuntimeError(
                    f"XTTS timed out after {XTTS_TIMEOUT_SECONDS}s (retry {int(retry_timeout)}s)"
                ) from retry_error
            except Exception as retry_error:
                raise RuntimeError(f"XTTS retry failed: {retry_error}") from retry_error
        except RuntimeError as runtime_error:
            if _is_cuda_oom_error(runtime_error) or _is_cuda_device_assert_error(runtime_error):
                _cleanup_inference_memory(force=True)
                if _xtts_device == "cuda":
                    logger.warning("XTTS CUDA failure; switching to CPU retry")
                    _xtts_device = "cpu"
                    try:
                        audio = await asyncio.wait_for(
                            asyncio.to_thread(_synthesize_sync_xtts, text, dialect, gender, emotion),
                            timeout=max(float(XTTS_TIMEOUT_SECONDS), 30.0),
                        )
                    except Exception as retry_error:
                        raise RuntimeError(f"XTTS CPU retry failed after CUDA OOM: {retry_error}") from retry_error
                else:
                    raise
            else:
                raise
        audio_format = "wav"
    elif provider == "chatterbox":
        try:
            audio = await asyncio.wait_for(
                asyncio.to_thread(_synthesize_sync_chatterbox, text, dialect, gender, emotion),
                timeout=float(CHATTERBOX_TIMEOUT_SECONDS),
            )
        except asyncio.TimeoutError as timeout_error:
            raise RuntimeError(f"Chatterbox timed out after {CHATTERBOX_TIMEOUT_SECONDS}s") from timeout_error
        audio_format = "wav"
    else:
        raise RuntimeError(f"Unsupported local provider: {provider}")

    _put_cached_tts(provider, text, dialect, gender, emotion, audio, audio_format)
    return audio, audio_format


def get_tts_backend_status() -> dict:
    xtts_model_dir = XTTS_MODEL_DIR
    reference_audio = _resolve_xtts_reference_audio("female")
    reference_audio_male = _resolve_xtts_reference_audio("male")

    return {
        "requested_provider": TTS_PROVIDER,
        "effective_provider": _resolve_provider(),
        "use_gpu": USE_GPU,
        "cuda_visible_devices": CUDA_VISIBLE_DEVICES,
        "available_local_providers": [
            provider
            for provider in ["chatterbox", "xtts"]
            if _is_local_provider_ready(provider)
        ],
        "fallback_chain": ["chatterbox", "xtts", "gemini", "edge", "gtts"],
        "auto_local_order": list(TTS_AUTO_LOCAL_ORDER),
        "edge_available": EDGE_TTS_AVAILABLE,
        "gemini_runtime_available": GEMINI_RUNTIME_AVAILABLE,
        "gemini_runtime_error": GEMINI_IMPORT_ERROR or None,
        "gemini_api_key_configured": bool(os.getenv("GEMINI_API_KEY", "").strip()),
        "gemini_ready": _is_gemini_ready(),
        "gemini_model": GEMINI_TTS_MODEL,
        "gemini_voice_default": GEMINI_TTS_VOICE,
        "gemini_voice_by_dialect": {
            "cairene": GEMINI_TTS_VOICE_CAIRENE or GEMINI_TTS_VOICE,
            "saidi": GEMINI_TTS_VOICE_SAIDI or GEMINI_TTS_VOICE,
            "alexandrian": GEMINI_TTS_VOICE_ALEXANDRIAN or GEMINI_TTS_VOICE,
            "bedouin": GEMINI_TTS_VOICE_BEDOUIN or GEMINI_TTS_VOICE,
        },
        "xtts_runtime_available": XTTS_RUNTIME_AVAILABLE,
        "xtts_runtime_error": XTTS_IMPORT_ERROR or TORCH_IMPORT_ERROR or None,
        "xtts_model_dir": str(xtts_model_dir),
        "xtts_model_dir_exists": xtts_model_dir.exists(),
        "xtts_model_files": _xtts_model_files_status(xtts_model_dir),
        "xtts_loaded": _xtts_model is not None,
        "xtts_model_error": _xtts_model_error,
        "xtts_device": _xtts_device,
        "xtts_device_requested": XTTS_DEVICE,
        "xtts_half_precision": XTTS_HALF_PRECISION,
        "xtts_enable_chunking": XTTS_ENABLE_CHUNKING,
        "xtts_use_deepspeed": XTTS_USE_DEEPSPEED,
        "xtts_sample_rate": _xtts_sample_rate,
        "xtts_timeout_seconds": XTTS_TIMEOUT_SECONDS,
        "xtts_input_max_chars": XTTS_INPUT_MAX_CHARS,
        "xtts_empty_cache_interval_seconds": XTTS_EMPTY_CACHE_INTERVAL_SECONDS,
        "xtts_reference_audio": str(reference_audio) if reference_audio else None,
        "xtts_reference_audio_exists": reference_audio.exists() if reference_audio else False,
        "xtts_reference_audio_male": str(reference_audio_male) if reference_audio_male else None,
        "xtts_reference_audio_male_exists": reference_audio_male.exists() if reference_audio_male else False,
        "tts_human_style_enabled": TTS_HUMAN_STYLE_ENABLED,
        "chatterbox_enabled": _is_chatterbox_ready(),
        "chatterbox_runtime_available": CHATTERBOX_RUNTIME_AVAILABLE,
        "chatterbox_runtime_error": CHATTERBOX_IMPORT_ERROR or None,
        "chatterbox_model_dir": str(CHATTERBOX_MODEL_DIR),
        "chatterbox_model_dir_exists": CHATTERBOX_MODEL_DIR.exists(),
        "chatterbox_model_files": _chatterbox_model_files_status(CHATTERBOX_MODEL_DIR),
        "chatterbox_loaded": _chatterbox_model is not None,
        "chatterbox_model_error": _chatterbox_model_error,
        "chatterbox_device": _chatterbox_device,
        "chatterbox_sample_rate": _chatterbox_sample_rate,
        "chatterbox_language": CHATTERBOX_LANGUAGE,
        "chatterbox_timeout_seconds": CHATTERBOX_TIMEOUT_SECONDS,
        "provider_cooldown_seconds": {
            "xtts": max(0, int(_local_provider_disabled_until.get("xtts", 0.0) - time.time())),
            "chatterbox": max(0, int(_local_provider_disabled_until.get("chatterbox", 0.0) - time.time())),
        },
        "transformers_runtime_available": TRANSFORMERS_RUNTIME_AVAILABLE,
        "transformers_runtime_error": TRANSFORMERS_IMPORT_ERROR or None,
        "hf_generation_loader": "AutoModelForCausalLM",
        "tts_return_silence_on_failure": TTS_RETURN_SILENCE_ON_FAILURE,
        "tts_failure_silence_ms": TTS_FAILURE_SILENCE_MS,
    }


def _generate_silence_wav(duration_ms: int = 650, sample_rate: int = 24000) -> bytes:
    num_samples = max(1, int(sample_rate * (duration_ms / 1000.0)))
    silence = np.zeros(num_samples, dtype=np.float32)
    return _float_wav_to_bytes(silence, sample_rate)


async def synthesize_speech(
    text: str,
    dialect: str = "cairene",
    gender: str = "female",
    emotion: str = "neutral",
) -> tuple[bytes, str]:
    safe_text = _validate_xtts_input_text(text)
    provider = _resolve_provider()

    cached = _get_cached_tts(provider, safe_text, dialect, gender, emotion)
    if cached is not None:
        return cached

    if provider in {"xtts", "chatterbox"}:
        try:
            return await _synthesize_local_provider(provider, safe_text, dialect, gender, emotion)
        except Exception as e:
            _mark_local_provider_failed(provider, str(e))
            alternate_local = "chatterbox" if provider == "xtts" else "xtts"
            if _is_local_provider_ready(alternate_local):
                try:
                    return await _synthesize_local_provider(alternate_local, safe_text, dialect, gender, emotion)
                except Exception as alt_error:
                    _mark_local_provider_failed(alternate_local, str(alt_error))
            logger.warning(
                "Local TTS provider '%s' failed, trying Gemini fallback then edge/gTTS: %s",
                provider,
                str(e),
            )
            if _is_gemini_ready():
                provider = "gemini"
            else:
                provider = "edge" if EDGE_TTS_AVAILABLE else "gtts"

    if provider == "gemini":
        cached = _get_cached_tts("gemini", safe_text, dialect, gender, emotion)
        if cached is not None:
            return cached

        try:
            audio, audio_format = await asyncio.to_thread(
                _synthesize_sync_gemini,
                safe_text,
                dialect,
                gender,
                emotion,
            )
            _put_cached_tts("gemini", safe_text, dialect, gender, emotion, audio, audio_format)
            return audio, audio_format
        except Exception as e:
            logger.warning("Gemini TTS failed, falling back to edge/gTTS: %s", str(e))
            provider = "edge" if EDGE_TTS_AVAILABLE else "gtts"

    if provider == "edge":
        cached = _get_cached_tts("edge", safe_text, dialect, gender, emotion)
        if cached is not None:
            return cached

        try:
            chunks: list[bytes] = []
            async for chunk in _stream_edge_tts(safe_text, dialect, gender, emotion):
                if chunk:
                    chunks.append(chunk)
            audio = b"".join(chunks)
            if audio:
                _put_cached_tts("edge", safe_text, dialect, gender, emotion, audio, "mp3")
                return audio, "mp3"
        except Exception as e:
            logger.warning("Edge TTS failed, falling back to gTTS: %s", str(e))

    cached = _get_cached_tts("gtts", safe_text, dialect, gender, emotion)
    if cached is not None:
        return cached

    try:
        audio = await asyncio.to_thread(_synthesize_sync_gtts, safe_text, dialect, emotion)
        _put_cached_tts("gtts", safe_text, dialect, gender, emotion, audio, "mp3")
        return audio, "mp3"
    except Exception as gtts_error:
        if TTS_RETURN_SILENCE_ON_FAILURE:
            logger.error("All TTS providers failed. Returning silence WAV fallback: %s", gtts_error)
            return _generate_silence_wav(TTS_FAILURE_SILENCE_MS, _xtts_sample_rate), "wav"
        raise RuntimeError(f"gTTS synthesis failed: {gtts_error}")


async def warmup_tts_engine():
    provider = _resolve_provider()
    if provider == "chatterbox":
        try:
            model = await asyncio.to_thread(_load_chatterbox_model_sync)
            if model is not None:
                await asyncio.to_thread(_prepare_chatterbox_reference_if_needed, model, "female")
            logger.info("Chatterbox warmup complete")
        except Exception as e:
            logger.warning("Chatterbox warmup skipped: %s", str(e))
        return

    if provider != "xtts":
        return

    try:
        model = await asyncio.to_thread(_load_xtts_model_sync)
        reference_audio = _resolve_xtts_reference_audio("female")
        if model is not None and reference_audio is not None:
            await asyncio.to_thread(_get_xtts_conditioning_sync, model, reference_audio)
        logger.info("XTTS warmup complete")
    except Exception as e:
        logger.warning("XTTS warmup skipped: %s", str(e))


async def synthesize_to_base64(
    text: str,
    dialect: str = "cairene",
    gender: str = "female",
    emotion: str = "neutral",
) -> tuple[str, str]:
    audio_bytes, audio_format = await synthesize_speech(text, dialect, gender, emotion)
    return base64.b64encode(audio_bytes).decode("utf-8"), audio_format


def _concat_wav_bytes_with_crossfade(wav_chunks: list[bytes], crossfade_ms: int = 120) -> bytes:
    """
    Concatenate WAV chunks with optional crossfade using numpy/scipy.
    Falls back to simple concatenation if pydub unavailable.
    """
    if not wav_chunks:
        return b""
    
    if len(wav_chunks) == 1:
        return wav_chunks[0]
    
    import wave
    
    # Try pydub-based concat first if available
    if AudioSegment is not None:
        try:
            combined_seg = None
            for chunk in wav_chunks:
                seg = AudioSegment.from_file(io.BytesIO(chunk), format="wav")
                if combined_seg is None:
                    combined_seg = seg
                else:
                    # Short crossfade to prevent clicks
                    combined_seg = combined_seg.append(seg, crossfade=min(crossfade_ms, 120))
            
            # Try to normalize
            try:
                from pydub import effects
                combined_seg = effects.normalize(combined_seg)
            except Exception:
                pass

            if combined_seg is None:
                raise RuntimeError("No audio segments produced during WAV concat")
            
            out_buf = io.BytesIO()
            combined_seg.export(out_buf, format="wav")
            return out_buf.getvalue()
        except Exception as e:
            logger.debug("pydub concat with crossfade failed: %s; falling back to raw concat", e)
    
    # Fallback: numpy-based crossfade without external dependencies
    try:
        import numpy as np
        
        wav_data_list = []
        params_list = []
        
        for chunk_idx, chunk in enumerate(wav_chunks):
            buf = io.BytesIO(chunk)
            with wave.open(buf, "rb") as w:
                params = w.getparams()
                frames = w.readframes(w.getnframes())
                
                if chunk_idx > 0:
                    # Verify compatibility
                    if params[:3] != params_list[0][:3]:
                        logger.warning("Incompatible WAV params at chunk %d; skipping crossfade", chunk_idx)
                        wav_data_list.append(np.frombuffer(frames, dtype=np.int16))
                        continue
                
                wav_array = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                wav_data_list.append(wav_array)
                params_list.append(params)
        
        if not wav_data_list:
            raise RuntimeError("No valid WAV data to concatenate")
        
        # Apply crossfade between adjacent segments
        crossfade_samples = int((crossfade_ms / 1000.0) * params_list[0].framerate)
        result = wav_data_list[0].copy()
        
        for i in range(1, len(wav_data_list)):
            current = wav_data_list[i]
            fade_len = min(crossfade_samples, len(result), len(current))
            
            if fade_len > 0:
                # Fade out end of previous, fade in start of current
                fade_out = np.linspace(1.0, 0.0, fade_len)
                fade_in = np.linspace(0.0, 1.0, fade_len)
                
                result[-fade_len:] *= fade_out
                current[:fade_len] *= fade_in
                result[-fade_len:] += current[:fade_len]
                result = np.concatenate([result, current[fade_len:]])
            else:
                result = np.concatenate([result, current])
        
        # Clip to [-1, 1] and convert back to int16
        result = np.clip(result, -1.0, 1.0)
        pcm16 = (result * 32767.0).astype(np.int16)
        
        out_buf = io.BytesIO()
        with wave.open(out_buf, "wb") as w:
            w.setnchannels(params_list[0].nchannels)
            w.setsampwidth(params_list[0].sampwidth)
            w.setframerate(params_list[0].framerate)
            w.writeframes(pcm16.tobytes())
        
        return out_buf.getvalue()
    except Exception as e:
        logger.debug("numpy-based crossfade failed: %s; falling back to raw concat", e)
    
    # Final fallback: simple concatenation
    out_buf = io.BytesIO()
    first_params = None
    combined_frames = bytearray()
    
    for chunk in wav_chunks:
        buf = io.BytesIO(chunk)
        with wave.open(buf, "rb") as w:
            params = w.getparams()
            frames = w.readframes(w.getnframes())
            if first_params is None:
                first_params = params
            else:
                if params[:3] != first_params[:3]:
                    logger.warning("Skipping WAV chunk with incompatible parameters")
                    continue
            combined_frames.extend(frames)
    
    if first_params is None:
        raise RuntimeError("No valid WAV chunks for concatenation")
    
    with wave.open(out_buf, "wb") as w:
        w.setnchannels(first_params.nchannels)
        w.setsampwidth(first_params.sampwidth)
        w.setframerate(first_params.framerate)
        w.writeframes(bytes(combined_frames))
    
    return out_buf.getvalue()


def _concat_mp3_bytes(mp3_chunks: list[bytes]) -> bytes:
    if not mp3_chunks:
        return b""
    if AudioSegment is None:
        raise RuntimeError("pydub is unavailable for MP3 concatenation")

    combined = AudioSegment.empty()
    for chunk in mp3_chunks:
        segment = AudioSegment.from_file(io.BytesIO(chunk), format="mp3")
        combined += segment

    out_buf = io.BytesIO()
    combined.export(out_buf, format="mp3")
    return out_buf.getvalue()


def _chunk_text_for_tts(text: str, max_chars: int) -> list[str]:
    import re

    cleaned = re.sub(r"\s+", " ", (text or "")).strip()
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]

    sentences = [part.strip() for part in re.split(r"(?<=[.!؟?])\s+|\n+", cleaned) if part.strip()]
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

        # split long sentence by commas/phrases
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

        # fallback to word-wise chunking
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


async def synthesize_long_text(
    text: str,
    dialect: str = "cairene",
    gender: str = "female",
    emotion: str = "neutral",
    max_chars: Optional[int] = None,
) -> tuple[bytes, str]:
    """Synthesize long text by chunking <= `max_chars` and reassembling audio.

    Returns concatenated audio bytes and an audio format (wav/mp3).
    """
    # Always use chunking in production to avoid clipping and protect VRAM
    try:
        requested = int(max_chars) if max_chars is not None else int(os.getenv("TTS_SEGMENT_MAX_CHARS", "120"))
    except Exception:
        requested = 120
    max_c = min(120, max(40, requested))

    segments = _chunk_text_for_tts(text or "", max_c)
    if not segments:
        raise RuntimeError("No text to synthesize")

    parts: list[bytes] = []
    formats: list[str] = []

    for seg in segments:
        audio, fmt = await synthesize_speech(seg, dialect, gender, emotion)
        parts.append(audio)
        formats.append(fmt or "wav")

    # If all formats are WAV, concatenate properly; otherwise join bytes and return mp3
    common_fmt = formats[0]
    if all(f == common_fmt for f in formats) and common_fmt == "wav":
        # Prefer pydub-based concatenation with short crossfade to prevent clicks
        if AudioSegment is not None:
            try:
                combined_seg = None
                for chunk in parts:
                    seg = AudioSegment.from_file(io.BytesIO(chunk), format="wav")
                    if combined_seg is None:
                        combined_seg = seg
                    else:
                        combined_seg = combined_seg.append(seg, crossfade=120)
                try:
                    from pydub import effects

                    combined_seg = effects.normalize(combined_seg)
                except Exception:
                    pass
                if combined_seg is None:
                    raise RuntimeError("No audio segments produced during WAV concat")
                out_buf = io.BytesIO()
                combined_seg.export(out_buf, format="wav")
                return out_buf.getvalue(), "wav"
            except Exception as exc:
                logger.warning("pydub WAV concat failed; falling back to raw concat: %s", exc)
        combined = _concat_wav_bytes_with_crossfade(parts)
        return combined, "wav"

    if all(f == common_fmt for f in formats) and common_fmt == "mp3":
        try:
            return _concat_mp3_bytes(parts), "mp3"
        except Exception as exc:
            logger.warning("MP3 TTS chunk concatenation failed; falling back to raw byte join: %s", exc)

    # fallback: simple concatenation (works reasonably for mp3/streamed chunks)
    return b"".join(parts), common_fmt or "mp3"


async def synthesize_chunked(
    text: str,
    dialect: str = "cairene",
    gender: str = "female",
    emotion: str = "neutral",
):
    """
    Async generator yielding (audio_chunk, format).
    - local providers: one WAV chunk
    - edge: streaming MP3 chunks
    - gtts: one MP3 chunk
    """
    provider = _resolve_provider()

    if provider in {"xtts", "chatterbox", "gemini"}:
        try:
            audio_bytes, audio_format = await synthesize_speech(text, dialect, gender, emotion)
            yield audio_bytes, audio_format
            return
        except Exception as e:
            logger.warning("Local TTS stream failed, falling back to edge/gtts: %s", str(e))
            provider = "edge" if EDGE_TTS_AVAILABLE else "gtts"

    if provider == "edge":
        try:
            async for chunk in _stream_edge_tts(text, dialect, gender, emotion):
                if chunk:
                    yield chunk, "mp3"
            return
        except Exception as e:
            logger.warning("Edge stream failed, falling back to gTTS: %s", str(e))

    audio_bytes = await asyncio.to_thread(_synthesize_sync_gtts, text, dialect, emotion)
    yield audio_bytes, "mp3"
