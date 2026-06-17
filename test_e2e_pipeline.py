"""
Servia Voice – End-to-End Pipeline Test
========================================
Tests the RUNNING server through its public HTTP and WebSocket APIs.
NO direct imports of tts_engine / stt_engine / dialect_mapper.
All traffic goes through the real FastAPI pipeline exactly as production.

Usage:
    python test_e2e_pipeline.py [--base-url http://127.0.0.1:8765]
"""

import argparse
import asyncio
import base64
import json
import struct
import sys
import time
import traceback
from typing import Any

# Only stdlib + lightweight HTTP/WS client
try:
    import aiohttp
except ImportError:
    print("[FATAL] aiohttp is required.  pip install aiohttp")
    sys.exit(1)


# ─────────────────────────── helpers ───────────────────────────

def generate_sine_wav(duration_s: float = 2.0, sample_rate: int = 16000,
                      freq: float = 440.0) -> bytes:
    """Generate a minimal WAV file with a sine tone (no numpy needed)."""
    import math
    num_samples = int(sample_rate * duration_s)
    samples = []
    for i in range(num_samples):
        t = i / sample_rate
        value = int(32767 * 0.5 * math.sin(2 * math.pi * freq * t))
        samples.append(struct.pack("<h", value))
    raw = b"".join(samples)

    # WAV header
    data_size = len(raw)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE",
        b"fmt ", 16,          # PCM sub-chunk
        1,                     # audio format (PCM)
        1,                     # mono
        sample_rate,
        sample_rate * 2,       # byte rate
        2,                     # block align
        16,                    # bits per sample
        b"data", data_size,
    )
    return header + raw


def separator(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def result_line(label: str, value: Any) -> None:
    print(f"  {label:<40s} : {value}")


# ─────────────────────────── tests ─────────────────────────────

async def test_health(session: aiohttp.ClientSession, base: str) -> bool:
    separator("1. Health Check — GET /")
    try:
        async with session.get(f"{base}/") as r:
            data = await r.json()
            result_line("Status code", r.status)
            result_line("Response", data)
            return r.status == 200
    except Exception as e:
        result_line("ERROR", str(e))
        return False


async def test_tts_status(session: aiohttp.ClientSession, base: str) -> dict:
    separator("2. TTS Status — GET /api/tts/status")
    try:
        async with session.get(f"{base}/api/tts/status") as r:
            data = await r.json()
            result_line("Status code", r.status)
            for k, v in data.items():
                result_line(k, v)
            return data
    except Exception as e:
        result_line("ERROR", str(e))
        return {}


async def test_stt_status(session: aiohttp.ClientSession, base: str) -> dict:
    separator("3. STT Status — GET /api/stt/status")
    try:
        async with session.get(f"{base}/api/stt/status") as r:
            data = await r.json()
            result_line("Status code", r.status)
            for k, v in data.items():
                result_line(k, v)
            return data
    except Exception as e:
        result_line("ERROR", str(e))
        return {}


async def test_tts_api(session: aiohttp.ClientSession, base: str) -> dict:
    separator("4. TTS via HTTP — POST /api/tts")
    payload = {
        "text": "أهلاً وسهلاً، إزيك النهارده؟",
        "dialect": "cairene",
        "gender": "female",
        "emotion": "neutral",
    }
    result_line("Input text", payload["text"])
    result_line("Dialect / Gender", f"{payload['dialect']} / {payload['gender']}")

    t0 = time.perf_counter()
    try:
        async with session.post(f"{base}/api/tts", json=payload, timeout=aiohttp.ClientTimeout(total=120)) as r:
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            result_line("Status code", r.status)
            result_line("Latency", f"{elapsed_ms} ms")

            if r.status != 200:
                body = await r.text()
                result_line("Error body", body[:500])
                return {"success": False, "error": body[:500]}

            data = await r.json()
            audio_b64 = data.get("audio_base64", "")
            audio_bytes = base64.b64decode(audio_b64) if audio_b64 else b""
            result_line("Audio format", data.get("audio_format", "?"))
            result_line("Audio size", f"{len(audio_bytes)} bytes")
            result_line("Text original", data.get("text_original", ""))
            result_line("Text dialectal", data.get("text_dialectal", ""))
            result_line("TTS audio generated?", "YES" if len(audio_bytes) > 100 else "NO / too small")
            return {"success": True, "audio_size": len(audio_bytes), "latency_ms": elapsed_ms, **data}
    except Exception as e:
        result_line("ERROR", str(e))
        traceback.print_exc()
        return {"success": False, "error": str(e)}


async def test_chat_api(session: aiohttp.ClientSession, base: str) -> dict:
    separator("5. Chat via HTTP — POST /api/chat (text only, no TTS)")
    payload = {
        "text": "إيه أحسن مكان أزوره في القاهرة؟",
        "session_id": "e2e-test-session",
        "dialect": "cairene",
        "gender": "female",
        "include_tts": False,
    }
    result_line("Input text", payload["text"])

    t0 = time.perf_counter()
    try:
        async with session.post(f"{base}/api/chat", json=payload, timeout=aiohttp.ClientTimeout(total=60)) as r:
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            result_line("Status code", r.status)
            result_line("Latency", f"{elapsed_ms} ms")

            if r.status != 200:
                body = await r.text()
                result_line("Error body", body[:500])
                return {"success": False, "error": body[:500]}

            data = await r.json()
            result_line("Session ID", data.get("session_id", "?"))
            result_line("LLM response", (data.get("response_text", "") or "")[:200])
            result_line("Analysis", json.dumps(data.get("analysis", {}), ensure_ascii=False)[:200])
            result_line("Memory state", json.dumps(data.get("memory", {}), ensure_ascii=False)[:200])
            return {"success": True, "latency_ms": elapsed_ms, **data}
    except Exception as e:
        result_line("ERROR", str(e))
        traceback.print_exc()
        return {"success": False, "error": str(e)}


async def test_chat_with_tts(session: aiohttp.ClientSession, base: str) -> dict:
    separator("6. Chat + TTS — POST /api/chat (include_tts=true)")
    payload = {
        "text": "قولي نكتة مصرية",
        "session_id": "e2e-test-session-tts",
        "dialect": "cairene",
        "gender": "female",
        "include_tts": True,
    }
    result_line("Input text", payload["text"])

    t0 = time.perf_counter()
    try:
        async with session.post(f"{base}/api/chat", json=payload, timeout=aiohttp.ClientTimeout(total=180)) as r:
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            result_line("Status code", r.status)
            result_line("Latency", f"{elapsed_ms} ms")

            if r.status != 200:
                body = await r.text()
                result_line("Error body", body[:500])
                return {"success": False, "error": body[:500]}

            data = await r.json()
            result_line("LLM response", (data.get("response_text", "") or "")[:200])
            tts_info = data.get("tts")
            if tts_info:
                audio_b64 = tts_info.get("audio_base64", "")
                audio_bytes = base64.b64decode(audio_b64) if audio_b64 else b""
                result_line("TTS audio format", tts_info.get("audio_format", "?"))
                result_line("TTS audio size", f"{len(audio_bytes)} bytes")
                result_line("Full voice response?", "YES" if len(audio_bytes) > 100 else "NO")
            else:
                result_line("TTS info", "None returned")
            return {"success": True, "latency_ms": elapsed_ms, **data}
    except Exception as e:
        result_line("ERROR", str(e))
        traceback.print_exc()
        return {"success": False, "error": str(e)}


async def test_websocket_text(base: str) -> dict:
    separator("7. WebSocket Pipeline — /ws/voice (text input)")
    ws_url = base.replace("http://", "ws://").replace("https://", "wss://") + "/ws/voice"
    result_line("WebSocket URL", ws_url)

    collected_messages = []
    audio_received = False
    tts_text_received = ""
    error_messages = []

    try:
        async with aiohttp.ClientSession() as ws_session:
            async with ws_session.ws_connect(ws_url, timeout=30) as ws:
                # Wait for connected message
                msg = await asyncio.wait_for(ws.receive(), timeout=10)
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    collected_messages.append(data)
                    result_line("Connected", f"session={data.get('session_id', '?')}")

                # Send a text message through the pipeline
                text_payload = json.dumps({
                    "type": "text",
                    "content": "مرحبا يا سيرفيا، إزيك؟"
                })
                result_line("Sending text", "مرحبا يا سيرفيا، إزيك؟")
                await ws.send_str(text_payload)

                # Collect responses for up to 120 seconds
                t0 = time.perf_counter()
                timeout_s = 120
                while (time.perf_counter() - t0) < timeout_s:
                    try:
                        msg = await asyncio.wait_for(ws.receive(), timeout=5)
                    except asyncio.TimeoutError:
                        # No message for 5s — if we already got audio, we're done
                        if audio_received:
                            break
                        continue

                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        collected_messages.append(data)
                        msg_type = data.get("type", "")

                        if msg_type == "error":
                            error_messages.append(data)
                            result_line("WS ERROR", json.dumps(data, ensure_ascii=False)[:300])

                        elif msg_type == "user_text":
                            result_line("User text echo", data.get("text", "")[:200])

                        elif msg_type == "analysis":
                            result_line("Analysis", json.dumps(data.get("analysis", {}), ensure_ascii=False)[:200])

                        elif msg_type == "response_text":
                            tts_text_received = data.get("text", "")
                            result_line("LLM Response", tts_text_received[:200])

                        elif msg_type == "tts_start":
                            result_line("TTS started", data.get("format", "?"))

                        elif msg_type == "tts_chunk":
                            audio_b64 = data.get("audio", "")
                            if audio_b64:
                                chunk_bytes = base64.b64decode(audio_b64)
                                result_line("TTS audio chunk", f"{len(chunk_bytes)} bytes")
                                audio_received = True

                        elif msg_type == "tts_end":
                            result_line("TTS ended", "✓")
                            break  # Full pipeline done

                        elif msg_type == "tts_audio":
                            audio_b64 = data.get("audio", "")
                            if audio_b64:
                                ab = base64.b64decode(audio_b64)
                                result_line("TTS full audio", f"{len(ab)} bytes")
                                audio_received = True

                        else:
                            result_line(f"WS msg [{msg_type}]", json.dumps(data, ensure_ascii=False)[:200])

                    elif msg.type == aiohttp.WSMsgType.BINARY:
                        result_line("Binary audio", f"{len(msg.data)} bytes")
                        audio_received = True

                    elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING, aiohttp.WSMsgType.CLOSED):
                        result_line("WS closed", str(msg.type))
                        break

                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        result_line("WS error", str(ws.exception()))
                        break

                elapsed = int((time.perf_counter() - t0) * 1000)
                result_line("Total WS pipeline time", f"{elapsed} ms")

    except Exception as e:
        result_line("ERROR", str(e))
        traceback.print_exc()
        return {"success": False, "error": str(e)}

    result_line("Audio received?", "YES" if audio_received else "NO")
    result_line("LLM text received?", "YES" if tts_text_received else "NO")
    result_line("Error messages", len(error_messages))

    return {
        "success": True,
        "audio_received": audio_received,
        "llm_response": tts_text_received,
        "errors": error_messages,
        "total_messages": len(collected_messages),
    }


# ─────────────────────────── main ──────────────────────────────

async def main(base_url: str):
    print(f"\n{'#' * 60}")
    print(f"  SERVIA VOICE — END-TO-END PIPELINE TEST")
    print(f"  Server: {base_url}")
    print(f"  Time:   {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 60}")

    results = {}

    async with aiohttp.ClientSession() as session:
        # 1. Health
        results["health"] = await test_health(session, base_url)

        if not results["health"]:
            print("\n[FATAL] Server is not reachable. Aborting remaining tests.")
            return results

        # 2. TTS status
        results["tts_status"] = await test_tts_status(session, base_url)

        # 3. STT status
        results["stt_status"] = await test_stt_status(session, base_url)

        # 4. TTS API
        results["tts_api"] = await test_tts_api(session, base_url)

        # 5. Chat API (text only)
        results["chat_api"] = await test_chat_api(session, base_url)

        # 6. Chat API + TTS
        results["chat_tts"] = await test_chat_with_tts(session, base_url)

    # 7. WebSocket full pipeline
    results["websocket"] = await test_websocket_text(base_url)

    # ─── Summary ───
    separator("SUMMARY")
    tts_status = results.get("tts_status", {})
    result_line("XTTS loaded?", tts_status.get("xtts_ready") or tts_status.get("xtts_loaded", "?"))
    result_line("XTTS device", tts_status.get("xtts_device", "?"))
    result_line("Active TTS backend", tts_status.get("active_backend") or tts_status.get("backend", "?"))
    result_line("TTS API success?", results.get("tts_api", {}).get("success", False))
    result_line("TTS audio size", f"{results.get('tts_api', {}).get('audio_size', 0)} bytes")
    result_line("Chat API success?", results.get("chat_api", {}).get("success", False))
    result_line("Chat+TTS success?", results.get("chat_tts", {}).get("success", False))
    result_line("WS pipeline success?", results.get("websocket", {}).get("success", False))
    result_line("WS audio received?", results.get("websocket", {}).get("audio_received", False))
    result_line("WS LLM response?", bool(results.get("websocket", {}).get("llm_response", "")))
    result_line("WS errors", len(results.get("websocket", {}).get("errors", [])))

    print(f"\n{'=' * 60}")
    all_ok = all([
        results.get("health"),
        results.get("tts_api", {}).get("success"),
        results.get("chat_api", {}).get("success"),
        results.get("chat_tts", {}).get("success"),
        results.get("websocket", {}).get("success"),
    ])
    if all_ok:
        print("  ✅ ALL PIPELINE TESTS PASSED")
    else:
        print("  ❌ SOME TESTS FAILED — check details above")
    print(f"{'=' * 60}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Servia E2E Pipeline Test")
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    args = parser.parse_args()

    asyncio.run(main(args.base_url))
