"""
End-to-End WebSocket test for Servia Voice.
Connects to the real running server via ws://localhost:8765/ws/voice
and sends a text message through the full pipeline:
  Text Input → LLM → TTS
Then records all responses and errors.
"""

import asyncio
import json
import time
import sys

try:
    import websockets
except ImportError:
    print("ERROR: websockets library not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "websockets"])
    import websockets


WS_URL = "ws://localhost:8765/ws/voice"
TEST_TEXT = "ازيك يا سيرفيا، عايز اعرف ايه اخبار الدنيا النهارده"
TIMEOUT_SECONDS = 120  # generous timeout for XTTS on CPU


async def run_e2e_test():
    results = {
        "connected": False,
        "session_id": None,
        "user_text_ack": False,
        "bot_response_text": None,
        "tts_segments_received": 0,
        "tts_format": None,
        "tts_total_bytes": 0,
        "tts_complete": False,
        "errors": [],
        "all_messages": [],
        "xtts_used": "unknown",
        "pad_token_error": False,
        "cuda_assert_error": False,
        "timeline": [],
    }

    def log(msg):
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] {msg}")
        results["timeline"].append(f"[{ts}] {msg}")

    try:
        log(f"Connecting to {WS_URL} ...")
        async with websockets.connect(WS_URL, open_timeout=15, close_timeout=5) as ws:
            log("WebSocket connected, waiting for 'connected' message...")

            # 1. Wait for connection acknowledgement
            raw = await asyncio.wait_for(ws.recv(), timeout=10)
            msg = json.loads(raw)
            results["all_messages"].append(msg)
            log(f"Received: type={msg.get('type')} | session_id={msg.get('session_id')}")

            if msg.get("type") == "connected":
                results["connected"] = True
                results["session_id"] = msg.get("session_id")
                log(f"✓ Connected! Session: {results['session_id']}")
                log(f"  Dialects: {msg.get('dialects')}")
                log(f"  Current dialect: {msg.get('current_dialect')}")
                log(f"  Current gender: {msg.get('current_gender')}")
            else:
                log(f"✗ Unexpected first message type: {msg.get('type')}")

            # 2. Send text message through the real pipeline
            text_msg = json.dumps({
                "type": "text",
                "content": TEST_TEXT,
            })
            log(f"Sending text: '{TEST_TEXT}'")
            await ws.send(text_msg)

            # 3. Collect all responses until tts_complete or timeout
            start = time.time()
            while (time.time() - start) < TIMEOUT_SECONDS:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=TIMEOUT_SECONDS)
                    msg = json.loads(raw)
                    results["all_messages"].append(msg)
                    msg_type = msg.get("type", "unknown")

                    if msg_type == "user_text":
                        results["user_text_ack"] = True
                        log(f"✓ user_text acknowledged: '{msg.get('text', '')[:60]}...'")
                        analysis = msg.get("analysis", {})
                        log(f"  Analysis: sentiment={analysis.get('sentiment_label')}, "
                            f"intent={analysis.get('intent_label')}, "
                            f"urgency={analysis.get('urgency')}")

                    elif msg_type == "bot_text":
                        results["bot_response_text"] = msg.get("text", "")
                        log(f"✓ bot_text: '{results['bot_response_text'][:100]}...'")
                        log(f"  Dialect: {msg.get('dialect')}, Emotion: {msg.get('emotion')}")

                    elif msg_type == "tts_audio":
                        results["tts_segments_received"] += 1
                        audio_b64 = msg.get("audio_base64", "")
                        segment_bytes = len(audio_b64) * 3 // 4  # approximate decoded size
                        results["tts_total_bytes"] += segment_bytes
                        results["tts_format"] = msg.get("format", "unknown")
                        log(f"✓ tts_audio segment {msg.get('segment_index', '?')}/{msg.get('segment_count', '?')} "
                            f"| format={msg.get('format')} | ~{segment_bytes} bytes | "
                            f"dialect={msg.get('dialect')} emotion={msg.get('emotion')}")

                    elif msg_type == "tts_complete":
                        results["tts_complete"] = True
                        log(f"✓ tts_complete received! Response ID: {msg.get('response_id')}")
                        break

                    elif msg_type == "error":
                        error_msg = msg.get("message", "unknown error")
                        results["errors"].append(error_msg)
                        log(f"✗ ERROR from server: {error_msg}")

                        # Check for specific errors
                        if "pad_token_id" in error_msg.lower():
                            results["pad_token_error"] = True
                            log("  ⚠ pad_token_id error detected!")
                        if "cuda" in error_msg.lower() and "assert" in error_msg.lower():
                            results["cuda_assert_error"] = True
                            log("  ⚠ CUDA device-side assert detected!")

                    else:
                        log(f"  [{msg_type}] {json.dumps(msg, ensure_ascii=False)[:120]}")

                except asyncio.TimeoutError:
                    log(f"✗ Timeout after {TIMEOUT_SECONDS}s waiting for response")
                    results["errors"].append(f"Timeout after {TIMEOUT_SECONDS}s")
                    break

    except ConnectionRefusedError:
        log("✗ Connection refused - is the server running on port 8765?")
        results["errors"].append("Connection refused")
    except Exception as e:
        error_str = str(e)
        log(f"✗ Exception: {error_str}")
        results["errors"].append(error_str)
        if "pad_token_id" in error_str.lower():
            results["pad_token_error"] = True
        if "cuda" in error_str.lower() and "assert" in error_str.lower():
            results["cuda_assert_error"] = True

    # Print summary
    print("\n" + "=" * 70)
    print("  END-TO-END TEST RESULTS")
    print("=" * 70)
    print(f"  Connected:              {results['connected']}")
    print(f"  Session ID:             {results['session_id']}")
    print(f"  User text acked:        {results['user_text_ack']}")
    print(f"  Bot response:           {(results['bot_response_text'] or 'NONE')[:100]}")
    print(f"  TTS segments received:  {results['tts_segments_received']}")
    print(f"  TTS format:             {results['tts_format']}")
    print(f"  TTS total bytes:        {results['tts_total_bytes']}")
    print(f"  TTS complete:           {results['tts_complete']}")
    print(f"  Errors:                 {results['errors'] if results['errors'] else 'NONE'}")
    print(f"  pad_token_id error:     {results['pad_token_error']}")
    print(f"  CUDA assert error:      {results['cuda_assert_error']}")
    print("=" * 70)

    # Determine overall pass/fail
    passed = (
        results["connected"]
        and results["user_text_ack"]
        and results["bot_response_text"]
        and results["tts_complete"]
        and results["tts_segments_received"] > 0
        and not results["pad_token_error"]
        and not results["cuda_assert_error"]
        and len(results["errors"]) == 0
    )

    if passed:
        print("  ✅ OVERALL: PASS - Full pipeline working end-to-end!")
    else:
        print("  ❌ OVERALL: FAIL - See details above")
    print("=" * 70)

    return results


if __name__ == "__main__":
    asyncio.run(run_e2e_test())
