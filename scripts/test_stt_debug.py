#!/usr/bin/env python3
"""
Quick test to debug STT issue.
"""

import sys
import os
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

# Load sample audio
sample_path = Path(__file__).parent.parent / "models" / "sample_06.wav"
with open(sample_path, 'rb') as f:
    audio_bytes = f.read()

print(f"Audio file size: {len(audio_bytes)} bytes")

# Import STT engine
from stt_engine import STT_ENGINE
import asyncio

async def test_stt():
    print("\n[Test 1] First transcription...")
    try:
        result1 = await STT_ENGINE.transcribe(audio_bytes, mime_type='audio/wav', language='ar')
        print(f"[OK] Result 1: confidence={result1.confidence}")
    except Exception as e:
        print(f"[ERROR] Failed: {e}")
        return
    
    print("\n[Test 2] Second transcription...")
    try:
        result2 = await STT_ENGINE.transcribe(audio_bytes, mime_type='audio/wav', language='ar')
        print(f"[OK] Result 2: confidence={result2.confidence}")
    except Exception as e:
        print(f"[ERROR] Failed: {e}")
        return
    
    print("\n[Test 3] Third transcription...")
    try:
        result3 = await STT_ENGINE.transcribe(audio_bytes, mime_type='audio/wav', language='ar')
        print(f"[OK] Result 3: confidence={result3.confidence}")
    except Exception as e:
        print(f"[ERROR] Failed: {e}")
        return

asyncio.run(test_stt())
