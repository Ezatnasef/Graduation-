#!/usr/bin/env python
"""Direct XTTS test to check for errors."""

import sys
import logging
import os

# Set environment variables
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTHONUNBUFFERED'] = '1'

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

print("=" * 60)
print("DIRECT XTTS TEST")
print("=" * 60)

try:
    print("\n1. Importing TTS engine...")
    from backend.tts_engine import TextToSpeechEngine
    print("   IMPORT SUCCESSFUL")
    
    print("\n2. Creating TTS engine instance...")
    engine = TextToSpeechEngine()
    print("   ENGINE CREATED")
    
    print("\n3. Checking XTTS availability...")
    status = engine.get_status()
    print(f"   Status: {status}")
    print(f"   Available providers: {status.get('available_local_providers', [])}")
    print(f"   Effective provider: {status.get('effective_provider', 'N/A')}")
    
    print("\n4. Testing XTTS synthesis...")
    audio_data = engine.synthesize(
        text="مرحبا",
        language="ar",
        provider="xtts"
    )
    if audio_data:
        print(f"   SYNTHESIS SUCCESSFUL")
        print(f"   Audio length: {len(audio_data)} bytes")
    else:
        print(f"   SYNTHESIS RETURNED NONE/EMPTY")
    
except Exception as e:
    print(f"\n   ERROR: {type(e).__name__}")
    print(f"   Message: {e}")
    import traceback
    print("\n   TRACEBACK:")
    traceback.print_exc()

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
