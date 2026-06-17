#!/usr/bin/env python
"""Quick startup test for Servia Voice backend"""
import os
import sys

# Set OpenMP workaround
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Change to backend directory
os.chdir(os.path.join(os.path.dirname(__file__), 'backend'))
sys.path.insert(0, os.getcwd())

print("=" * 60)
print("SERVIA VOICE STARTUP TEST")
print("=" * 60)
print(f"Working directory: {os.getcwd()}")
print(f"Python version: {sys.version}")

try:
    print("\n[1/5] Importing logging...")
    import logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("startup-test")
    print("✓ Logging configured")
    
    print("\n[2/5] Importing tts_engine...")
    import tts_engine
    print(f"✓ TTS engine imported")
    
    print("\n[3/5] Checking tokenizer patch...")
    from tts_engine import _patch_xtts_generation_runtime
    print("✓ Tokenizer patch function found")
    
    print("\n[4/5] Importing stt_engine...")
    import stt_engine
    print(f"✓ STT engine imported")
    
    print("\n[5/5] Importing FastAPI app...")
    import main
    print("✓ FastAPI app imported")
    
    print("\n" + "=" * 60)
    print("✓ ALL IMPORTS SUCCESSFUL")
    print("=" * 60)
    
except Exception as e:
    print(f"\n✗ ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
