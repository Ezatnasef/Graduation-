#!/usr/bin/env python3
"""
Quick test to verify XTTS dtype fix and test single inference.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from tts_engine import _load_xtts_model_sync, _synthesize_sync_xtts

async def test_xtts():
    print("=" * 60)
    print("XTTS Dtype Fix Test")
    print("=" * 60)
    
    try:
        print("\n[1] Loading XTTS model...")
        model = _load_xtts_model_sync()
        print(f"✓ Model loaded successfully")
        print(f"  Device: {model.device if hasattr(model, 'device') else 'unknown'}")
        
        print("\n[2] Testing XTTS inference with short text...")
        text = "أهلا وسهلا بك"
        audio_bytes = _synthesize_sync_xtts(text, dialect="cairene", gender="female", emotion="neutral")
        print(f"✓ Inference succeeded!")
        print(f"  Output audio size: {len(audio_bytes)} bytes")
        
        print("\n[3] Testing with longer text...")
        longer_text = "مرحبا بك في خدمة الصوت. كيف يمكنني مساعدتك اليوم؟"
        audio_bytes2 = _synthesize_sync_xtts(longer_text, dialect="cairene", gender="female")
        print(f"✓ Longer text inference succeeded!")
        print(f"  Output audio size: {len(audio_bytes2)} bytes")
        
        print("\n" + "=" * 60)
        print("✓ All XTTS tests passed!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_xtts())
    sys.exit(0 if success else 1)
