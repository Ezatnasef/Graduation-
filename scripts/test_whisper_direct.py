#!/usr/bin/env python3
"""
Direct test of faster_whisper to debug the issue.
"""

import sys
import os
import tempfile
from pathlib import Path

# Load sample audio
sample_path = Path(__file__).parent.parent / "models" / "sample_06.wav"
with open(sample_path, 'rb') as f:
    audio_bytes = f.read()

print(f"Audio file size: {len(audio_bytes)} bytes")

# Test faster_whisper directly
print("\n[1] Loading faster-whisper model...")
try:
    from faster_whisper import WhisperModel
    model = WhisperModel("base", device="cpu", compute_type="int8")
    print("✓ Model loaded on CPU")
except Exception as e:
    print(f"✗ Failed to load: {e}")
    sys.exit(1)

print("\n[2] Transcribing audio...")
try:
    # Write audio to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_bytes)
        temp_path = f.name
    
    print(f"  Temp file: {temp_path}")
    
    # Transcribe
    segments_iter, info = model.transcribe(
        temp_path,
        language="ar",
        beam_size=3,
        vad_filter=True,
    )
    
    # Collect segments
    segments = list(segments_iter)
    print(f"✓ Transcription complete!")
    print(f"  Language: {info.language}")
    print(f"  Segments: {len(segments)}")
    
    for i, seg in enumerate(segments):
        print(f"    [{i}] {seg.text}")
    
    # Clean up
    os.remove(temp_path)
    
except Exception as e:
    print(f"✗ Transcription failed: {e}")
    import traceback
    traceback.print_exc()
