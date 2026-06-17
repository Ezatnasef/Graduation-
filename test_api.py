#!/usr/bin/env python
"""Test backend API health endpoint."""

import time
import requests
import json

print("Waiting for server to stabilize...")
time.sleep(3)

try:
    print("\n1. Testing /health endpoint...")
    response = requests.get('http://127.0.0.1:8765/health', timeout=5)
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.text}")
except Exception as e:
    print(f"   Error: {e}")

try:
    print("\n2. Testing TTS synthesis (quick test)...")
    response = requests.post('http://127.0.0.1:8765/synthesize', 
        json={
            "text": "مرحبا",
            "language": "ar",
            "speaker_name": "female"
        },
        timeout=30)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        print(f"   Audio length: {len(response.content)} bytes")
        print("   ✓ TTS synthesis successful!")
    else:
        print(f"   Response: {response.text}")
except Exception as e:
    print(f"   Error: {e}")

print("\n✓ API tests completed")
