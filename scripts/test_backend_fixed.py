#!/usr/bin/env python3
"""Test backend HTTP and WebSocket connectivity after fixes."""

import asyncio
import json
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import aiohttp
    import websockets
except ImportError:
    print("⚠️  Missing dependencies. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "aiohttp", "websockets"])
    import aiohttp
    import websockets


BACKEND_URL = "http://localhost:8765"
WS_URL = "ws://localhost:8765/ws/voice"
TEST_TIMEOUT = 5.0


async def test_http_endpoint():
    """Test HTTP GET / endpoint."""
    print("🧪 Testing HTTP root endpoint...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(BACKEND_URL, timeout=aiohttp.ClientTimeout(total=TEST_TIMEOUT)) as response:
                data = await response.json()
                print(f"✅ HTTP OK: {response.status}")
                print(f"   Response: {json.dumps(data, indent=2)}")
                return True
    except asyncio.TimeoutError:
        print("❌ HTTP timeout - backend may still be loading")
        return False
    except Exception as e:
        print(f"❌ HTTP error: {e}")
        return False


async def test_websocket():
    """Test WebSocket /ws/voice endpoint."""
    print("\n🧪 Testing WebSocket endpoint...")
    try:
        async with websockets.connect(WS_URL, ping_interval=None) as websocket:
            print(f"✅ WebSocket connected")
            
            # Wait for welcome message
            welcome = await asyncio.wait_for(websocket.recv(), timeout=TEST_TIMEOUT)
            msg = json.loads(welcome)
            print(f"✅ Received welcome: {json.dumps(msg, indent=2)}")
            
            # Send test message
            test_msg = {"type": "set_dialect", "dialect": "cairene"}
            await websocket.send(json.dumps(test_msg))
            print(f"✅ Sent test message: {json.dumps(test_msg)}")
            
            # Try to receive response
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                print(f"✅ Received response: {response[:100]}")
            except asyncio.TimeoutError:
                print("⚠️  No response received (expected for set_dialect)")
            
            await websocket.close()
            return True
    except asyncio.TimeoutError:
        print("❌ WebSocket timeout - server may not be ready")
        return False
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        return False


async def main():
    """Run all tests."""
    print("=" * 60)
    print("Backend Connectivity Test (After Fixes)")
    print("=" * 60)
    
    # Try HTTP first
    http_ok = await test_http_endpoint()
    
    if not http_ok:
        print("\n⏳ Backend still loading, retrying in 5 seconds...")
        await asyncio.sleep(5)
        http_ok = await test_http_endpoint()
    
    if not http_ok:
        print("\n❌ Backend HTTP unreachable. Make sure backend is running:")
        print("   cd d:\\Desktop\\programing\\CSAP\\Servia_Voice")
        print("   python backend/main.py")
        return False
    
    # Try WebSocket
    ws_ok = await test_websocket()
    
    print("\n" + "=" * 60)
    if http_ok and ws_ok:
        print("✅ ALL TESTS PASSED - Backend ready for connection")
        return True
    elif http_ok:
        print("⚠️  HTTP OK but WebSocket failed - check backend logs for errors")
        return False
    else:
        print("❌ Backend connectivity failed")
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏸️  Test interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
