"""Quick API health checker for Servia backend."""

from __future__ import annotations

import json
import urllib.request

BASE = "http://127.0.0.1:8765"


def fetch(path: str):
    with urllib.request.urlopen(f"{BASE}{path}", timeout=8) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    endpoints = [
        "/",
        "/api/dialects",
        "/api/tts/status",
        "/api/stt/status",
        "/api/analytics/summary?hours=1",
    ]

    for ep in endpoints:
        data = fetch(ep)
        print(f"[OK] {ep}")
        print(json.dumps(data, ensure_ascii=False)[:300])


if __name__ == "__main__":
    main()
