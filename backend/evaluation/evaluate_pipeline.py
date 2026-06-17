"""Lightweight quality and latency evaluation for Servia API."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any


def post_json(url: str, payload: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def score_relevance(response_text: str, expected_keywords: list[str]) -> float:
    text = (response_text or "").lower()
    if not expected_keywords:
        return 1.0
    hits = sum(1 for keyword in expected_keywords if keyword.lower() in text)
    return hits / len(expected_keywords)


def run_eval(base_url: str, dataset_path: Path) -> dict[str, Any]:
    rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not rows:
        raise RuntimeError("Evaluation dataset is empty")

    latencies: list[int] = []
    relevance_scores: list[float] = []
    intent_hits = 0
    results = []

    for row in rows:
        text = (row.get("text") or "").strip()
        if not text:
            continue

        started = time.perf_counter()
        data = post_json(
            f"{base_url.rstrip('/')}/api/chat",
            {
                "text": text,
                "dialect": "cairene",
                "gender": "female",
                "include_tts": False,
            },
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        latencies.append(latency_ms)

        analysis = data.get("analysis") or {}
        predicted_intent = analysis.get("intent_label", "other")
        expected_intent = row.get("expected_intent", "other")
        if predicted_intent == expected_intent:
            intent_hits += 1

        response_text = data.get("response_text") or ""
        rel = score_relevance(response_text, row.get("expected_keywords") or [])
        relevance_scores.append(rel)

        results.append(
            {
                "id": row.get("id"),
                "latency_ms": latency_ms,
                "predicted_intent": predicted_intent,
                "expected_intent": expected_intent,
                "relevance": round(rel, 3),
            }
        )

    return {
        "samples": len(results),
        "intent_accuracy": round(intent_hits / max(1, len(results)), 3),
        "avg_relevance": round(sum(relevance_scores) / max(1, len(relevance_scores)), 3),
        "avg_latency_ms": round(sum(latencies) / max(1, len(latencies)), 1),
        "p95_latency_ms": round(statistics.quantiles(latencies, n=20)[-1], 1) if len(latencies) > 1 else latencies[0],
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).resolve().parent / "sample_eval_dataset.json"),
    )
    args = parser.parse_args()

    report = run_eval(args.base_url, Path(args.dataset))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
