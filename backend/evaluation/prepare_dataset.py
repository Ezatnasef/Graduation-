"""Prepare noisy conversational datasets for classification/evaluation use.

This script does not produce fine-tuning-ready generation data directly.
It cleans duplicates, repetitive agent answers, and basic dialect noise.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def normalize_text(text: str) -> str:
    value = (text or "").strip()
    value = re.sub(r"\s+", " ", value)
    value = value.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    return value


def clean_dataset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    cleaned: list[dict[str, Any]] = []

    for row in rows:
        conv = row.get("conversation") or []
        customer = ""
        agent = ""
        for turn in conv:
            if turn.get("role") == "customer" and not customer:
                customer = normalize_text(turn.get("text") or "")
            if turn.get("role") == "agent" and not agent:
                agent = normalize_text(turn.get("text") or "")

        if not customer or not agent:
            continue

        # Filter extremely repetitive boilerplate agent replies.
        if len(set(agent.split())) <= 4 and len(agent.split()) >= 6:
            continue

        key = (
            normalize_text(row.get("dialect") or ""),
            customer,
            normalize_text(row.get("emotion") or "neutral"),
        )
        if key in seen:
            continue
        seen.add(key)

        cleaned.append(
            {
                "id": row.get("id"),
                "dialect": row.get("dialect"),
                "domain": row.get("domain"),
                "emotion": row.get("emotion"),
                "customer_text": customer,
                "agent_text": agent,
            }
        )

    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    src = Path(args.input)
    rows = json.loads(src.read_text(encoding="utf-8"))
    cleaned = clean_dataset(rows)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Input rows: {len(rows)}")
    print(f"Clean rows: {len(cleaned)}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
