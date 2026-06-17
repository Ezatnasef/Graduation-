"""
Import mixed Arabic/English dataset samples from Hugging Face into
codeswitch_dataset_samples.txt for runtime context retrieval.

Usage (PowerShell):
  $env:HF_TOKEN = "hf_xxx"
  python backend/import_hf_codeswitch_dataset.py

Optional env vars:
  HF_DATASET_ID=raniahossam/egypt_dialect
  HF_SPLIT=train
  CODESWITCH_OUTPUT_PATH=backend/codeswitch_dataset_samples.txt
  CODESWITCH_MAX_LINES=12000
  CODESWITCH_OVERWRITE=true
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request
from typing import Iterable
from urllib.error import HTTPError, URLError

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset


DEFAULT_DATASET_ID = "raniahossam/egypt_dialect"
DEFAULT_OUTPUT_NAME = "codeswitch_dataset_samples.txt"
TEXT_CANDIDATE_COLUMNS = [
    "sentence",
    "text",
    "transcript",
    "transcription",
    "utterance",
    "normalized_text",
    "content",
]
ROWS_API_BASE = "https://datasets-server.huggingface.co"


def _normalize_text(value: str) -> str:
    text = (value or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _truthy(value: str) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_output_path() -> str:
    configured = (os.getenv("CODESWITCH_OUTPUT_PATH", "") or "").strip()
    if configured:
        if os.path.isabs(configured):
            return configured
        return os.path.abspath(configured)

    backend_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(backend_dir, DEFAULT_OUTPUT_NAME)


def _pick_text_columns(columns: Iterable[str]) -> list[str]:
    available = {c.lower(): c for c in columns}
    picked = []
    for candidate in TEXT_CANDIDATE_COLUMNS:
        original = available.get(candidate.lower())
        if original:
            picked.append(original)
    if picked:
        return picked

    # Fallback: any string-like name that hints at text.
    for col in columns:
        low = col.lower()
        if any(k in low for k in ["text", "sent", "trans", "utter", "content"]):
            picked.append(col)
    return picked


def _iter_rows(ds_obj):
    if isinstance(ds_obj, (Dataset, IterableDataset)):
        for row in ds_obj:
            yield row
        return

    if isinstance(ds_obj, (DatasetDict, IterableDatasetDict)):
        for split_name, split_ds in ds_obj.items():
            print(f"[import] Reading split: {split_name}")
            for row in split_ds:
                yield row


def _build_rows_api_headers(token: str) -> dict[str, str]:
    return {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
    }


def _api_get_json(path: str, params: dict[str, str], token: str) -> dict:
    query = urllib.parse.urlencode(params)
    url = f"{ROWS_API_BASE}/{path}?{query}"
    req = urllib.request.Request(url=url, headers=_build_rows_api_headers(token), method="GET")

    attempts = 0
    while True:
        attempts += 1
        try:
            with urllib.request.urlopen(req, timeout=40) as resp:
                body = resp.read().decode("utf-8")
            break
        except HTTPError as e:
            if e.code in {429, 500, 502, 503, 504} and attempts < 8:
                wait_seconds = min(20.0, 1.5 ** attempts)
                print(f"[import] API retry {attempts} for {path} (status={e.code}), sleeping {wait_seconds:.1f}s")
                time.sleep(wait_seconds)
                continue
            raise
        except URLError:
            if attempts < 6:
                wait_seconds = min(12.0, 1.5 ** attempts)
                print(f"[import] Network retry {attempts} for {path}, sleeping {wait_seconds:.1f}s")
                time.sleep(wait_seconds)
                continue
            raise

    data = json.loads(body)
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected API payload type from {path}")
    if data.get("error"):
        raise RuntimeError(str(data["error"]))
    return data


def _fetch_available_splits(dataset_id: str, token: str) -> list[tuple[str, str]]:
    payload = _api_get_json("splits", {"dataset": dataset_id}, token)
    result: list[tuple[str, str]] = []
    for item in payload.get("splits", []):
        if not isinstance(item, dict):
            continue
        config = str(item.get("config", "")).strip()
        split = str(item.get("split", "")).strip()
        if config and split:
            result.append((config, split))
    return result


def _extract_rows_via_api(
    dataset_id: str,
    token: str,
    split: str,
    config: str,
    max_lines: int,
    start_offset: int = 0,
) -> list[str]:
    # Request first page to discover features + total rows.
    first = _api_get_json(
        "rows",
        {
            "dataset": dataset_id,
            "config": config,
            "split": split,
            "offset": str(max(0, start_offset)),
            "length": "100",
        },
        token,
    )

    feature_names = [
        str(f.get("name", "")).strip()
        for f in first.get("features", [])
        if isinstance(f, dict)
    ]
    text_columns = _pick_text_columns(feature_names)
    if not text_columns:
        raise RuntimeError(
            f"Could not detect text columns in split '{split}' / config '{config}'. "
            f"Available columns: {feature_names}"
        )

    num_total = int(first.get("num_rows_total", 0) or 0)
    page_size = int(first.get("num_rows_per_page", 100) or 100)

    seen = set()
    lines: list[str] = []

    def consume_rows(rows_payload: list[dict]):
        for item in rows_payload:
            if not isinstance(item, dict):
                continue
            row = item.get("row")
            if not isinstance(row, dict):
                continue
            for col in text_columns:
                raw = row.get(col)
                if raw is None:
                    continue
                line = _normalize_text(str(raw))
                if not line or line in seen:
                    continue
                seen.add(line)
                lines.append(line)
                break
            if len(lines) >= max_lines:
                return

    consume_rows(first.get("rows", []))
    if len(lines) >= max_lines:
        return lines

    offset = max(0, start_offset) + page_size
    while offset < num_total and len(lines) < max_lines:
        page = _api_get_json(
            "rows",
            {
                "dataset": dataset_id,
                "config": config,
                "split": split,
                "offset": str(offset),
                "length": str(page_size),
            },
            token,
        )
        consume_rows(page.get("rows", []))
        offset += page_size
        # Friendly pacing to reduce API throttling.
        time.sleep(0.15)

    return lines


def main() -> int:
    dataset_id = (os.getenv("HF_DATASET_ID", DEFAULT_DATASET_ID) or "").strip()
    split = (os.getenv("HF_SPLIT", "") or "").strip()
    token = (os.getenv("HF_TOKEN", "") or "").strip()

    if not token:
        raise RuntimeError(
            "HF_TOKEN is required for this gated dataset. "
            "Accept the dataset terms on Hugging Face first, then set HF_TOKEN."
        )

    output_path = _resolve_output_path()
    max_lines = int((os.getenv("CODESWITCH_MAX_LINES", "12000") or "12000").strip())
    overwrite = _truthy(os.getenv("CODESWITCH_OVERWRITE", "true"))
    streaming = _truthy(os.getenv("CODESWITCH_STREAMING", "true"))
    allow_fallback = _truthy(os.getenv("CODESWITCH_ALLOW_FALLBACK", "false"))
    config = (os.getenv("HF_CONFIG", "default") or "default").strip()
    start_offset = int((os.getenv("HF_OFFSET", "0") or "0").strip())

    print(f"[import] Dataset: {dataset_id}")
    print(f"[import] Split: {split or 'all splits'}")
    print(f"[import] Output: {output_path}")
    print(f"[import] Max lines: {max_lines}")
    print(f"[import] Streaming: {streaming}")
    print(f"[import] Fallback loader enabled: {allow_fallback}")
    print(f"[import] Config: {config}")
    print(f"[import] Start offset: {start_offset}")

    lines: list[str] = []

    # Prefer datasets-server rows API because it handles gated datasets without
    # downloading multi-GB parquet files to local disk.
    try:
        targets: list[tuple[str, str]]
        if split:
            targets = [(config, split)]
        else:
            all_splits = _fetch_available_splits(dataset_id, token)
            targets = [(cfg, sp) for (cfg, sp) in all_splits if cfg == config]
            if not targets and all_splits:
                targets = all_splits

        for cfg, sp in targets:
            if len(lines) >= max_lines:
                break
            print(f"[import] API rows from config={cfg}, split={sp}")
            chunk = _extract_rows_via_api(
                dataset_id=dataset_id,
                token=token,
                split=sp,
                config=cfg,
                max_lines=max_lines - len(lines),
                start_offset=start_offset,
            )
            lines.extend(chunk)
    except Exception as api_error:
        print(f"[import] API mode failed: {api_error}")
        if not allow_fallback:
            raise RuntimeError(
                "Rows API failed and fallback is disabled. "
                "Set CODESWITCH_ALLOW_FALLBACK=true if you want datasets-loader fallback."
            ) from api_error
        print("[import] Falling back to datasets loader")

    # Fallback path if API gave nothing.
    if not lines:
        if split:
            ds_obj = load_dataset(dataset_id, split=split, token=token, streaming=streaming)
        else:
            ds_obj = load_dataset(dataset_id, token=token, streaming=streaming)

        sample_columns = []
        if isinstance(ds_obj, (Dataset, IterableDataset)):
            sample_columns = list(getattr(ds_obj, "column_names", []) or [])
        elif isinstance(ds_obj, (DatasetDict, IterableDatasetDict)):
            for _, split_ds in ds_obj.items():
                sample_columns = list(getattr(split_ds, "column_names", []) or [])
                if sample_columns:
                    break

        text_columns = _pick_text_columns(sample_columns)
        if not text_columns:
            raise RuntimeError(
                "Could not detect text column automatically. "
                f"Available columns: {sample_columns}"
            )

        print(f"[import] Using text columns: {text_columns}")

        seen = set()
        for row in _iter_rows(ds_obj):
            if not isinstance(row, dict):
                continue

            for col in text_columns:
                raw = row.get(col)
                if raw is None:
                    continue
                line = _normalize_text(str(raw))
                if not line or line in seen:
                    continue
                seen.add(line)
                lines.append(line)
                break

            if len(lines) >= max_lines:
                break

    if not lines:
        raise RuntimeError("No text lines were extracted from the dataset.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    mode = "w" if overwrite else "a"
    with open(output_path, mode, encoding="utf-8") as f:
        if overwrite:
            f.write("# Auto-generated from Hugging Face dataset\n")
            f.write(f"# source={dataset_id}\n")
            f.write("\n")
        for line in lines:
            f.write(line + "\n")

    print(f"[import] Wrote {len(lines)} lines to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
