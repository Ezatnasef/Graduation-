from __future__ import annotations

from typing import Any, Optional


def apply_xtts_production_safe_patch(model: Any) -> None:
    """Attach a production-safe generation wrapper + tokenizer self-healing.

    Idempotent: can be called multiple times.

    This patch is intentionally defensive to avoid CUDA device-side asserts
    and generation crashes caused by:
      - pad_token_id == eos_token_id conflict
      - missing/invalid attention_mask
      - corrupted/unsafe input_ids
      - out-of-vocab indices
    """

    gpt = getattr(model, "gpt", None)
    if gpt is None:
        return
    if getattr(gpt, "_servia_safe_genlayer_attached", False):
        return

    # Optional torch import (works CPU-only too)
    try:
        import torch  # type: ignore
    except Exception:
        torch = None  # type: ignore

    # -------------------- Tokenizer Safety Layer --------------------
    tokenizer = getattr(gpt, "tokenizer", None)

    def _safe_int(x: Any) -> Optional[int]:
        try:
            if x is None:
                return None
            return int(x)
        except Exception:
            return None

    safe_pad_id: int = 0
    eos_id: Optional[int] = None

    if tokenizer is not None:
        tok_pad = _safe_int(getattr(tokenizer, "pad_token_id", None))
        tok_eos = _safe_int(getattr(tokenizer, "eos_token_id", None))
        eos_id = tok_eos

        # NEVER allow pad_token_id == eos_token_id
        if tok_eos is not None and tok_pad is not None and tok_pad == tok_eos:
            tok_pad = None

        # If PAD missing or conflicting, auto-create a new PAD token.
        if tok_pad is None:
            try:
                if hasattr(tokenizer, "add_special_tokens"):
                    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            except Exception:
                pass
            tok_pad = _safe_int(getattr(tokenizer, "pad_token_id", None))

        # Final guard: ensure pad != eos; if still conflicting, pick a distinct id.
        if tok_pad is None:
            for cand in [getattr(tokenizer, "unk_token_id", None), 0, 1, 2, 3]:
                cand_i = _safe_int(cand)
                if cand_i is None:
                    continue
                if eos_id is not None and cand_i == eos_id:
                    continue
                tok_pad = cand_i
                break

        safe_pad_id = int(tok_pad) if tok_pad is not None else 0

        # Apply tokenizer fixes
        try:
            tokenizer.pad_token_id = safe_pad_id
            tokenizer.padding_side = "right"
        except Exception:
            pass

        # Resize embeddings if new pad token was created.
        try:
            if hasattr(gpt, "resize_token_embeddings"):
                if hasattr(tokenizer, "__len__"):
                    gpt.resize_token_embeddings(len(tokenizer))
        except Exception:
            pass

    # -------------------- Vocab bounds (for CUDA pre-flight) --------------------
    vocab_size: int = 0
    gpt_config = getattr(gpt, "config", None)
    if gpt_config is not None:
        vocab_size = _safe_int(getattr(gpt_config, "vocab_size", None)) or 0

    if vocab_size <= 0:
        try:
            embeds = gpt.get_input_embeddings()
            if embeds is not None and hasattr(embeds, "weight"):
                vocab_size = int(embeds.weight.shape[0])
        except Exception:
            vocab_size = 0

    # Ensure config/generation_config token ids match safe values
    try:
        for cfg in [gpt_config, getattr(gpt, "generation_config", None)]:
            if cfg is None:
                continue
            try:
                setattr(cfg, "pad_token_id", int(safe_pad_id))
            except Exception:
                pass
            if eos_id is not None:
                try:
                    setattr(cfg, "eos_token_id", int(eos_id))
                except Exception:
                    pass
    except Exception:
        pass

    original_generate = getattr(gpt, "generate", None)
    if not callable(original_generate):
        return

    # -------------------- CUDA Crash Prevention + Sanitization --------------------
    def _repair_input_ids(input_ids: Any, device: Any) -> Any:
        if torch is None:
            return input_ids
        try:
            x = input_ids.to(device=device, dtype=torch.long, non_blocking=True)
            if not x.is_contiguous():
                x = x.contiguous()

            # Pre-flight bounds: avoid invalid embedding indices
            # Replace:
            #   - negative values
            #   - zeros
            #   - out-of-vocab
            if vocab_size and vocab_size > 0:
                bad = (x < 0) | (x >= vocab_size) | (x == 0)
                if bad.any():
                    x = torch.where(bad, torch.full_like(x, max(1, safe_pad_id)), x)
            else:
                bad = (x < 0) | (x == 0)
                if bad.any():
                    x = torch.where(bad, torch.full_like(x, max(1, safe_pad_id)), x)

            # Ensure input[0] != 0 (some models assert this)
            if x.numel() > 0:
                flat = x.view(-1)
                if int(flat[0].item()) == 0:
                    flat[0] = max(1, safe_pad_id)
                x = flat.view(x.shape)

            return x.contiguous()
        except Exception:
            # Never crash in pre-flight
            try:
                if torch.is_tensor(input_ids):
                    y = input_ids
                    if y.dtype != torch.long:
                        y = y.long()
                    if not y.is_contiguous():
                        y = y.contiguous()
                    return y
            except Exception:
                pass
            return input_ids

    # -------------------- Safe Generate Wrapper (idempotent) --------------------
    def _safe_generate(*args: Any, **kwargs: Any):
        # Remove conflicting kwargs
        kwargs.pop("pad_token_id", None)
        kwargs.pop("eos_token_id", None)

        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            cand0 = args[0]
            if hasattr(cand0, "shape"):
                input_ids = cand0

        # Repair input ids + always create attention_mask
        if input_ids is not None and torch is not None:
            device = None
            try:
                device = input_ids.device if torch.is_tensor(input_ids) else getattr(gpt, "device", None)
            except Exception:
                device = getattr(gpt, "device", None)
            if device is None:
                device = "cpu"

            input_ids = _repair_input_ids(input_ids, device)
            kwargs["input_ids"] = input_ids

            # attention_mask = (input_ids != pad_token_id).long()
            try:
                pad_id_tensor = torch.tensor(int(safe_pad_id), device=input_ids.device, dtype=input_ids.dtype)
                attention_mask = input_ids.ne(pad_id_tensor).to(dtype=torch.long, device=input_ids.device)
                kwargs["attention_mask"] = attention_mask
            except Exception:
                kwargs["attention_mask"] = torch.ones_like(input_ids, dtype=torch.long)

            # Keep ids in safe range for downstream generate internals
            try:
                kwargs["pad_token_id"] = int(safe_pad_id)
                if eos_id is not None:
                    kwargs["eos_token_id"] = int(eos_id)
            except Exception:
                pass

        return original_generate(*args, **kwargs)

    # Patch only once
    gpt.generate = _safe_generate
    setattr(gpt, "_servia_safe_genlayer_attached", True)

