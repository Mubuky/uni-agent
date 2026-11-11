"""Optional generation-level signals from MiMo's long-horizon training recipe."""

from dataclasses import replace

import numpy as np
import torch
from omegaconf import OmegaConf

from uni_agent.gateway.session.types import Trajectory


def signal_context(trajectory):
    return trajectory.extra_fields.get("runner_reward_info", {}).get("reward_context", {})


_REPETITION_HASH_BASE = np.uint64(1099511628211)  # FNV-1a 64-bit prime, odd so invertible mod 2**64
_REPETITION_HASH_BASE_INV = np.uint64(pow(int(_REPETITION_HASH_BASE), -1, 1 << 64))
_REPETITION_DETECT_DEFAULTS: dict[str, object] = {
    "enable": False,
    "ngram_size": 300,
    "window_size": 8300,
    "min_content_length": 16384,
    "min_repeat": 15,
    "zero_reward": False,
}


def _normalize_repetition_detect_config(raw: object) -> dict[str, object]:
    """Validate ``agent_framework.repetition_detect`` and fill the reference defaults."""
    if raw is None:
        raw = {}
    if OmegaConf.is_config(raw):
        raw = OmegaConf.to_container(raw, resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("actor_rollout_ref.rollout.custom.agent_framework.repetition_detect must be a mapping")
    unknown = set(raw) - set(_REPETITION_DETECT_DEFAULTS)
    if unknown:
        raise ValueError(f"repetition_detect has unknown keys: {sorted(unknown)}")
    cfg = dict(_REPETITION_DETECT_DEFAULTS)
    cfg.update(raw)
    for key in ("enable", "zero_reward"):
        if type(cfg[key]) is not bool:
            raise ValueError(f"repetition_detect.{key} must be a bool")
    for key in ("ngram_size", "window_size", "min_repeat"):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"repetition_detect.{key} must be a positive int")
    gate = cfg["min_content_length"]
    if isinstance(gate, bool) or not isinstance(gate, int) or gate < 0:
        raise ValueError("repetition_detect.min_content_length must be a non-negative int")
    if cfg["window_size"] < cfg["ngram_size"]:
        raise ValueError("repetition_detect.window_size must be >= ngram_size")
    return cfg


def _rolling_ngram_hashes(token_ids: np.ndarray, ngram_size: int) -> np.ndarray:
    """64-bit polynomial hash of every ``ngram_size``-token window, wrapping mod 2**64.

    H[i] = sum_{j<n} (t[i+j]+1) * B**(n-1-j). Computed as
    B**(i+n-1) * (S[i+n] - S[i]) with S the prefix sum of (t[j]+1) * B**(-j); B is odd so
    its inverse modulo 2**64 exists and every product wraps consistently in uint64.
    """
    n = int(ngram_size)
    tokens = np.asarray(token_ids, dtype=np.uint64) + np.uint64(1)
    count = tokens.shape[0] - n + 1
    if count <= 0:
        return np.empty(0, dtype=np.uint64)
    with np.errstate(over="ignore"):
        inv_powers = np.cumprod(np.full(tokens.shape[0], _REPETITION_HASH_BASE_INV, dtype=np.uint64))
        inv_powers = np.concatenate([np.ones(1, dtype=np.uint64), inv_powers[:-1]])  # B**(-j)
        prefix = np.concatenate([np.zeros(1, dtype=np.uint64), np.cumsum(tokens * inv_powers, dtype=np.uint64)])
        powers = np.cumprod(np.full(tokens.shape[0], _REPETITION_HASH_BASE, dtype=np.uint64))  # B**(i+1)
        scale = powers[n - 2 : n - 2 + count] if n >= 2 else np.ones(count, dtype=np.uint64)
        return (prefix[n : n + count] - prefix[:count]) * scale


def repetition_first_hit(
    token_ids,
    *,
    ngram_size: int = 300,
    window_size: int = 8300,
    min_content_length: int = 16384,
    min_repeat: int = 15,
) -> int:
    """Token offset (exclusive end of the repeated n-gram) of the first repetition hit, or -1.

    Same predicate as the reference ``_check_repetition_tokens``: at position ``i`` the window
    holds the n-grams starting in ``[i + ngram_size - window_size, i]``; a hit is the first
    ``i`` whose n-gram occurs at least ``min_repeat`` times in that window.
    """
    seq_len = len(token_ids)
    if seq_len < min_content_length or seq_len < ngram_size:
        return -1
    if min_repeat <= 1:
        return ngram_size
    hashes = _rolling_ngram_hashes(token_ids, ngram_size)
    order = np.argsort(hashes, kind="stable")  # groups equal hashes, positions ascending within a group
    sorted_hashes = hashes[order]
    positions = order.astype(np.int64)
    lag = min_repeat - 1
    if positions.shape[0] <= lag:
        return -1
    same = sorted_hashes[lag:] == sorted_hashes[:-lag]
    within = (positions[lag:] - positions[:-lag]) <= (window_size - ngram_size)
    candidates = positions[lag:][same & within]
    if candidates.size == 0:
        return -1
    return int(candidates.min()) + ngram_size


def _repetition_hit_segments(trajectory: Trajectory, cfg: dict[str, object]) -> list[int]:
    """Indices of generation spans whose tokens trip ``repetition_first_hit``."""
    hits: list[int] = []
    response_ids = np.asarray(trajectory.response_ids, dtype=np.int64)
    for index, span in enumerate(trajectory.generation_spans):
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            continue
        start, end = int(span[0]), int(span[1])
        if start < 0 or end <= start or end > response_ids.shape[0]:
            continue
        offset = repetition_first_hit(
            response_ids[start:end],
            ngram_size=int(cfg["ngram_size"]),
            window_size=int(cfg["window_size"]),
            min_content_length=int(cfg["min_content_length"]),
            min_repeat=int(cfg["min_repeat"]),
        )
        if offset >= 0:
            hits.append(index)
    return hits


def _repetition_mask(trajectory: Trajectory) -> tuple[torch.Tensor, str, int]:
    """Expand ``reward_info.repetition_hit_segments`` onto response-token offsets.

    Mirrors ``_tool_call_error_mask``: the framework decided the hits at scoring time
    (``_annotate_repetition``); here they are only projected through ``generation_spans``.
    Fail closed on any inconsistency so a bad span cannot mask unrelated tokens.
    """
    mask = torch.zeros(len(trajectory.response_ids), dtype=torch.float32)
    reward_info = signal_context(trajectory) if isinstance(signal_context(trajectory), dict) else {}
    hits = reward_info.get("repetition_hit_segments")
    if hits is None:
        return mask, "not_detected", 0
    if not isinstance(hits, (list, tuple)):
        return mask, "invalid_hits", 0
    spans = trajectory.generation_spans
    marked = 0
    for index in hits:
        if not isinstance(index, int) or isinstance(index, bool) or index < 0 or index >= len(spans):
            return torch.zeros_like(mask), "invalid_hit_index", 0
        span = spans[index]
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            return torch.zeros_like(mask), "invalid_span", 0
        try:
            start, end = int(span[0]), int(span[1])
        except (TypeError, ValueError, OverflowError):
            return torch.zeros_like(mask), "invalid_span", 0
        if start < 0 or end < start or end > len(mask):
            return torch.zeros_like(mask), "invalid_span", 0
        mask[start:end] = 1.0
        marked += 1
    return mask, "ok", marked


def _turn_index(trajectory: Trajectory) -> tuple[torch.Tensor, str]:
    """Per-response-token generation-turn index projected through ``generation_spans``.

    Tokens inside span ``i`` get value ``i``; tokens outside every span (observations,
    tool results) get ``-1``. Consumed by the trainer's experimental
    ``algorithm.deep_failure_mask`` channel. Fails closed (all ``-1``) on any span
    inconsistency so a bad span can never mark unrelated tokens.
    """
    n = len(trajectory.response_ids)
    index = torch.full((n,), -1, dtype=torch.long)
    spans = trajectory.generation_spans
    if not isinstance(spans, (list, tuple)):
        return index, "invalid_spans"
    prev_end = 0
    for i, span in enumerate(spans):
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            return torch.full((n,), -1, dtype=torch.long), "invalid_span"
        try:
            start, end = int(span[0]), int(span[1])
        except (TypeError, ValueError, OverflowError):
            return torch.full((n,), -1, dtype=torch.long), "invalid_span"
        if start < prev_end or end < start or end > n:
            return torch.full((n,), -1, dtype=torch.long), "invalid_span"
        index[start:end] = i
        prev_end = end
    return index, "ok"


def _tool_call_error_mask(trajectory: Trajectory) -> tuple[torch.Tensor, str, int]:
    """Expand per-generation tool errors onto response-token offsets.

    The runner owns error classification while the gateway owns token spans.
    Refuse to guess when their lengths disagree: a zero mask plus an explicit
    alignment status is safer than penalizing the wrong assistant turn.
    """
    mask = torch.zeros(len(trajectory.response_ids), dtype=torch.float32)
    reward_info = signal_context(trajectory) if isinstance(signal_context(trajectory), dict) else {}
    flags = reward_info.get("tool_call_error_flags")
    spans = trajectory.generation_spans
    if flags is None:
        return mask, "missing_flags", 0
    if not isinstance(flags, (list, tuple)):
        return mask, "invalid_flags", 0
    if len(flags) != len(spans):
        return mask, "length_mismatch", 0

    marked = 0
    for flag, span in zip(flags, spans, strict=True):
        if not flag:
            continue
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            return torch.zeros_like(mask), "invalid_span", 0
        try:
            start, end = int(span[0]), int(span[1])
        except (TypeError, ValueError, OverflowError):
            return torch.zeros_like(mask), "invalid_span", 0
        if start < 0 or end < start or end > len(mask):
            return torch.zeros_like(mask), "invalid_span", 0
        mask[start:end] = 1.0
        marked += 1
    return mask, "ok", marked


def annotate_repetition(trajectories, config):
    if not config["enable"]:
        return trajectories
    hits = [_repetition_hit_segments(trajectory, config) for trajectory in trajectories]
    zero_reward = config["zero_reward"] and any(hits)
    result = []
    for trajectory, segments in zip(trajectories, hits, strict=True):
        context = {**signal_context(trajectory), "repetition_hit_segments": segments, "repetition_hit": bool(segments)}
        if zero_reward:
            context.update(repetition_original_reward=trajectory.reward_score, repetition_zeroed_reward=True)
        extra = {
            **trajectory.extra_fields,
            "runner_reward_info": {**trajectory.extra_fields.get("runner_reward_info", {}), "reward_context": context},
        }
        result.append(
            replace(trajectory, extra_fields=extra, reward_score=0.0 if zero_reward else trajectory.reward_score)
        )
    return result
