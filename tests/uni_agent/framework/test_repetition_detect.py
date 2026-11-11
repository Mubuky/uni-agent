"""Repetition guard: repetition detector port + reward zeroing + mask projection."""

import collections
from dataclasses import replace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from uni_agent.framework.trajectory_signals import (
    _normalize_repetition_detect_config,
    _repetition_hit_segments,
    _repetition_mask,
    _turn_index,
    annotate_repetition,
    repetition_first_hit,
    signal_context,
)
from uni_agent.gateway.session import Trajectory


def _reference_implementation(token_ids, ngram_size, window_size, min_content_length, min_repeat):
    """Verbatim port of the reference repetition detector,
    returning the first hit offset (i + ngram_size) instead of a bool."""
    seq_len = len(token_ids)
    if seq_len < min_content_length or seq_len < ngram_size:
        return -1
    ngram_counts: dict[tuple, int] = {}
    queue: collections.deque = collections.deque()
    queue_start = 0
    for i in range(seq_len - ngram_size + 1):
        ngram = tuple(token_ids[i : i + ngram_size])
        queue.append(ngram)
        ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
        min_start = (i + ngram_size) - window_size
        while queue_start < min_start:
            old = queue.popleft()
            count = ngram_counts[old] - 1
            if count:
                ngram_counts[old] = count
            else:
                del ngram_counts[old]
            queue_start += 1
        if ngram_counts[ngram] >= min_repeat:
            return i + ngram_size
    return -1


def test_repetition_first_hit_matches_reference_on_random_sequences():
    rng = np.random.default_rng(1234)
    for _ in range(300):
        n = int(rng.integers(40, 300))
        tokens = rng.integers(0, 6, size=n).tolist()
        if rng.random() < 0.7:
            period = int(rng.integers(1, 5))
            unit = rng.integers(0, 6, size=period).tolist()
            length = int(rng.integers(15, 150))
            start = int(rng.integers(0, max(1, n - length)))
            tokens[start : start + length] = (unit * (length // period + 1))[:length]
        ngram = int(rng.integers(2, 10))
        window = int(rng.integers(ngram, 60))
        min_repeat = int(rng.integers(2, 7))
        expected = _reference_implementation(tokens, ngram, window, 0, min_repeat)
        actual = repetition_first_hit(
            tokens, ngram_size=ngram, window_size=window, min_content_length=0, min_repeat=min_repeat
        )
        assert actual == expected, (tokens, ngram, window, min_repeat)


def test_repetition_first_hit_honours_length_gate_and_window():
    loop = [7] * 5000
    assert repetition_first_hit(loop, ngram_size=300, window_size=8300, min_content_length=16384, min_repeat=15) == -1
    assert repetition_first_hit(loop, ngram_size=300, window_size=8300, min_content_length=0, min_repeat=15) == 314
    # 15 occurrences need starts spanning 14 positions: a window smaller than that never hits.
    assert repetition_first_hit(loop, ngram_size=300, window_size=310, min_content_length=0, min_repeat=15) == -1
    assert (
        repetition_first_hit(list(range(5000)), ngram_size=300, window_size=8300, min_content_length=0, min_repeat=2)
        == -1
    )


def _trajectory(*, response_ids, spans, reward=1.0, reward_info=None):
    return Trajectory(
        prompt_ids=[1],
        response_ids=list(response_ids),
        response_mask=[1] * len(response_ids),
        generation_spans=list(spans),
        extra_fields={"runner_reward_info": {"reward_context": dict(reward_info or {})}},
        reward_score=reward,
    )


def _cfg(**overrides):
    cfg = {"enable": True, "ngram_size": 3, "window_size": 30, "min_content_length": 0, "min_repeat": 4}
    cfg.update(overrides)
    return _normalize_repetition_detect_config(cfg)


def test_repetition_hit_segments_scans_each_generation_span_separately():
    clean = list(range(100, 140))
    loop = [9, 9, 9] * 20
    traj = _trajectory(response_ids=clean + loop + clean, spans=[(0, 40), (40, 100), (100, 140)])
    assert _repetition_hit_segments(traj, _cfg()) == [1]


def test_repetition_mask_projects_hits_and_fails_closed():
    traj = _trajectory(response_ids=list(range(10)), spans=[(0, 4), (4, 10)])
    untouched, status, count = _repetition_mask(traj)
    assert status == "not_detected" and count == 0 and not untouched.any()

    hit = replace(traj, extra_fields={"runner_reward_info": {"reward_context": {"repetition_hit_segments": [1]}}})
    mask, status, count = _repetition_mask(hit)
    assert status == "ok" and count == 1
    assert torch.equal(mask, torch.tensor([0.0] * 4 + [1.0] * 6))

    bad = replace(traj, extra_fields={"runner_reward_info": {"reward_context": {"repetition_hit_segments": [5]}}})
    mask, status, _ = _repetition_mask(bad)
    assert status == "invalid_hit_index" and not mask.any()


def _framework(repetition_detect):
    return _normalize_repetition_detect_config(repetition_detect)


def test_annotate_repetition_zeroes_session_reward_and_stamps_hits():
    loop = [3, 3, 3] * 20
    hit_traj = _trajectory(response_ids=loop, spans=[(0, 60)], reward=1.0, reward_info={"reward": 1.0})
    clean_traj = _trajectory(response_ids=list(range(60)), spans=[(0, 60)], reward=1.0)
    fw = _framework(
        {
            "enable": True,
            "ngram_size": 3,
            "window_size": 30,
            "min_content_length": 0,
            "min_repeat": 4,
            "zero_reward": True,
        }
    )

    out = annotate_repetition([clean_traj, hit_traj], fw)

    assert [t.reward_score for t in out] == [0.0, 0.0]  # session-wide override
    assert signal_context(out[0])["repetition_hit_segments"] == []
    assert signal_context(out[1])["repetition_hit_segments"] == [0]
    assert signal_context(out[1])["repetition_hit"] is True
    assert signal_context(out[1])["repetition_original_reward"] == 1.0
    assert signal_context(out[1])["repetition_zeroed_reward"] is True
    assert signal_context(out[1])["reward"] == 1.0  # the runner's posted value is left as evidence

    mask, status, _ = _repetition_mask(out[1])
    assert status == "ok" and mask.sum() == 60


def test_annotate_repetition_keeps_reward_when_zero_reward_disabled():
    loop = [3, 3, 3] * 20
    traj = _trajectory(response_ids=loop, spans=[(0, 60)], reward=1.0)
    fw = _framework({"enable": True, "ngram_size": 3, "window_size": 30, "min_content_length": 0, "min_repeat": 4})
    (out,) = annotate_repetition([traj], fw)
    assert out.reward_score == 1.0
    assert signal_context(out)["repetition_hit_segments"] == [0]
    assert "repetition_zeroed_reward" not in signal_context(out)


def test_annotate_repetition_disabled_is_identity():
    traj = _trajectory(response_ids=[3, 3, 3] * 20, spans=[(0, 60)], reward=1.0)
    fw = _framework({"enable": False})
    assert annotate_repetition([traj], fw) == [traj]
    assert "repetition_hit_segments" not in signal_context(traj)


@pytest.mark.parametrize(
    "bad",
    [
        {"enable": "yes"},
        {"ngram_size": 0},
        {"window_size": 100, "ngram_size": 300},
        {"unknown_key": 1},
        {"min_repeat": True},
    ],
)
def test_normalize_repetition_detect_config_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        _normalize_repetition_detect_config(bad)


def test_normalize_repetition_detect_config_fills_defaults():
    cfg = _normalize_repetition_detect_config(OmegaConf.create({"enable": True}))
    assert cfg == {
        "enable": True,
        "ngram_size": 300,
        "window_size": 8300,
        "min_content_length": 16384,
        "min_repeat": 15,
        "zero_reward": False,
    }


def test_turn_index_projects_spans_and_fails_closed():
    traj = _trajectory(response_ids=list(range(10)), spans=[(0, 4), (6, 10)])
    index, status = _turn_index(traj)
    assert status == "ok"
    assert index.tolist() == [0, 0, 0, 0, -1, -1, 1, 1, 1, 1]

    overlapping = replace(traj, generation_spans=[(0, 5), (4, 10)])
    index, status = _turn_index(overlapping)
    assert status == "invalid_span" and (index == -1).all()

    beyond = replace(traj, generation_spans=[(0, 4), (6, 11)])
    index, status = _turn_index(beyond)
    assert status == "invalid_span" and (index == -1).all()


def test_single_token_repetitions_are_position_independent():
    assert repetition_first_hit([5, 9, 5, 9, 5], ngram_size=1, window_size=5, min_content_length=0, min_repeat=3) == 5
