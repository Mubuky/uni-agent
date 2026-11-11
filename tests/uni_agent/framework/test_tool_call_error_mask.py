import torch

from uni_agent.framework.trajectory_signals import _tool_call_error_mask
from uni_agent.gateway.session import Trajectory


def _trajectory(*, flags, spans=((0, 2), (2, 4)), response_len=4):
    return Trajectory(
        prompt_ids=[1],
        response_ids=list(range(response_len)),
        response_mask=[1] * response_len,
        generation_spans=list(spans),
        extra_fields={
            "runner_reward_info": {"reward_context": {"tool_call_error_flags": flags} if flags is not None else {}}
        },
    )


def test_tool_call_error_mask_expands_flags_over_generation_spans():
    mask, status, segments = _tool_call_error_mask(_trajectory(flags=[True, False]))

    assert torch.equal(mask, torch.tensor([1.0, 1.0, 0.0, 0.0]))
    assert status == "ok"
    assert segments == 1


def test_tool_call_error_mask_fails_closed_for_missing_or_mismatched_flags():
    missing, missing_status, _ = _tool_call_error_mask(_trajectory(flags=None))
    mismatch, mismatch_status, _ = _tool_call_error_mask(_trajectory(flags=[True]))

    assert not missing.any()
    assert missing_status == "missing_flags"
    assert not mismatch.any()
    assert mismatch_status == "length_mismatch"


def test_tool_call_error_mask_fails_closed_for_invalid_span():
    trajectory = _trajectory(flags=[True, False], spans=((0, 99), (2, 4)))

    mask, status, segments = _tool_call_error_mask(trajectory)

    assert not mask.any()
    assert status == "invalid_span"
    assert segments == 0


def test_tool_call_error_mask_fails_closed_for_non_numeric_span():
    trajectory = _trajectory(flags=[True, False], spans=(("bad", 2), (2, 4)))

    mask, status, segments = _tool_call_error_mask(trajectory)

    assert not mask.any()
    assert status == "invalid_span"
    assert segments == 0
