"""MiMo protocol and generation provenance on the upstream session managers."""

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch

from tests.uni_agent.support import FakeTokenizer
from uni_agent.framework.framework import AgentRequestTimeout, GatewayAgentFramework
from uni_agent.gateway.adapters.openai_response import responses_build_response, responses_to_internal
from uni_agent.gateway.session import GatewaySession, MessageCodec, SessionHandle
from uni_agent.gateway.session.tree_session import TreeGatewaySession
from uni_agent.tasks import TaskResult
from verl.workers.rollout.replica import TokenOutput


class Backend:
    def __init__(self, *outputs):
        self.outputs = iter(outputs)
        self.prompts = []

    async def generate(self, *, prompt_ids, **kwargs):
        self.prompts.append(list(prompt_ids))
        ids = [ord(c) for c in next(self.outputs)]
        return TokenOutput(token_ids=ids, log_probs=[-0.25] * len(ids), stop_reason="stop")


def request(items, tools=None):
    return responses_to_internal({"input": items, "tools": tools or []}, base_sampling_params={"logprobs": True})


@pytest.mark.asyncio
@pytest.mark.parametrize("session_cls", [GatewaySession, TreeGatewaySession])
async def test_responses_compaction_preserves_tokens_masks_and_generation_spans(session_cls):
    codec = MessageCodec(FakeTokenizer())
    session = session_cls(SessionHandle("compact"), codec, sampling_params={"logprobs": True})
    backend = Backend("first", "second", "after summary")
    initial = [{"role": "user", "content": "work"}]
    await session.run_generation(request(initial), backend)
    await session.run_generation(
        request(
            initial
            + [
                {"role": "assistant", "content": "first"},
                {"role": "user", "content": "tool observation"},
            ]
        ),
        backend,
    )
    await session.run_generation(request([{"role": "user", "content": "compact summary"}]), backend)
    trajectories = await session.finalize()
    assert len(trajectories) == 2
    assert [len(t.generation_spans) for t in trajectories] == [2, 1]
    for trajectory in trajectories:
        expected = [0] * len(trajectory.response_ids)
        for start, end in trajectory.generation_spans:
            expected[start:end] = [1] * (end - start)
        assert trajectory.response_mask == expected
        assert len(trajectory.response_logprobs) == len(expected)
        assert all(
            lp == (-0.25 if mask else 0.0) for lp, mask in zip(trajectory.response_logprobs, expected, strict=True)
        )
        generated = [token for token, mask in zip(trajectory.response_ids, expected, strict=True) if mask]
        text = "".join(map(chr, generated))
        assert text in {"firstsecond", "after summary"}
    framework = GatewayAgentFramework(None, runner_registry={}, ship_turn_index=True)
    field, tag = framework._trajectory_to_tq_field_and_tag(
        trajectory=replace(trajectories[0], reward_score=1.0, finished=True),
        sample_fields={},
        session_index=0,
        global_steps=1,
        uid="question",
    )
    assert torch.equal(field["loss_mask"], field["response_mask"])
    assert (field["turn_index"][field["response_mask"] == 0] == -1).all()
    assert field["session_id"] == 0 and tag["uid"] == "question"


@pytest.mark.asyncio
async def test_custom_exec_raw_body_and_thinking_survive_responses_roundtrip():
    codec = MessageCodec(FakeTokenizer(), tool_parser_name="qwen3_xml")
    raw = 'const x = await tools.exec_command({cmd: "printf hello"});\ntext(x);'
    text = "<think>plan</think><tool_call><function=exec>" + raw + "</function></tool_call>"
    tools = [{"type": "custom", "name": "exec", "description": "Run JavaScript", "format": {"type": "text"}}]
    internal = request([{"role": "user", "content": "run"}], tools)
    message, finish = await codec.decode_response([ord(c) for c in text], tools=internal["tools"])
    assert message["reasoning_content"] == "plan"
    assert message["tool_calls"][0]["function"]["arguments"] == raw
    outcome = SimpleNamespace(assistant_msg=message, finish_reason=finish, prompt_tokens=1, completion_tokens=len(text))
    wire = responses_build_response(outcome, payload={"tools": tools}, model="fixture")
    call = next(item for item in wire["output"] if item["type"] == "custom_tool_call")
    assert call["input"] == raw
    replay = request([{"role": "user", "content": "run"}, *wire["output"]], tools)
    assistant = next(m for m in replay["messages"] if m["role"] == "assistant" and m.get("tool_calls"))
    assert codec.canonicalize_message_for_prefix_comparison(
        assistant
    ) == codec.canonicalize_message_for_prefix_comparison(message)


@pytest.mark.asyncio
@pytest.mark.parametrize("session_cls", [GatewaySession, TreeGatewaySession])
async def test_abort_returns_unscored_snapshot_and_releases_session(session_cls):
    session = session_cls(SessionHandle("aborted"), MessageCodec(FakeTokenizer()), sampling_params={"logprobs": True})
    await session.run_generation(request([{"role": "user", "content": "work"}]), Backend("partial"))
    snapshots = await session.abort()
    assert len(snapshots) == 1 and snapshots[0].reward_score is None
    assert snapshots[0].generation_spans == [(0, len("partial"))]
    assert await session.abort() == []
    assert not session.active_chains


@pytest.mark.asyncio
async def test_idle_wait_returns_typed_result_and_ignores_grading():
    framework = GatewayAgentFramework(
        SimpleNamespace(
            get_session_state=AsyncMock(
                return_value={
                    "agent_finished": True,
                    "awaiting_external_followup": True,
                    "awaiting_external_followup_seconds": 99.0,
                }
            )
        ),
        runner_registry={},
    )
    framework._RUNNER_ACTIVITY_POLL_SECONDS = 0.001

    async def grading():
        await asyncio.sleep(0.01)
        return TaskResult(reward=0.5, finished=True)

    result = await framework._wait_for_runner_task(
        grading(), session_id="s", session_timeout_seconds=1.0, request_idle_timeout_seconds=0.001
    )
    assert result.reward == 0.5
    framework.gateway_manager.get_session_state.return_value["agent_finished"] = False
    pending = asyncio.create_task(asyncio.sleep(1))
    try:
        with pytest.raises(AgentRequestTimeout):
            await framework._wait_for_runner_task(
                pending, session_id="s", session_timeout_seconds=1.0, request_idle_timeout_seconds=0.001
            )
    finally:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
