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
    return responses_to_internal(
        {"input": items, "tools": tools or []},
        base_sampling_params={"logprobs": True},
        allowed_sampling_keys=frozenset(),
    )


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


@pytest.mark.asyncio
async def test_stream_sends_progress_before_generation_and_cancels_on_disconnect():
    import json

    from uni_agent.gateway.adapters.openai_response import responses_stream_response

    started, cancelled = asyncio.Event(), asyncio.Event()

    async def generate():
        started.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    response = responses_stream_response(generate, model="fixture", heartbeat_interval_s=0.001)
    stream = response.body_iterator
    created = json.loads((await anext(stream)).decode().split("data: ")[1])
    assert created["type"] == "response.created" and not started.is_set()
    await anext(stream)
    heartbeat = json.loads((await anext(stream)).decode().split("data: ")[1])
    assert heartbeat["type"] == "response.in_progress" and started.is_set()
    await stream.aclose()
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_stream_failure_is_reported_without_a_completed_event():
    from uni_agent.gateway.adapters.openai_response import responses_stream_response

    async def generate():
        raise ValueError("backend failure")

    response = responses_stream_response(generate, model="fixture")
    wire = b"".join([chunk async for chunk in response.body_iterator])
    assert b"event: response.failed" in wire and b"event: error" in wire
    assert b"event: response.completed" not in wire


@pytest.mark.asyncio
async def test_incomplete_xml_call_returns_harness_feedback():
    codec = MessageCodec(FakeTokenizer(), tool_parser_name="qwen3_xml")
    text = "<tool_call><function=exec>await tools.exec_command("
    message, finish = await codec.decode_response(
        list(map(ord, text)), tools=[{"type": "function", "function": {"name": "exec"}}], stop_reason="length"
    )
    assert finish == "length"
    assert message["tool_calls"][0]["function"]["name"] == "__malformed_tool_call__"
    assert text in message["tool_calls"][0]["function"]["arguments"]["raw_body"]


def test_custom_template_view_keeps_raw_input_and_schema():
    codec = MessageCodec(FakeTokenizer())
    tools = [{"type": "custom", "name": "exec", "format": {"type": "text"}}]
    internal = request(
        [
            {"role": "user", "content": "work"},
            {"type": "custom_tool_call", "call_id": "a", "name": "exec", "input": "text(1);"},
        ],
        tools,
    )
    codec._freeform_tool_calls = True
    assert codec.model_tools_for(internal["tools"]) == tools
    projected = codec.model_messages_for(internal["messages"], internal["tools"])
    assert projected[-1]["tool_calls"] == [{"name": "exec", "input": "text(1);"}]
    codec._freeform_tool_calls = False
    assert codec.model_tools_for(internal["tools"])[0]["function"]["parameters"]["required"] == ["input"]
    assert "custom" in internal["tools"][0]


@pytest.mark.asyncio
@pytest.mark.parametrize("tree", [False, True])
async def test_http_responses_and_activity_routes_use_native_session(tree):
    import httpx

    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.gateway import _GatewayActor

    actor = _GatewayActor(
        GatewayActorConfig(tokenizer=FakeTokenizer(), use_tree_trajectory_manager=tree), Backend("done")
    )
    actor._server_base_url = "http://gateway.test"
    handle = await actor.create_session("http", sampling_params={"logprobs": True})
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=actor._app), base_url=actor._server_base_url
    ) as client:
        bad = await client.post(handle.base_url + "/responses", json={"input": "go", "max_output_tokens": 0})
        assert bad.status_code == 400
        response = await client.post(handle.base_url + "/responses", json={"input": "go", "stream": True})
        assert response.status_code == 200
        assert "event: response.created" in response.text
        assert "event: response.completed" in response.text
        activity = await client.post(handle.activity_url, json={"agent_finished": True})
        assert activity.status_code == 200
        assert (await actor.get_session_state("http"))["agent_finished"] is True
    trajectories = await actor.finalize_session("http")
    assert trajectories[0].response_ids == list(map(ord, "done"))
    assert trajectories[0].response_logprobs == [-0.25] * 4
