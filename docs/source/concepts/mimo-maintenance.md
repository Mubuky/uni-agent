# MiMo maintenance branch

`Mubuky/uni-agent:mimo-maint` builds on upstream Uni-Agent and retains the generic
MiMo features that upstream has not replaced. The branch uses upstream's native
Continuous Token codec, linear and tree session managers, typed runner results,
TransferQueue serialization, and multi-policy interfaces.

## Source revisions

Reviewed on 2026-11-11:

| Source | Revision |
| --- | --- |
| `verl-project/uni-agent:main` | `78348025ac6a5383ad62f17f78e67e943472caef` |
| `XiaomiMiMo/uni-agent:mimo-oss` | `c63e0b01c375ebede95e01fe92bc367df24e5bf3` |
| Common ancestor | `eac798589593503924dba0e340c6a6b8f186896e` |
| Upstream Uni-Agent's verl base | `2bc6ac46043e31855cea194236ff7548c91817c8` |
| Companion verl gitlink | `d799866371163b0762302f2a7e5a1f09e15ee9d8` |
| Existing downstream MiMo verl | `79090e58195e80a9cfe251c95ae7a78a2132b6e2` |

The MiMo delta is `eac79858..c63e0b01`: 26 files, 5,002 insertions and
104 deletions. The transplant was reviewed against that delta, then compared
with upstream main. It is not a reverse diff replacing main with the older fork.
The upstream main head was checked again after implementation and remained
`78348025`.

## Retained patches

| Capability | Implementation |
| --- | --- |
| Raw custom tools | Retain custom declarations alongside upstream's parser-compatible function view. Use the flat declaration and raw input when the tokenizer template supports freeform calls; otherwise reuse the native function envelope. Preserve raw direct XML JSON/JavaScript bodies and the normal backend parser for parameter-tag bodies. |
| Reasoning and replay | Split template-owned thinking from visible output, retain Anthropic thinking blocks and emit thinking/signature delta events, remove trailing EOS from visible tool content, and compare raw/replayed arguments semantically without changing generated token IDs. |
| Long Responses requests | Start SSE before waiting for model completion, emit `response.in_progress` heartbeats, cancel the generation when the stream closes, and report failures through `response.failed` and `error`. The native output-item serializer, namespaces, custom-tool round trip and completion status are reused. |
| Generation provenance | Export actual generation spans for linear and tree trajectories. Carry them through context merges, rollback, compaction and shared-prefix materialization. Export semantic and template-facing transcripts alongside native token arrays. |
| Training signals | Retain MiMo's optional repetition detector, tool-error masks and turn indices in `trajectory_signals.py`. Read annotations from native `runner_reward_info.reward_context`. Keep ordinary `response_mask` and `loss_mask` under native framework ownership. |
| Failure lifecycle | Keep failed snapshots unscored, preserve lifecycle records, and release aborted routes. Runner waiting and timeouts use the native lifecycle. |

The repetition detector remains disabled by default. `repetition_detect.zero_reward`
is an explicit opt-in session-wide reward transformation. `ship_turn_index` is
also disabled by default. Tool-error flags are projected only when their count
matches captured generation spans; an alignment failure produces a zero auxiliary
mask and an explicit status. These auxiliary masks do not reactivate context or
padding tokens in the native loss mask.

The single-token n-gram case was corrected during the transplant: its rolling
hash now cancels the position-dependent multiplier, with a regression test.
Generation-span tests also cover native context-boundary trimming and empty
assistant turns. When Responses lowering removes an empty assistant item, native
rollback replaces that turn; the exported turn indices follow the surviving
trajectory rather than counting a discarded generation.

### Streaming errors

Request-shape validation still runs before opening a stream. Errors discovered
inside generation, including unsupported media or extractor failures, are reported
in SSE after headers have been sent. Client errors keep `invalid_request_error`;
backend/extractor failures keep `internal_error`. Neither produces a completed
response or a scored training trajectory. Non-streaming requests retain HTTP
error responses. This is the intentional transport difference needed for MiMo's
long-generation heartbeat behavior.

## Changes replaced by native upstream features

| MiMo change | Upstream replacement |
| --- | --- |
| Separate `responses.py` adapter and route | `openai_response.py` and the native gateway route. No duplicate adapter or old-module alias. |
| Developer-role conversion, namespace and hosted-tool normalization | Native Responses lowering and canonical system-message ordering. |
| Custom-tool wrapper and tool-output conversion | Native Responses function projection, raw input round trip and multimodal content validation; only freeform template metadata remains as a patch. |
| Local incremental token encoding | Native verl Continuous Token construction, assistant/context merges and metadata alignment. |
| Multiple context chains | Native linear manager and shared-prefix tree manager; `trajectory_selection=all` and `{uid}_{session_index}_{index}` keys remain native. |
| Multimodal URL projection | Native image/video URL normalization and incremental-media checks. |
| Model EOS injection through the framework | Native backend/model termination configuration; no second tokenizer-to-sampling path. |
| Reward-info HTTP protocol | Native `TaskResult(reward, accuracy, finished, extra_info)` return. |
| Parser auto-detection and fallback | Explicit native rollout-backend dispatch and parser cache. A parser API/import failure remains an error. |
| Mixed-harness recipe labels | Native runner selection, harness identity and framework extension hooks. Recipe-specific data-source rewriting remains with the recipe. |
| Default log-root override | Native explicit `log_dir` / `UNI_AGENT_LOG_DIR`. |

GPU-worker placement, Ray CPU reservation and dataset-specific wall-time overrides
are not transplanted. Execution placement, scheduling, sandbox integration and
business scoring belong to the consuming application. No platform or project
resource identifiers are present in this branch.

## Source-file audit

Every production file in the MiMo delta is accounted for below. The associated
MiMo tests were either adapted to the native interfaces or replaced by upstream
and new regression coverage; the old test files are not copied wholesale.

| MiMo source file | Disposition |
| --- | --- |
| `framework/entry.py` | Native model/backend configuration replaces tokenizer injection. |
| `framework/framework.py` | Retain signals and failure evidence; reuse native framework/result/TQ flow; exclude recipe scheduling and metadata policy. |
| `gateway/adapters/__init__.py` | Reuse upstream exports. |
| `gateway/adapters/anthropic.py` | Retain thinking round trip; keep upstream's parallel tool-result ordering fix. |
| `gateway/adapters/openai.py` | Reuse upstream `openai_chat.py`, including message-level tools. |
| `gateway/adapters/responses.py` | Reuse upstream `openai_response.py`; retain lazy SSE and freeform-template information there. |
| `gateway/adapters/types.py` | Reuse upstream request types. |
| `gateway/gateway.py` | Reuse native routing and policy binding; add lazy Responses generation and abort snapshots. |
| `gateway/manager.py` | Retain state inspection and abort-route cleanup; keep upstream placement. |
| `gateway/session/codec.py` | Retain missing protocol/projection behavior; reuse native Continuous Token and backend parsers. |
| `gateway/session/session.py` | Retain generation spans and failure snapshots on the upstream manager. |
| `gateway/session/types.py` | Retain generation/transcript fields; keep typed reward fields. |

Source tests cover dataset timeout, runner CPU allocation, request-idle timeout,
repetition, tool errors, framework generation, Anthropic/Responses adapters,
gateway actor/manager, continuous tokens, tool dispatch, activity state and
multiple chains. Dataset-timeout, CPU-allocation and activity-idle tests are excluded with their
recipe-level features. Native runner timeout behavior is retained. Retained algorithm tests and new protocol tests are in
`tests/uni_agent/framework/{test_repetition_detect,test_tool_call_error_mask,test_rollout_grouping}.py`
and `tests/uni_agent/gateway/test_mimo_contract.py`; native suites continue to cover
rollback, coalescing, media, tree branches and framework admission.

## Compatibility and validation

The reviewed combination is Uni-Agent code `60f4438`, companion verl
`d799866371163b0762302f2a7e5a1f09e15ee9d8`, and unchanged
`XiaomiMiMo/mimoagent@c3df8469f250b15e7906487a6d52be4c1827a509`.
Commit `710a0bf` pins that companion verl in this repository. Its `uni_agent`
source tree is identical to the tested code (`fb29bc7c978f134070265c38c796d26e717c95df`).

The [verl maintenance branch](https://github.com/Mubuky/verl/tree/mimo-maint)
uses the current Continuous Token API and returns native `TaskResult` from its
MiMo runner. Consumers use `trajectory_postprocessor_fqn` and
`extra_fields.runner_reward_info.reward_context`. Runner waiting, timeouts and
reward delivery follow the upstream lifecycle. The former MiMo verl revision
`79090e58` predates the required codec API.

The isolated runtime uses Python 3.12.12, Torch 2.11.0, vLLM 0.20.2,
Transformers 5.10.4, Ray 2.58.0, TensorDict 0.10.0 and TransferQueue 0.1.11.
All 244 registry dependencies and the three component import paths were checked
against the frozen consumer manifest. Existing FSDP/offload and vLLM staged-wake
fixes remain in the consumer. Installation of Torch 2.13.0/vLLM 0.29.0 was blocked
by remote wheel availability; that runtime combination remains unverified.

| Scope | Result |
| --- | --- |
| Gateway/framework, including native HTTP/SSE, Responses, codec, linear/tree contexts and real Ray gateway actors | 717 passed, 9 skipped, 17 deselected |
| Native verl multi-trajectory GRPO, loss aggregation, Token-TIS, padding and ReplayBuffer with a real TransferQueue 0.1.11 service | 95 passed |
| MiMo runner, consumer routing, token contract, native-result monitoring and optimizer membership | 36 passed, 1 skipped |
| Isolated Codex/model rollout with default compaction threshold and four-GPU optimizer update | Passed: 16 consumed rollouts, nonzero gradient and saved parameter change |
| Forced native Codex compaction | Blocked by upstream Responses rejection of an empty-tool request |

The nine gateway skips require optional `qwen_vl_utils` or `vllm.parser.qwen3`.
The 17 deselected level1 checks exercise provider tunnels or live worker-ready
probes. The consumer skip requires an explicitly configured Docker test image.
A live SGLang deployment is outside this validation scope.

The forced-compaction Codex 0.154.0 trial reached model generation, then failed when native
compaction sent `tools=[]` with `parallel_tool_calls=false`. The current upstream
adapter rejects that flag before generation. A minimal empty-tool request
reproduces the rejection; the earlier MiMo adapter accepted it. Codex constructs
its [compaction prompt](https://github.com/openai/codex/blob/rust-v0.154.0/codex-rs/core/src/compact.rs#L271)
using [Prompt defaults](https://github.com/openai/codex/blob/rust-v0.154.0/codex-rs/core/src/client_common.rs#L37),
so this is not a mimoagent result-contract issue. The branch currently retains
upstream's validation. Live compaction and optimizer consumption of compacted multi-chain trajectories
remain switching requirements.

A separate trial using Codex's default compaction threshold completed one native
optimizer step on four H200 GPUs. It consumed two groups of eight rollouts with
rewards 6/8 and 1/8, with no padding. The gradient norm was `0.28590438`;
202,871,959 elements changed in the native BF16 HF export (L2 delta
`0.02075249`). The saved optimizer state was 75,281,850,592 bytes. This proves a
parameter update, not an improvement in task performance.

All 29 completed sessions passed span/mask/logprob checks: 680,195 generated
tokens and 1,867,213 context tokens. Rollout/training probability correlation was
`0.99843889`, mean absolute probability difference `0.01030278`, and Token-TIS
mean weight `0.98054516` with effective sample size `0.99796887`. The optimizer
consumed 16 of those sessions; three unfinished prefetched sessions remained
ungraded. All captured completed sessions had one context chain. The test job,
GPU allocation and sandbox containers were released after artifact collection.

The uneven-context regression gives one successful rollout three contexts and
seven unsuccessful rollouts one context each. Native GRPO assigns `0.875` to
each successful context and `-0.125` to each unsuccessful rollout. Context,
tool-result and padding positions have zero gradient; Token-TIS preserves those
masks. Compacted request histories use the same native chain construction and
alignment paths as the retained generation-span export.

```bash
# Run from this repository with the companion verl importable.
python -m pytest -q tests/uni_agent/gateway tests/uni_agent/framework -m 'not level1'

# Run from the companion verl checkout.
python -m pytest -q \
  tests/trainer/test_multi_trajectories_advantage_on_cpu.py \
  tests/trainer/ppo/test_loss_aggregation_on_cpu.py \
  tests/trainer/ppo/test_rollout_corr.py \
  tests/trainer/ppo/test_rollout_corr_integration.py \
  tests/trainer/ppo/v1/test_replay_buffer_on_cpu.py \
  tests/trainer/ppo/v1/test_agent_loop_tq_on_cpu.py \
  tests/utils/test_padding_on_cpu.py
```
