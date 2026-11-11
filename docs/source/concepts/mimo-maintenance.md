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
| Upstream Uni-Agent's unchanged verl gitlink | `2bc6ac46043e31855cea194236ff7548c91817c8` |
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
| Activity and lifecycle | Track requests in flight, pending tool calls and activity-based idle time. Expose a separate activity endpoint for `agent_finished` so grading is not mistaken for a stalled tool. Keep failed snapshots unscored, preserve lifecycle records, and release aborted routes. |

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
| Reward-info HTTP protocol | Native `TaskResult(reward, accuracy, finished, extra_info)` return. Activity reporting is separate from scoring. |
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
| `framework/framework.py` | Retain signals, idle waiting and failure evidence; reuse native framework/result/TQ flow; exclude recipe scheduling and metadata policy. |
| `gateway/adapters/__init__.py` | Reuse upstream exports. |
| `gateway/adapters/anthropic.py` | Retain thinking round trip; keep upstream's parallel tool-result ordering fix. |
| `gateway/adapters/openai.py` | Reuse upstream `openai_chat.py`, including message-level tools. |
| `gateway/adapters/responses.py` | Reuse upstream `openai_response.py`; retain lazy SSE and freeform-template information there. |
| `gateway/adapters/types.py` | Reuse upstream request types. |
| `gateway/gateway.py` | Reuse native routing and policy binding; add activity reporting, lazy Responses generation and abort snapshots. |
| `gateway/manager.py` | Retain state inspection and abort-route cleanup; keep upstream placement. |
| `gateway/session/codec.py` | Retain missing protocol/projection behavior; reuse native Continuous Token and backend parsers. |
| `gateway/session/session.py` | Retain activity, generation spans and failure snapshots on the upstream manager. |
| `gateway/session/types.py` | Retain generation/transcript fields and activity URL; keep typed reward fields. |

Source tests cover dataset timeout, runner CPU allocation, request-idle timeout,
repetition, tool errors, framework generation, Anthropic/Responses adapters,
gateway actor/manager, continuous tokens, tool dispatch, activity state and
multiple chains. Dataset-timeout and CPU-allocation tests are excluded with their
recipe-level features. Retained algorithm tests and new protocol tests are in
`tests/uni_agent/framework/{test_repetition_detect,test_tool_call_error_mask,test_rollout_grouping}.py`
and `tests/uni_agent/gateway/test_mimo_contract.py`; native suites continue to cover
rollback, coalescing, media, tree branches and framework admission.

## Compatibility and validation

The existing MiMo verl revision cannot import the new Uni-Agent codec:
`VLContinuousTokenMixin` is absent. Its `create_continuous_token_builder` factory
also predates the current `hf_model_type`/processor API. The branch keeps the
upstream verl gitlink; it does not add an old-codec fallback.

The existing MiMo code runner posts to `SessionHandle.reward_info_url` and returns
`None`. It needs a consuming-project bridge that returns native `TaskResult` and
uses `activity_url` for completion activity. Old imports of
`OpenAICompatibleAgentFramework` and `gateway.adapters.responses`, and consumers
of `Trajectory.reward_info`, must migrate to native names and typed reward
context. These are integration changes, not framework compatibility aliases.

Both verl revisions pass the uneven-context GRPO/loss checks: one successful
rollout with three contexts plus seven unsuccessful rollouts produces advantage
`0.875` on each successful context and `-0.125` on each unsuccessful rollout.
Zero-mask padding and tool/context positions contribute no gradient. Token-level
importance correction also preserves zero-mask positions.

Validation runs use CPU synthetic backends and tokenizer fixtures, plus native
HTTP/ASGI and Ray gateway tests. They do not invoke a model or execute agent tools.
A real Codex CLI/model rollout, GPU optimizer update and learning comparison are
separate switch gates. Tests use the existing environment without changing its
packages: Python 3.12.12, Torch 2.11.0+cu130, Ray 2.58.0, TensorDict 0.10.0 and
TransferQueue 0.1.8. Upstream's test requirements specify TransferQueue 0.1.11;
a production TransferQueue service and trainer deployment with that version have
not been validated here.

Test commands (use separate frozen checkouts):

```bash
PYTHONPATH="$UNI_AGENT:$UPSTREAM_VERL" python -m pytest -q \
  tests/uni_agent/gateway tests/uni_agent/framework -m 'not level1'

# The optimizer checks can also run against the existing MiMo verl checkout.
PYTHONPATH="$MIMO_VERL" python -c '
import runpy
checks = runpy.run_path("tests/uni_agent/framework/test_rollout_grouping.py")
for name, check in checks.items():
    if name.startswith("test_"):
        check()
'
```

### Results

| Frozen Uni-Agent code | Scope | Result |
| --- | --- | --- |
| `9c8a211` | Gateway and framework suites with `-m 'not level1'`, including native HTTP/SSE and real Ray gateway tests | 716 passed, 9 skipped, 17 deselected |
| `eff394c29630e1258b7996aa6ecc2767940fa087` | Same selection after the final context-boundary span fix, excluding `test_gateway_actor_on_cpu.py` and `test_gateway_manager_on_cpu.py` | 634 passed, 9 skipped, 17 deselected |
| Existing MiMo verl `79090e58` | Standalone native grouped-GRPO, gradient-mask and token-importance checks | 2 passed |
| Final Python changes | Ruff, compilation and whitespace checks | Passed |

The final delta after `9c8a211` clips generation spans when native context merging
replaces a generated suffix, and adds two regression cases using native metadata
alignment. All affected codec, session and framework suites were rerun on
`eff394c`; the unchanged HTTP/Ray test coverage is reused from `9c8a211`.
The following documentation commit does not change executable code.

The nine skips are three optional `qwen_vl_utils` cases and six optional
`vllm.parser.qwen3` cases. SGLang payload/API-error distinctions are covered with
parser fixtures; a live SGLang parser/model deployment is not validated. The
17 `level1` cases (provider tunnel checks and live worker-ready probes) are outside
the selected regression scope. No test failure remains in the selected scopes.

This branch is ready for consuming-project integration work, not a direct
replacement for the existing MiMo verl/runner combination. Coordinate the verl
API upgrade, migrate runner and audit consumers, then validate the actual model,
Codex compaction and training service before switching production dependencies.
