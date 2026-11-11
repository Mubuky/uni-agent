"""Model-scoped codec for tokenizer, processor, tool-parser, and decode paths.

This layer stays within the model boundary: it applies chat templates, handles
processor-backed multimodal inputs, parses tools, and decodes backend outputs.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from uni_agent.gateway.session.types import UnsupportedMediaError
from uni_agent.gateway.utils import normalize_tool_arguments
from verl.utils.tokenizer import normalize_token_ids
from verl.utils.tokenizer.chat_template import apply_chat_template as _apply_chat_template
from verl.utils.tokenizer.chat_template import initialize_turn_separator
from verl.utils.tokenizer.continuous_token import MergeResult, VLContinuousTokenMixin
from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder

# Map backend stop_reason values into the gateway's internal finish_reason vocabulary.
_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FUNCTION_BLOCK_RE = re.compile(r"<function=([^>]+)>(.*?)</function>", re.DOTALL)

_FREEFORM_PROBE_NAME = "__uni_agent_freeform_probe__"
_FREEFORM_PROBE_BODY = "__uni_agent_freeform_body__"

_FINISH_REASON_MAP = {
    "completed": "stop",
    "stop": "stop",
    "matched_stop": "stop",
    "eos": "stop",
    "length": "length",
    "max_tokens": "length",
    "aborted": "stop",
    "abort": "stop",
}

_SGLANG_TOOL_PARSER_ALIASES = {
    "qwen3_xml": "qwen3_coder",
}

_VLLM_TOOL_PARSER_ALIASES = {
    "qwen": "qwen3_xml",
    "qwen25": "qwen3_xml",
    "qwen3": "qwen3_xml",
}


def _normalize_url_value(value: Any, *, field: str) -> str:
    """Normalize a URL field to a non-empty string."""
    if isinstance(value, dict):
        if "url" not in value:
            raise ValueError(f"{field} must contain a url field")
        value = value["url"]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field}.url must be a non-empty string")
    return value


def _normalize_messages_for_qwen_vision_info(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize OpenAI URL fields only on the view passed to Qwen extraction."""
    normalized_messages = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            normalized_messages.append(message)
            continue
        normalized_content = []
        for block in content:
            if isinstance(block, dict) and ("image_url" in block or "video_url" in block):
                block = dict(block)
                if "image_url" in block:
                    block["image_url"] = _normalize_url_value(block["image_url"], field="image_url")
                if "video_url" in block:
                    block["video"] = _normalize_url_value(block.pop("video_url"), field="video_url")
            normalized_content.append(block)
        normalized_messages.append({**message, "content": normalized_content})
    return normalized_messages


def _canonical_tools_hash(tools: list[dict[str, Any]]) -> str:
    """Return a stable hash for a tool schema independent of dict key order."""
    canonical = json.dumps(tools, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def initialize_generation_prompt(processing_class, **apply_chat_template_kwargs) -> list[int]:
    """Initialize the token suffix inserted by ``add_generation_prompt=True``."""
    without_generation_prompt = normalize_token_ids(
        _apply_chat_template(
            processing_class,
            [{"role": "user", "content": ""}],
            add_generation_prompt=False,
            **apply_chat_template_kwargs,
        )
    )
    with_generation_prompt = normalize_token_ids(
        _apply_chat_template(
            processing_class,
            [{"role": "user", "content": ""}],
            add_generation_prompt=True,
            **apply_chat_template_kwargs,
        )
    )
    if with_generation_prompt[: len(without_generation_prompt)] != without_generation_prompt:
        raise ValueError("Generation prompt is not a stable token suffix")
    return with_generation_prompt[len(without_generation_prompt) :]


def _normalize_messages_for_continuous_tokens(
    messages: list[dict[str, Any]], *, force_system_anchor: bool = False
) -> list[dict[str, Any]]:
    """Keep a leading system message before the Qwen synthetic user anchor.

    ``verl`` can add a dummy user when a chat template requires one.  That
    fallback must come after an existing system message (Qwen3.5 rejects a
    system message that follows a user message).  Return copies so the session
    history owned by the framework is never mutated.
    """

    # TODO: Remove this workaround once our pinned verl revision includes the upstream fix
    # and Qwen3.5 system-only initialization and continuation are verified to preserve
    # system ordering and the immutable Continuous Token prefix without this wrapper.
    normalized = []
    for message in messages:
        copied = dict(message)
        if isinstance(message.get("tool_calls"), list):
            copied["tool_calls"] = list(message["tool_calls"])
        normalized.append(copied)
    if not force_system_anchor and any(message.get("role") == "user" for message in normalized):
        return normalized
    first_non_system = next(
        (index for index, message in enumerate(normalized) if message.get("role") != "system"),
        len(normalized),
    )
    if first_non_system == 0:
        return normalized
    return [
        *normalized[:first_non_system],
        {"role": "user", "content": [{"type": "text", "text": ""}]},
        *normalized[first_non_system:],
    ]


def _split_generation_thinking(text: str, *, prompt_opens_thinking: bool) -> tuple[str | None, str]:
    """Normalize model text when a chat template owns the think opening tag.

    Qwen3.5 appends ``<think>`` to the generation prompt, so its completion
    starts with reasoning and normally contains only ``</think>``. Other
    templates (including MiMo-V2) leave both tags to the model. Keep this
    helper model-agnostic and fail open on malformed output so no generated
    text is silently discarded.
    """
    if not isinstance(text, str) or not text:
        return None, text or ""

    if prompt_opens_thinking:
        opening = "<think>"
        if text.startswith(opening):
            text = text[len(opening) :]
        close = text.find("</think>")
        if close < 0:
            return None, text
        reasoning = text[:close].rstrip("\n")
        content = text[close + len("</think>") :].lstrip("\n")
        return reasoning or None, content

    if text.startswith("<think>"):
        close = text.find("</think>", len("<think>"))
        if close < 0:
            return None, text
        reasoning = text[len("<think>") : close].strip("\n")
        content = text[close + len("</think>") :].lstrip("\n")
        return reasoning or None, content
    return None, text


def _canonicalize_tool_arguments_for_comparison(arguments: Any) -> tuple[str, Any]:
    if isinstance(arguments, dict | list):
        canonical: tuple[str, Any] = ("json", arguments)
    elif isinstance(arguments, str):
        try:
            canonical = ("json", json.loads(arguments))
        except json.JSONDecodeError:
            canonical = ("raw", arguments)
    else:
        canonical = ("raw", arguments)

    if canonical[0] == "json" and isinstance(canonical[1], dict) and set(canonical[1]) == {"input"}:
        return _canonicalize_tool_arguments_for_comparison(canonical[1]["input"])
    return canonical


def _supports_freeform_tool_calls(processing_class) -> bool:
    """Probe whether the active Jinja renders ``tool_call.input`` verbatim."""
    probe = [
        {"role": "user", "content": "__uni_agent_probe_user__"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"name": _FREEFORM_PROBE_NAME, "input": _FREEFORM_PROBE_BODY}],
        },
    ]
    try:
        rendered = processing_class.apply_chat_template(
            probe,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        return False
    return f"<function={_FREEFORM_PROBE_NAME}>{_FREEFORM_PROBE_BODY}</function>" in rendered


def _freeform_call_input(arguments: Any) -> str:
    """Unwrap the internal ``{"input": raw}`` envelope for v2.6 Jinja."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
    if isinstance(arguments, dict) and len(arguments) == 1 and isinstance(arguments.get("input"), str):
        return arguments["input"]
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments, ensure_ascii=False) if arguments is not None else ""


def _tool_name(tool: Any) -> str | None:
    if not isinstance(tool, dict):
        return None
    if isinstance(tool.get("function"), dict):
        name = tool["function"].get("name")
        return str(name) if name else None
    name = tool.get("name") or tool.get("type")
    return str(name) if name else None


class MessageCodec:
    """Model-scoped request codec used by gateway sessions.

    ``_GatewayActor`` owns one codec per actor and injects it into
    ``GatewaySession`` instances. The codec renders chat templates, handles
    multimodal processor inputs, and decodes backend token outputs without
    reading session state.
    """

    def __init__(
        self,
        tokenizer,
        *,
        processor=None,
        vision_info_extractor=None,
        vision_info_extractor_kwargs: dict[str, Any] | None = None,
        tool_parser_name: str | None = None,
        rollout_backend: str | None = None,
        enable_tool_parser_cache: bool = True,
        hf_model_type: str | None = None,
        apply_chat_template_kwargs: dict[str, Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ):
        self._tokenizer = tokenizer
        self._apply_chat_template_kwargs = dict(apply_chat_template_kwargs or {})
        normalized_model_type = (hf_model_type or "").strip().lower()
        reasoning_effort = self._apply_chat_template_kwargs.get("reasoning_effort")
        deepseek_v4_thinking = (
            normalized_model_type == "deepseek_v4"
            and bool(
                self._apply_chat_template_kwargs.get("enable_thinking")
                or self._apply_chat_template_kwargs.get("thinking")
            )
            and reasoning_effort != "none"
        )
        self._mm_processor_kwargs = dict(mm_processor_kwargs or {})
        self._continuous_token_builder = create_continuous_token_builder(
            tokenizer,
            hf_model_type=hf_model_type,
            chat_template_kwargs=self._apply_chat_template_kwargs,
            mm_processor_kwargs=self._mm_processor_kwargs,
            processor=processor,
        )
        self._processor = processor
        self._freeform_tool_calls = _supports_freeform_tool_calls(processor if processor is not None else tokenizer)
        self._vision_info_extractor = vision_info_extractor or self._default_vision_info_extractor
        self._vision_info_extractor_kwargs = dict(vision_info_extractor_kwargs or {})
        processing_class = processor if processor is not None else tokenizer
        if normalized_model_type == "deepseek_v4":
            generation_prompt = "<｜Assistant｜>" + ("<think>" if deepseek_v4_thinking else "</think>")
            self._generation_prompt = normalize_token_ids(tokenizer.encode(generation_prompt, add_special_tokens=False))
            self._turn_separator = []
        elif hasattr(processing_class, "chat_template") and processing_class.chat_template is None:
            self._generation_prompt = []
            self._turn_separator = []
        else:
            self._generation_prompt = initialize_generation_prompt(
                processing_class,
                **self._apply_chat_template_kwargs,
            )
            self._turn_separator = initialize_turn_separator(
                processing_class,
                **self._apply_chat_template_kwargs,
            )
        prompt_text = tokenizer.decode(self._generation_prompt, skip_special_tokens=False)
        self._generation_prompt_opens_thinking = "<think>" in prompt_text and "</think>" not in prompt_text
        self._tool_parser_name = tool_parser_name
        self._rollout_backend = rollout_backend
        self._enable_tool_parser_cache = enable_tool_parser_cache
        # Backend parser construction performs expensive setup, so reuse parsers
        # within this actor-scoped codec. SGLang/vLLM bind tool schemas at
        # construction, while verl receives schemas per extraction call; this is
        # why their cache keys differ. Keep the cache codec-scoped because parser
        # instances may retain mutable request state and dynamic schemas can grow
        # the mapping over the codec lifetime. Callers can disable this
        # optimization for parser implementations that require request-scoped
        # instances.
        self._tool_parser_cache: dict[tuple[str, ...], Any] = {}

    @property
    def mm_processor_kwargs(self) -> dict[str, Any]:
        """Return processor kwargs shared by CT rendering and inference."""
        return dict(self._mm_processor_kwargs)

    @property
    def generation_prompt(self) -> list[int]:
        """Return the configured chat template's generation-prompt token suffix."""
        return list(self._generation_prompt)

    @property
    def turn_separator(self) -> list[int]:
        """Return the configured chat template's inter-turn separator tokens."""
        return list(self._turn_separator)

    async def _default_vision_info_extractor(
        self,
        messages: list[dict[str, Any]],
        *,
        image_patch_size: int,
        **_extra: Any,
    ) -> tuple[list[Any] | None, list[Any] | None]:
        """Resolve Qwen media using a temporary URL-normalized message view."""
        # Keep image dependencies out of text-only requests.
        from qwen_vl_utils import process_vision_info

        return process_vision_info(
            _normalize_messages_for_qwen_vision_info(messages),
            image_patch_size=image_patch_size,
            return_video_metadata=True,
        )

    async def extract_multi_modal_data(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[Any] | None, list[Any] | None]:
        """Return (images, videos) in message/block order, one object per block.

        Custom extractors accept image_patch_size; explicit kwargs override the
        processor value. Videos may include metadata as (video, metadata) tuples.
        """
        has_media = any(
            isinstance(block, dict) and block.get("type") in {"image", "image_url", "video", "video_url"}
            for message in messages
            if isinstance(message.get("content"), list)
            for block in message["content"]
        )
        if not has_media:
            return None, None
        if not self._continuous_token_builder.supports_multimodal():
            raise UnsupportedMediaError("image or video content requires a multimodal processor")

        extractor_kwargs = dict(self._vision_info_extractor_kwargs)
        if "image_patch_size" not in extractor_kwargs:
            extractor_kwargs["image_patch_size"] = self._processor.image_processor.patch_size
        return await self._vision_info_extractor(messages, **extractor_kwargs)

    def model_tools_for(self, tools):
        if tools is None:
            return None
        return [
            tool["custom"]
            if self._freeform_tool_calls and "custom" in tool
            else {key: value for key, value in tool.items() if key != "custom"}
            for tool in tools
        ]

    def model_messages_for(self, messages, tools):
        custom_names = {_tool_name(tool) for tool in tools or [] if "custom" in tool}
        if not custom_names or not self._freeform_tool_calls:
            return messages
        result = []
        for message in messages:
            calls = message.get("tool_calls")
            if not isinstance(calls, list):
                result.append(message)
                continue
            projected = []
            for call in calls:
                function = call.get("function") or {}
                if function.get("name") in custom_names:
                    projected.append(
                        {"name": function["name"], "input": _freeform_call_input(function.get("arguments"))}
                    )
                else:
                    projected.append(call)
            result.append({**message, "tool_calls": projected})
        return result

    def build_initial_tokens(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        image_data: list[Any] | None = None,
        video_data: list[Any] | None = None,
    ) -> list[int]:
        """Build the initial runtime token stream."""
        return self._continuous_token_builder.build_initial_tokens(
            _normalize_messages_for_continuous_tokens(self.model_messages_for(messages, tools)),
            tools=self.model_tools_for(tools),
            images=image_data,
            videos=video_data,
        )

    def merge_assistant_tokens(
        self,
        runtime_token_ids: list[int],
        assistant_token_ids: list[int],
        response_mask: list[int] | None = None,
        response_logprobs: list[float] | None = None,
        *,
        assistant_logprobs: list[float] | None = None,
    ) -> tuple[list[int], list[int], list[float] | None] | MergeResult:
        """Merge model-generated tokens and align response metadata."""
        merge_result = self._continuous_token_builder.merge_assistant_tokens(
            runtime_token_ids,
            assistant_token_ids,
        )
        if response_mask is None:
            return merge_result
        response_mask, response_logprobs = self._continuous_token_builder.align_response_metadata(
            merge_result,
            response_mask,
            response_logprobs,
            assistant_logprobs=assistant_logprobs,
        )
        return merge_result.token_ids, response_mask, response_logprobs

    def validate_incremental_media(self, messages: list[dict[str, Any]]) -> None:
        """Reject appended media unsupported by the Gateway or selected builder."""
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                media_type = block.get("type")
                if media_type in {"video", "video_url", "audio", "audio_url", "input_audio", "output_audio"}:
                    raise UnsupportedMediaError(
                        "Gateway Continuous Token does not support incremental video or audio data"
                    )
                if media_type in {"image", "image_url"} and not isinstance(
                    self._continuous_token_builder, VLContinuousTokenMixin
                ):
                    raise UnsupportedMediaError(
                        "Gateway incremental image inputs require a VL Continuous Token builder"
                    )

    def merge_context_tokens(
        self,
        previous_messages: list[dict[str, Any]],
        updated_messages: list[dict[str, Any]],
        runtime_token_ids: list[int],
        response_mask: list[int] | None = None,
        response_logprobs: list[float] | None = None,
        *,
        tools: list[dict[str, Any]] | None = None,
        image_data: list[Any] | None = None,
        video_data: list[Any] | None = None,
    ) -> tuple[list[int], list[int], list[float] | None] | MergeResult:
        """Merge prepared context and align response metadata.

        image_data covers all updated_messages, including history, in block order.
        Image-bearing VL contexts treat None as [] to disable raw-reference fallback.
        The session validates appended media first; non-empty video_data is unsupported.
        """
        if video_data:
            raise UnsupportedMediaError(
                "Continuous Token context merging does not currently support incremental video data"
            )
        previous_messages = self.model_messages_for(previous_messages, tools)
        updated_messages = self.model_messages_for(updated_messages, tools)
        tools = self.model_tools_for(tools)
        force_system_anchor = any(message.get("role") == "system" for message in previous_messages) and not any(
            message.get("role") == "user" for message in previous_messages
        )
        previous_messages = _normalize_messages_for_continuous_tokens(
            previous_messages, force_system_anchor=force_system_anchor
        )
        updated_messages = _normalize_messages_for_continuous_tokens(
            updated_messages, force_system_anchor=force_system_anchor
        )
        builder = self._continuous_token_builder
        has_images = bool(image_data) or any(
            isinstance(block, dict) and block.get("type") in {"image", "image_url"}
            for message in updated_messages
            if isinstance(message.get("content"), list)
            for block in message["content"]
        )
        if isinstance(builder, VLContinuousTokenMixin) and has_images:
            merge_result = builder.merge_context_tokens(
                previous_messages,
                updated_messages,
                runtime_token_ids,
                tools=tools,
                images=image_data if image_data is not None else [],
            )
        else:
            # Keep text-only contexts compatible with the legacy CT API.
            merge_result = builder.merge_context_tokens(
                previous_messages,
                updated_messages,
                runtime_token_ids,
                tools=tools,
            )
        if response_mask is None:
            return merge_result
        response_mask, response_logprobs = self._continuous_token_builder.align_response_metadata(
            merge_result,
            response_mask,
            response_logprobs,
        )
        return merge_result.token_ids, response_mask, response_logprobs

    def _process_tool_calls_sglang(
        self,
        text: str,
        tools: list[dict[str, Any]],
        parser_name: str,
    ) -> tuple[str, list[Any]]:
        cache_key = ("sglang", parser_name, _canonical_tools_hash(tools))
        parser = self._tool_parser_cache.get(cache_key) if self._enable_tool_parser_cache else None
        if parser is None:
            from sglang.srt.entrypoints.openai.protocol import Function as SglFunction
            from sglang.srt.entrypoints.openai.protocol import Tool as SglTool
            from sglang.srt.function_call.function_call_parser import FunctionCallParser

            sglang_tools = [SglTool(type=tool["type"], function=SglFunction(**tool["function"])) for tool in tools]
            parser = FunctionCallParser(sglang_tools, parser_name)
            if self._enable_tool_parser_cache:
                self._tool_parser_cache[cache_key] = parser

        if not parser.has_tool_call(text):
            return text, []
        content, calls = parser.parse_non_stream(text)
        return content, [SimpleNamespace(name=call.name, arguments=call.parameters) for call in calls]

    def _process_tool_calls_vllm(
        self,
        text: str,
        tools: list[dict[str, Any]],
        parser_name: str,
    ) -> tuple[str, list[Any]]:
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionToolsParam

        cache_key = ("vllm", parser_name, _canonical_tools_hash(tools))
        parser = self._tool_parser_cache.get(cache_key) if self._enable_tool_parser_cache else None
        vllm_tools = [ChatCompletionToolsParam(**tool) if isinstance(tool, dict) else tool for tool in tools]
        if parser is None:
            from vllm.tool_parsers import ToolParserManager

            parser_cls = ToolParserManager.get_tool_parser(parser_name)
            parser_parameters = inspect.signature(parser_cls).parameters
            if "tools" in parser_parameters:
                parser = parser_cls(self._tokenizer, tools=vllm_tools)
            else:
                parser = parser_cls(self._tokenizer)
            if self._enable_tool_parser_cache:
                self._tool_parser_cache[cache_key] = parser

        request = SimpleNamespace(tools=vllm_tools, tool_choice="auto", skip_special_tokens=True)

        # Complete responses do not need streaming argument deltas. Repeated partial
        # conversions can become quadratic for long tool arguments; disabling them
        # preserves final conversion. Detect engine capabilities rather than parser
        # names so aliases and other compatible parsers benefit, while older parsers
        # keep the standard path. Restore the flag below for cached parser reuse.
        engine = getattr(parser, "_parser_engine", None)
        should_disable_deltas = (
            engine is not None
            and hasattr(engine, "_arg_converter")
            and getattr(engine, "_arg_converter", None) is not None
            and hasattr(engine, "_stream_arg_deltas")
        )

        if should_disable_deltas:
            saved_stream_arg_deltas = engine._stream_arg_deltas
            engine._stream_arg_deltas = False
            try:
                parsed = parser.extract_tool_calls(text, request)
            finally:
                engine._stream_arg_deltas = saved_stream_arg_deltas
        else:
            parsed = parser.extract_tool_calls(text, request)
        if not parsed.tools_called:
            return text, []
        return parsed.content or "", [tool_call.function for tool_call in parsed.tool_calls]

    async def _process_tool_calls_verl(
        self,
        response_ids: list[int],
        tools: list[dict[str, Any]],
        parser_name: str,
    ) -> tuple[str, list[Any]]:
        """Parse tool calls with verl's built-in tool-parser registry."""
        from verl.experimental.agent_loop.tool_parser import ToolParser
        from verl.tools.schemas import OpenAIFunctionToolSchema

        cache_key = ("verl", parser_name)
        parser = self._tool_parser_cache.get(cache_key) if self._enable_tool_parser_cache else None
        if parser is None:
            parser = ToolParser.get_tool_parser(parser_name, self._tokenizer)
            if self._enable_tool_parser_cache:
                self._tool_parser_cache[cache_key] = parser

        tool_schemas = [OpenAIFunctionToolSchema.model_validate(tool) for tool in tools]
        content, calls = await parser.extract_tool_calls(response_ids, tool_schemas)
        return content, [SimpleNamespace(name=call.name, arguments=call.arguments) for call in calls]

    async def _extract_tool_calls(
        self,
        response_ids: list[int],
        tools: list[dict[str, Any]],
        parser_name: str,
    ) -> tuple[str, list[Any]]:
        tools = [{key: value for key, value in tool.items() if key != "custom"} for tool in tools]
        text = self._tokenizer.decode(response_ids, skip_special_tokens=False)
        pattern = r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>"
        blocks = list(re.finditer(pattern, text, re.DOTALL))
        xml_parser = parser_name in {"qwen", "qwen25", "qwen3", "qwen3_xml", "qwen3_coder"}
        if xml_parser and not blocks and self._has_malformed_tool_envelope(text):
            return self._malformed_tool_call_result(text)
        if xml_parser and any(match[2].strip() and "<parameter=" not in match[2] for match in blocks):
            calls = []
            for match in blocks:
                if match[2].strip() and "<parameter=" not in match[2]:
                    calls.append(SimpleNamespace(name=match[1].strip(), arguments=match[2], raw_arguments=True))
                else:
                    ids = self._tokenizer.encode(match[0], add_special_tokens=False)
                    _, parsed = await self._extract_tool_calls(ids, tools, parser_name)
                    calls.extend(parsed)
            remaining = re.sub(pattern, "", text, flags=re.DOTALL)
            if self._has_malformed_tool_envelope(remaining):
                remaining, malformed = self._malformed_tool_call_result(remaining)
                calls.extend(malformed)
            return remaining, calls
        parser_backend = self._rollout_backend if self._rollout_backend in {"sglang", "vllm"} else "verl"

        try:
            if parser_backend == "sglang":
                sglang_name = _SGLANG_TOOL_PARSER_ALIASES.get(parser_name, parser_name)
                return self._process_tool_calls_sglang(text, tools, sglang_name)
            if parser_backend == "vllm":
                vllm_name = _VLLM_TOOL_PARSER_ALIASES.get(parser_name, parser_name)
                return self._process_tool_calls_vllm(text, tools, vllm_name)
            return await self._process_tool_calls_verl(response_ids, tools, parser_name)
        except Exception as exc:
            raise RuntimeError(f"{parser_backend} tool parser {parser_name!r} failed") from exc

    async def decode_response(
        self,
        response_ids: list[int],
        *,
        tools: list[dict[str, Any]] | None = None,
        stop_reason: str | None = None,
    ) -> tuple[dict[str, Any], str]:
        """Decode model output tokens into an assistant message and finish reason."""
        if self._tool_parser_name and tools:
            content, function_calls = await self._extract_tool_calls(
                response_ids,
                tools,
                self._tool_parser_name,
            )
            if function_calls:
                tool_calls = [
                    {
                        "id": f"call_{uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": fc.name,
                            "arguments": fc.arguments
                            if getattr(fc, "raw_arguments", False)
                            else normalize_tool_arguments(fc.arguments),
                        },
                    }
                    for fc in function_calls
                ]
                reasoning, visible = _split_generation_thinking(
                    self._strip_trailing_stop_token(content or ""),
                    prompt_opens_thinking=self._generation_prompt_opens_thinking,
                )
                message = {
                    "role": "assistant",
                    "content": visible,
                    "tool_calls": tool_calls,
                }
                if reasoning:
                    message["reasoning_content"] = reasoning
                return message, "length" if _FINISH_REASON_MAP.get(stop_reason) == "length" else "tool_calls"
        response_text = self._tokenizer.decode(response_ids, skip_special_tokens=True)
        finish_reason = _FINISH_REASON_MAP.get(stop_reason, stop_reason) if stop_reason else "stop"
        reasoning, visible = _split_generation_thinking(
            response_text,
            prompt_opens_thinking=self._generation_prompt_opens_thinking,
        )
        message = {"role": "assistant", "content": visible}
        if reasoning:
            message["reasoning_content"] = reasoning
        return message, finish_reason

    def canonicalize_message_for_prefix_comparison(self, message: dict[str, Any]) -> dict[str, Any]:
        """Canonicalize one message before session prefix comparison."""
        normalized = dict(message)
        normalized.pop("tool_call_id", None)
        tool_calls = normalized.get("tool_calls")
        if not isinstance(tool_calls, list):
            return normalized

        normalized_tool_calls: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            normalized_tool_call = dict(tool_call)
            normalized_tool_call.pop("id", None)
            normalized_tool_call["type"] = "function"
            function = normalized_tool_call.get("function")
            if isinstance(function, dict) and "arguments" in function:
                normalized_function = dict(function)
                normalized_function["arguments"] = _canonicalize_tool_arguments_for_comparison(function["arguments"])
                normalized_tool_call["function"] = normalized_function
            normalized_tool_calls.append(normalized_tool_call)
        normalized["tool_calls"] = normalized_tool_calls
        return normalized

    def _strip_trailing_stop_token(self, content: str) -> str:
        """Remove terminal EOS text from the visible tool-call content."""
        eos = getattr(self._tokenizer, "eos_token", None)
        if not isinstance(eos, str) or not eos.strip():
            return content
        stripped = content.rstrip()
        while stripped.endswith(eos):
            stripped = stripped[: -len(eos)].rstrip()
        return stripped if stripped != content.rstrip() else content

    @staticmethod
    def _malformed_tool_call(name: str, body: str) -> SimpleNamespace:
        del name
        return SimpleNamespace(
            name="__malformed_tool_call__",
            arguments=json.dumps(
                {"error": "malformed or incomplete tool call", "raw_body": body},
                ensure_ascii=False,
            ),
        )

    @classmethod
    def _malformed_tool_call_from_text(cls, text: str) -> SimpleNamespace:
        function_match = _FUNCTION_BLOCK_RE.search(text)
        if function_match is None:
            return cls._malformed_tool_call("", text)
        return cls._malformed_tool_call(function_match.group(1).strip(), function_match.group(2))

    @classmethod
    def _malformed_tool_call_result(cls, text: str) -> tuple[str, list[SimpleNamespace]]:
        start = text.find("<tool_call>")
        return text[:start], [cls._malformed_tool_call_from_text(text[start:])]

    @staticmethod
    def _has_malformed_tool_envelope(text: str) -> bool:
        """Detect an incomplete XML tool envelope without parsing arguments."""
        if text.count("<tool_call>") > text.count("</tool_call>"):
            return True
        for match in _TOOL_CALL_BLOCK_RE.finditer(text):
            inner = match.group(1)
            if "<function=" in inner and _FUNCTION_BLOCK_RE.search(inner) is None:
                return True
        return "<tool_call>" in text and text.count("<function=") > text.count("</function>")
