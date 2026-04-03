from __future__ import annotations

import json
from typing import Any

from toolcall_gateway._tool_choice import (
    normalize_tool_choice,
    select_tools_for_choice,
    tool_choice_prompt_guidance,
    tool_choice_user_suffix,
)
from toolcall_gateway.models import OpenAIContentPart, OpenAIMessage, OpenAIToolCall

TAGGED_TOOL_PROMPT_PARALLEL = """You are a tool-capable assistant.

You must respond using only the following XML-like tags:
- <think>...</think>
- <tool_calls>[{"name":"ToolName","arguments":{...}}]</tool_calls>
- <final_answer>...</final_answer>

Rules:
- You may output one or more <think> blocks.
- You must then output exactly one terminal block: either <tool_calls> or <final_answer>.
- Do not output any text outside these tags.
- In <tool_calls>, the content must be a valid JSON array. Each item must be an object with keys "name" and "arguments".
- If you need only one tool, still use <tool_calls> with an array of length 1.
- In string values inside <tool_calls>, you must escape quotes, backslashes, and newlines exactly as JSON requires.
- After </tool_calls> or </final_answer>, stop immediately.
- Never generate Observation, tool results, or a second terminal block in the same response.
- Never output <observation>; the system will provide tool results in the next turn.
"""

TAGGED_TOOL_PROMPT_SINGLE = """You are a tool-capable assistant.

You must respond using only the following XML-like tags:
- <think>...</think>
- <tool_call>{"name":"ToolName","arguments":{...}}</tool_call>
- <final_answer>...</final_answer>

Rules:
- You may output one or more <think> blocks.
- You must then output exactly one terminal block: either <tool_call> or <final_answer>.
- Do not output any text outside these tags.
- In <tool_call>, the content must be a valid JSON object with keys "name" and "arguments".
- In string values inside <tool_call>, you must escape quotes, backslashes, and newlines exactly as JSON requires.
- After </tool_call> or </final_answer>, stop immediately.
- Never generate Observation, tool results, or a second terminal block in the same response.
- Never output <observation>; the system will provide tool results in the next turn.
"""

STRICT_SUFFIX_PARALLEL = (
    "Use the strict tagged tool protocol when responding. "
    "Your final response must be optional <think> blocks followed by exactly one "
    "terminal block: <tool_calls> or <final_answer>."
)

STRICT_SUFFIX_SINGLE = (
    "Use the strict tagged tool protocol when responding. "
    "Your final response must be optional <think> blocks followed by exactly one "
    "terminal block: <tool_call> or <final_answer>."
)


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _tool_function_dict(tool: dict[str, Any]) -> dict[str, Any]:
    if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
        return tool["function"]
    return tool


def _normalize_content(
    content: str | list[OpenAIContentPart] | list[dict[str, Any]] | None,
) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content

    parts: list[str] = []
    for item in content:
        if isinstance(item, OpenAIContentPart):
            part_type = item.type
            text = item.text or ""
        elif isinstance(item, dict):
            part_type = str(item.get("type") or "")
            text = str(item.get("text") or "")
        else:
            continue

        if part_type == "text" and text:
            parts.append(text)
    return "\n".join(parts)


def _normalize_tool_calls(
    tool_calls: list[OpenAIToolCall | dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in tool_calls:
        raw = (
            item.model_dump(exclude_none=True)
            if isinstance(item, OpenAIToolCall)
            else item
        )
        function = raw.get("function") or {}
        raw_args = function.get("arguments", {})
        if isinstance(raw_args, str):
            try:
                arguments = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                arguments = {"raw": raw_args}
        elif isinstance(raw_args, dict):
            arguments = raw_args
        else:
            arguments = {"raw": str(raw_args)}

        normalized.append(
            {
                "id": str(raw.get("id") or ""),
                "name": str(function.get("name") or ""),
                "arguments": arguments,
            }
        )
    return normalized


def format_tools_for_prompt(tools: list[dict[str, Any]]) -> str:
    if not tools:
        return ""

    lines: list[str] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = _tool_function_dict(tool)
        if not isinstance(function, dict):
            continue

        name = str(function.get("name") or "").strip()
        if not name:
            continue
        description = str(function.get("description") or function.get("summary") or "")
        params = function.get("parameters") or function.get("input_schema") or {}
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                params = {}
        props_raw = params.get("properties") if isinstance(params, dict) else None
        props = props_raw if isinstance(props_raw, dict) else {}
        required = (
            set(params.get("required") or []) if isinstance(params, dict) else set()
        )
        args_desc = ", ".join(
            f"{key}: {value.get('type', 'any')}"
            + (" (required)" if key in required else "")
            for key, value in props.items()
            if isinstance(value, dict)
        )
        suffix = "..." if len(description) > 200 else ""
        lines.append(f"- {name}({args_desc}): {description[:200]}{suffix}")
    return "\n".join(lines)


def build_tagged_prompt(
    tools: list[dict[str, Any]],
    tools_text: str | None = None,
    *,
    allow_parallel_tool_calls: bool = True,
    tool_choice: str | dict[str, Any] | None = None,
) -> str:
    normalized_choice = normalize_tool_choice(tool_choice)
    effective_tools = select_tools_for_choice(tools, normalized_choice)
    if tools_text is None:
        tools_text = format_tools_for_prompt(effective_tools)

    prompt = (
        TAGGED_TOOL_PROMPT_PARALLEL
        if allow_parallel_tool_calls
        else TAGGED_TOOL_PROMPT_SINGLE
    )
    choice_guidance = tool_choice_prompt_guidance(
        normalized_choice,
        allow_parallel_tool_calls=allow_parallel_tool_calls,
    )
    if choice_guidance:
        prompt += "\n\n## Tool choice\n\n" + choice_guidance + "\n"
    if not tools_text:
        return prompt
    return prompt + "\n\n---\n\n## Available tools\n\n" + tools_text + "\n"


def _assistant_tool_block(
    tool_calls: list[dict[str, Any]],
    *,
    allow_parallel_tool_calls: bool,
) -> str:
    payload = [
        {"name": tool_call["name"], "arguments": tool_call["arguments"]}
        for tool_call in tool_calls
    ]
    if len(payload) == 1 and not allow_parallel_tool_calls:
        return "<tool_call>" + _json_dump(payload[0]) + "</tool_call>"
    return "<tool_calls>" + _json_dump(payload) + "</tool_calls>"


def _tool_result_followup(*, allow_parallel_tool_calls: bool) -> str:
    terminal_desc = (
        "<tool_calls>[...]</tool_calls>"
        if allow_parallel_tool_calls
        else "<tool_call>{...}</tool_call>"
    )
    json_target = "<tool_calls>" if allow_parallel_tool_calls else "<tool_call>"
    single_tool_rule = (
        "If only one tool is needed, still use a JSON array with one item.\n"
        if allow_parallel_tool_calls
        else ""
    )
    terminal_name = "tool_calls" if allow_parallel_tool_calls else "tool_call"
    return (
        "Now output exactly one response using only the tagged protocol:\n"
        "- optional <think>...</think>\n"
        f"- then exactly one {terminal_desc} or <final_answer>...</final_answer>\n"
        "Do not output Observation.\n"
        "Do not output <tool_result>.\n"
        "Do not output tool results.\n"
        "Do not output a second terminal block.\n"
        f"Stop immediately after </{terminal_name}> or </final_answer>.\n"
        f"Inside {json_target}, the content must be valid JSON.\n"
        f"{single_tool_rule}"
        "If a string value contains quotes, backslashes, or newlines, escape them exactly as JSON requires."
    )


def _tool_result_followup_for_choice(
    *,
    allow_parallel_tool_calls: bool,
    tool_choice: str | dict[str, Any] | None,
) -> str:
    guidance = _tool_result_followup(
        allow_parallel_tool_calls=allow_parallel_tool_calls
    )
    choice_suffix = tool_choice_user_suffix(
        normalize_tool_choice(tool_choice),
        allow_parallel_tool_calls=allow_parallel_tool_calls,
    )
    if not choice_suffix:
        return guidance
    return guidance + "\n" + choice_suffix


def build_prompt(
    messages: list[OpenAIMessage | dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    allow_parallel_tool_calls: bool = True,
    include_tool_prompt: bool = True,
    tool_choice: str | dict[str, Any] | None = None,
) -> str:
    validated_messages = [
        msg if isinstance(msg, OpenAIMessage) else OpenAIMessage.model_validate(msg)
        for msg in messages
    ]
    raw_tools = tools or []
    normalized_choice = normalize_tool_choice(tool_choice)
    effective_tools = select_tools_for_choice(raw_tools, normalized_choice)

    parts: list[str] = []
    if include_tool_prompt and (effective_tools or normalized_choice.mode != "auto"):
        parts.append(
            build_tagged_prompt(
                raw_tools,
                allow_parallel_tool_calls=allow_parallel_tool_calls,
                tool_choice=tool_choice,
            )
        )

    for message in validated_messages:
        content_text = _normalize_content(message.content)
        if message.role == "system":
            if content_text:
                parts.append(f"System:\n{content_text}")
            continue

        if message.role == "user":
            if content_text:
                strict_suffix = (
                    STRICT_SUFFIX_PARALLEL
                    if allow_parallel_tool_calls
                    else STRICT_SUFFIX_SINGLE
                )
                choice_suffix = tool_choice_user_suffix(
                    normalized_choice,
                    allow_parallel_tool_calls=allow_parallel_tool_calls,
                )
                suffix_parts = [
                    suffix
                    for suffix in (
                        strict_suffix if raw_tools else "",
                        choice_suffix,
                    )
                    if suffix
                ]
                if suffix_parts:
                    parts.append(f"User:\n{content_text}\n\n" + "\n".join(suffix_parts))
                else:
                    parts.append(f"User:\n{content_text}")
            continue

        if message.role == "assistant":
            if content_text:
                parts.append(f"Assistant:\n{content_text}")
            tool_calls = list(message.tool_calls or [])
            if tool_calls:
                normalized = _normalize_tool_calls(tool_calls)
                call_ids = [
                    tool_call["id"] for tool_call in normalized if tool_call["id"]
                ]
                if call_ids:
                    parts.append("Assistant tool calls: " + ", ".join(call_ids))
                parts.append(
                    _assistant_tool_block(
                        normalized,
                        allow_parallel_tool_calls=allow_parallel_tool_calls,
                    )
                )
            continue

        if message.role == "tool":
            if not content_text:
                continue
            call_id = message.tool_call_id or ""
            parts.append(
                f"Tool result for call_id={call_id}:\n"
                "<tool_result>\n"
                f"{content_text}\n"
                "</tool_result>\n\n"
                + _tool_result_followup_for_choice(
                    allow_parallel_tool_calls=allow_parallel_tool_calls,
                    tool_choice=tool_choice,
                )
            )

    return "\n\n".join(part for part in parts if part)
