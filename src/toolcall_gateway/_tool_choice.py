from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from toolcall_gateway.errors import ToolChoiceError


@dataclass(frozen=True, slots=True)
class NormalizedToolChoice:
    mode: Literal["auto", "required", "none", "function"]
    function_name: str | None = None


def normalize_tool_choice(
    tool_choice: str | dict[str, Any] | None,
) -> NormalizedToolChoice:
    if tool_choice is None or tool_choice == "auto":
        return NormalizedToolChoice(mode="auto")

    if isinstance(tool_choice, str):
        if tool_choice == "required":
            return NormalizedToolChoice(mode="required")
        if tool_choice == "none":
            return NormalizedToolChoice(mode="none")
        raise ToolChoiceError(
            "tool_choice must be one of: auto, required, none, or a function selector"
        )

    if not isinstance(tool_choice, dict):
        raise ToolChoiceError(
            "tool_choice must be a string, a function selector object, or None"
        )

    if tool_choice.get("type") != "function":
        raise ToolChoiceError("tool_choice object must have type='function'")

    function = tool_choice.get("function")
    if isinstance(function, dict):
        name = function.get("name")
    else:
        name = tool_choice.get("name")

    function_name = str(name or "").strip()
    if not function_name:
        raise ToolChoiceError(
            "tool_choice function selector must include a non-empty name"
        )

    return NormalizedToolChoice(mode="function", function_name=function_name)


def select_tools_for_choice(
    tools: list[dict[str, Any]],
    tool_choice: NormalizedToolChoice,
) -> list[dict[str, Any]]:
    if tool_choice.mode == "none":
        return []

    if tool_choice.mode == "required" and not tools:
        raise ToolChoiceError("tool_choice='required' requires at least one tool")

    if tool_choice.mode != "function":
        return list(tools)

    selected: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") if tool.get("type") == "function" else tool
        if not isinstance(function, dict):
            continue
        if str(function.get("name") or "").strip() == tool_choice.function_name:
            selected.append(tool)

    if not selected:
        raise ToolChoiceError(
            f"tool_choice selected function '{tool_choice.function_name}', but it is not present in tools"
        )
    return selected


def tool_choice_prompt_guidance(
    tool_choice: NormalizedToolChoice,
    *,
    allow_parallel_tool_calls: bool,
) -> str:
    if tool_choice.mode == "auto":
        return ""

    tool_tag = "<tool_calls>" if allow_parallel_tool_calls else "<tool_call>"
    tool_terminal = (
        "<tool_calls>[...]</tool_calls>"
        if allow_parallel_tool_calls
        else "<tool_call>{...}</tool_call>"
    )
    if tool_choice.mode == "none":
        return (
            "Tool choice for this response is fixed to none.\n"
            "You may output one or more <think> blocks.\n"
            "Your terminal block must be <final_answer>.\n"
            f"Do not output {tool_tag} in this response."
        )

    if tool_choice.mode == "required":
        return (
            "Tool choice for this response is required.\n"
            "You may output one or more <think> blocks.\n"
            f"Your terminal block must be {tool_terminal}.\n"
            "Do not output <final_answer> in this response."
        )

    return (
        f"Tool choice for this response is fixed to function '{tool_choice.function_name}'.\n"
        "You may output one or more <think> blocks.\n"
        f"Your terminal block must be {tool_terminal}.\n"
        f"Every tool call in {tool_tag} must use the function name '{tool_choice.function_name}'.\n"
        "Do not output <final_answer> in this response."
    )


def tool_choice_user_suffix(
    tool_choice: NormalizedToolChoice,
    *,
    allow_parallel_tool_calls: bool,
) -> str:
    if tool_choice.mode == "auto":
        return ""
    if tool_choice.mode == "none":
        return "For this response, tool_choice is none. End with <final_answer>."
    if tool_choice.mode == "required":
        terminal = (
            "<tool_calls>[...]</tool_calls>"
            if allow_parallel_tool_calls
            else "<tool_call>{...}</tool_call>"
        )
        return f"For this response, tool_choice is required. End with {terminal}, not <final_answer>."
    return (
        "For this response, tool_choice requires calling "
        f"'{tool_choice.function_name}'. Do not call any other function."
    )
