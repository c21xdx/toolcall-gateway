from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Literal

from toolcall_gateway._tool_choice import normalize_tool_choice, select_tools_for_choice
from toolcall_gateway.errors import TaggedOutputError, ToolChoiceError
from toolcall_gateway.models import (
    OpenAIFunctionCall,
    OpenAIParsedAssistantTurn,
    OpenAIToolCall,
)


@dataclass(slots=True)
class TaggedToolCall:
    name: str
    arguments: dict[str, Any]
    raw_json: str


@dataclass(slots=True)
class TaggedOutput:
    thinking: str | None = None
    tool_calls: list[TaggedToolCall] = field(default_factory=list)
    final_answer: str | None = None

    @property
    def is_tool_call(self) -> bool:
        return bool(self.tool_calls)

    @property
    def tool_call(self) -> TaggedToolCall | None:
        return self.tool_calls[0] if self.tool_calls else None

    @property
    def is_final_answer(self) -> bool:
        return self.final_answer is not None


def _parse_tool_call_item(payload: Any) -> TaggedToolCall:
    if not isinstance(payload, dict):
        raise TaggedOutputError("tool call payload must be an object")

    name = payload.get("name")
    arguments = payload.get("arguments")
    if not isinstance(name, str) or not name.strip():
        raise TaggedOutputError("tool_call.name must be a non-empty string")
    if not isinstance(arguments, dict):
        raise TaggedOutputError("tool_call.arguments must be an object")

    return TaggedToolCall(
        name=name.strip(),
        arguments=arguments,
        raw_json=json.dumps(payload, ensure_ascii=False),
    )


def _parse_tool_call_block(raw_json: str) -> list[TaggedToolCall]:
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise TaggedOutputError(f"invalid tool_call json: {exc}") from exc
    return [_parse_tool_call_item(payload)]


def _parse_tool_calls_block(raw_json: str) -> list[TaggedToolCall]:
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise TaggedOutputError(f"invalid tool_calls json: {exc}") from exc
    if not isinstance(payload, list):
        raise TaggedOutputError("tool_calls payload must be an array")
    if not payload:
        raise TaggedOutputError("tool_calls payload must not be empty")
    return [_parse_tool_call_item(item) for item in payload]


def parse_tagged_output(text: str) -> TaggedOutput:
    if not text or not text.strip():
        raise TaggedOutputError("empty tagged output")

    content = text.strip()
    n = len(content)

    def skip_ws(pos: int) -> int:
        while pos < n and content[pos].isspace():
            pos += 1
        return pos

    def read_block(pos: int, tag: str) -> tuple[str, int]:
        open_tag = f"<{tag}>"
        close_tag = f"</{tag}>"
        if not content.startswith(open_tag, pos):
            raise TaggedOutputError(f"expected {open_tag}")
        start = pos + len(open_tag)
        end = content.find(close_tag, start)
        if end < 0:
            raise TaggedOutputError(f"missing {close_tag}")
        return content[start:end], end + len(close_tag)

    pos = skip_ws(0)
    thinking_blocks: list[str] = []
    tool_calls: list[TaggedToolCall] = []
    final_answer: str | None = None

    while pos < n:
        if content.startswith("<think>", pos):
            raw_thinking, pos = read_block(pos, "think")
            thinking = raw_thinking.strip()
            if thinking:
                thinking_blocks.append(thinking)
            pos = skip_ws(pos)
            continue

        if content.startswith("<tool_calls>", pos):
            raw_tool_json, pos = read_block(pos, "tool_calls")
            tool_calls = _parse_tool_calls_block(raw_tool_json.strip())
            break

        if content.startswith("<tool_call>", pos):
            raw_tool_json, pos = read_block(pos, "tool_call")
            tool_calls = _parse_tool_call_block(raw_tool_json.strip())
            break

        if content.startswith("<final_answer>", pos):
            raw_answer, pos = read_block(pos, "final_answer")
            final_answer = raw_answer.strip()
            break

        if content[pos].isspace():
            pos += 1
            continue

        raise TaggedOutputError("text outside tags is not allowed")

    if not tool_calls and final_answer is None:
        raise TaggedOutputError("expected <tool_calls>, <tool_call>, or <final_answer>")

    return TaggedOutput(
        thinking="\n\n".join(thinking_blocks) or None,
        tool_calls=tool_calls,
        final_answer=final_answer,
    )


def _thinking_content(thinking: str | None) -> str | None:
    if not thinking:
        return None
    return f"<think>{thinking}</think>"


def parse_to_openai_assistant_turn(
    text: str,
    *,
    include_thinking: bool = True,
    tool_choice: str | dict[str, Any] | None = None,
    available_tools: list[dict[str, Any]] | None = None,
) -> OpenAIParsedAssistantTurn:
    parsed = parse_tagged_output(text)
    thinking_content = _thinking_content(parsed.thinking) if include_thinking else None
    normalized_choice = normalize_tool_choice(tool_choice)
    allowed_tool_names: set[str] = set()
    if available_tools is not None:
        for tool in select_tools_for_choice(available_tools, normalized_choice):
            function = tool.get("function") if tool.get("type") == "function" else tool
            if isinstance(function, dict):
                name = str(function.get("name") or "").strip()
                if name:
                    allowed_tool_names.add(name)

    if parsed.is_tool_call:
        if normalized_choice.mode == "none":
            raise ToolChoiceError("tool_choice='none' does not allow tool calls")
        for tool_call in parsed.tool_calls:
            if (
                normalized_choice.mode == "function"
                and tool_call.name != normalized_choice.function_name
            ):
                raise ToolChoiceError(
                    f"tool_choice requires function '{normalized_choice.function_name}', got '{tool_call.name}'"
                )
            if allowed_tool_names and tool_call.name not in allowed_tool_names:
                raise ToolChoiceError(
                    f"tool call '{tool_call.name}' is not present in available_tools"
                )
        return OpenAIParsedAssistantTurn(
            content=thinking_content,
            tool_calls=[
                OpenAIToolCall(
                    id=f"call_{uuid.uuid4().hex[:24]}",
                    function=OpenAIFunctionCall(
                        name=tool_call.name,
                        arguments=json.dumps(tool_call.arguments, ensure_ascii=False),
                    ),
                )
                for tool_call in parsed.tool_calls
            ],
            finish_reason="tool_calls",
            thinking=parsed.thinking,
        )

    if normalized_choice.mode in {"required", "function"}:
        raise ToolChoiceError(
            "tool_choice requires a tool call, but the model returned <final_answer>"
        )

    parts = []
    if thinking_content:
        parts.append(thinking_content)
    parts.append(parsed.final_answer or "")
    return OpenAIParsedAssistantTurn(
        content="\n\n".join(part for part in parts if part),
        tool_calls=[],
        finish_reason="stop",
        thinking=parsed.thinking,
    )


class _State(Enum):
    OUTSIDE = auto()
    IN_TAG = auto()
    IN_THINK = auto()
    IN_TOOL_CALL = auto()
    IN_TOOL_CALLS = auto()
    IN_FINAL_ANSWER = auto()


@dataclass(slots=True)
class TaggedStreamEvent:
    type: Literal[
        "message_start",
        "block_start",
        "block_delta",
        "block_end",
        "tool_call",
        "message_stop",
        "error",
    ]
    block_type: Literal["thinking", "text"] | None = None
    text: str | None = None
    call_index: int | None = None
    name: str | None = None
    arguments: dict[str, Any] | None = None
    stop_reason: Literal["tool_use", "end_turn"] | None = None
    error: str | None = None


class TaggedStreamParser:
    def __init__(self) -> None:
        self._state = _State.OUTSIDE
        self._state_before_tag = _State.OUTSIDE
        self._tag_buf = ""
        self._text_buf = ""
        self._message_started = False
        self._preamble_open = False
        self._terminal_kind: Literal["tool_use", "end_turn"] | None = None
        self._saw_terminal = False
        self._terminal_closed = False
        self._message_stopped = False

    def feed(self, chunk: str) -> list[TaggedStreamEvent]:
        events: list[TaggedStreamEvent] = []
        for char in chunk:
            events.extend(self._on_char(char))
        if self._state in {_State.IN_THINK, _State.IN_FINAL_ANSWER} and self._text_buf:
            events.extend(self._flush_text_buffer())
        return events

    def finish(self) -> list[TaggedStreamEvent]:
        events: list[TaggedStreamEvent] = []
        if self._state == _State.IN_TAG:
            raise TaggedOutputError("incomplete tag at end of stream")
        if self._state == _State.IN_THINK:
            raise TaggedOutputError("missing </think>")
        if self._state == _State.IN_TOOL_CALL:
            raise TaggedOutputError("missing </tool_call>")
        if self._state == _State.IN_TOOL_CALLS:
            raise TaggedOutputError("missing </tool_calls>")
        if self._state == _State.IN_FINAL_ANSWER and not self._terminal_closed:
            if self._saw_terminal and self._terminal_kind == "end_turn":
                events.extend(self._flush_text_buffer())
                self._state = _State.OUTSIDE
                events.append(TaggedStreamEvent(type="block_end", block_type="text"))
            elif self._preamble_open:
                events.extend(self._flush_text_buffer())
                self._preamble_open = False
                self._state = _State.OUTSIDE
                events.append(TaggedStreamEvent(type="block_end", block_type="text"))
            else:
                raise TaggedOutputError("missing </final_answer>")
        if not self._saw_terminal:
            if self._message_started:
                self._saw_terminal = True
                self._terminal_kind = self._terminal_kind or "end_turn"
            else:
                raise TaggedOutputError("missing terminal block")
        if self._message_stopped:
            return events
        events.append(self._message_stop_event())
        return events

    def _on_char(self, char: str) -> list[TaggedStreamEvent]:
        if self._terminal_closed:
            return []

        if self._state == _State.OUTSIDE:
            if char.isspace():
                return []
            if char == "<":
                self._state_before_tag = self._state
                self._state = _State.IN_TAG
                self._tag_buf = "<"
                return []
            events: list[TaggedStreamEvent] = []
            self._ensure_message_started(events)
            self._state = _State.IN_FINAL_ANSWER
            self._preamble_open = True
            self._text_buf = char
            events.append(TaggedStreamEvent(type="block_start", block_type="text"))
            return events

        if self._state == _State.IN_TAG:
            self._tag_buf += char
            if char != ">":
                return []
            tag = self._tag_buf
            self._tag_buf = ""
            return self._handle_tag(tag)

        if self._state in {_State.IN_THINK, _State.IN_FINAL_ANSWER}:
            if char == "<":
                events = self._flush_text_buffer()
                if (
                    self._preamble_open
                    and self._state == _State.IN_FINAL_ANSWER
                    and not self._saw_terminal
                ):
                    self._preamble_open = False
                    events.append(TaggedStreamEvent(type="block_end", block_type="text"))
                    self._state = _State.OUTSIDE
                self._state_before_tag = self._state
                self._state = _State.IN_TAG
                self._tag_buf = "<"
                return events
            self._text_buf += char
            return []

        if self._state in {_State.IN_TOOL_CALL, _State.IN_TOOL_CALLS}:
            if char == "<":
                self._state_before_tag = self._state
                self._state = _State.IN_TAG
                self._tag_buf = "<"
                return []
            self._text_buf += char
            return []

        raise TaggedOutputError(f"unexpected parser state: {self._state}")

    def _handle_tag(self, tag: str) -> list[TaggedStreamEvent]:
        events: list[TaggedStreamEvent] = []

        if tag == "<think>":
            self._ensure_message_started(events)
            self._state = _State.IN_THINK
            self._text_buf = ""
            events.append(TaggedStreamEvent(type="block_start", block_type="thinking"))
            return events

        if tag == "</think>":
            if self._state_before_tag != _State.IN_THINK:
                if self._state_before_tag == _State.OUTSIDE:
                    return []
                raise TaggedOutputError("unexpected </think>")
            events.extend(self._flush_text_buffer())
            self._state = _State.OUTSIDE
            events.append(TaggedStreamEvent(type="block_end", block_type="thinking"))
            return events

        if tag == "<tool_call>":
            if self._saw_terminal:
                raise TaggedOutputError("only one terminal block is allowed")
            self._ensure_message_started(events)
            self._saw_terminal = True
            self._terminal_kind = "tool_use"
            self._state = _State.IN_TOOL_CALL
            self._text_buf = ""
            return events

        if tag == "<tool_calls>":
            if self._saw_terminal:
                raise TaggedOutputError("only one terminal block is allowed")
            self._ensure_message_started(events)
            self._saw_terminal = True
            self._terminal_kind = "tool_use"
            self._state = _State.IN_TOOL_CALLS
            self._text_buf = ""
            return events

        if tag == "</tool_call>":
            if self._state_before_tag != _State.IN_TOOL_CALL:
                raise TaggedOutputError("unexpected </tool_call>")
            tool_calls = _parse_tool_call_block(self._text_buf.strip())
            self._text_buf = ""
            self._state = _State.OUTSIDE
            self._terminal_closed = True
            for index, tool_call in enumerate(tool_calls):
                events.append(
                    TaggedStreamEvent(
                        type="tool_call",
                        call_index=index,
                        name=tool_call.name,
                        arguments=tool_call.arguments,
                    )
                )
            return events

        if tag == "</tool_calls>":
            if self._state_before_tag != _State.IN_TOOL_CALLS:
                raise TaggedOutputError("unexpected </tool_calls>")
            tool_calls = _parse_tool_calls_block(self._text_buf.strip())
            self._text_buf = ""
            self._state = _State.OUTSIDE
            self._terminal_closed = True
            for index, tool_call in enumerate(tool_calls):
                events.append(
                    TaggedStreamEvent(
                        type="tool_call",
                        call_index=index,
                        name=tool_call.name,
                        arguments=tool_call.arguments,
                    )
                )
            return events

        if tag == "<final_answer>":
            if self._saw_terminal:
                raise TaggedOutputError("only one terminal block is allowed")
            self._ensure_message_started(events)
            self._saw_terminal = True
            self._terminal_kind = "end_turn"
            self._state = _State.IN_FINAL_ANSWER
            self._text_buf = ""
            events.append(TaggedStreamEvent(type="block_start", block_type="text"))
            return events

        if tag == "</final_answer>":
            if self._state_before_tag != _State.IN_FINAL_ANSWER:
                raise TaggedOutputError("unexpected </final_answer>")
            events.extend(self._flush_text_buffer())
            self._state = _State.OUTSIDE
            self._terminal_closed = True
            events.append(TaggedStreamEvent(type="block_end", block_type="text"))
            return events

        if self._state_before_tag in {
            _State.IN_THINK,
            _State.IN_TOOL_CALL,
            _State.IN_TOOL_CALLS,
            _State.IN_FINAL_ANSWER,
        }:
            self._state = self._state_before_tag
            self._text_buf += tag
            return events

        raise TaggedOutputError(f"unsupported tag: {tag}")

    def _ensure_message_started(self, events: list[TaggedStreamEvent]) -> None:
        if self._message_started:
            return
        self._message_started = True
        events.append(TaggedStreamEvent(type="message_start"))

    def _flush_text_buffer(self) -> list[TaggedStreamEvent]:
        if not self._text_buf:
            return []
        if self._state_before_tag == _State.IN_THINK or self._state == _State.IN_THINK:
            block_type: Literal["thinking", "text"] = "thinking"
        elif (
            self._state_before_tag == _State.IN_FINAL_ANSWER
            or self._state == _State.IN_FINAL_ANSWER
        ):
            block_type = "text"
        else:
            return []
        text = self._text_buf
        self._text_buf = ""
        return [
            TaggedStreamEvent(
                type="block_delta",
                block_type=block_type,
                text=text,
            )
        ]

    def _message_stop_event(self) -> TaggedStreamEvent:
        if self._terminal_kind is None:
            raise TaggedOutputError("missing stop reason")
        self._message_stopped = True
        return TaggedStreamEvent(
            type="message_stop",
            stop_reason=self._terminal_kind,
        )
