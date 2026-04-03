from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class OpenAIContentPart(BaseModel):
    type: str
    text: str | None = None
    image_url: dict[str, Any] | str | None = None

    model_config = {"extra": "allow"}


class OpenAIFunctionSpec(BaseModel):
    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    strict: bool = False

    model_config = {"extra": "allow"}


class OpenAIToolSpec(BaseModel):
    type: Literal["function"] = "function"
    function: OpenAIFunctionSpec

    model_config = {"extra": "allow"}


class OpenAIFunctionCall(BaseModel):
    name: str = ""
    arguments: str = "{}"

    model_config = {"extra": "allow"}


class OpenAIToolCall(BaseModel):
    id: str = ""
    type: Literal["function"] = "function"
    function: OpenAIFunctionCall

    model_config = {"extra": "allow"}


class OpenAIMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[OpenAIContentPart] | None = ""
    tool_calls: list[OpenAIToolCall | dict[str, Any]] | None = None
    tool_call_id: str | None = None

    model_config = {"extra": "allow"}


class OpenAIParsedAssistantTurn(BaseModel):
    content: str | None = None
    tool_calls: list[OpenAIToolCall] = Field(default_factory=list)
    finish_reason: Literal["tool_calls", "stop"] = "stop"
    thinking: str | None = None

    model_config = {"extra": "allow"}
