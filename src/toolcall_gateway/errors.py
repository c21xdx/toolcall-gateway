from __future__ import annotations


class ToolcallGatewayError(ValueError):
    """Base error for the public toolcall-gateway API."""


class TaggedOutputError(ToolcallGatewayError):
    """Raised when model output violates the tagged text protocol."""


class ToolChoiceError(ToolcallGatewayError):
    """Raised when tool_choice input or output constraints are invalid."""
