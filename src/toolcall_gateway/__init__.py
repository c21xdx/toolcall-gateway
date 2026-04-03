from importlib.metadata import PackageNotFoundError, version

from toolcall_gateway.errors import (
    TaggedOutputError,
    ToolChoiceError,
    ToolcallGatewayError,
)
from toolcall_gateway.models import (
    OpenAIContentPart,
    OpenAIFunctionCall,
    OpenAIFunctionSpec,
    OpenAIMessage,
    OpenAIParsedAssistantTurn,
    OpenAIToolCall,
    OpenAIToolSpec,
)
from toolcall_gateway.text2tool import (
    TaggedOutput,
    TaggedStreamEvent,
    TaggedStreamParser,
    TaggedToolCall,
    parse_tagged_output,
    parse_to_openai_assistant_turn,
)
from toolcall_gateway.tool2text import (
    build_prompt,
    build_tagged_prompt,
    format_tools_for_prompt,
)

try:
    __version__ = version("toolcall-gateway")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    "OpenAIContentPart",
    "OpenAIFunctionCall",
    "OpenAIFunctionSpec",
    "OpenAIMessage",
    "OpenAIParsedAssistantTurn",
    "OpenAIToolCall",
    "OpenAIToolSpec",
    "TaggedOutput",
    "TaggedOutputError",
    "TaggedStreamEvent",
    "TaggedStreamParser",
    "TaggedToolCall",
    "ToolChoiceError",
    "ToolcallGatewayError",
    "build_prompt",
    "build_tagged_prompt",
    "format_tools_for_prompt",
    "parse_tagged_output",
    "parse_to_openai_assistant_turn",
]
