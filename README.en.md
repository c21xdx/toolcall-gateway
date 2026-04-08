# toolcall-gateway

[![PyPI version](https://img.shields.io/pypi/v/toolcall-gateway.svg)](https://pypi.org/project/toolcall-gateway/)
[![Python 3.12+](https://img.shields.io/pypi/pyversions/toolcall-gateway.svg)](https://pypi.org/project/toolcall-gateway/)

[中文（默认）](README.md) · **English**

## What it is

**toolcall-gateway** is a small, typed Python library that sits between **OpenAI-style tool calling** (function definitions, `tool_calls` on assistant turns, tool results in the chat) and **plain-text model I/O**.

It gives you a strict, reproducible **tagged text protocol** (`<think>`, `<tool_calls>` / `<tool_call>`, `<final_answer>`) so you can:

1. **tool2text** — Turn tools + message history into one prompt string for **text-only** or **non-native-tool** models.
2. **text2tool** — Parse the model’s tagged reply back into **OpenAI-style** assistant semantics (e.g. `finish_reason`, `tool_calls`).

The package is **[published on PyPI](https://pypi.org/project/toolcall-gateway/)** as `toolcall-gateway`. It is **not** an HTTP API or gateway server: you plug it into your own runtime, agent loop, or middleware.

## Problems it solves

| Situation | How this helps |
|-----------|----------------|
| Your backend speaks OpenAI tools, but the model only accepts a single text prompt | Serialize tools and history into one tagged prompt with `build_prompt` / related helpers. |
| The model returns “XML-like” tool calls instead of structured API fields | Parse with `parse_to_openai_assistant_turn` (or lower-level `parse_tagged_output`) and feed the result into your existing tool runner. |
| You need `tool_choice` (`auto`, `required`, `none`, or a specific function) | The library encodes constraints into the prompt on the way out and can **validate** parsed output on the way back. |
| You want stable errors and typing for production code | Public exceptions (`ToolcallGatewayError`, `TaggedOutputError`, `ToolChoiceError`) and `py.typed` for type checkers. |

## Install (PyPI)

Requires **Python 3.12+**.

```bash
pip install toolcall-gateway
```

```bash
uv add toolcall-gateway
```

To work from a git checkout instead:

```bash
uv pip install -e .
```

## Quick start

```python
from toolcall_gateway import build_prompt, parse_to_openai_assistant_turn

messages = [
    {"role": "user", "content": "Read a.py and summarize it."},
]
tools = [
    {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
]

prompt = build_prompt(
    messages,
    tools=tools,
    tool_choice={"type": "function", "function": {"name": "Read"}},
)

# Send `prompt` to your text-only model...
model_output = (
    "<think>I need the file first</think>"
    '<tool_calls>[{"name":"Read","arguments":{"path":"a.py"}}]</tool_calls>'
)

turn = parse_to_openai_assistant_turn(model_output)
print(turn.finish_reason)  # tool_calls
print(turn.tool_calls[0].function.name)  # Read
```

## Main APIs

- **Prompt building:** `build_prompt`, `build_tagged_prompt`, `format_tools_for_prompt` (`toolcall_gateway.tool2text`).
- **Parsing:** `parse_to_openai_assistant_turn`, `parse_tagged_output`, streaming via `TaggedStreamParser` (`toolcall_gateway.text2tool`).
- **Models / types:** `OpenAIMessage`, `OpenAIToolSpec`, `OpenAIParsedAssistantTurn`, etc. (`toolcall_gateway.models`).

## Tagged protocol (summary)

The model is instructed to use only these tags (parallel variant shown):

```xml
<think>...</think>
<tool_calls>[{"name":"Read","arguments":{"path":"a.py"}}]</tool_calls>
```

or a final reply:

```xml
<think>...</think>
<final_answer>...</final_answer>
```

A single-tool variant uses `<tool_call>...</tool_call>` instead of `<tool_calls>[...]</tool_calls>`.

## Error handling

```python
from toolcall_gateway import ToolChoiceError, TaggedOutputError

try:
    turn = parse_to_openai_assistant_turn(
        model_output,
        tool_choice={"type": "function", "function": {"name": "Read"}},
    )
except ToolChoiceError:
    # Model output violated the requested tool_choice.
    ...
except TaggedOutputError:
    # Invalid or malformed tagged text.
    ...
```

## Demos & verification

```bash
uv run python examples/demo_tool2text.py
uv run python examples/demo_text2tool.py
```

```bash
make verify
```

or:

```bash
uv run --with ruff ruff check .
uv run python -m unittest discover -s tests
uv build
```
