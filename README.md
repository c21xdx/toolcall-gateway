# toolcall-gateway

`toolcall-gateway` is a small Python library for one specific job:

- `tool2text`: convert OpenAI-style tools and tool history into a strict tagged prompt for text-only models
- `text2tool`: parse tagged model output back into OpenAI-style tool-call semantics

This project does not provide an HTTP API layer.
You bring your own middleware, gateway, or runtime.

It also understands `tool_choice` semantics on both sides:

- `tool2text`: writes `auto / required / none / specific function` constraints into the tagged prompt
- `text2tool`: can validate the model output against the same constraints while parsing

## Production Notes

- The public API raises stable exceptions from [`toolcall_gateway.errors`](./src/toolcall_gateway/errors.py):
  - `ToolcallGatewayError`
  - `TaggedOutputError`
  - `ToolChoiceError`
- The package ships a `py.typed` marker for static type checkers.
- The repository includes runnable demos, tests, and a buildable wheel/sdist.

## Tagged Protocol

The library uses a strict XML-like DSL:

```xml
<think>...</think>
<tool_calls>[{"name":"Read","arguments":{"path":"a.py"}}]</tool_calls>
```

or:

```xml
<think>...</think>
<final_answer>...</final_answer>
```

## Install

```bash
uv pip install -e .
```

## Example

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

# Send prompt to your own text-only model runtime...
model_output = (
    "<think>I need the file first</think>"
    '<tool_calls>[{"name":"Read","arguments":{"path":"a.py"}}]</tool_calls>'
)

turn = parse_to_openai_assistant_turn(model_output)
print(turn.finish_reason)  # tool_calls
print(turn.tool_calls[0].function.name)  # Read
```

## Error Handling

```python
from toolcall_gateway import ToolChoiceError, TaggedOutputError

try:
    turn = parse_to_openai_assistant_turn(
        model_output,
        tool_choice={"type": "function", "function": {"name": "Read"}},
    )
except ToolChoiceError:
    # The model violated the requested tool_choice constraint.
    ...
except TaggedOutputError:
    # The model produced invalid tagged text.
    ...
```

## Demos

```bash
uv run python examples/demo_tool2text.py
uv run python examples/demo_text2tool.py
```

## Verification

```bash
make verify
```

or:

```bash
uv run --with ruff ruff check .
uv run python -m unittest discover -s tests
uv build
```

## Tests

```bash
uv run python -m unittest discover -s tests
```
