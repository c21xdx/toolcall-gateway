# toolcall-gateway

[![PyPI version](https://img.shields.io/pypi/v/toolcall-gateway.svg)](https://pypi.org/project/toolcall-gateway/)
[![Python 3.12+](https://img.shields.io/pypi/pyversions/toolcall-gateway.svg)](https://pypi.org/project/toolcall-gateway/)

**默认语言：中文** · [English](README.en.md)

## 这是什么

**toolcall-gateway** 是一个体量小、带类型标注的 Python 库，作用是在 **OpenAI 风格的工具调用**（函数定义、助手消息里的 `tool_calls`、工具角色返回的结果等）与 **纯文本模型输入输出** 之间做转换。

它约定了一套严格的 **标签文本协议**（`<redacted_thinking>`、`<tool_calls>` / `<tool_call>`、`<final_answer>`），让你可以：

1. **tool2text** — 把工具定义和多轮对话打成 **一段** 适合 **只接受文本** 或 **不支持原生 tool_calls** 的模型的 prompt。
2. **text2tool** — 把模型按协议吐出的标签文本 **解析回** OpenAI 风格的助手轮次语义（例如 `finish_reason`、`tool_calls`）。

本库已 **[发布至 PyPI](https://pypi.org/project/toolcall-gateway/)**，包名为 `toolcall-gateway`。它 **不提供** HTTP 接口或网关进程；你需要在自己的运行时、智能体循环或中间件里调用这些函数。

## 解决什么问题

| 场景 | 本库的作用 |
|------|------------|
| 业务侧已是 OpenAI 式 tools，但模型只能收「一整段字符串」 | 用 `build_prompt` 等把工具与历史会话序列化成带标签的单一 prompt。 |
| 模型用「类 XML」表示工具调用，而不是 API 里的结构化字段 | 用 `parse_to_openai_assistant_turn`（或更底层的 `parse_tagged_output`）解析后，交给现有工具执行逻辑。 |
| 需要 `tool_choice`（`auto` / `required` / `none` / 指定函数） | 生成 prompt 时写入约束；解析时可按相同约束 **校验** 模型输出。 |
| 生产环境需要稳定异常与类型信息 | 公开异常类型（`ToolcallGatewayError`、`TaggedOutputError`、`ToolChoiceError`）及 `py.typed` 便于静态检查。 |

## 安装（PyPI）

需要 **Python 3.12+**。

```bash
pip install toolcall-gateway
```

```bash
uv add toolcall-gateway
```

若从本仓库源码开发安装：

```bash
uv pip install -e .
```

## 快速上手

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

# 将 prompt 发给你的纯文本模型...
model_output = (
    "<redacted_thinking>I need the file first</redacted_thinking>"
    '<tool_calls>[{"name":"Read","arguments":{"path":"a.py"}}]</tool_calls>'
)

turn = parse_to_openai_assistant_turn(model_output)
print(turn.finish_reason)  # tool_calls
print(turn.tool_calls[0].function.name)  # Read
```

## 主要 API

- **构造 prompt：** `build_prompt`、`build_tagged_prompt`、`format_tools_for_prompt`（`toolcall_gateway.tool2text`）。
- **解析输出：** `parse_to_openai_assistant_turn`、`parse_tagged_output`，流式可用 `TaggedStreamParser`（`toolcall_gateway.text2tool`）。
- **数据模型：** `OpenAIMessage`、`OpenAIToolSpec`、`OpenAIParsedAssistantTurn` 等（`toolcall_gateway.models`）。

## 标签协议（摘要）

并行多工具时使用 `<tool_calls>`，内容为 JSON 数组，例如：

```xml
<redacted_thinking>...</redacted_thinking>
<tool_calls>[{"name":"Read","arguments":{"path":"a.py"}}]</tool_calls>
```

若只需最终自然语言回答：

```xml
<redacted_thinking>...</redacted_thinking>
<final_answer>...</final_answer>
```

单工具场景可使用 `<tool_call>...</tool_call>`（单个 JSON 对象），而非 `<tool_calls>[...]</tool_calls>`。

## 异常处理

```python
from toolcall_gateway import ToolChoiceError, TaggedOutputError

try:
    turn = parse_to_openai_assistant_turn(
        model_output,
        tool_choice={"type": "function", "function": {"name": "Read"}},
    )
except ToolChoiceError:
    # 模型输出违反了 tool_choice 约束
    ...
except TaggedOutputError:
    # 标签格式非法或无法解析
    ...
```

## 示例与校验

```bash
uv run python examples/demo_tool2text.py
uv run python examples/demo_text2tool.py
```

```bash
make verify
```

或：

```bash
uv run --with ruff ruff check .
uv run python -m unittest discover -s tests
uv build
```
