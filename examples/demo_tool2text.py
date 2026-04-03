#!/usr/bin/env python3
"""可本地运行的 tool2text 小 demo：把 OpenAI 风格 tools + 对话打成带标签的 prompt。

配对示例（模型输出 → OpenAI 语义）见 ``demo_text2tool.py``。

在项目根目录执行::

    uv run python examples/demo_tool2text.py
"""

from __future__ import annotations

from toolcall_gateway import build_prompt, build_tagged_prompt, format_tools_for_prompt

TOOLS = [
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


def main() -> None:
    print("=== 1. format_tools_for_prompt：工具列表可读摘要 ===\n")
    rendered = format_tools_for_prompt(TOOLS)
    print(rendered)

    print("\n=== 2. build_tagged_prompt：系统提示 + 协议说明 + 工具块 ===\n")
    tagged = build_tagged_prompt(TOOLS)
    # 输出较长，只展示关键片段
    for needle in ("<tool_calls>", "## Available tools", "Read(path:"):
        assert needle in tagged
    print(tagged[:1200])
    if len(tagged) > 1200:
        print(f"\n... （共 {len(tagged)} 字符，已截断）\n")

    print("\n=== 3. build_prompt：多轮 + assistant tool_calls + tool 结果回放 ===\n")
    prompt = build_prompt(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "读一下 a.py"},
            {
                "role": "assistant",
                "content": "<think>先读文件</think>",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "Read",
                            "arguments": '{"path":"a.py"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "content": "print('hello')",
            },
            {"role": "user", "content": "总结一下"},
        ],
        tools=TOOLS,
        allow_parallel_tool_calls=True,
    )
    print(prompt)

    print("\n=== 4. tool_choice：强制调用指定函数 ===\n")
    forced = build_prompt(
        [{"role": "user", "content": "只允许读取 README.md"}],
        tools=TOOLS,
        tool_choice={"type": "function", "function": {"name": "Read"}},
    )
    print(forced)


if __name__ == "__main__":
    main()
