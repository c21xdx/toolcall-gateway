#!/usr/bin/env python3
"""可本地运行的 text2tool 小 demo：解析带标签的模型输出。

配对示例（tools + 对话 → prompt）见 ``demo_tool2text.py``。

在项目根目录执行::

    uv run python examples/demo_text2tool.py
"""

from __future__ import annotations

import json
from dataclasses import asdict

from toolcall_gateway import (
    TaggedStreamParser,
    parse_tagged_output,
    parse_to_openai_assistant_turn,
)


def main() -> None:
    print("=== 1. parse_tagged_output：单工具 + thinking ===\n")
    raw = (
        "<think>先读文件</think>"
        '<tool_call>{"name":"Read","arguments":{"path":"README.md"}}</tool_call>'
    )
    parsed = parse_tagged_output(raw)
    print("输入:\n", raw, "\n")
    print("thinking:", parsed.thinking)
    if parsed.tool_call:
        print("tool:", parsed.tool_call.name, parsed.tool_call.arguments)

    print("\n=== 2. parse_to_openai_assistant_turn：最终回答 ===\n")
    final_raw = "<think>可以回答了</think><final_answer>你好，世界。</final_answer>"
    turn = parse_to_openai_assistant_turn(final_raw)
    print("输入:\n", final_raw, "\n")
    print("finish_reason:", turn.finish_reason)
    print("content (assistant):\n", turn.content)
    print("tool_calls:", turn.tool_calls)

    print("\n=== 3. TaggedStreamParser：分块流式 ===\n")
    parser = TaggedStreamParser()
    chunks = [
        "<thi",
        "nk>流式</thi",
        "nk><tool_calls>[",
        '{"name":"Read","arguments":{"path":"a.py"}}]</tool_calls>',
    ]
    events = []
    for c in chunks:
        events.extend(parser.feed(c))
    events.extend(parser.finish())
    print("分块:", chunks)
    print("事件序列:")
    for ev in events:
        d = {k: v for k, v in asdict(ev).items() if v is not None}
        print(" ", ev.type, json.dumps(d, ensure_ascii=False))

    print("\n=== 4. parse_to_openai_assistant_turn：按 tool_choice 做校验 ===\n")
    constrained = (
        "<think>只允许 Read</think>"
        '<tool_call>{"name":"Read","arguments":{"path":"README.md"}}</tool_call>'
    )
    checked = parse_to_openai_assistant_turn(
        constrained,
        tool_choice={"type": "function", "function": {"name": "Read"}},
    )
    print("finish_reason:", checked.finish_reason)
    print("tool:", checked.tool_calls[0].function.name)


if __name__ == "__main__":
    main()
