import unittest

from toolcall_gateway import (
    ToolChoiceError,
    build_prompt,
    build_tagged_prompt,
    format_tools_for_prompt,
)


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


class TestTool2Text(unittest.TestCase):
    def test_format_tools_for_prompt(self) -> None:
        rendered = format_tools_for_prompt(TOOLS)
        self.assertIn("Read(path: string (required))", rendered)
        self.assertIn("Read a file", rendered)

    def test_format_tools_for_prompt_properties_none(self) -> None:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "NoProps",
                    "description": "Tool with null properties",
                    "parameters": {
                        "type": "object",
                        "properties": None,
                        "required": [],
                    },
                },
            }
        ]
        rendered = format_tools_for_prompt(tools)
        self.assertIn("NoProps():", rendered)
        self.assertIn("Tool with null properties", rendered)

    def test_build_tagged_prompt(self) -> None:
        prompt = build_tagged_prompt(TOOLS)
        self.assertIn("<tool_calls>", prompt)
        self.assertIn("## Available tools", prompt)
        self.assertIn("Read(path: string (required))", prompt)

    def test_build_tagged_prompt_tool_choice_none_hides_tools(self) -> None:
        prompt = build_tagged_prompt(TOOLS, tool_choice="none")
        self.assertIn("Tool choice for this response is fixed to none.", prompt)
        self.assertIn("Your terminal block must be <final_answer>.", prompt)
        self.assertNotIn("## Available tools", prompt)

    def test_build_prompt_tool_choice_required(self) -> None:
        prompt = build_prompt(
            [{"role": "user", "content": "Find the answer"}],
            tools=TOOLS,
            tool_choice="required",
            allow_parallel_tool_calls=True,
        )
        self.assertIn("tool_choice is required", prompt)
        self.assertIn("Do not output <final_answer> in this response.", prompt)

    def test_build_prompt_tool_choice_specific_function_filters_tool_list(self) -> None:
        prompt = build_prompt(
            [{"role": "user", "content": "Read a.py"}],
            tools=TOOLS
            + [
                {
                    "type": "function",
                    "function": {
                        "name": "Write",
                        "description": "Write a file",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            tool_choice={"type": "function", "function": {"name": "Read"}},
        )
        self.assertIn("function 'Read'", prompt)
        self.assertIn("Read(path: string (required))", prompt)
        self.assertNotIn("Write()", prompt)

    def test_build_prompt_tool_choice_specific_function_missing_raises(self) -> None:
        with self.assertRaises(ToolChoiceError):
            build_prompt(
                [{"role": "user", "content": "Read a.py"}],
                tools=TOOLS,
                tool_choice={"type": "function", "function": {"name": "Write"}},
            )

    def test_build_prompt_replays_tool_history(self) -> None:
        prompt = build_prompt(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Find the answer"},
                {
                    "role": "assistant",
                    "content": "<think>Need to inspect the file</think>",
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
                    "content": "file contents",
                },
                {"role": "user", "content": "continue"},
            ],
            tools=TOOLS,
            allow_parallel_tool_calls=True,
        )

        self.assertIn("System:\nYou are helpful.", prompt)
        self.assertIn("User:\nFind the answer", prompt)
        self.assertIn("Assistant:\n<think>Need to inspect the file</think>", prompt)
        self.assertIn(
            '<tool_calls>[{"name": "Read", "arguments": {"path": "a.py"}}]</tool_calls>',
            prompt,
        )
        self.assertIn("Tool result for call_id=call_123:", prompt)
        self.assertIn("<tool_result>\nfile contents\n</tool_result>", prompt)
        self.assertIn(
            "Now output exactly one response using only the tagged protocol", prompt
        )
        self.assertIn("tool_choice is required", build_prompt(
            [{"role": "user", "content": "Find the answer"}],
            tools=TOOLS,
            tool_choice="required",
        ))


if __name__ == "__main__":
    unittest.main()
