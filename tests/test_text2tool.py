import json
import unittest

from toolcall_gateway import (
    TaggedOutputError,
    TaggedStreamParser,
    ToolChoiceError,
    parse_tagged_output,
    parse_to_openai_assistant_turn,
)


class TestText2Tool(unittest.TestCase):
    def test_parse_single_tool_call_with_thinking(self) -> None:
        parsed = parse_tagged_output(
            "<think>Read file first</think>"
            '<tool_call>{"name":"Read","arguments":{"path":"a.py"}}</tool_call>'
        )

        self.assertEqual(parsed.thinking, "Read file first")
        assert parsed.tool_call is not None
        self.assertEqual(parsed.tool_call.name, "Read")
        self.assertEqual(parsed.tool_call.arguments, {"path": "a.py"})
        self.assertEqual(len(parsed.tool_calls), 1)
        self.assertIsNone(parsed.final_answer)

    def test_parse_final_answer(self) -> None:
        turn = parse_to_openai_assistant_turn(
            "<think>Done</think><final_answer>Hello world</final_answer>"
        )

        self.assertEqual(turn.finish_reason, "stop")
        self.assertEqual(turn.thinking, "Done")
        self.assertEqual(turn.content, "<think>Done</think>\n\nHello world")
        self.assertEqual(turn.tool_calls, [])

    def test_parse_tool_calls_to_openai_semantics(self) -> None:
        turn = parse_to_openai_assistant_turn(
            "<think>Need file</think>"
            '<tool_calls>[{"name":"Read","arguments":{"path":"a.py"}},'
            '{"name":"Read","arguments":{"path":"b.py"}}]</tool_calls>'
        )

        self.assertEqual(turn.finish_reason, "tool_calls")
        self.assertEqual(turn.content, "<think>Need file</think>")
        self.assertEqual(len(turn.tool_calls), 2)
        self.assertEqual(turn.tool_calls[0].function.name, "Read")
        self.assertEqual(
            json.loads(turn.tool_calls[0].function.arguments),
            {"path": "a.py"},
        )
        self.assertEqual(
            json.loads(turn.tool_calls[1].function.arguments),
            {"path": "b.py"},
        )

    def test_rejects_text_outside_tags(self) -> None:
        with self.assertRaises(TaggedOutputError):
            parse_tagged_output("prefix <final_answer>Hello world</final_answer>")

    def test_parse_to_openai_assistant_turn_respects_tool_choice_none(self) -> None:
        with self.assertRaises(ToolChoiceError):
            parse_to_openai_assistant_turn(
                '<tool_call>{"name":"Read","arguments":{"path":"a.py"}}</tool_call>',
                tool_choice="none",
            )

    def test_parse_to_openai_assistant_turn_respects_tool_choice_required(self) -> None:
        with self.assertRaises(ToolChoiceError):
            parse_to_openai_assistant_turn(
                "<final_answer>Hello world</final_answer>",
                tool_choice="required",
            )

    def test_parse_to_openai_assistant_turn_respects_specific_function_choice(self) -> None:
        with self.assertRaises(ToolChoiceError):
            parse_to_openai_assistant_turn(
                '<tool_call>{"name":"Write","arguments":{"path":"a.py"}}</tool_call>',
                tool_choice={"type": "function", "function": {"name": "Read"}},
            )

    def test_parse_to_openai_assistant_turn_validates_available_tools(self) -> None:
        with self.assertRaises(ToolChoiceError):
            parse_to_openai_assistant_turn(
                '<tool_call>{"name":"Write","arguments":{"path":"a.py"}}</tool_call>',
                available_tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "Read",
                            "description": "Read a file",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            )

    def test_stream_parser_handles_chunked_tool_calls(self) -> None:
        parser = TaggedStreamParser()
        events = []
        for chunk in [
            "<thi",
            "nk>Need file</thi",
            "nk><tool_",
            'calls>[{"name":"Read","arguments":{"path":"a.py"}},',
            '{"name":"Read","arguments":{"path":"b.py"}}]</tool_calls>',
        ]:
            events.extend(parser.feed(chunk))
        events.extend(parser.finish())

        self.assertEqual(
            [event.type for event in events],
            [
                "message_start",
                "block_start",
                "block_delta",
                "block_end",
                "tool_call",
                "tool_call",
                "message_stop",
            ],
        )
        self.assertEqual(events[2].text, "Need file")
        self.assertEqual(events[4].name, "Read")
        self.assertEqual(events[4].arguments, {"path": "a.py"})
        self.assertEqual(events[5].arguments, {"path": "b.py"})
        self.assertEqual(events[6].stop_reason, "tool_use")


if __name__ == "__main__":
    unittest.main()
