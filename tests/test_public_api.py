import unittest

from toolcall_gateway import (
    TaggedOutputError,
    ToolChoiceError,
    ToolcallGatewayError,
    __version__,
)


class TestPublicApi(unittest.TestCase):
    def test_version_is_exposed(self) -> None:
        self.assertIsInstance(__version__, str)
        self.assertTrue(__version__)

    def test_error_hierarchy_is_stable(self) -> None:
        self.assertTrue(issubclass(ToolcallGatewayError, ValueError))
        self.assertTrue(issubclass(TaggedOutputError, ToolcallGatewayError))
        self.assertTrue(issubclass(ToolChoiceError, ToolcallGatewayError))


if __name__ == "__main__":
    unittest.main()
