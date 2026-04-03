.PHONY: lint test build verify demo

lint:
	uv run --with ruff ruff check .

test:
	uv run python -m unittest discover -s tests

build:
	uv build

verify: lint test build

demo:
	uv run python examples/demo_tool2text.py
	uv run python examples/demo_text2tool.py
