.PHONY: setup test parity serve demo doctor clean

setup:
	uv sync

test:
	uv run ruff check src tests
	uv run pytest tests/unit -q

parity:
	uv run pytest tests/parity -q -m emulator

serve:
	uv run tinyml serve

doctor:
	uv run tinyml doctor

demo:
	uv run tinyml demo-data pump --out workspace/demo
	uv run tinyml demo-data driving --out workspace/demo
	uv run tinyml demo-data activity --out workspace/demo

clean:
	rm -rf workspace .pytest_cache .ruff_cache
