@AGENTS.md

# Claude Code notes

- Before editing, check which folder you are in against the ownership table and stay inside it.
- For any change to features, codegen, firmware or toolchains, run `make parity` (or at least `uv run pytest tests/unit`) before saying you are done, and quote the result.
- Use the `parity-reviewer` agent in `.claude/agents/` to review changes that touch Python features or generated C.
- Do not edit `docs/contracts/` unless the task explicitly says the lead approved the contract change.
