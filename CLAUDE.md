# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the interactive game (human vs. bot, pygame window)
python main.py

# Run the whole test suite
pytest

# Run a single test file / test
pytest tests/test_moves.py
pytest tests/test_moves.py::test_is_jump_move

# Lint (must stay clean)
ruff check .
ruff check . --fix
```

`pytest` and `ruff check .` are the two CI-relevant commands; there is no build
step and no type checking. The ruff rule set is pinned explicitly in `ruff.toml`
rather than left to ruff's shifting defaults. Note that `N` (pep8-naming) is
deliberately **not** enabled: this project uses camelCase throughout, which is
un-pythonic but consistent, and enabling `N` would flag ~300 violations and
invite a rename touching every file for no functional gain. Match the
surrounding camelCase in new code.

## Architecture

**See `ARCHITECTURE.md`** — it is the single source of truth for how the code is
structured: the layer diagram, the board's geometry, the three field-addressing
schemes, the two move representations, the load-bearing invariants, and the
current state of each area. Do not restate that content here; keep this file to
commands and working conventions so the two cannot drift apart.

The three things worth having in mind before reading any code:

1. The engine (`game/`) is pure Python with no pygame or RL dependency, and has
   **three consumers** — `heuristics/`, `visual/`, `env/`. Changing engine
   behaviour affects all three, but only `visual/` shows it.
2. Fields are addressed three ways (`coord`, `id`, `fieldNumber`). Use `id`
   unless you are doing geometry.
3. Moves come in two forms — endpoints `(start, end)` and full jump path
   `[start, ..., end]`. Picking the wrong one is the classic mistake here.

## Keeping the docs current

`ARCHITECTURE.md` is maintained deliberately, not automatically. When a change
alters structure — a module's responsibility, an invariant, a layer boundary,
the state of `env/` or `visual/` — update it in the same commit. Wording fixes
and internal refactors that leave the structure intact do not need an update.
