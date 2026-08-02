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

# Play against a trained policy instead of a heuristic bot
python -m scripts.playAgainstAgent
python -m scripts.playAgainstAgent --model models/tunedEnt001 --sampled

# Run the whole test suite
pytest

# Run a single test file / test
pytest tests/test_moves.py
pytest tests/test_moves.py::test_is_jump_move

# Lint (must stay clean)
ruff check .
ruff check . --fix

# Format (must stay clean)
ruff format .
ruff format --check .

# Type check (must stay clean)
basedpyright .

# Bot strength: every pairing of the heuristics (the yardstick)
python -m scripts.baseline --games 150

# Clone a heuristic bot into the policy (PPO from scratch does not get there)
python -m scripts.pretrain --samples 150000 --epochs 12

# Fine-tune that clone with PPO and score it against the yardstick
python -m scripts.train --steps 300000 --games 200 --init models/cloned

# Fine-tune against a pool of bots instead of one, so the agent does not
# just specialise to --opponent
python -m scripts.train --steps 300000 --init models/cloned \
    --opponentPool advancedDistScore sparsityScore bottleneck
```

All four must stay green; there is no build step. The pre-commit hook in
`.githooks/` runs exactly these, so enable it once per clone:

```bash
git config core.hooksPath .githooks
```

Both the ruff rule set and the formatter settings are pinned in `ruff.toml`
rather than left to ruff's shifting defaults; the type checker is configured in
`pyrightconfig.json`, which the editor reads as well.

**Ruff does not check types.** That is why the type checker is a separate gate:
things like assigning an `int | str` into a `list[str]` pass lint cleanly and
would otherwise only surface as a squiggle in the editor. `env/` is excluded
from type checking on purpose — `HalmaEnv` does not satisfy the Gymnasium API
yet, and fixing that is the first task of the RL work.

Layout is the formatter's decision — do not hand-tune blank lines, quotes or
line breaks, and run `ruff format .` before committing. Naming is the one thing
the tooling does not enforce: `N` (pep8-naming) is deliberately **not** enabled,
because this project uses camelCase throughout. That is un-pythonic but
consistent, and enabling `N` would flag ~300 violations and invite a rename
touching every file for no functional gain. Match the surrounding camelCase in
new code.

## Type annotations

`game/` and `heuristics/` annotate every method signature, using the aliases in
`game/boardTypes.py` (`FieldId`, `Coord`, `PlayerId`, `MoveEndpoints`,
`MovePath`, `AnyMove`). Keep new engine code annotated — the point is that the
two move representations become checkable rather than merely documented.

Annotate signatures, not obvious locals. There is no type checker in CI; the
value is what the editor reports while writing. `visual/` and `env/` are
deliberately unannotated, since both are due to be rewritten.

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
