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

# Play against a trained policy instead of a heuristic bot. The default is
# models/Talos2.0_round5, the strongest checkpoint that exists.
python -m scripts.playAgainstAgent
python -m scripts.playAgainstAgent --model models/Talos2.0_round4 --sampled

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

# Clone a heuristic bot into the policy (PPO from scratch does not get there).
# The teacher defaults to `calibrated`, alone -- a mixture was measured and
# bought nothing; see RESULTS.md.
python -m scripts.pretrain --samples 150000 --epochs 12

# The control for the parity penalty in the potential -- see RESULTS.md.
# Default is --parity 0.25; 0 restores the travel-only potential every result
# recorded before 2026-08-07 was measured with
python -m scripts.train --steps 300000 --init models/cloned --parity 0

# Fine-tune that clone with PPO and score it against the yardstick
python -m scripts.train --steps 300000 --games 200 --init models/cloned

# Fine-tune against a pool of bots instead of one, so the agent does not
# just specialise to --opponent. The pool is the frozen four -- see the
# panel note below.
python -m scripts.train --steps 300000 --init models/cloned \
    --opponentPool calibrated calibratedCluster calibratedJump calibratedClusterJump

# The three stages that build a .0 model, in order -- see RESULTS.md for
# what each one measured. Stage 2 is two runs: the clone is fine-tuned against
# heuristics alone, then against heuristics and checkpoints mixed.
python -m scripts.pretrain --samples 500000 --epochs 12 \
    --expert calibrated --name clone
python -m scripts.train --steps 300000 --games 200 --init models/clone \
    --opponentPool calibrated calibratedCluster calibratedJump calibratedClusterJump \
    --lr 1e-4 --targetKl 0.02 --name stage2a
python -m scripts.train --steps 500000 --games 200 --init models/stage2a \
    --opponentPool calibrated calibratedCluster calibratedJump calibratedClusterJump random \
    --opponentModelPool models/clone --lr 1e-4 --targetKl 0.02 --name stage2b
python -m scripts.talos2League --init models/stage2b --basename myRun_round

# Everything in the league is a flag, and the defaults reproduce the recorded
# Talos2.0 run. Two the TODO asks for: --parity 0 is the control arm of the
# replication, and --poolCap keeps the anchor plus only the last N rounds, so a
# long league stops spending most of its episodes re-beating its own history.
python -m scripts.talos2League --parity 0 --basename parity0_round
python -m scripts.talos2League --rounds 8 --poolCap 3 --basename extended_round

# Note that --name is prefixed with models/ by scripts.train, so pass a bare
# name: --name models/foo writes models/models/foo.zip.

# Open the first six plies at random -- three per side, counted in total --
# so training does not keep replaying the same few positions. Measured once
# and it did not produce a stronger model; see RESULTS.md before reaching
# for it again.
python -m scripts.train --steps 300000 --init models/Talos2.0_round5 \
    --opponentModel models/Talos2.0_round4 --randomOpening 6 \
    --lr 1e-4 --targetKl 0.02 --entropy 0.03 --name next

# Let a checkpoint opponent play its distribution in half the episodes, so
# training is not answered the same way in every position it revisits
python -m scripts.train --steps 300000 --init models/Talos2.0_round5 \
    --noHeuristicOpponents --opponentModelPool models/Talos2.0_round5 \
    --opponentSampling 0.5 --lr 1e-4 --targetKl 0.02 --name next

# Score checkpoints against the heuristics -- the yardstick, on its own rather
# than as the tail of a training run
python -m scripts.evaluateAgainstBots models/Talos2.0_round5 models/Talos2.0_round4

# Seat a checkpoint behind one ply of search, against the bots. --noSearch is
# the control through the same loop. As measured, the search plays worse than
# the policy it wraps -- see RESULTS.md before building on it
python -m scripts.evaluateSearch models/Talos2.0_round5 --bots lookahead2 --games 25
python -m scripts.evaluateSearch models/Talos2.0_round5 --bots lookahead2 --noSearch

# Compare two checkpoints over all 800 two-ply openings -- the yardstick once
# the heuristics saturate at 100%
python -m scripts.openingSweep models/Talos2.0_round5 models/Talos2.0_round4

# Compare them off the beaten track instead: random starting positions, each
# played twice with the seats swapped. Baseline in RESULTS.md
python -m scripts.randomPositionSweep models/Talos2.0_round5 models/Talos2.0_round4

# Watch one of those games, stepping the history with the arrow keys
python -m scripts.replayGame --opening 6,26 66,67 --start 70 \
    models/Talos2.0_round5 models/Talos2.0_round4
```

**Only generation-2 checkpoints load.** The 2026-08-07 observation and critic
changes broke `Talos1.0`, `Talos1.1` and `Talos1.2`; they are still on disk but
no current script can read them, so every command above names a `Talos2.0_*`
file. `models/Talos2.0_round5` is the strongest and `round4` the only sparring
partner of the right strength.

**The bot panel was frozen on 2026-08-21** at `calibrated`,
`calibratedCluster`, `calibratedJump` and `calibratedClusterJump`. They share
one scoring function and differ only in their weight vector, and they were
selected on pairwise *move agreement* rather than win rate — the pool exists to
be varied, not uniformly strong. `distance`, `tipDistance`, `shaped`,
`straggler`, `random` and `lookahead2` still exist and are still the evaluation
yardstick; they are simply no longer what a training pool is built from.

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
would otherwise only surface as a squiggle in the editor.

Layout is the formatter's decision — do not hand-tune blank lines, quotes or
line breaks, and run `ruff format .` before committing. Naming is the one thing
the tooling does not enforce: `N` (pep8-naming) is deliberately **not** enabled,
because this project uses camelCase throughout. That is un-pythonic but
consistent, and enabling `N` would flag ~300 violations and invite a rename
touching every file for no functional gain. Match the surrounding camelCase in
new code.

## Type annotations

`game/`, `heuristics/`, and `env/` annotate every method signature, using the
aliases in `game/boardTypes.py` (`FieldId`, `Coord`, `PlayerId`,
`MoveEndpoints`, `MovePath`, `AnyMove`) wherever engine types are in play. Keep
new code annotated — the point is that the two move representations become
checkable rather than merely documented.

Annotate signatures, not obvious locals. There is no type checker in CI; the
value is what the editor reports while writing. `visual/` is the one area
still deliberately unannotated, since it is due to be replaced by the browser
GUI.

## Architecture

**See `ARCHITECTURE.md`** — it is the single source of truth for how the code is
structured: the layer diagram, the board's geometry, the three field-addressing
schemes, the two move representations and the load-bearing invariants. What was
*measured* lives in `RESULTS.md`. Do not restate either here; keep this file to
commands and working conventions so they cannot drift apart.

The three things worth having in mind before reading any code:

1. The engine (`game/`) is pure Python with no pygame or RL dependency, and has
   **three consumers** — `heuristics/`, `visual/`, `env/`. Changing engine
   behaviour affects all three, but only `visual/` shows it.
2. Fields are addressed three ways (`coord`, `id`, `fieldNumber`). Use `id`
   unless you are doing geometry.
3. Moves come in two forms — endpoints `(start, end)` and full jump path
   `[start, ..., end]`. Picking the wrong one is the classic mistake here.

## Keeping the docs current

Four files, four jobs, and they are kept apart on purpose so they cannot drift
into each other:

| File | Holds | Grows? |
|---|---|---|
| `ARCHITECTURE.md` | **Structure**: layers, geometry, the two move representations, the invariants | No — it should stay roughly its current length |
| `RESULTS.md` | **Evidence**: what was measured, what it means, what failed | Yes, only ever appended |
| `TODO.md` | **What is pending.** Items are deleted when done | Churns |
| `CLAUDE.md` | Commands and working conventions. This file | No |

`ARCHITECTURE.md` is maintained deliberately, not automatically. When a change
alters structure — a module's responsibility, an invariant, a layer boundary,
the state of `env/` or `visual/` — update it in the same commit. Wording fixes
and internal refactors that leave the structure intact do not need an update.
The `doc-sync` subagent (`.claude/agents/`) checks exactly that and reports
which sections went stale; it does not edit.

When a measurement finishes, the result goes to `RESULTS.md` and the item comes
out of `TODO.md`. Do not put results in `ARCHITECTURE.md` — that is the drift
this split was made to stop.

**A measured figure has one home.** In code, state the claim qualitatively —
"the strongest one-ply bot", "much better on sampled play" — and leave the
digits to `ARCHITECTURE.md` or `RESULTS.md`. A qualitative claim survives a
re-measurement; a percentage does not, and the bot panel has already changed
twice. Before this rule, `shaped`'s strength lived in four files at once.

Three kinds of number stay in the code, because they are not bot strengths and
do not go stale when a bot is re-measured: the `STRAGGLER_WEIGHT` and
`SHAPE_WEIGHT` sweeps, which sit at the constant they warn about and *are* the
warning; costs and confidence levels, which are properties of the code; and the
evidence behind a load-bearing design choice, such as the potential's sign or
`_needsFlip`'s home corner.

**Before quoting any number from `RESULTS.md`, read its "How to read the
numbers here".** In particular, argmax margins recorded before 2026-08-22 are
far too tight: against a deterministic opponent, 30 argmax games are about five
distinct game lines rather than 30 independent samples.
