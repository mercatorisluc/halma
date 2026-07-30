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
```

There is no lint/type-check config or build step; `pytest` is the only CI-relevant command.

## Architecture

The engine (`game/`) has **no dependency on pygame or numpy-heavy math beyond distance
caching** — it is pure game logic, reused unchanged by the pygame visualization
(`visual/`) and the Gymnasium RL environment (`env/`). When changing engine behavior,
check all three consumers.

### Board addressing (the main source of confusion)

Fields are addressed **three different ways**, converted between by
`game/initializer.py` and `game/fieldPositionsMapper.py`:

- **`coord`** — axial hex coordinate `(x, y)`, used for geometry (rotation, distance,
  flipping).
- **`id`** — stable sequential index `0-120`, used everywhere else (player
  positions, moves, `board.fields[id]`, the RL action encoding).
- **`fieldNumber`** — `coord` embedded into a 17x17 grid (`(x+8) + (y+8)*17`), used
  only internally by `Initializer.initEdges` to find neighbours via offset
  arithmetic (`directionMapper`) before everything is re-expressed in `id`s.

A move is a tuple/list of `id`s; `(start, end)` for the endpoints-only form used by
`board.allValidMoves`, or the full jump path `[start, ..., end]` used by
`board.allValidMovesWithWay` and `game/move.py`'s `Move`.

### Game engine (`game/`)

- `HalmaGame` (`gameManager.py`) is the base engine: owns the board, players, turn
  order and move history, enforces win conditions. `ComputedGame` (all-bot, used by
  simulations and the RL env) and `InteractiveGame` (one human via the
  visualization) only differ in which players they seat — behavior lives entirely
  in the base class.
- `HalmaBoard` (`board.py`) does move generation (BFS over chained jumps) and holds
  the heuristic scoring functions (`simpleDistanceScore`, `advancedDistanceScore`,
  `sparsityScore`, `potentialJumpScore`, `homeBonusScore`) that the bot strategies
  score positions with. `player.distanceScore` is maintained incrementally on each
  move (`updatePlayerDistanceScore`) rather than recomputed, for performance.
- `HalmaPlayer`/`Computer`/`HumanPlayer` (`player.py`) track a player's piece
  positions and derived sets (`nonArrived`, `openEndPositions`) kept in sync via
  `updatePositionWithMove`. `Computer` delegates move choice to
  `heuristics/strategy.py`'s `Strategy`, selected by name (`"advancedDistScore"`,
  `"sparsityScore"`, `"simpleDistScore"`, `"random"`).
- Field permissions (`Initializer.initPermissions`): a field surrounded entirely by
  one player's own start/end cells is exclusive to that player; every other field
  is open to all. This is what stops a player finishing by squatting in an
  opponent's home triangle.

### Visualization (`visual/`)

`GameVisualization` wires together four focused modules and runs the pygame main
loop: `BoardProjector` (coordinate math + board rotation), `BoardRenderer`
(drawing), `GamePlaybackController` (steps through `game.moves` history, forward/
back), `HumanInputHandler` (click-to-move). The bot moves automatically once the
move cursor (`playback.moveTraveler`) is at the live front of the game; stepping
back through history does not trigger bot auto-play.

### RL environment (`env/`)

`HalmaEnv` wraps `ComputedGame` as a Gymnasium env. `env/boardNormalizer.py`'s
`Normalizer` exploits the board's symmetry: it precomputes field-id permutations
for each player's viewpoint (`player1WithFlip`, `player2WithoutFlip`, etc., plus
their inverses) so that observations/actions can be normalized to a canonical
"player 1, no flip" frame regardless of which player/orientation is actually
moving — meant to let a single policy learn from a symmetry-reduced state space.
This normalization is not yet load-bearing anywhere (`dummyNN` just picks a random
legal move); actions are `start*121 + end` encoded/decoded via
`encode_action`/`decode_action`.

## Current status / in-flight direction

Engine, game manager and the heuristic strategies are refactored and
characterization-tested (see `tests/`, covering `board`/`move`/`player`/
`gameManager`/`heuristics.strategy` behavior). Not yet covered by tests:
`visual/`, `env/` — left untested on purpose, since both are about to be
reworked (see below) rather than kept as-is.

Three-phase plan toward a strong browser-playable bot, in order:

1. **Tests** (done) — close the coverage gaps above the engine.
2. **RL bot** (next) — `HalmaEnv.step()` currently takes `(player,
   permutationKey, action)` instead of the standard Gymnasium `step(action)`;
   fix the API, then train via self-play PPO (stable-baselines3) against the
   existing heuristic bots as a baseline opponent. `dummyNN` (random legal
   move) and the empty `models/` directory are placeholders for this.
3. **Browser GUI** (after) — replace the pygame `visual/` frontend with a
   FastAPI + WebSocket backend around the untouched `game/` engine and the
   trained bot, plus an HTML5-canvas frontend.
