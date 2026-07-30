# Halma

A Python implementation of the board game [Halma](https://en.wikipedia.org/wiki/Halma)
on a star-shaped (Chinese Checkers style) board, with a pygame visualization,
heuristic-driven bot players, and a Gymnasium environment for reinforcement-learning
experiments.

## Project layout

| Path          | Responsibility                                                        |
|---------------|-----------------------------------------------------------------------|
| `game/`       | Core game engine: board, fields, moves, players, rules (pure Python). |
| `heuristics/` | Move-selection strategies used by the bot players.                    |
| `visual/`     | pygame-based visualization and human interaction.                     |
| `env/`        | Gymnasium environment wrapping the engine for RL.                     |
| `main.py`     | Entry point — launches an interactive game.                           |

The game engine (`game/`) has no dependency on pygame; the visualization is one
possible frontend on top of it.

For how it all fits together — layer diagram, board geometry, the field
addressing schemes and the invariants to respect — see
[ARCHITECTURE.md](ARCHITECTURE.md).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

You play as the human against a bot, which moves automatically when it is its
turn. Controls:

- **Mouse** — click one of your pieces, then a highlighted destination, to move
- **←/→** — step backward / forward through the move history
- **↑/↓** — rotate the board by 60°

## Tests

```bash
pytest
```
