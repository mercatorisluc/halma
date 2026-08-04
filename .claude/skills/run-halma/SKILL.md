---
name: run-halma
description: Build, run and drive the Halma game — launch the pygame front-end headlessly, click pieces, take screenshots, step the RL environment, seat a trained policy, and run the tests and lint/format/type gates. Use when asked to run or start Halma, play a move, screenshot the board, check the app still works after a change, or exercise game/, heuristics/, visual/ or env/.
---

The app is a pygame Halma board (`main.py`) whose main loop blocks forever — no
use to an agent. Drive it instead with **`.claude/skills/run-halma/driver.py`**,
which runs the same front-end headless (`SDL_VIDEODRIVER=dummy`), one frame at a
time, over a line REPL on stdin: synthetic clicks and arrow keys in, board state
and PNG screenshots out. It also has one-shot modes for the non-visual layers
(`engine`, `env`, `policy`), which is what most changes here actually touch.

All paths below are relative to the repo root, and every command assumes the
venv's interpreter (`.venv/bin/python`) — the driver needs `pygame`, and `env`
mode needs `gymnasium`.

## Prerequisites

macOS or Linux, Python 3.13 (3.11 also works — there are `cpython-311` caches in
the tree). No system packages needed: the driver never opens a window, so no
X server, no `xvfb`, no `libgl`. `tmux` is **not** required.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`sb3-contrib` pulls torch (~530 MB). Skip it only if you will not touch `env/`;
the driver's `repl`, `smoke` and `engine` modes never import it.

No build step.

## Run (agent path)

### One-shot: does the whole app still work?

```bash
.venv/bin/python .claude/skills/run-halma/driver.py smoke /tmp/halma-shots
```

Launches the front-end, plays a human move through synthetic clicks, lets the
bot reply, plays 20 more moves, rotates the board, steps the history back and
forward, and asserts each of those took effect. Ends with `SMOKE OK` and four
PNGs in `/tmp/halma-shots/` (`01-opening`, `02-after-human-move`, `03-midgame`,
`04-rotated`). Takes ~20 s — `lookahead2` thinks for ~130 ms a move. **Open the
screenshots.** A green exit with a blank board is still a failure.

### Scripted: a specific position, then look at it

Pipe commands in. This is the normal way to drive it:

```bash
printf 'new seed=3\nmoves 5\nmove 6 28\nss /tmp/halma-shots/human.png\nquit\n' \
  | .venv/bin/python .claude/skills/run-halma/driver.py
```

```
halma driver ready. `help` for commands.
seat 1 on move (human=True)  moves=0 cursor=0  home={1: 0, 2: 0}  winner=None
20 legal: [(6, 28), (6, 26), (7, 29), (7, 27), (8, 30)]
clicked field 6 at (240.0, 473.2050807568877)
clicked field 28 at (280.0, 403.92304845413264)
seat 1 on move (human=True)  moves=2 cursor=2  home={1: 0, 2: 0}  winner=None
wrote /tmp/halma-shots/human.png (13307 bytes)
```

`moves=2` after one human move is correct: the bot's reply is the second.

| command | what it does |
|---|---|
| `new [seed=N] [agent=PATH] [sampled]` | start a game. Bare: human on seat 1 vs `lookahead2`, as `main.py`. With `agent=models/Talos1.1`: the policy takes **seat 1** and the human seat 2, as `scripts/playAgainstAgent` — the observation is built for seat 1 and `NeuralComputer` refuses any other. |
| `tick [n=1]` | run n main-loop frames. A bot on move plays one move per frame; a human on move idles. |
| `state` | seat on move, human?, move count, playback cursor, pieces home per seat, winner |
| `moves [n=10]` | legal `(start, end)` endpoint moves for whoever is on move |
| `move START END` | click START, then END — a real human move through `HumanInputHandler`, then one tick for the bot's reply. Reports `rejected` if the pair is not legal. |
| `click FIELD_ID` | a single synthetic click at that field's pixel, via the projector |
| `key LEFT\|RIGHT\|UP\|DOWN` | history back / forward, rotate ±60° |
| `auto [n=1]` | play n human moves picked at random from the legal set, with the bot replying |
| `ss PATH` | screenshot the current frame (creates parent dirs) |

### Stepwise: keep one game alive across several commands

There is no tmux here. Use a FIFO — the `sleep` holds the write end open so the
driver does not see EOF and exit:

```bash
mkfifo /tmp/halma.in
(sleep 600 > /tmp/halma.in &)
.venv/bin/python .claude/skills/run-halma/driver.py < /tmp/halma.in > /tmp/halma.out 2>&1 &
sleep 2; echo "new seed=3" > /tmp/halma.in; sleep 1; cat /tmp/halma.out
```

Then one command per step, reading `/tmp/halma.out` between them:

```bash
echo "move 6 28" > /tmp/halma.in; sleep 1; echo "ss /tmp/halma-shots/fifo.png" > /tmp/halma.in; sleep 1; tail -5 /tmp/halma.out
```

Tear down with `echo quit > /tmp/halma.in; rm -f /tmp/halma.in`.

## Direct invocation — the layers without pygame

Most changes here land in `game/`, `heuristics/` or `env/`, and none of those
need the front-end. Each mode is a few seconds:

```bash
.venv/bin/python .claude/skills/run-halma/driver.py engine 1
# → winner=2 after 132 moves          (ComputedGame, advancedDistScore vs sparsityScore)

.venv/bin/python .claude/skills/run-halma/driver.py env 40
# → obs keys=['board', 'scalars'] shapes={'board': (3, 17, 17), 'scalars': (4,)}
# → 40 steps ok, return=0.100, last info keys=['action_mask', 'illegalAction', 'outcome']

.venv/bin/python .claude/skills/run-halma/driver.py policy models/Talos1.1
# → models/Talos1.1 vs advancedDistScore: winner=1 in 98 moves
```

Fast smoke runs of the training scripts, when you have changed them:

```bash
.venv/bin/python -m scripts.pretrain --samples 2000 --epochs 1 --games 2 --name smokeClone   # ~7 s
.venv/bin/python -m scripts.train --steps 2048 --games 2 --envs 2 --name smokeRun            # ~18 s
rm -f models/smokeClone.zip models/smokeRun.zip
```

Both **write into `models/`** — delete the checkpoints afterwards, as above, or
you leave junk next to the real ones. The win rates at these sizes are 0% and
mean nothing; you are checking that the loop runs, not that it learns.

The bot yardstick is the expensive one — it plays every pairing of six
strategies, so cost scales with `--games`. `--games 2` already takes 43 s:

```bash
.venv/bin/python -m scripts.baseline --games 2
```

## Run (human path)

```bash
.venv/bin/python main.py                       # → a 600×600 pygame window, human vs lookahead2
.venv/bin/python -m scripts.playAgainstAgent --model models/tunedEnt001
```

Click a piece then a highlighted destination; ←/→ step history, ↑/↓ rotate.
The loop never returns until the window is closed or a player wins, so do not
run this in the foreground of a tool call — background it and `kill` it:

```bash
.venv/bin/python main.py & P=$!; sleep 6; kill $P
```

## Test

```bash
.venv/bin/python -m pytest        # 89 passed in 3.85s
.venv/bin/ruff check .            # must stay clean
.venv/bin/ruff format --check .   # must stay clean
.venv/bin/basedpyright .          # 0 errors, 0 warnings, 0 notes
```

All four are the pre-commit hook (`git config core.hooksPath .githooks`). There
are **no tests for `visual/`** — the driver is the only thing that exercises the
front-end, which is why a change there needs a screenshot, not a green suite.

## Gotchas

- **The front-end's state lags the drawn frame by one tick.** `runGame` draws
  *before* it handles events, so a screenshot taken right after a click or key
  shows the previous state. It cost two byte-identical PNGs to notice: rotating
  with `key UP` and screenshotting produced the same file as before the
  rotation. The driver's `ss` redraws first, so this is already handled — but
  any code you write against `GameVisualization` directly will hit it.
- **Two clicks in one frame do not make a move.** `handleClickedField` only
  reads the second click as the end of a pair when `waitingForHumanMove` is
  already `True`, and that flag is refreshed by `adaptToHumanInteraction`
  *after* the events are drained. So a fresh pair posted in one frame just
  overwrites `clickedField`. The driver posts one click per tick
  (`Driver.clickPair`); do the same if you drive the loop yourself.
- **Play order is randomised**, so seat 1 is not reliably on move at `moves=0`.
  `new seed=N` fixes it; without a seed, `tick` until `state` says
  `human=True`.
- **`seed=` does not fix the position, only who starts.** Bots break ties at
  random from the game's own `rng`, so two runs at the same seed diverge.
  Compare screenshots by eye, not by hash.
- **`agent=` flips which seat is human.** With a policy seated, the human is
  seat 2, so `move` and `auto` act on seat 2's pieces. `state` tells you.
- **`SDL_VIDEODRIVER=dummy` must be set before pygame touches the display.**
  The driver sets it at import time, above its own pygame import; setting it
  after `pygame.display.set_mode()` does nothing and you get a real window.
- **The board rotates but does not re-scale**, and `key UP`/`DOWN` change only
  `projector.flipAngle`. Clicks still work after a rotation because
  `Driver.pixelOf` goes through `projector.visualPosition`, the same table
  hit-testing uses. Hardcoded pixel coordinates would not.
- **`driver.py tick` duplicates the ten-line body of
  `GameVisualization.runGame`** — there is no seam to step the loop one frame.
  If you change `runGame`, change `Driver.tick` and `Driver.render` with it.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'game'`** — running a script by path
  puts the script's directory on `sys.path`, not the repo root. The driver
  prepends the root itself; if you write your own script under
  `.claude/skills/`, do the same or run it as `-m` from the root.
- **`error: move X->Y rejected -- not legal from here`** — either the pair is
  genuinely illegal or it is not the human's turn. Run `moves` first and pick a
  pair from what it prints; run `state` to check the seat.
- **`error: not the human's turn -- tick until it is`** — the bot has the move.
  `tick 1` plays it.
- **The REPL exits immediately when driven by FIFO** — the writer closed and the
  driver saw EOF. Keep a long `sleep` redirected into the FIFO, as in the
  stepwise recipe above.
- **`timeout: command not found`** on macOS — no coreutils. Use
  `cmd & P=$!; sleep N; kill $P` instead.
