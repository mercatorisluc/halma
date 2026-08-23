---
name: halma-measure
description: Run a long Halma measurement or training job correctly — bot baselines, checkpoint sweeps, weight fits, PPO runs, league rounds. Covers how to launch it so it does not hang invisibly or fork-bomb, roughly what it will cost in wall time, and how to read the result without over-reading it. Use whenever starting anything under scripts/ that takes more than a minute.
---

Everything under `scripts/` that plays games is slow, and three of the ways it
goes wrong look identical to "it is just taking a while". This is how to launch
one, what it should cost, and how to read what comes back.

Assume the venv's interpreter throughout (`.venv/bin/python`); nothing here
works with the system Python.

## Launching

**Always `python -u`.** Without it stdout is block-buffered, and a 45-minute
evaluation prints nothing at all until it exits — indistinguishable from a
hang. One run was killed on exactly that mistake.

**Long runs go in the background**, with output to a log at the repo root
(`*.log` is gitignored, so this is safe):

```bash
.venv/bin/python -u -m scripts.evaluateAgainstBots models/Talos2.0_round5 \
    --bots lookahead2 --games 200 > lookahead2.log 2>&1
```

Prefer launching it with the Bash tool's `run_in_background: true` over a
manual `&`. The harness then re-invokes on exit instead of leaving you to
poll, and polling a 45-minute run is pure waste.

**Never run a `multiprocessing.Pool` from a `python - <<'PY'` heredoc.** macOS
starts workers with *spawn*, each worker re-imports the main module, and
`<stdin>` cannot be re-imported: the worker dies, respawns, and loops forever.
Observed once — 1.3 GB of tracebacks and an orphaned parent spinning for hours
at load average 23, which silently starved every later run down to 17% CPU.
Parallel code goes in a real file under `scripts/` with an
`if __name__ == "__main__":` guard, or runs via `python -m scripts.<name>`.

**Never edit an imported source file while a forking run is in flight.** Every
new worker re-imports it, so a half-finished edit to `heuristics/strategy.py`
killed a running fit with `unknown strategy 'calibrated'`. Finish all edits,
then launch.

**Before starting anything long**, confirm no orphan is holding the machine:

```bash
pgrep -f multiprocessing | wc -l    # want 0
ps -eo pid,ppid,etime,command | grep -i python
```

If workers sit far below 100% CPU, suspect an orphan from an earlier failure
before suspecting the workload, and kill the root parent — the workers respawn.

## What things cost

| Command | Rough cost |
|---|---|
| `scripts.baseline --games 150` | minutes; `lookahead2` pairings dominate |
| `scripts.evaluateAgainstBots --games 200` against `lookahead2` | ~29 min **per checkpoint** |
| `scripts.openingSweep` (800 openings) | tens of minutes |
| `scripts.train --steps 300000` | ~1 h, network-bound |
| one league round (`talos2League.py`) | ~13 min |
| `scripts.pretrain --samples 500000` | ~80 min of fitting after collection |

**`--games N` in `evaluateAgainstBots` is N per bot per mode**, and both argmax
and sampled always run — so `--games 200` against one bot plays 400 games, at
~4.3 s/game.

The environment steps at ~1,650/s alone and ~90–120/s with PPO in the loop, so
a training run is bound by the network, not by the game. Speeding up the engine
does not speed up training.

## Reading the result

**Argmax margins against a deterministic opponent are far too tight.** The
opening is fixed and both sides answer identically every time, so a seed varies
only the play order: 30 argmax games produce about 5 distinct game lines, not
30 independent samples. Every `±` on an argmax number computed as
`1.96·sqrt(p(1-p)/n)` understates the real uncertainty. Use `--sampled` for a
genuine win rate, or a forced-opening sweep for a census.

**The 20-game inline reports during a training round mean very little.** Round
1 of one league reported 90–95% against its anchor inline; the 800-opening
sweep put it at 58.5%.

**Read `info["outcome"]`, never the reward.** The reward includes potential
shaping and rises when the agent merely advances.

**A win rate against a saturated panel says nothing.** Every checkpoint since
`Talos1.0` scores 98–100% argmax against the scoring bots. Once that happens
the measurement of record is head-to-head: `scripts.openingSweep` for the
census, `scripts.randomPositionSweep` for play off the beaten track.

## After it finishes

The result goes in `RESULTS.md` (what was measured and why), and the
finished item comes out of `TODO.md`. The log file is scratch — do not commit
it and do not treat it as the record.
