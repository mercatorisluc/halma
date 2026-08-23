# Checkpoint manifest

The `.zip` files here are gitignored — binary, large, and mostly regenerable.
This file is not, and it is the only place that records **which code can read
which checkpoint**.

That question has no other answer. A `.zip` carries an observation space and a
policy architecture, and the only way to discover them is to try loading it.
Three checkpoints stopped loading on 2026-08-07 and nobody noticed for two
weeks: `scripts/playAgainstAgent.py` still defaulted to one, so the documented
way to play a human game crashed on startup, and a third of `CLAUDE.md`'s
commands named files no script could open. Keeping this table current is
cheaper than rediscovering that.

**When you produce a checkpoint, add a row.** When a change to `env/` alters
the observation, the action encoding or the policy architecture, re-run the
load check below and mark what broke.

```bash
.venv/bin/python -c '
from pathlib import Path
from sb3_contrib import MaskablePPO
for p in sorted(Path("models").glob("*.zip")):
    try:
        MaskablePPO.load(p); print(f"{p.stem:<26} loads")
    except Exception as e:
        print(f"{p.stem:<26} FAILS: {type(e).__name__}")
'
```

## Generation 2 — loads on current code

Observation `Dict{board: (14, 17, 17), scalars: (5,)}`, `FactoredMaskablePolicy`
with the split trunk. Everything below was built after the 2026-08-07 change.

| Checkpoint | Size | How it was made |
|---|---|---|
| `Talos2.0_round5` | 24.7 MB | **The strongest checkpoint that exists.** Round 5 of the stage-3 league. 70.5% over its anchor across all 800 two-ply openings, 64.5% off the beaten track, 72.0 ± 6.2 argmax / 58.0 ± 6.8 sampled against `lookahead2` |
| `Talos2.0_round4` | 24.7 MB | Round 4. Kept as the only sparring partner of the right strength; every other round is superseded |
| `Talos2.0_round1..3` | 24.7 MB each | Earlier league rounds. Referenced by nothing; their numbers are in `RESULTS.md`. Safe to delete |
| `talos2_stage6` | 24.7 MB | End of stage 2, and the league's anchor. `scripts/talos2League.py --init` defaults to it |
| `talos2_parity025` | 24.7 MB | Stage 2a, parity 0.25 — the arm the whole generation-2 lineage descends from |
| `talos2_parity0` | 24.7 MB | Stage 2a, parity 0. The untouched control arm of the replication still open in `TODO.md` |
| `clone_from_multi_500k` | 8.3 MB | The stage-1 clone, 500k samples, three teachers |
| `clone_from_multi` | 8.3 MB | The same at 150k samples. Superseded |
| `teacherSingle` | 8.3 MB | 2026-08-22 teacher comparison, single teacher (`calibrated`) |
| `teacherMixed` | 8.3 MB | The same comparison, five teachers mixed. Lost on breadth *and* strength — see `RESULTS.md` |

## Generation 1 — **cannot be loaded by any current code**

Observation `Box(3, 17, 17)`, undivided extractor. The 2026-08-07 observation
and critic changes made all three unreadable; `MaskablePPO.load` raises
`RuntimeError` on the state dict. They are also the only artefacts here that
**cannot be regenerated** — the scripts that built them (`progressivePhase1.py`,
`progressivePhase2.py`) were deleted on 2026-08-23 because their own `--init`
had stopped loading too, and steps 1–6 of that lineage were never reproducible.

| Checkpoint | Size | Status |
|---|---|---|
| `Talos1.0` | 8.3 MB | Dead weight. Kept only because deleting is irreversible |
| `Talos1.1` | 8.3 MB | Dead weight |
| `Talos1.2` | 8.3 MB | Dead weight. Was the strongest of its generation |

Their recorded numbers are in `RESULTS.md` and remain the only thing about them
that still matters. Note that those numbers were measured against the *old* bot
panel, so they are not comparable with anything measured after 2026-08-14.

## The one risk this file does not cover

`Talos2.0_round5` exists in a single copy, on one disk, outside version
control, and the recipe that produced it takes about five hours to re-run.
Backing it up somewhere else is the cheapest insurance in this project.
