# TODO

The work queue, kept here rather than in a chat session so that clearing a
session costs nothing. `RESULTS.md` stays the record of what was
*measured* and why, and `ARCHITECTURE.md` the record of how the code is
structured; this file is only what is *pending*. When an item finishes, delete
it here and write the result up in `RESULTS.md`.

Convention: `[ ]` open, `[x]` done but not yet written up, `[~]` running.

State as of 2026-08-11: the generation-2 league is finished and pushed
(`6acf060`). `models/Talos2.0_round5` is the strongest checkpoint that exists —
70.5% over its anchor across 800 openings, 64.5% off the beaten track, and
72.0 ± 6.2 argmax / 58.0 ± 6.8 sampled against `lookahead2`. All of it is
written up in RESULTS.md.

**2026-08-14: the heuristic panel was rebuilt and is stronger.** `distance` lost
its static distance half, `shaped` was re-weighted and is now the best one-ply
bot, `stragglerTravelScore` became a pruned search at 5.9x, and every bot and
primitive was renamed. All measured and written up in RESULTS.md; what it
leaves open is the section below.

**2026-08-20: the scores were put on a common scale and the weights fitted.**
Every primitive now maps onto [0, 1] through its own measured distribution
(`heuristics/calibration.py`), so a weight is a weight rather than an accident
of whatever divisor a primitive carried. On top of that sits a family of bots
that share one scoring function and differ only in their weight vector —
`calibrated` plus deliberately partial members, fitted by
`scripts/fitWeights.py` against an opponent pool. The measurements are in
RESULTS.md; what is left is below under "Calibration: what is left".

**2026-08-21: the panel is frozen at four bots.** `calibrated`,
`calibratedCluster`, `calibratedJump` and `calibratedClusterJump`. Two variants
were removed for duplicating `distance` rather than for being weak, one was
fitted to replace them, and the measurement that decides membership is now move
agreement rather than win rate — `scripts/variantAgreement.py`. What the pool
cannot buy by reweighting is bounded, and why, is in ARCHITECTURE.md under
"Diversity is bounded by playability".

**2026-08-22: the clone copies one teacher, and it is `calibrated`.** A mixture
of five was measured against it at equal budget and bought nothing — breadth
minimum 40.7% against 42.4%, and all four sampled strength comparisons the
other way. `scripts/teacherAgreement.py` is the measurement. The same run
turned up a caveat that colours every argmax number on record: against
a deterministic bot, 30 argmax games produce about 5 distinct game lines, so
those margins are far too tight.

**Where to pick up:** everything blocking the rebuild is now decided — panel,
pool, teacher. What is left before committing to it is the 500k rerun of the
teacher comparison, under "Calibration: what is left", and then the rebuild
itself. Items above that section predate the panel freeze and should be read
with that in mind.

**2026-08-23: a full-codebase review, and its cleanup, are done.** Two broken
entry points fixed, 482 lines of unrunnable scripts and four pieces of dead code
deleted, five scripts moved onto one match harness, the league made flag-driven,
and `ARCHITECTURE.md` split into structure plus `RESULTS.md`. What it found and
what survives it are under "Code health" at the bottom. None of it touched what
a bot or a policy plays, so no recorded measurement was invalidated.

**Two conventions changed with it.** Results now go to `RESULTS.md`, not
`ARCHITECTURE.md`, which is meant to stop growing. And a `doc-sync` subagent
(`.claude/agents/`) will tell you which doc passages a change has just made
false — worth running before committing anything that touches `game/`,
`heuristics/` or `env/`.

## The panel moved — decide what that costs

- [ ] **`lookahead2` is no longer the same bot**, and it is the anchor the two
      model generations were compared on. Its leaf is `straggler`, whose
      distance term changed, so `Talos2.0_round5`'s recorded 72.0 ± 6.2 argmax /
      58.0 ± 6.8 sampled were measured against a bot that no longer exists.
      Either re-measure that one pairing and restate the comparison, or state in
      RESULTS.md that generation-1-vs-2 is frozen at the old panel. Doing
      neither leaves two incomparable numbers side by side.
- [ ] **Should `lookahead2` search on `shaped` instead of `straggler`?**
      `shaped` wins 58.2% ± 4.8 head to head at one ply, but costs 14.7 µs a
      candidate against 1.7 µs, and the search evaluates ~b² leaves. A move
      would go from ~35 ms to something like 250 ms, which is past the point
      where it can generate training data at all. Worth one measurement at
      reduced games before deciding.

## Heuristics still worth consolidating

- [ ] **`shaped` computes its shape terms even when they cannot matter.** They
      are multiplied by `unfilledTargetScore`, so they fade to nothing as pieces
      arrive — but all 14 µs of them are still computed when that factor is near
      0. An early exit below a threshold makes the late game almost free at
      identical play. Pure speed, no strength question, and `shaped` is the bot
      most in need of it.

## Calibration: what is left

- [ ] **Buy diversity at the episode level instead.** Follows from "Diversity is
      bounded by playability" in ARCHITECTURE.md: every sound vector in this
      family has to be distance-dominated, so reweighting has a ceiling and the
      pool is close to it. The untried levers are outside the bots —
      `--opponentSampling` (already implemented, and the league's default)
      and varied openings (see the plausible-opening-plies
      idea; uniform random openings were measured once and did not help).
      Neither has been measured *for diversity*, only for strength.
- [ ] **Or add a genuinely new primitive.** The other way past the same bound.
      The five terms are all some flavour of progress-or-shape; nothing scores
      blocking, tempo, or the opponent's position at all. A bigger piece of work
      than a refit, and the teacher measurement it was queued behind is now
      done, so nothing is holding it up but priority.
- [ ] **`calibratedCluster`'s fit found nothing.** The control vector won its
      round outright, which is either a real optimum or too small a search for
      two free weights. One re-run at higher `--candidates` settles it; leave the
      control in place until then.
- [ ] **Decide whether `straggler` and `lookahead2` get calibrated too.** Only
      the `shaped` lineage was converted. `straggler` pays +23% per candidate
      against `shaped`'s +5%, and it is `lookahead2`'s leaf, so the search pays
      it b² times a move. Measure before assuming it is affordable.
- [ ] **Decide what `jumpPotentialScore` is for.** 6.00 µs of `shaped`'s
      16.01 µs, and the fit weighted it down to 0.04 — near off — where the
      ablation on the *uncalibrated* bot said it was worth 4.5 points. Those two
      results are not in conflict, but they do mean nobody has yet measured what
      dropping it costs the calibrated bot. Removing it makes `shaped` a third
      cheaper, which is what stands between it and being `lookahead2`'s leaf or
      `pretrain`'s teacher.

      **2026-08-21 makes this sharper, not softer.** `calibratedJump` is both the
      strongest partial variant (83.0% against `distance`) and by a wide margin
      the most distinctive player in the panel. The term the full fit nearly
      switched off is the one buying both. Either the fit was wrong about it or
      it pays only when it is not competing with `clustering` and
      `stragglerLag` — and those imply different answers about dropping it.

      The `calibratedClusterJump` fit is a third data point and it favours the
      second reading: given only `clustering` to compete with, all three
      finalists put `jumpPotential` at 0.032-0.062. Every fit that has ever had
      an alternative has discarded it; it wins only when it is the only shape
      term on offer. The measurement that would settle it is an ablation on the
      calibrated bot — drop the term, refit the rest, see what it costs.
- [ ] **Rerun the teacher comparison at 500k before the rebuild commits to it.**
      The 2026-08-22 result — one teacher beats a mixture on breadth and on
      strength, written up in RESULTS.md under "A mixed teacher does not
      make a broader clone" — is at equal *cost*, which is the right basis for
      choosing today but not for the question itself: both clones were improving
      at epoch 12 and fitting five teachers is the harder problem, so the
      mixture was the more data-starved arm. 500k is the budget the recorded
      generation-2 clone used and the one the rebuild will use anyway. Cheap to
      decide against — if the minimum still does not move, the question is
      closed for good.
- [ ] **Re-clone and retrain.** This is the whole point of the exercise — Talos2
      gets rebuilt from scratch on the new bots, and the panel it needed has
      been frozen since 2026-08-21. Both blocking decisions are made: the clone
      copies `calibrated` alone, and the opponent pool is the four frozen
      variants. What is left is running it, at the 500k budget the recorded
      generation-2 clone used.

## Now

- [ ] **Name the winner.** `models/Talos2.0_round5` has earned a name on
      everything measurable, with one caveat worth stating plainly: its argmax
      play is statistically *level* with `Talos1.0` and `Talos1.1`
      (72.0 ± 6.2 against 68.3 ± 11.8 and 75.0 ± 11.0), and the unambiguous win
      over generation 1 is in sampled play — 58.0% against their best of 33.3%.
      Against its own anchor it is ahead on everything. Decide whether that is
      a `.0` under the versioning rule; if so, rename the file and record the
      rename in RESULTS.md.
- [ ] **Extend the league — the conditions are now met.** `lookahead2` came
      back +13.5 over the anchor, so the gains are real and not sibling-only,
      and the head-to-head margins were flat-to-rising (54.8, 52.2, 55.4, 56.6)
      rather than plateauing. A round is ~13 minutes.

      **Cap the opponent pool first** — the machinery now exists. The draw is
      uniform (`env/halmaEnv.py`, `reset()`), so round 5 already spent 4/5 of
      its episodes re-beating history and a sixth would make it 5/6.
      `--poolCap 3` keeps the anchor plus the last three rounds and drops the
      middle, and `--basename` names the extended rounds apart from
      `Talos2.0_roundN`, which they must be since the recipe differs:
      ```
      python -m scripts.talos2League --rounds 8 --poolCap 3 \
          --basename extended_round
      ```
- [ ] **Prune `models/`.** Rounds 1-4 are 98 MB and only round 5 and the anchor
      are still referenced. Their numbers are all in RESULTS.md, and
      `models/manifest.md` records which still load. Keep
      round 4 if the league is extended, since it is the only sparring partner
      of the right strength.

## Making the search fast enough to be worth having

Profiled 2026-08-11, at ply 12 of a real game: branching factor **75**, one
`lookahead2` move is 1 + b + b² ≈ **5,700 leaf scorings at 70 ms**, and 55% of
that was `stragglerTravelScore` — a max-over-mins across `nonArrived ×
openEndPositions`, 225 lookups, recomputed 9,710 times per move.

**That hot spot is dealt with** (2026-08-14): it is now a pruned search seeded
with a lower bound, exact and 5.9x faster, which took the same move from 75.4 ms
to 35.3 ms. The leaf count is unchanged, so everything below still holds — the
argument was never about the constant factor.

**Depth 10 is not reachable by making this faster**, and the gap is ~15 orders
of magnitude rather than a constant factor: 75¹⁰ ≈ 5.6×10¹⁸ nodes, which is
~178 years even at 1 ns/node, and alpha-beta with perfect ordering only gets it
to 75⁵ ≈ 2.4×10⁹, still 40 minutes per move at C speed. Vectorising the hot
spot with numpy was tried and gives only 1.9× (2.5× with indices cached) — the
15×15 slice is too small to beat a Python loop. So the leverage is not there.

- [ ] **Decide which search is actually wanted**, because the two answers
      diverge:
      * *A stronger bot / yardstick* → alpha-beta with move ordering and a
        transposition table. `lookahead2` today does **zero** pruning; it
        evaluates all 75 replies for all 75 candidates. Order candidates by the
        already-incremental `plainDistance` and depth 4-5 becomes reachable.
        Halma transposes heavily, so a TT should pay well.
      * *A stronger agent* → policy-guided MCTS/PUCT, the network as prior and
        the critic as leaf value. It reaches depth 10+ along the principal line
        because the policy prunes to a handful of moves per ply. A fixed beam
        of width 4 is the cheap version: 4¹⁰ = 1M leaves.
- [ ] **Either path needs batched position evaluation first**, which is the
      refactor already recorded in `env/searchPlayer.py`'s docstring: candidates
      are scored by mutating the real board and undoing it, so two hypothetical
      positions cannot exist at once. That blocks batching the network (one
      forward pass per node today, where a batch of 64 costs barely more than
      one) and blocks any parallelism. A copyable lightweight position is the
      unlock, and it is worth more than any micro-optimisation of the scorers.

## The parity = 0 replication

The parity penalty was A/B'd once, at stage 2a only, and came back level
everywhere except sampled play against `lookahead2`, where the control was
*much* better (45.0% vs 18.3%, 60 games). That gap is unexplained and the whole
generation-2 lineage now carries `--parity 0.25` on the strength of an A/B that
did not actually favour it.

So: run the same three stages again with `--parity 0` throughout, and compare
the two pipelines end to end rather than at one stage.

- [ ] **Stage 2b, parity 0.** `models/talos2_parity0` already exists and is the
      parity-0 twin of stage 2a. Run the 500k mixed-pool stage from it:
      ```
      python -m scripts.train --steps 500000 --games 200 \
          --init models/talos2_parity0 \
          --opponentPool distance tipDistance shaped straggler random \
          --opponentModelPool models/clone_from_multi_500k models/talos2_parity025 \
          --lr 1e-4 --targetKl 0.02 --parity 0 --seed 0 --name stage6_parity0
      ```
      Note the model pool mirrors the 0.25 arm's: each arm spars against the
      other's stage-2a checkpoint, so neither gets an easier pool.
- [x] **Stage 3, parity 0** is now a one-liner — the flag asked for here
      exists:
      ```
      python -m scripts.talos2League --init models/stage6_parity0 \
          --parity 0 --basename parity0_round
      ```
      Still to *run*; the 0.25 league stays reproducible from the defaults.
- [ ] **Compare the two pipelines**: `openingSweep` and `randomPositionSweep`
      arm against arm, plus `evaluateAgainstBots --bots lookahead2 --games 200`
      on both. The question is whether the shaping term changes where a *full*
      pipeline converges, not just where one 300k stage lands.

## Open questions inherited from generation 1

- [ ] **The sibling-ranking probe on the critic.** Take a position, take the
      policy's two best moves, play both out, check whether the critic's
      ordering matches the outcome. At ~50% no architecture change helps and
      the objective has to change; well above 50% and the capacity work was
      worth it. The critic has had its own branch and a `vf=[256, 256]` head
      since 2026-08-07 and there is still no evidence capacity was the
      constraint. Not built — and it is the same machinery the MCTS route above
      would need, so the two are worth planning together.
- [ ] **A history / repetition signal in the observation.** The compatibility
      cost is already paid — every old checkpoint is unloadable anyway — so
      only the design is open. `env/searchPlayer.py`'s repetition rule removed
      the deadlocks at play time (draws 8 -> 0/2), which says the fix works
      without being learned. New evidence that it still matters: draws appear
      in the league sweeps as the lineage converges — 0 in the round-1 and
      round-2 sweeps, 2 in round 4 vs 3, 12 in round 5 vs 4.
- [ ] **Seat asymmetry.** Still unexplained, and the league sweeps add data
      rather than settling it: round 2 beats the anchor 68.5% moving first
      against 57.2% moving second, where round 1 was 60.0 / 57.0. Invariant 7
      has a history here. Nothing has checked whether it is a seat-2 weakness
      or just the normal first-move advantage.

## Housekeeping

- [ ] **The three `Talos1.x` files, 23.5 MB together, cannot be loaded** by any
      current script — the 2026-08-07 observation and critic changes broke
      them, and only their recorded numbers still matter. They are also the one
      thing here that cannot be regenerated. Kept for now because deleting them
      is irreversible and buys almost nothing; revisit if space ever bites.
- [ ] **Write generation 2's stages 1-2 up in `RESULTS.md`**: the clone,
      the parity A/B, and the stage-2 numbers. Stage 3 and the `lookahead2`
      comparison are written up already, and `4faaf0f`'s message carries the
      observation and critic rationale.

## Code health — worked through 2026-08-23

The 11-step cleanup from the full review is **done**, except for the two items
kept below. All four gates stayed green throughout and no measurement had to be
re-run: nothing here changed what any bot or policy plays.

What it did, in one place so the reasons are not lost:

- **Two broken entry points fixed.** `scripts/playAgainstAgent.py` and the
  `run-halma` driver both defaulted to a `Talos1.x` checkpoint that stopped
  loading on 2026-08-07, so the documented way to play against a policy had
  been crashing on startup for two weeks. Both now name `Talos2.0_round5`.
- **CLAUDE.md's command list re-pointed.** Ten of its commands named dead
  checkpoints. The `pretrain` recipes moved to `--expert calibrated`, and the
  default in `main()` moved with them, which closes the teacher question.
- **The three `progressivePhase` scripts deleted** — 482 lines that all failed
  at their first `--init`. `scripts/talos2League.py` is the living successor.
- **Dead code removed**: `turnBoard120DegreesPermutation`, `gameObservation`,
  `Strategy.ALIASES`, and the logistic calibration branch (`LOGISTIC_FIT` was
  an empty dict — the fit had never once been selected).
- **`scripts/matchStats.py` added**, holding `Result`, `marginOfError`,
  `playMatch` and `playBothSeats`. Seven copies of the Wald interval, two
  identical `Result` dataclasses and five `playMatch` variants collapsed into
  it; nine scripts now import from it, five of them for the match harness and
  the rest for `marginOfError` alone. `tests/test_match_stats.py` pins the seat-swap contract every recorded
  win rate depends on — the control is a self-match, which must come out at
  exactly 50.0%.
- **`talos2League.py` is flag-driven.** `--parity`, `--init`, `--basename`,
  `--rounds` and `--poolCap` replace edit-in-place constants; the defaults
  still reproduce the recorded league, and `tests/test_league.py` pins that.
- **`ARCHITECTURE.md` split.** 2,068 lines became 1,272 of structure plus
  `RESULTS.md`, the lab notebook, which carries a "How to read the numbers
  here" preamble — including the caveat that argmax margins recorded before
  2026-08-22 are far too tight.
- **`models/manifest.md` added**, recording which checkpoints load and why the
  generation-1 ones cannot. This is the thing that would have made the
  2026-08-07 breakage visible on the day it happened.

### What is left from that review

- [x] **General docstring thinning: examined and dropped as not worth doing.**
      The volume is proportionate — the longest docstrings run about 1:1 with
      their function (`_potential` 54 lines on 63, `_buildGeometryPlanes` 53 on
      67), not 5:1 — and the content is mostly negative results that cost hours
      of compute to rediscover. `_potential` alone records the sign error that
      taught three training runs to stall. With measurements costing 29 minutes
      to five hours and sessions cleared often, that prose is cheap insurance.
      The 0.55 prose-per-code figure is also a poor metric: `game/boardTypes.py`
      scores 6.25 because its "code" is type aliases whose whole purpose is
      documentation. Revisit only if a second contributor joins, or if `game/`
      and `env/` start changing quickly again — both make stale prose expensive
      in a way it currently is not.
- [x] **Measured figures now have one home**, which was the real defect behind
      that item: 51 numbers appeared in more than one file, and `shaped`'s
      strength alone lived in four. Pulled out of `game/board.py`,
      `game/gameManager.py`, `heuristics/strategy.py`,
      `scripts/{fitWeights,pretrain,playAgainstAgent,talos2League}.py` — 87
      figures in code down to 61. **The standing rule:** a docstring states the
      claim qualitatively ("the strongest one-ply bot"), and the digits live in
      `ARCHITECTURE.md` or `RESULTS.md`. A qualitative claim survives a
      re-measurement; a number does not, and the panel has already changed
      twice. What deliberately stayed: the `STRAGGLER_WEIGHT` and
      `SHAPE_WEIGHT` sweeps, because they sit at the constant they warn about
      and the numbers *are* the warning; confidence levels and cost figures,
      which are properties of the code rather than of a bot; and the evidence
      for the potential's sign, `_needsFlip`, and the deadlock rates.

- [ ] **Back `Talos2.0_round5` up somewhere off this disk.** It is the only
      strong checkpoint, it exists in one copy, it is not in git, and the
      recipe takes about five hours to re-run. `models/manifest.md` states the
      risk; nothing yet addresses it.

## Traps worth remembering

- **`--games N` in `scripts/evaluateAgainstBots.py` is N per bot *per mode***,
  and both argmax and sampled always run, so `--games 200` plays 400 games —
  ~29 minutes per checkpoint against `lookahead2` at the measured 4.3 s/game.
- **Redirect long runs with `python -u`.** Without it stdout is block-buffered
  and a 45-minute evaluation shows nothing at all until it exits, which is
  indistinguishable from a hang. One run was killed on that basis.
- **The 20-game inline reports in a training round mean very little.** Round 1
  reported 90-95% against the anchor inline; the 800-opening sweep put it at
  58.5%. The bots in those reports are saturated besides.
- **`*.log` is gitignored now**, so detached runs can write to the repo root.
