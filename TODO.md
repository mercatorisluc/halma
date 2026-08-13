# TODO

The work queue, kept here rather than in a chat session so that clearing a
session costs nothing. `ARCHITECTURE.md` stays the record of what was
*measured* and why; this file is only what is *pending*. When an item finishes,
delete it here and write the result there.

Convention: `[ ]` open, `[x]` done but not yet written up, `[~]` running.

State as of 2026-08-11: the generation-2 league is finished and pushed
(`6acf060`). `models/Talos2.0_round5` is the strongest checkpoint that exists —
70.5% over its anchor across 800 openings, 64.5% off the beaten track, and
72.0 ± 6.2 argmax / 58.0 ± 6.8 sampled against `lookahead2`. All of it is
written up in ARCHITECTURE.md.

**2026-08-14: the heuristic panel was rebuilt and is stronger.** `distance` lost
its static distance half, `shaped` was re-weighted and is now the best one-ply
bot, `stragglerTravelScore` became a pruned search at 5.9x, and every bot and
primitive was renamed. All measured and written up in ARCHITECTURE.md; what it
leaves open is the section below.

## The panel moved — decide what that costs

- [ ] **`lookahead2` is no longer the same bot**, and it is the anchor the two
      model generations were compared on. Its leaf is `straggler`, whose
      distance term changed, so `Talos2.0_round5`'s recorded 72.0 ± 6.2 argmax /
      58.0 ± 6.8 sampled were measured against a bot that no longer exists.
      Either re-measure that one pairing and restate the comparison, or state in
      ARCHITECTURE.md that generation-1-vs-2 is frozen at the old panel. Doing
      neither leaves two incomparable numbers side by side.
- [ ] **Reconsider what `scripts/pretrain.py` clones from.** It clones
      `distance`, and two bots in the panel now beat it — `shaped` 92.0% and
      `straggler` ~80%. The clone is the root of every Talos lineage, so the
      teacher is not a detail. `shaped` is 45x the cost per candidate, which is
      the reason to measure rather than assume: sample generation is already the
      slow part of pretraining.
- [ ] **Should `lookahead2` search on `shaped` instead of `straggler`?**
      `shaped` wins 58.2% ± 4.8 head to head at one ply, but costs 14.7 µs a
      candidate against 1.7 µs, and the search evaluates ~b² leaves. A move
      would go from ~35 ms to something like 250 ms, which is past the point
      where it can generate training data at all. Worth one measurement at
      reduced games before deciding.

## Heuristics still worth consolidating

- [ ] **Two straggler terms, never measured against each other.**
      `stragglerTravelScore` (absolute remaining travel, in `straggler`) and
      `stragglerLagScore` (the same piece's lag behind the pack's mean, in
      `shaped`) describe the same phenomenon differently. If one subsumes the
      other, a term disappears from a scorer. `shaped` is the expensive bot, so
      the interesting direction is whether it can drop `stragglerLagScore`.
- [ ] **`shaped` computes its shape terms even when they cannot matter.** They
      are multiplied by `unfilledTargetScore`, so they fade to nothing as pieces
      arrive — but all 14 µs of them are still computed when that factor is near
      0. An early exit below a threshold makes the late game almost free at
      identical play. Pure speed, no strength question, and `shaped` is the bot
      most in need of it.
- [ ] **Remove `Strategy.ALIASES`** once the recorded recipes have been moved to
      the new bot names. It exists so that commands written before 2026-08-14
      keep running; it is not meant to be permanent.

## Now

- [ ] **Name the winner.** `models/Talos2.0_round5` has earned a name on
      everything measurable, with one caveat worth stating plainly: its argmax
      play is statistically *level* with `Talos1.0` and `Talos1.1`
      (72.0 ± 6.2 against 68.3 ± 11.8 and 75.0 ± 11.0), and the unambiguous win
      over generation 1 is in sampled play — 58.0% against their best of 33.3%.
      Against its own anchor it is ahead on everything. Decide whether that is
      a `.0` under the versioning rule; if so, rename the file and record the
      rename in ARCHITECTURE.md.
- [ ] **Extend the league — the conditions are now met.** `lookahead2` came
      back +13.5 over the anchor, so the gains are real and not sibling-only,
      and the head-to-head margins were flat-to-rising (54.8, 52.2, 55.4, 56.6)
      rather than plateauing. A round is ~13 minutes.

      **Cap the opponent pool first.** The draw is uniform
      (`env/halmaEnv.py`, `reset()`), so round 5 already spent 4/5 of its
      episodes re-beating history and a sixth round would make it 5/6. Keep the
      anchor — it earns a permanent slot as a fixed reference — plus the last
      three rounds, and drop the middle. That is a small change to the pool
      construction in `main()`, and it changes the recipe, so the extended
      rounds should be named apart from `Talos2.0_roundN` rather than
      continuing the numbering.
- [ ] **Prune `models/`.** Rounds 1-4 are 98 MB and only round 5 and the anchor
      are still referenced. Their numbers are all in ARCHITECTURE.md. Keep
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
- [ ] **Stage 3, parity 0.** `talos2League.py` with `INIT` and `PARITY`
      switched. Do not edit the file in place for this — copy the constants or
      add a flag, so that the 0.25 league stays reproducible from what is on
      disk.
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
- [ ] **Write generation 2's stages 1-2 up in `ARCHITECTURE.md`**: the clone,
      the parity A/B, and the stage-2 numbers. Stage 3 and the `lookahead2`
      comparison are written up already, and `4faaf0f`'s message carries the
      observation and critic rationale.

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
