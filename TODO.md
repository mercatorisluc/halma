# TODO

The work queue, kept here rather than in a chat session so that clearing a
session costs nothing. `ARCHITECTURE.md` stays the record of what was
*measured* and why; this file is only what is *pending*. When an item finishes,
delete it here and write the result there.

Convention: `[ ]` open, `[x]` done but not yet written up, `[~]` running.

## Now

- [x] **`lookahead2`, 200 games — done, and the league holds up.** Round 5 at
      72.0 +/- 6.2 argmax and 58.0 +/- 6.8 sampled against the anchor's
      58.5/44.0. Written up in ARCHITECTURE.md.
- [ ] **Name the winner.** `models/Talos2.0_round5` has earned a name on
      everything measurable: strongest of the five rounds, +70.5% over its
      anchor across 800 openings, 64.5% off the beaten track, and +13.5 points
      on the only bot that bridges the generations. The one caveat is that its
      argmax play is statistically *level* with `Talos1.0` and `Talos1.1`
      rather than ahead — the win over generation 1 is in sampled play. Decide
      whether that is a `.0` under the versioning rule; if so, rename the file
      and record the rename in ARCHITECTURE.md.
- [ ] **Extend the league, if and only if `lookahead2` holds.** It had not
      plateaued, and a round is ~13 minutes. Two conditions before spending the
      hour: `lookahead2` at or above the anchor, and a **capped opponent pool**
      — the draw is uniform, so round 5 already spent 4/5 of its episodes
      re-beating history. Keep the anchor plus the last three rounds.

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
          --opponentPool advancedDistScore simpleDistScore sparsityScore bottleneck random \
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
      constraint. Not built.
- [ ] **A history / repetition signal in the observation.** The compatibility
      cost is already paid — every old checkpoint is unloadable anyway — so
      only the design is open. `env/searchPlayer.py`'s repetition rule removed
      the deadlocks at play time (draws 8 -> 0/2), which says the fix works
      without being learned.
- [ ] **Seat asymmetry.** `talos2_stage6` beats its anchor 68.5% moving first
      and 59.8% moving second over the same 800 openings. Invariant 7 has a
      history here. Nothing has checked whether it is a seat-2 weakness or just
      the normal first-move advantage.
- [x] **`env/searchPlayer.py` has tests** — `tests/test_search_player.py`, 12 of
      them. Each was checked by mutation: breaking the shaping correction, the
      repetition rule, the reply-ply minimum, the `attachTo` reset, the
      all-repeat fallback, or making `_rank` return endpoints instead of the
      paths it was given each fails exactly one test. Two of the first drafts
      passed against a broken source and were rewritten — the opening offers no
      multi-field jump path, so a path and its endpoints are indistinguishable
      there, and the untrained critic scores siblings so evenly that "declined
      the repeat" could not be told from "preferred another move" without
      stubbing the value.

## Housekeeping

- [x] **`scripts/train.py` rejects a `--name` carrying a separator** rather
      than silently writing `models/models/foo.zip`. `--checkpointBasename` is
      joined onto `MODELS` the same way and is checked too. Rejecting rather
      than dropping the prefix, so every existing invocation keeps working; the
      check runs at parse time, not 300k steps later at the save.
- [ ] **The three `Talos1.x` files, 23.5 MB together, cannot be loaded** by any
      current script — the 2026-08-07 observation and critic changes broke
      them, and only their recorded numbers still matter. They are also the one
      thing here that cannot be regenerated. Kept for now because deleting them
      is irreversible and buys almost nothing; revisit if space ever bites.
      Everything else was pruned on 2026-08-09: eleven intermediate checkpoints,
      370 MB -> 111 MB.
- [ ] **Write generation 2 up in `ARCHITECTURE.md`**: the three-stage recipe,
      the parity A/B, the stage-2 numbers, and the `lookahead2` gap. Stage 3
      (the league) is written up already; what is still missing is stages 1-2
      and the naming, which waits on the `lookahead2` measurement.
