# Results

The lab notebook: what was measured here, and what it means. `ARCHITECTURE.md`
is the companion and holds the *structure* — the layer diagram, the board's
geometry, the two move representations, the invariants. This file holds the
evidence, and it only grows.

Kept **roughly chronological, oldest first, and grouped by thread** rather than
strictly by date — the entries build on one another and several refer back by
name ("the capacity candidate raised under … below"), so a strict re-sort would
break those references. The rebuild thread is the one place this shows: the
2026-08-22 teacher comparison sits with the clone entry it belongs to, ahead of
the older league entries. Use the index, which is in date order, to jump.

---

## How to read the numbers here

**Argmax margins recorded before 2026-08-22 are far too tight.** Against a
deterministic opponent the opening is fixed and both sides answer identically,
so a seed varies only the play order: 30 argmax games are about **5 distinct
game lines**, not 30 independent samples. Every `±` computed as
`1.96·sqrt(p(1−p)/n)` on an argmax number therefore understates the real
uncertainty, in some cases badly. This was found on 2026-08-22 and it colours
every argmax figure below. Sampled numbers are unaffected.

**A win rate against the heuristic panel stopped meaning anything after
`Talos1.0`.** Every checkpoint since scores 98–100% argmax on the scoring bots.
Where a number needs to separate two checkpoints, it comes from the
800-opening census (`scripts/openingSweep.py`) or from random positions
(`scripts/randomPositionSweep.py`), not from the panel.

**The bot panel changed on 2026-08-14 and again on 2026-08-21.** `distance`
lost its static half, `shaped` was re-weighted, `stragglerTravelScore` became a
pruned search, every bot was renamed, and the pool was then frozen at the four
calibrated variants. A number measured against a bot name before those dates
was measured against a different bot.

**`lookahead2` is not exempt.** Its leaf is `straggler`, whose distance term
changed on 2026-08-14, so figures against `lookahead2` from before then — the
generation-1-versus-2 comparison among them — are against a bot that no longer
exists.

---

## Index

| Date | Entry |
|---|---|
| 2026-08-04 | [Where things stood then](#where-things-stood-then) — frozen; the *current* summary is in ARCHITECTURE.md |
| 2026-08-05 | [How the Talos checkpoints were actually built](#how-the-talos-checkpoints-were-actually-built-and-what-is-still-on-disk) |
| 2026-08-06 | [`lookahead2` is the one column that still moves](#lookahead2-is-the-one-column-that-still-moves-and-talos12-fell-on-it) |
| 2026-08-07 | [Search on top of the critic makes the policy weaker](#search-on-top-of-the-critic-makes-the-policy-weaker-not-stronger) |
| 2026-08-07 | [Next](#next) — where the lineage stopped and why |
| 2026-08-09 | [Step 1 of the rebuild: the clone](#step-1-of-the-rebuild-the-clone-and-what-33-the-data-bought) |
| 2026-08-22 | [A mixed teacher does not make a broader clone](#a-mixed-teacher-does-not-make-a-broader-clone) |
| 2026-08-10 | [Stage 3, the generation-2 league](#stage-3-the-generation-2-league-five-rounds-705-over-its-own-anchor) |
| 2026-08-11 | [The league is real: +13.5 on `lookahead2`](#the-league-is-real-135-on-lookahead2-measured-at-200-games) |

---

## Where things stood then

**This section is a snapshot, not a status.** It was the "Current state" of
`ARCHITECTURE.md` when the clone was first fine-tuned, and it stopped being
updated long before `Talos1.0` was named. It is kept because the narrative
below builds on it. For where the project actually stands, see
`ARCHITECTURE.md`'s "Where things stand".

| Area | State (as of 2026-08-04) |
|---|---|
| `game/` | Refactored, documented, characterization-tested |
| `heuristics/` | Working; five bots, strength measured against each other |
| `visual/` | Working; refactored into focused modules. No tests |
| `env/` | Satisfies the Gymnasium API, masked and shaped; trained policies playable via `NeuralComputer` |
| Agent | Cloned from a bot, then fine-tuned by PPO to 99% against `distance` |

`env/` passes `gymnasium.utils.env_checker.check_env` and has action masking, a
canonical observation and reproducible seeding.

`scripts/baseline.py` plays every pairing of the bots and is the yardstick a
trained agent has to clear. The number that shaped the plan: **a random agent
wins none of 700 games**, so an untrained policy sees the same reward in
essentially every episode. That is why shaping came before training.

`scripts/evaluateAgainstBots.py` is the other half of that yardstick: a
checkpoint against the bots, argmax and sampled, for as many checkpoints as
asked for. `baseline.py` covers bot against bot and `compareCheckpoints.py`
checkpoint against checkpoint; this third side existed only as the tail of a
training run, so re-measuring an existing checkpoint meant training something
to get the report. `lookahead2` is ~20x slower per game than the scoring bots,
which is what `--bots` is for.

`scripts/randomPositionSweep.py` measures the one thing none of those do:
whether a checkpoint plays well from positions it has no reason to be familiar
with. Every other yardstick here starts a game at the opening or two plies into
it — the position a policy has seen most. This one deals each game a start of
4–20 uniformly random legal plies, drawn from its own `--seed`ed generator so
the same deal is handed to any pair of checkpoints. Uniformly random rather
than sampled from either contestant, deliberately: a position distribution one
side generated would favour that side, which is the one thing a benchmark must
not do. The cost is that some deals are positions no sensible game reaches —
that is the breadth being measured, but it does mean the score answers "how
well does it cope off its own track", not "how strong is it", and
`openingSweep.py` stays the answer to the second.

**Each deal is played twice with the checkpoints swapping seats**, because a
random position is not fair — it can hand one seat a won game before either
policy moves — and neither is the side to move. The mirror cancels both
exactly. The control that pins this: the same checkpoint on both sides scores
50.0% ± 0.0 with every pair split, since two identical policies make the two
games of a pair literally the same game. Unlike the opening sweep's census of
all 800 two-ply openings, this is a *sample* of a much larger space, so the
score does carry a confidence interval — and the unit for it is the pair, not
the game, since the two games of a pair share a deal and are not independent.

**Shaping was not enough on its own.** With the reward shaped, the geometry
given to a CNN and the action head factored, PPO from scratch still ended a
quarter hour of training at 0 wins and single-digit percent of pieces home. The
signal is there, but the region of policy space where Halma is played is not
somewhere random exploration arrives.

**What worked was copying a bot first.** `scripts/pretrain.py` plays games with
`straggler` on the agent's seat, records its choice in every position, and fits
the policy to those choices by cross-entropy over the masked distribution. On
150k positions and 12 epochs the policy agrees with the bot on 73% of positions
and goes from 0% wins to 86% against `distance` (argmax, 50 games).
Scored over the same games, the teacher itself takes 68% — but it wins 64% against
`shaped` where the clone takes 54%, so the clone is at roughly teacher
strength and specialised to the opponent its data was collected against, not
generally stronger. `scripts/train.py --init` fine-tunes from that checkpoint.

Mixing the bot into PPO's *own* rollouts — playing the bot's move some fraction
of the time — is the obvious-looking alternative and does not work:
`collect_rollouts` stores the log-probability of the action the policy sampled,
and the update forms `exp(log_prob - old_log_prob)` against it, so a substituted
action leaves the ratio comparing the wrong pair of distributions and nothing
raises. Mixing during *collection* is a real method (DAgger) but its labels come
from the expert and its loss is the supervised one; `pretrain.py --mix --rounds`
implements that form.

**PPO on top of the clone clears its teacher.** 300k steps from `models/cloned`
take it from 82% to **99% against `distance`** (198W 2L of 200 games,
argmax; 97.5% sampled), with the fine-tuned agent also finishing games in 50
steps against the clone's 60. The margins do not overlap, so this is the answer
to the question the shaping and speed work was in service of: reinforcement
learning does add something here, once it starts somewhere it can learn from.

Two things that measurement also settled. The entropy coefficient, set to 0.01
so a from-noise policy would not collapse, turned out not to matter on a cloned
one: 0.001 and 0 land at 99.0% and 99.5%, indistinguishable. What does matter is
step size. At sb3's default learning rate `approx_kl` ran 0.06-0.37 against a
healthy ~0.01 and `clip_fraction` 0.2-0.37, and both runs were thrown back
repeatedly along the way — one from 90% to 40% and back within 50k steps. A
policy that starts sharp needs smaller steps than one that starts diffuse;
`--lr` and `--targetKl` exist for that and the default is still the from-noise
one.

The agent is also **specialised to the opponent it trained against**: 99%
against `distance` but 90-92% against `random`, where the clone was at
96%. Beating one bot decisively is not the same as playing Halma well, so the
measurement worth having was against opponents it never saw.

Measured, 30 games against each of four bots (argmax), mean win rate last:

| checkpoint | plainDistance | straggler | shaped | random | mean |
|---|---|---|---|---|---|
| `cloned` | 80.0 | 46.7 | 60.0 | **96.7** | 70.8 |
| `tunedEnt000` | **100.0** | 90.0 | 76.7 | 86.7 | **88.3** |
| `tunedEnt001` | 96.7 | 90.0 | 76.7 | 90.0 | **88.3** |

At ±11 to ±18 on each cell, only the gaps between the clone and the tuned pair
mean anything. Those say the specialisation is **narrower than it looked**: the
clone is ahead only against `random`, and fine-tuning nearly doubles the score
against `straggler` — the strongest one-ply bot, which neither checkpoint ever
trained on. So PPO on top of the clone did not merely sharpen it against
`distance`; what it lost is ground against the weakest opponent, where
the shortest path to a win is least like anything a bot would play. The two
tuned checkpoints are indistinguishable, which is the entropy finding again.

The comparison goes through a common panel of bots because that is what answers
the question. Each cell is an absolute number against a fixed opponent — and for
three of the four, one no checkpoint ever trained on — so a column is a yardstick
rather than a relative ordering, and all three checkpoints are measurable on it
at once. A head-to-head result says only which of two policies is ahead, and says
it against an opponent that moves as training does.

75k further steps on `straggler` — a heuristic the agent had only ever seen
baked into `cloned`'s imitation data, never as a PPO opponent — pushed
`tunedOnBottleneck` past `tunedEnt000` on every one of those four bots except
`lookahead2` (46% argmax), the one bot none of this lineage has ever trained
against: 100% on `distance` and `shaped`, 96% on `straggler`
itself, 90% on `random`.

Head-to-head is also available now (`scripts/compareCheckpoints.py`, using
`selfSeat`), and building it is what found invariant 7 above: the first
round-robin looked like a bug report rather than a result — seat 1 won every
matchup and seat 2 never won once, regardless of which checkpoint sat where.
The `_needsFlip` fix resolved it; a rerun stopped being seat-biased (draws
went from many, seat 2 repeatedly timing out at the move cap, to zero) and
produced a real ranking. Grown since to eleven checkpoints, every pairing, 30
games at seed 0:

| checkpoint | record |
|---|---|
| `multiVsModel` | 8-2 |
| `tunedEnt000` | 7-3 |
| `pooledFinetuned` | 7-3 |
| `pooledWithLookahead` | 7-3 |
| `tunedEnt001` | 6-4 |
| `multiTeacherTuned` | 5-5 |
| `cloned` | 5-5 |
| `tunedOnBottleneck` | 4-6 |
| `multiTuned` | 4-6 |
| `clone_from_multi` | 2-8 |
| `maskedPPO` | 0-10 |

Single-seed argmax, so treat this as directional rather than a tight
estimate — most matchups were 100/0 or close, a few near-even at 56.7%. Those
near-even numbers have since been explained, and they were not close matches:
under argmax the opening is fixed and both policies are deterministic, so the
seed varies only the play order and a pairing has exactly **two** possible
games, whichever `--games` says. 56.7% is 17/30, the share of those seeds that
let seat 1 start — i.e. the first mover won every game and the two policies
were indistinguishable. The non-transitivity below is the same artefact: about
one bit per matchup, reported with an interval that assumed `--games`
independent samples. `compareCheckpoints.py` now plays every pairing in both
seat directions, and in argmax mode reports the winner per play order instead
of a percentage; `--sampled` is what yields a real win rate (the same 20 seeds
give 17 distinct games). The ranking above predates that and is kept as it was
measured.

The non-transitivity (`pooledWithLookahead` beats `tunedEnt000`, `tunedEnt000`
beats `tunedOnBottleneck`, `tunedOnBottleneck` beats `pooledWithLookahead`) is
therefore mostly measurement noise here, though genuine non-transitivity is
also expected for adversarial policies trained by different processes. `maskedPPO` and `maskedPPO_300k`
predate later architecture changes; the latter now loads but its observation
space no longer matches (`Box(246,)` against the current `Dict`) and is excluded,
the former loads and plays but is the oldest checkpoint here and finished last.
`pooledFinetuned` is simply `pooledWithLookahead` given 100k further PPO steps
against its original pool minus `lookahead2` (dropped for training-loop speed,
~6x slower per step than the other four heuristics) — nothing structurally
new, and it took the top spot outright.

**A multi-teacher clone did not repeat the pooled-opponent generalisation
story.** `scripts/pretrain.py --expert` now takes a list of bots instead of
one; `collect()` draws one teacher per game from a seeded `rng`, so the
recorded moves — and the fitted policy — are a blend rather than one bot's
blind spots. `clone_from_multi` (`distance` + `shaped` +
`straggler` as teachers) reached 68.7% agreement with its blended targets
after 12 epochs, against ~73% for the single-teacher `cloned` — a harder
target, as expected. Fine-tuning it with PPO's default learning rate
reproduced the sharp-policy instability noted above for `cloned`, except this
time it didn't recover: `approx_kl` ran 0.09-0.23 for the full 100k steps and
argmax win rate against bots in the training pool *fell* (`shaped`
54%→22%), with sampled win rates on trained-on bots collapsing to 1%.
Restarting from `clone_from_multi` with `--lr 1e-4 --targetKl 0.03` (200k
steps, four-heuristic pool) fixed it — `approx_kl` held near 0.02-0.03 — and
produced `multiTeacherTuned`, which clears its training panel decisively
(99-100% argmax on three of four bots). But head-to-head against the
existing roster it lands mid-table at 5-4, clearly behind every checkpoint
descended from a single-bot or single-pool clone. The rest of the
multi-teacher lineage does worse still (`clone_from_multi` 1-8, the collapsed
`multiTuned` 4-5). So crushing a fixed heuristic panel and holding up against
other trained policies turned out to be different things here — diversifying
the imitation-learning *teacher* did not buy what diversifying the PPO
*opponent* pool did.

**`HalmaEnv` can now seat a frozen checkpoint as the training opponent**
(`opponentModel`, `scripts/train.py --opponentModel PATH` — see the `env/`
section above), not just a heuristic. That is a fixed sparring partner, not
self-play: the opponent's weights never move during the run. True self-play —
the opponent kept in step with training rather than frozen, the route that
does not inherit a teacher's ceiling — is still unbuilt, and search is a
separate, further-out direction.

**And a fixed model opponent beat both heuristic approaches.** `multiVsModel`
is `clone_from_multi` fine-tuned for 200k steps (`--lr 1e-4 --targetKl 0.03`
from the start this time) against a frozen `pooledFinetuned` -- then the best
checkpoint in the roster -- instead of any heuristic. It went 25%→80% sampled
against that opponent over the run, ended **100-0 argmax against
`pooledFinetuned` itself** in the post-run report, and topped the eleven-way
round robin outright at 8-2, ahead of every heuristic-only lineage including
the one it started from (`clone_from_multi` alone was 2-8). It still
generalises to bots it never trained on -- 91% argmax vs `distance`,
93% vs `random`, though only 25% vs `lookahead2`, which nothing in this
project has ever trained against. Its one clear head-to-head loss is to
`tunedEnt000`. One run is not enough to call this the better method in
general rather than a strong-opponent-plus-lower-lr combination that happened
to work once, but it is the first thing in this whole investigation to beat
`pooledFinetuned` decisively, and the mechanism (fight something that is
already good, rather than a fixed heuristic panel or a blend of teachers) is
the closest analogue to self-play buildable without also updating the
opponent's weights. One thing agreed for later is making
position evaluation parallelisable, which today's `moveApplied` prevents.

**A checkpoint-only league produced `Talos1.1`, and `--targetKl` turned
out to be load-bearing rather than optional.** `scripts/progressivePhase1.py`
runs six rounds of 50k/75k/100k/125k/150k/175k steps, each initialised from the
previous round's checkpoint and trained against the accumulated pool of
`Talos1.0` plus every earlier round — no heuristic anywhere in the draw
(`--noHeuristicOpponents`). The first attempt omitted `--targetKl`, and round 2
collapsed: argmax against `distance` fell 97%→46%, `shaped` to
32%, and it lost 14% of games to `random`, with `approx_kl` running 0.044–0.064
against the ~0.01 that is healthy here. Rerunning that same round with
`--targetKl 0.02` and nothing else changed restored it to 100%. A 2x2 over
{entropy 0.01, 0.03} x {no cap, cap} put the cause beyond doubt: both capped
cells score ~100% across the panel, and entropy 0.03 — the value blamed first —
is fine once updates are capped. The entropy runaway (`entropy_loss` -1.97 to
-2.94) was a symptom of the oversized updates, not the driver. A control that
added a `straggler` anchor to the training pool without the cap recovered only
partially (73%/60%/100%), so pure checkpoint self-play was never the problem
either. `multiVsModel` above had already used `--targetKl 0.03` from the start;
the progressive script simply failed to inherit that.

The heuristic panel cannot measure this lineage any more — every round scores
99–100% argmax on all three bots, as does `Talos1.0` itself, so the numbers say
only that nothing broke. Head-to-head is the only usable yardstick here.
Sampled, both seat directions, 60 games per pairing: `Talos1.1` (round 6,
kept under that name; the intermediate rounds were discarded)
beats `Talos1.0` **65% ± 12.1**, round 1 83%, round 3 95%. But rounds 1 and 3
are statistically level with `Talos1.0` (53.3% and 58.3%, both intervals
spanning 50%), so the first ~225k steps bought nothing measurable and the gain
came from the longer late rounds — worth remembering when picking the next
schedule. Part of round 6's margin is also league specialisation: it beats
round 3 (a pool member) 95% but `Talos1.0` only 65%, while those two are level
with each other, so 65% is the honest figure for general strength.

**Forcing the opening is a better way to compare two checkpoints than either
sampling or plain argmax** (`scripts/openingSweep.py`). Each side has exactly
20 legal opening moves, and
the count does not depend on what the other played, so fixing both gives 400
games per play order and 800 in total — all still played out deterministically,
so the variety costs no sampling noise. Over those 800, `Talos1.1` beats
`Talos1.0` **70.5%** (27.5% lost, 2.0% drawn), which corroborates the 65% above
from a completely different direction. It is a census rather than a sample:
those are *all* the two-ply openings, so no confidence interval applies, and
the open question is whether forced openings represent free play rather than
anything statistical. It also sizes the first-mover advantage properly at about
**5 points** (`Talos1.1` 73.2% when starting against 67.8% when not) — the two
games plain argmax produces make it look decisive, which it is not.

**Random openings did not produce a stronger checkpoint, and the run is kept
only as this paragraph.** 300k steps from `Talos1.1` against a frozen
`Talos1.0`, six random opening plies (three per side), `--lr 1e-4 --targetKl
0.02 --entropy 0.03 --seed 42`. The reasoning was sound and the result was not.
Against the opponent it trained on it improved a lot — 800-opening sweep
**85.1%** against `Talos1.0`, where `Talos1.1` scores 70.5% — but against
`Talos1.1` itself it finished level: 49.9% to 45.8%, 4.4% drawn. So the 300k
steps bought specialisation against one frozen sparring partner rather than
strength, the same trap `--opponentPool` exists for, and randomising the
opening did not prevent it: it varies the *positions*, not the opponent.

The heuristic panel says the same thing from the other side, and adds a cost
the sweep cannot see. Argmax is unchanged and at the ceiling — 98–100% across
the five fast bots, identical to `Talos1.1`. *Sampled* is worse on every
non-trivial bot: `shaped` **42.0% ± 9.7** against `Talos1.1`'s 69.0% ±
9.1, `straggler` 61% against 75%, `distance` 76% against 85%. The
`shaped` gap is far outside both intervals, so the policy's best move is
as good as before while its *distribution* got measurably worse — probability
mass spread onto moves that are poor from the standard opening, which is the
plausible cost of training away from that opening at entropy 0.03.

Two things follow. The checkpoint was discarded rather than versioned, so
`models/` still holds the Talos pair alone. And `randomOpeningPlies` was kept:
the negative result is about this pairing and this budget, not about the
mechanism, and the environment cannot answer the question a second time if the
knob is removed with the checkpoint.

The intermediate stands are worth recording, because they do not move in one
direction: sweeping each against `Talos1.1` gives 53.0% at 100k, 46.9% at 200k,
49.9% at 300k. Three points, no trend, all near level — nothing in the run
suggests a longer budget would have arrived somewhere better. The 20-game
progress column swung 25–55% across the same run while the true strength barely
moved, which is worth remembering before reading anything into that column.

**The baseline off the beaten track, measured 2026-08-05 before any of the
breadth work:** over 200 mirrored deals of 4–20 random plies (400 games,
seed 0), `Talos1.1` scores **60.4% ± 3.5** against `Talos1.0`. The edge is real
and it is roughly ten points smaller than the same pairing's 70.5% on the
opening sweep — so part of what six rounds of league play bought was strength
on the track the league was played on. Depth of the deal does not visibly
change it (`Talos1.0` scores 38.1% ± 5.0 on the shallow half and 41.5% ± 5.0 on
the deep half, intervals overlapping), which says these two are not separated by
how far off-track a position is, only by whether it is off-track at all.

Two things about that measurement cap what it can ever show. **144 of the 200
pairs were split** — each checkpoint won the side the deal put ahead — so most
random positions are decided before either policy moves, and only the remaining
quarter is where a strength difference can register at all. And **51 of the 400
games were drawn, 12.8% against the opening sweep's 2%**: deterministic
deadlock is six times more common from a random position than from the opening,
which is a breadth weakness in its own right rather than a measurement
artefact. Both argue for reading this score as a coarse instrument — a
ten-point move means something, a two-point move does not.

**`opponentSampling` measured neutral over three seeds** (Phase A,
2026-08-05). Two arms of three seeds, 300k each from `Talos1.0` against a
frozen `Talos1.0`, identical but for the knob, every checkpoint scored against
the same reference `Talos1.1`:

| seed | control breadth | sampled breadth | control opening | sampled opening |
|---|---|---|---|---|
| 42 | 40.8% | 51.2% | 33.9% | 48.8% |
| 43 | 48.1% | 48.5% | 43.4% | 39.6% |
| 44 | 51.0% | 47.4% | 41.9% | 40.9% |
| mean | 46.6% | 49.0% | 39.7% | 43.1% |

+2.4 points of breadth, which no rank test can separate — with three seeds a
side, complete separation is the only significant outcome available (p = 1/20)
and seed 43 breaks it. **The finding that outlasts the null result is the
spread**: the control arm alone ranges over ten points on both instruments,
which is as large as the gap seed 42 appeared to show. Single-run conclusions
in this project are not reliable, and the three intermediate stands of the
randomOpening run above said the same thing before anyone was listening.

**`Talos1.2` came out of five progressive rounds of 300k from `Talos1.0`**
(`scripts/progressivePhase2.py`, 2026-08-05), with `--opponentSampling 0.5`
kept in despite the null result — a deliberate bet, recorded as one in that
script's docstring, on the argument that Phase A tested the knob in the setting
where it has least to offer (one sparring partner, one round). Measured against
`Talos1.1` after every round:

| round | steps | breadth | opening |
|---|---|---|---|
| — | 0 (`Talos1.0`) | 39.6% | 29.5% |
| 1 | 300k | 47.1% | 39.5% |
| 2 | 600k | 54.5% | 51.0% |
| 3 | 900k | 63.7% | 68.2% |
| 4 | 1.2M | 63.1% | 70.6% |
| 5 | 1.5M | **64.9%** | **69.4%** |

Final: `Talos1.2` beats `Talos1.1` **69.4%** over the 800-opening census and
**64.9% ± 3.7** on random positions; against the anchor `Talos1.0` it is 82.9%
and 69.2%. The generation step is the same size as `Talos1.1`'s over
`Talos1.0` (70.5%), which is why it is 1.2 and not 2.0.

**The heuristic panel finally says something again, and it is the sampled
column.** Argmax has been at 100% since `Talos1.0`, but sampled play went from
`Talos1.1`'s 85% / 69% / 75% (`distance` / `shaped` /
`straggler`) to **98.3% / 90.0% / 93.3%**. That is the same measurement the
failed randomOpening run drove *down* to 42%, and it is the sharpest evidence
that what improved is the policy's distribution rather than only its best move.

**A fixed reference saturates, and rounds 3-4 show exactly what that looks
like.** Against `Talos1.1` those two rounds scored 63.7% then 63.1% breadth,
68.2% then 70.6% opening — flat, and read on its own it would say the league
had run out. Head to head, `round4` beats `round3` **57.4%**, and `round5`
beats `round4` **72.9%**. So the flatness was the instrument: once a candidate
takes ~70% of a census, the reference has little resolution left to give. From
round 5 the measurement of record became the previous round — a reference that
grows with the run — with `Talos1.1` kept only as the bridge to everything
recorded above it.

The 2% draws are all the same failure and are worth knowing about: two
deterministic policies deadlock. `openingSweep.py` prints which openings drew
and `scripts/replayGame.py --opening` replays one into the pygame window, which
is how this was read off rather than inferred. In the one examined, 248
half-moves visited
only 79 distinct positions, one of them 44 times, entering a **4-half-move
cycle at half-move 75** — one piece per side shuffling between two fields
(17↔29 and 38↔78) while the score stayed frozen at 6-4 for the remaining ~170
moves. Reaching the move cap is already priced as a loss, so the incentive is
right; the cycle survives because a cycle needs *both* sides deterministic, and
that never happens in training — PPO samples the learner's actions while the
frozen opponent is argmax, so the randomness breaks it. Measured: 0 draws in
100 games in the training configuration, average length 113 of a 250 cap. The
agent therefore gets no gradient signal about this at all, and more steps will
not address it; it is an artefact of evaluating a policy in a mode it never
trained in. Fixing it properly means a repetition signal in the observation,
which would change the observation space and invalidate every existing
checkpoint — deliberately deferred to the next generation.

The observation space did change on 2026-08-07 (the geometry planes above), so
that cost has now been paid and the repetition signal is free to follow. It was
**not** included in that change: what belongs in a history encoding — how many
previous plies, whose pieces, or a repetition count rather than raw positions —
is an open question, and bundling an unsettled design into the change that
broke compatibility would have made both harder to read afterwards.

### How the Talos checkpoints were actually built, and what is still on disk

The results above were measured over some sixteen checkpoints, which are named
throughout as if they were still there. They are not: `models/` was pruned to
`Talos1.0` and `Talos1.1` on 2026-08-05 (`Talos1.2` joined them the same day),
because everything else was either
unloadable, superseded, or reachable only through a checkpoint that survives.
The numbers stay valid — they are recorded here and in the commit messages,
which is what `.gitignore` says the history is for — but re-running an old
comparison means retraining its participants.

Two of the discarded ones could not have been re-run in any case:
`shapedByTravel` predates `de6aecd` and fails to load at all (`size mismatch`
in the feature extractor), and `maskedPPO_300k` still carries the pre-geometry
`Box(246,)` observation space, which no current script can feed.

The lineage that produced the surviving pair, reconstructed from the
checkpoints' own metadata and the commands that made them — worth recording
because none of it is derivable from the two files that remain:

| # | checkpoint | how |
|---|---|---|
| 1 | `clone_from_multi` | `pretrain --expert advancedDistScore sparsityScore bottleneck --samples 150000 --epochs 12` (the bot names of the day; `distance shaped straggler` now) |
| 2 | `multiVsModel` | 200k PPO from 1, opponent the frozen `pooledFinetuned`, `--lr 1e-4 --targetKl 0.03` |
| 3 | `multiVsModel2` | 100k from 2, opponent the frozen `tunedEnt000`, same step size |
| 4 | — | 150k from 3, against the five-heuristic pool |
| 5 | — | 100k from 4, against `random` alone |
| 6 | **`Talos1.0`** | 500k from 5, heuristic pool **and** model pool (`multiVsModel`, `multiVsModel2`, `pooledFinetuned`), `--lr 1e-4 --targetKl 0.03` |
| 7 | **`Talos1.1`** | six league rounds from 6, `scripts/progressivePhase1.py` |
| 8 | **`Talos1.2`** | five 300k league rounds from **6**, `scripts/progressivePhase2.py` |

Step 8 branches from step 6, not from step 7: **`Talos1.2` is not a descendant
of `Talos1.1` and never trained against it.** That was deliberate — `Talos1.1`
is the reference every number above is expressed against, and a candidate
trained against its own yardstick scores higher without being stronger, which
is precisely what round 3 of the phase-1 league did (95% against pool members,
65% against `Talos1.0`). It also means the two can be compared without an
asterisk, and that `Talos1.1` can eventually be deleted without orphaning the
lineage — but only once a measurement against `Talos1.0` exists to bridge the
record, which is why the final numbers above report both.

Steps 4-6 all wrote to the same name and were renamed to `Talos1.0` at the end,
so the intermediate stages no longer exist as files; step 7's six round
checkpoints were discarded with `226b632`. The frozen opponents in steps 2, 3
and 6 belong to the heuristic-cloning lineage (`cloned` →
`pooledWithLookahead` → `pooledFinetuned`, and `cloned` → `tunedEnt000`), which
is therefore an ancestor of the *training signal* rather than of the weights.

**None of this lineage is reproducible any more.** Steps 7 and 8 used to be —
`progressivePhase1.py` and `progressivePhase2.py` both fixed seed 42 and
rebuilt every round from `Talos1.0` — but the 2026-08-07 observation change
made `Talos1.0` unloadable, so the scripts had nothing left to branch from and
were deleted on 2026-08-23 (recoverable from git at `d194c5d`). Steps 1-6 were
never reproducible. `Talos1.0`, `1.1` and `1.2` are therefore the only
artefacts here that can neither be regenerated nor read; they are kept because
deleting them is irreversible and buys 24 MB.

### `lookahead2` is the one column that still moves, and `Talos1.2` fell on it

Measured 2026-08-06, 60 games per mode, seed 10000, `scripts/evaluateAgainstBots`:

| checkpoint | argmax | sampled | argmax home% |
|---|---|---|---|
| `Talos1.0` | 68.3 ± 11.8 | 33.3 ± 11.9 | 93.3 |
| `Talos1.1` | **75.0 ± 11.0** | 13.3 ± 8.6 | 91.4 |
| `Talos1.2` | **41.7 ± 12.5** | 33.3 ± 11.9 | 80.2 |

This is the first measurement in which `Talos1.2` is worse than what it was
built from, and the intervals do not overlap in either direction. It is also
the one bot nothing in this project has ever trained against, and the only one
that *searches* rather than scoring one ply — every other column has been at
98–100% argmax since `Talos1.0`, and the two Talos-against-Talos instruments
only ever compare the family with itself. So 1.5M rounds of league play bought
strength against the league's own members and the fast bots (the sampled column
above, 69%→90% against `shaped`) while losing ground against the one
opponent that plays differently. The old lineage's 25–46% against `lookahead2`
is not the comparison to draw here: `Talos1.0` and `Talos1.1` are fine.

The intervals are real ones. `Strategy.pickLowest` breaks ties from
`player.rng`, so 60 games against `lookahead2` are 60 distinct games — unlike
two argmax policies facing each other, where a pairing has only two.

### Search on top of the critic makes the policy weaker, not stronger

`env/searchPlayer.py` and `scripts/evaluateSearch.py`, 25 games per seat
direction (50 total) against `lookahead2`, seed 10000, `Talos1.2`:

| selection | win% | W | L | D | s/game |
|---|---|---|---|---|---|
| policy alone (`--noSearch`) | **30.0 ± 12.7** | 15 | 27 | 8 | 4.8 |
| critic alone (6 × 0 replies) | 20.0 ± 11.1 | 10 | 40 | 0 | 4.8 |
| critic + 1 reply | 20.0 ± 11.1 | 10 | 38 | 2 | 5.3 |
| critic + 3 replies | 18.0 ± 10.6 | 9 | 39 | 2 | 5.6 |

**The depth of the opponent ply changes nothing** — 0, 1 and 3 replies all land
at 18–20%. The loss appears the moment the critic takes over move selection, so
the failure is the evaluation, not the search, and not the "opponent plays as I
would" assumption that was the first suspect. Any single arm overlaps the
control's interval; three arms ten points below it do not read as noise.

The likely reason is the training objective rather than the network. PPO fits
the critic as a *baseline* for advantage estimation: it needs to be right on the
trajectory distribution its own policy visits, and its errors largely cancel in
`A = R - V`. Nothing ever asks it to rank two sibling positions, which is the
only thing a search wants from it. Shaping sharpens the point — with `V' = V -
w*phi` and `phi` added back explicitly, what remains of the critic's own
contribution is exactly the part with the least training pressure.

Capacity is a second candidate and is worth the numbers: of 689,403 parameters,
632,392 sit in the feature extractor **shared with the policy**
(`share_features_extractor=True`), and the value head is 20,673 — 256→64→64→1
behind a straggler whose shape the policy determines. The 1×1 squeeze to 8
channels before the flatten was a parameter-count decision that suits a policy
("which piece, where") better than a value ("is this structure won").

**Addressed on 2026-08-07, and only this candidate.** The critic now has its
own branch out of a shared trunk and a `vf=[256, 256]` MLP behind it, taking
the critic-only path to 1,371,217 parameters — see "The trunk is shared, the
branches are not" above. The numbers in the table were measured on `Talos1.2`,
which the same change made unloadable, so they are the record of the old
architecture rather than a control the new one can be compared against. The
sibling-ranking probe below is still unbuilt and is still what would say
whether capacity was the binding constraint or the objective was.

**The measurement that should come before any redesign** is whether the critic
ranks siblings at better than chance: take a position, take the policy's two
best moves, play both out, and check whether the critic's ordering matches the
outcome. At ~50% no architecture change helps and the objective has to change
(expert iteration, where a search supplies targets for policy *and* value); well
above 50% and the capacity and observation work is worth doing. Not yet built.

One thing the search did deliver: the repetition rule removes the deadlocks.
Draws went 8 → 0/2 across the arms. That is the first evidence for the
observation-level fix deferred to the next generation — and it says the fix does
not need to be learned to work at play time.

Two loose ends recorded rather than resolved. The two harnesses disagree about
`Talos1.2` against `lookahead2` — 41.7% from `evaluateAgainstBots` (agent on
`AGENT_SEAT`, play order drawn per game) against 30.0% from `evaluateSearch`
(both seat directions, half the games on seat 2). A seat-2 weakness would
explain the gap exactly and has history here (invariant 7), but nothing has
checked it. And `env/searchPlayer.py` has no tests yet, where the rest of `env/`
does.

### Next

**The Talos lineage ends here.** The 2026-08-07 observation and critic changes
above make `Talos1.0`/`1.1`/`1.2` unloadable, so `scripts/progressivePhase3.py`
— written, never run, and branching from `Talos1.2` — had nothing to branch
from any more. It and its two predecessors were deleted on 2026-08-23;
`scripts/talos2League.py` is the living successor and carries the same
schedule. What was gone was never the scripts but the checkpoints they pointed
at.

So the immediate path is a rebuild rather than a continuation:

1. A fresh clone — **done, see below**.
2. PPO from that clone, at the `--lr 1e-4 --targetKl 0.02` settings the sharp-
   policy instability above established, to a first `Talos2.0`.
3. The yardsticks, unchanged — `evaluateAgainstBots`, `openingSweep`,
   `randomPositionSweep`. The bots are the only reference that survived the
   break, which makes them the bridge between the two generations: `Talos1.x`'s
   recorded numbers against them are still the thing to beat, even though the
   checkpoints that produced them can no longer be played.

Three open questions that the rebuild is the natural moment for, none settled:
a **history/repetition signal** in the observation (see the deadlock discussion
above — the compatibility cost is already paid, only the design is open), the
**sibling-ranking probe** on the critic, which now has capacity and still has
no evidence that capacity was what it lacked, and whether the **parity penalty
earns its place** — `--parity 0` is the control, and nothing has yet run the
pair. It is policy-invariant by construction, so it can only change how fast
the agent learns, never what it converges to; the question is purely whether
the extra signal is worth anything.

After that, replace the pygame front-end with a browser-based one — a backend
around the unchanged engine plus a canvas front-end.

### Stage 2's two numbers, rescued from a docstring

The full write-up of generation 2's stages 1–2 is still open (`TODO.md`). These
two figures had been living only in `scripts/talos2League.py`'s docstring and
in `TODO.md`, both of which churn, so they are parked here before they are lost:

- **`models/talos2_stage6` beats `models/talos2_parity025` 64.1%** over the
  800-opening census. That is what the extra 500k steps of stage 2b bought, and
  it is why the league's anchor is stage 6 rather than the 300k checkpoint
  before it.
- **The parity-0 control arm was much better on sampled play against
  `lookahead2`: 45.0% against 18.3% over 60 games**, while measuring level
  everywhere else. That gap is unexplained, it is the reason the parity-0
  replication is still queued, and it is the first thing to suspect if the
  generation-2 lineage disappoints.

### Step 1 of the rebuild: the clone, and what 3.3× the data bought

`models/clone_from_multi_500k`, 2026-08-09. Same recipe as the old generation's
step 1 — `pretrain --expert advancedDistScore sparsityScore bottleneck
--epochs 12`, in the bot names of the day — run twice, at 150k and 500k
samples, because the network is now
2.05M parameters against the old 689k and the agreement curve was still rising
at epoch 12.

| | old gen, 150k | new gen, 150k | new gen, **500k** |
|---|---|---|---|
| agreement with the teachers | 68.7% | 66.6% | **72.5%** |
| vs `distance`, argmax | — | 84.0% | **88.0%** |
| vs `shaped`, argmax | — | 42.0% | **90.0%** |
| vs `random`, argmax | — | 86.0% | **96.0%** |
| vs `distance`, sampled | — | 50.0% | 66.0% |
| vs `shaped`, sampled | — | 16.0% | 32.0% |

50 games per cell. The three teachers scored 46% / 78% / 76% against
`distance` over the same 50 games, so the 500k clone at 88% is above
all of them on that pairing — cloning cannot exceed its teacher at *imitation*,
but a blend of three can beat any one of them at play.

**The data mattered far more than the extra epochs would suggest.** At equal
epochs the 500k run is 3.3× the gradient steps, so a per-epoch comparison
flatters it; the endpoint is the honest read, and 66.6% → 72.5% on held-out
*games* — the bot columns — is not a step-count artefact. `shaped` is
the striking one: 42% → 90% argmax. The 150k clone had a genuine blind spot
there and more of the same data closed it.

**The sampled columns are where this clone is still weak**, and `shaped`
at 32% is the number to watch through PPO. Sampled play is what
`progressivePhase2` moved most on the old lineage, so there is precedent for
it recovering; nothing here says it will.

The agreement curve had not flattened at epoch 12 (last step +0.6 points, from
+10.3 between epochs 1 and 2), so the run ended because the epochs ran out, not
because it converged. Whether more would help is untested and cheap to test.

Two things about `scripts/pretrain.py` changed to make 500k possible at all,
both recorded because the second is load-bearing:

- **`collect()` stores the dynamic half only.** A full observation is 14 planes
  of float32 and eleven of them are constant, so keeping them per sample stores
  the same block half a million times; the action mask is 14,641 entries of
  which ~65 are legal. Together that is 15.4 GB at 500k, on a 17 GB machine.
  Storing the two piece planes as `uint8` and `packbits`-ing the mask brings it
  to 1.2 GB, and `fit()` reassembles each batch. Measured: no cost in time — an
  epoch is 418s at 500k against the ~125s that 150k implies, which is linear.
- **The premise is that planes 2 and up never move.** If that ever stops being
  true, the fit would silently train on stale geometry and the loss would still
  fall. `tests/test_pretrain.py` pins both halves: the reconstruction equals the
  environment's own observation bit for bit, and the constant planes really are
  constant across a played game.

`--saveEvery N` was added at the same time. `main()` saves once, after the last
epoch *and* after 300 evaluation games, which for an 80-minute fit is a long
way to fall.

### A mixed teacher does not make a broader clone

Measured 2026-08-22, and it is the measurement that decides what
`scripts/pretrain.py` should clone from. Two arms at an identical budget — 150k
samples, 12 epochs, seed 0 — differing only in `--expert`:

- `models/teacherSingle`, cloned from `calibrated` alone
- `models/teacherMixed`, cloned from `calibrated calibratedCluster
  calibratedJump calibratedClusterJump straggler`

A clone learns its teacher's move distribution, so one strong teacher should
give a narrow clone however well it plays. That is the premise, and it did not
survive contact.

**Breadth.** `scripts/teacherAgreement.py`, per-teacher argmax agreement over
1,500 shared positions. The average `pretrain` prints cannot answer this — a
clone that copies one teacher and ignores four scores the same average as one
covering all five — so the number to read is the **minimum**.

| teacher | `teacherSingle` | `teacherMixed` |
|---|---|---|
| `calibrated` | 71.5% | 60.6% |
| `calibratedCluster` | 44.5% | 50.1% |
| `calibratedJump` | 43.7% | 40.7% |
| `calibratedClusterJump` | 51.4% | 47.5% |
| `straggler` | 42.4% | 42.2% |
| **minimum** | **42.4%** | **40.7%** |

Level to slightly worse. What the mixture did was flatten the profile — 10.9
points given up on `calibrated`, 5.6 gained on `calibratedCluster`, everything
else inside the noise. That is redistribution, not breadth.

**The reason is the panel, not the method.** The single-teacher clone already
agrees 42–51% with four teachers it never saw. Every sound bot in this family
has to be distance-dominated — see "Diversity is bounded by playability" — so
imitating one already captures most of what the others do, and there is little
breadth left at the level of individual moves for a mixture to buy.

**Strength**, 100 games per bot per mode, seed 80000:

| | vs `distance` | vs `shaped` | vs `calibrated` | vs `calibratedJump` |
|---|---|---|---|---|
| `teacherSingle` sampled | **61.0 ± 9.6** | **21.0 ± 8.0** | **18.0 ± 7.5** | **23.0 ± 8.2** |
| `teacherMixed` sampled | 49.0 ± 9.8 | 16.0 ± 7.2 | 11.0 ± 6.1 | 10.0 ± 5.9 |
| `teacherSingle` argmax | 88.0 | 76.0 | 100.0 | 31.0 |
| `teacherMixed` argmax | 68.0 | 100.0 | 100.0 | 30.0 |

Four of four sampled comparisons favour the single-teacher clone, two of them
outside the margins. **At equal budget the mixture bought nothing on either
axis**, so `pretrain` should keep cloning one bot, and `calibrated` is the one
to clone.

Two things stop that being the last word. Both clones were still improving at
epoch 12, and fitting five teachers is the harder problem, so an equal budget
is not an equal opportunity — equal *cost* is the right comparison for choosing
today, but the question itself deserves a 500k rerun. And the mixed clone's one
clear argmax win, 100-0 against `shaped` where the single clone went 76, is
real but rests on very few distinct games.

#### Read argmax margins with suspicion

That last point generalises and is worth stating on its own, because it applies
to every argmax number in this document. **Against a heuristic, argmax play is
very nearly deterministic**: the policy has no randomness, the bot has none
either bar tie-breaks, and only the seat draw varies. Counted directly, 30
argmax games against `calibrated` produced **5 distinct game lengths**, and 30
against `calibratedJump` produced 10.

So a 100-game argmax result is not 100 independent trials, and the binomial
margin that `evaluate` prints is far too tight — three cells above read
`100.0 ± 0.0`, which really means "won all of about five distinct lines". The
sampled column does not have this problem, since sampling varies every game,
which is a second reason to weigh it as heavily as the docstring of
`scripts/evaluateAgainstBots.py` already suggests for mid-training policies.
Nothing here is wrong, but a confident-looking argmax margin against a
deterministic opponent is measuring reproducibility as much as strength.

### Stage 3, the generation-2 league: five rounds, +70.5% over its own anchor

`scripts/talos2League.py`, five 300k rounds of checkpoint-only self-play from
`models/talos2_stage6`, finished 2026-08-10. Round 1 ran on 2026-08-09; rounds
2-5 were resumed the next day with the `--startRound` flag added for the
purpose, which rebuilds the opponent pool from the `Talos2.0_roundN` files on
disk so a resumed round sees the pool an uninterrupted run would have given it.

`scripts/openingSweep.py`, 800 openings per pairing:

| round | vs anchor `talos2_stage6` | vs the previous round |
|---|---|---|
| 1 | 58.5 | — |
| 2 | 62.9 | 54.8 |
| 3 | 65.1 | 52.2 |
| 4 | 67.0 | 55.4 |
| 5 | **70.5** | **56.6** |

The league did not plateau. The gain against the *anchor* decelerates (+4.4,
+2.2, +1.9, +3.5) but that curve flatters itself, because every round inherits
its predecessor's lead over a checkpoint that stopped improving five rounds
ago. The honest column is the head-to-head, and it is flat-to-rising: round 3
was a weak round (52.2%), not the onset of a plateau. Round 5 also beats round
1 66.5%, so the four rounds resumed here bought about as much as round 1 did
over the anchor.

`scripts/randomPositionSweep.py` agrees off the beaten track: round 5 over the
anchor **64.5% ± 4.2** (400 games from 200 mirrored positions, seed 0), 65.6%
at 4-12 random plies and 63.2% at 13-20, so the advantage is not an opening
book.

**Every number above is Talos-against-Talos**, and the three training bots have
been saturated at 98-100% argmax since the anchor — the inline reports across
all five rounds are 100% almost everywhere, with a single 95% against `random`
in rounds 4 and 5. So the league on its own could not distinguish "stronger"
from "better at beating its own siblings", and the generation-1 precedent is
exact: `Talos1.2` came out of 1.5M steps of league play beating its own league
members while its `lookahead2` score fell from `Talos1.0`'s 68.3% to 41.7%.

### The league is real: +13.5 on `lookahead2`, measured at 200 games

`scripts/evaluateAgainstBots.py`, 200 games per mode (400 per checkpoint),
seed 10000, 2026-08-10. `lookahead2` is the only bot that still moves and the
only instrument that bridges the two generations:

| checkpoint | argmax | sampled | argmax home% |
|---|---|---|---|
| `Talos1.0` (60 games) | 68.3 ± 11.8 | 33.3 ± 11.9 | 93.3 |
| `Talos1.1` (60 games) | 75.0 ± 11.0 | 13.3 ± 8.6 | 91.4 |
| `Talos1.2` (60 games) | 41.7 ± 12.5 | 33.3 ± 11.9 | 80.2 |
| `talos2_stage6`, the anchor | 58.5 ± 6.8 | 44.0 ± 6.9 | 93.4 |
| **`Talos2.0_round5`** | **72.0 ± 6.2** | **58.0 ± 6.8** | 94.6 |

Three readings, in decreasing order of confidence.

**The league bought real strength, not just sibling wins.** Round 5 beats its
own anchor by 13.5 points argmax and 14.0 sampled, and the intervals barely
touch (65.8-78.2 against 51.7-65.3) — where `Talos1.2` had *lost* 26 points
across the equivalent stage. The 1.5M steps went somewhere.

**Sampled play is the clear cross-generation win.** 58.0 ± 6.8 against
generation 1's best of 33.3 ± 11.9 — nowhere near overlapping. The policy's
whole distribution now beats the strongest bot more often than not, where
generation 1's distribution lost two games in three. The observation rebuild is
the likeliest cause, since it is what changed about what the network sees.

**Argmax is level with generation 1, not ahead of it.** 72.0 ± 6.2 spans
65.8-78.2; `Talos1.0` spans 56.5-80.1 and `Talos1.1` 64.0-86.0. Both overlap
heavily, so generation 2 has *matched* the old lineage's best argmax play, not
beaten it — and the honest statement is that the 60-game generation-1 numbers
are too coarse to separate from it either way. Re-measuring them is impossible;
the checkpoints no longer load.

Also worth recording: the anchor's own 200-game figure of 58.5% is comfortably
inside the 53.3 ± 12.6 measured over 60 games, so the old number was not wrong,
only too coarse to build on — which is exactly why 60 games could not answer
whether generation 2 had fallen behind `Talos1.0`.

Training health, against the instability that once collapsed argmax strength
from 97% to 46%: rounds 2-5 ran `approx_kl` at 0.012-0.021 with occasional
overshoots to 0.025-0.037 that `--targetKl 0.02` aborted, plus one 0.090 spike
on the first update of round 5 that the abort caught at step 1 and that
recovered immediately. Round 1's flat 0.0198-0.0199 — pinned against the limit
— did not recur, so that warning sign was specific to round 1.

Two design notes for anyone extending the league:

- **The opponent pool is drawn uniformly** (`env/halmaEnv.py`, `reset()`), so
  round 5 spent only 1/5 of its episodes against round 4, its strongest
  sparring partner, and the rest re-beating history. A sixth round would make
  that 1/6. If the league is extended, cap the pool at the anchor plus the last
  three rounds rather than letting it grow — the anchor earns a permanent slot
  as a fixed reference, the old middle rounds do not.
- **Draws appear as the lineage converges**: 0 in every round-1 and round-2
  sweep, then 2 (round 4 vs 3), 12 (round 5 vs 4). Siblings are starting to
  play each other into repetition, which is the same deadlock the observation
  still has no history signal for.
