# Architecture

How this codebase is put together and why. `README.md` covers what the project
is and how to run it; this file explains the internals. `CLAUDE.md` holds the
day-to-day conventions for working in the repo and points here for structure.

---

## The shape of the system

The game engine is pure Python and knows nothing about pygame, numpy-heavy
tensors or reinforcement learning. Everything else is a consumer of it.

```mermaid
graph TD
    subgraph core["game/ — the engine (no I/O, no rendering)"]
        GM[HalmaGame<br/>turn order, history, win check]
        B[HalmaBoard<br/>move generation, scoring]
        P[HalmaPlayer<br/>pieces, target, derived sets]
        F[HalmaField<br/>occupancy, neighbours, permissions]
        I[Initializer<br/>builds board + starting layout]
        MV[Move<br/>one recorded move]
    end

    H["heuristics/<br/>Strategy — scores candidate moves"]
    V["visual/<br/>pygame front-end"]
    E["env/<br/>Gymnasium wrapper for RL"]

    GM --> B & P & MV
    B --> F
    I --> B & P
    P --> H
    H --> B
    V --> GM
    E --> GM

    style core fill:#f5f5f5,stroke:#999
```

The single most important consequence: **changing engine behaviour affects three
consumers**, and only one of them (`visual/`) is something you can see. Check
`heuristics/`, `visual/` and `env/` when you touch `game/`.

---

## The board

121 fields forming a six-pointed star (Chinese-Checkers style). Each field has
up to six neighbours — two at the star's tips.

`Initializer.buildFields` builds it as a **9×9 rhombus (81 fields) plus four
ten-field triangles (40)**. Note that this construction does *not* line up with
the six home regions, which is the confusing part:

- Up to three players, **15 pieces each** — a triangle of 1+2+3+4+5.
- For players 1 and 2 that home region **straddles the seam**: 10 fields come
  from an explicitly built triangle, the remaining 5 from the rhombus's edge
  row (`(i, -4)` and `(-4, i)` respectively).
- Player 3's home region lies **entirely inside a corner of the rhombus**
  (`(4-j, i)`), with no explicit triangle at all — which is why its coordinates
  look unlike the other two.

So "the four triangles" are a construction detail of the geometry, not the
players' home bases. A player's target is the star point opposite their start.

### Three ways to address a field

This is the main source of confusion in the codebase. A field carries the first
two; `HalmaBoard` converts between them.

| Name | Form | Used for |
|---|---|---|
| `coord` | axial hex `(x, y)` | geometry: distance, rotation, flipping, drawing, and finding the field a jump passed over |
| `id` | `0`–`120` | **everything else**: `board.fields[id]`, player positions, moves, RL actions |
| `fieldNumber` | `(x+8) + (y+8)*17` | build-time only — lets `initEdges` find neighbours by offset arithmetic on a 17×17 grid |

`fieldNumber` is pure scaffolding and does not exist outside `Initializer`: it
is derived from `coord` where adjacency is being wired and is deliberately not
stored on the field, because nothing after construction has a use for it. New
code should not reintroduce it.

`coord`, by contrast, is *not* build-time only — a common assumption worth
correcting. Rendering, click hit-testing, the distance matrix, the RL symmetry
permutations and `Move`'s jumped-over calculation all need it while the game is
being played.

The three are tied together by one rule worth knowing: **a field's `id` is its
rank in `fieldNumber` order**. `Initializer.buildFields` builds the fields in
that order, so the id is just the index — the "fields must be ordered by id"
contract on `setFields` is satisfied by construction rather than by a later
sort. That in turn makes `board.coordFromId(id)` a plain `fields[id].coord`
with no lookup, and only the reverse direction needs an index
(`board.idFromCoord`, built once in `setFields`).

The board owns that index because it is board data. `Initializer` keeps no
state at all: it builds fields and hands them over, and when it later needs a
coordinate translated — placing the players' starting layout — it asks the
board, the same way `initEdges` and `initPermissions` already take the board as
a parameter. The dependency only ever points from initializer to board.

### Field permissions

`Initializer.initPermissions` marks a field as exclusive to one player when
*every* neighbour of that field is also in that player's own start/end set —
i.e. the interior of their home triangles. All other fields are open to
everyone. This is what stops a player from parking a piece in someone else's
target and blocking them out forever.

---

## Moves: two representations

A move is a sequence of field `id`s, but it exists in **two forms**, and picking
the wrong one is an easy mistake:

- **Endpoints only** — `(start, end)`. Produced by `board.allValidMoves`. This
  is what the RL env and the scoring heuristics use.
- **Full path** — `[start, land, land, ..., end]`. Produced by
  `board.allValidMovesWithWay`. The visualization needs it to draw a multi-hop
  jump, and `Move` needs it to know the intermediate landings.

`game/boardTypes.py` names both (`MoveEndpoints` = `tuple`, `MovePath` =
`list`), so the distinction is checkable by the editor and not just described
here. The same file names `FieldId`, `Coord` and `PlayerId`.

`PlayerId` is an `int` — a seat number, 1 to 3. Never a name, and never 0,
because 0 is what an empty `HalmaField` holds. `board.boardState()` collects
those ids into a numpy array that the RL layer does arithmetic on, so a string
identifier would turn it into a string array in which every empty field
compares unequal to 0 and reads as an opponent's piece. Whether a player is
human is answered by `isHuman()`, not by their identifier.

There is **one** generator, not two: `allValidMovesWithWay` runs the BFS over
chained jumps, and `allValidMoves` is its endpoints view. A single step goes to
an empty neighbour; a jump hops over an occupied field onto the empty one
directly beyond, and jumps chain. Only the piece's own field is queued, so a
step is never chained into a jump.

`Move` stores whatever it was given. If it only got endpoints,
`reconstructFullMove` recovers the path by BFS and records the jumped-over
fields — see the invariant on `jumpedOvers` below.

---

## Layer by layer

### `game/` — the engine

`HalmaGame` owns the board, the players, the turn order and the move history,
and decides who has won. `ComputedGame` (all bots) and `InteractiveGame` (one
human) differ *only* in which players they seat; all behaviour lives in the base
class.

Turn order is randomised once at setup (`computePlayersOrder`) and then rotates
by move count — `currentPlayer()` is derived from `len(self.moves)`, not stored.
That is what makes stepping backwards through history work without extra
bookkeeping.

A player wins by filling their target base, **or** when their target base is
completely full including opponent pieces (`playerIsWinningByBlockedFields`) —
otherwise a single squatter could deadlock the game forever.

`HalmaBoard` does double duty: move generation *and* the heuristic scoring
functions the bots rank positions with (`simpleDistanceScore`,
`advancedDistanceScore`, `sparsityScore`, `playerSparsityScore`,
`potentialJumpScore`, `homeBonusScore`, `bottleneckScore`). For all of them
**lower is better**.

### `heuristics/` — the bots

`Strategy.SCORERS` maps a strategy name to the method implementing it and is
the single source of truth for which strategies exist — construction validates
against it and `scripts/baseline.py` enumerates `STRATEGY_NAMES`, so no second
list can drift. `bestMove` evaluates every candidate by **applying it to the
real board, scoring, then reversing it**, and picks a minimum at random among
ties.

Two findings are worth recording, because both cost real time to establish and
would otherwise be rediscovered the expensive way:

- **Opponent-awareness cannot help at one ply.** An opponent's distance and
  home-bonus terms are *identical* across all of the mover's candidates — the
  opposition does not move while you evaluate your own options — so subtracting
  them shifts every candidate equally and cannot change which move wins.
  Measured: one distinct value across 72 candidates. It only pays inside a
  search, where the replies differ, which is what `LookaheadStrategy` is for.
- **Rewarding long available jump chains makes the bot worse**, badly: 8.8% and
  0.0% win rates against `advancedDistScore` at two weights. Preferring
  positions that *have* long chains keeps pieces hoarding jump potential
  instead of advancing.

What does work is `bottleneckScore`: the distance still facing the piece left
furthest behind. The game ends only when every piece is home, so the sum of
distances is the wrong late objective — it collapses while one straggler
decides the length of the game. Adding it to `advancedDist` wins 84% (±5.9 over
150 games) against plain `advancedDist`, robust across weights from 0.02 to 1.0.

Strength order, all measured: `lookahead2` > `bottleneck` > `advancedDistScore`
> `simpleDistScore` > `sparsityScore` >> `random`. `InteractiveGame` seats the
strongest; `ComputedGame` keeps the cheap one-ply bots because it feeds the RL
environment, where a two-ply search at ~130ms a move is out of the question.

Seats confer no advantage, which is what makes `scripts/baseline.py` readable:
in 400 mirror games seat 1 won 47.8% (±4.9) and the player on move 53.5%
(±4.9) — both consistent with an even game.

The mutation is scoped by `board.moveApplied(move, player)`, a context manager
that undoes the move in a `finally`. Copying the board per candidate would be
far more expensive, and the undo is exact — see the invariants. The consequence
to keep in mind is that scoring is *not* read-only: a board can only be scored
by one caller at a time, so candidate moves for a single position cannot be
evaluated in parallel. Running several independent games at once is unaffected,
since each has its own board.

### `visual/` — the pygame front-end

`GameVisualization` is the orchestrator and owns the main loop. It holds no
drawing logic itself; it wires up four single-purpose collaborators:

- **`BoardProjector`** — the only thing that knows about screen geometry and
  the current rotation. Both drawing and click hit-testing go through its
  `coordToPos`/`posToCoord` tables, so a rotated board projects and un-projects
  consistently.
- **`BoardRenderer`** — draws; holds no state of its own.
- **`GamePlaybackController`** — owns the history cursor `moveTraveler` and
  steps the board forwards/backwards by applying and un-applying recorded moves.
- **`HumanInputHandler`** — turns a pair of clicks into a validated move.

The bot only auto-plays when the cursor is at the live front of the game
(`moveTraveler == len(game.moves)`), so reviewing history does not trigger new
moves.

### `env/` — the RL wrapper

`HalmaEnv` wraps a `ComputedGame` as a Gymnasium environment. Actions are
encoded as `start * fieldCount + end`.

Gymnasium models one agent against a world; Halma has two players. So the agent
owns one seat and the heuristic opponent moves *inside* `step` — one env step is
a full round. Only ~65 of 14641 encoded actions are legal at a time, so
`action_masks()` is not optional: without it the policy would spend itself
learning which actions are illegal rather than which are good.

`boardNormalizer.Normalizer` precomputes field-id permutations for each player's
viewpoint (`player1WithFlip`, `player2WithoutFlip`, … plus inverses) that map
any player/orientation into one canonical frame, so a single policy learns one
orientation rather than several.

**The reward is shaped, and it has to be.** Winning is the only true reward and
it arrives once per ~69 decisions — and a random agent, measured over 700 games,
never wins at all. A constant signal teaches nothing, so `_shaping` adds
`weight * (gamma * phi(s') - phi(s))` on every step. This is the potential-based
form (Ng, Harada & Russell 1999) whose terms telescope, so it changes no policy's
ranking — the agent is hurried, not redirected. Verified over 32 completed games:
the discounted return shifts by exactly `-phi(s0)`, to machine precision. Two
conditions: `gamma` must match the training discount, and the potential is zeroed
on termination but deliberately *not* on time-limit truncation, where the agent
should still bootstrap. Set `shapingWeight=0` to turn it off and measure whether
it earns its place. `info["outcome"]` carries the unshaped ±1 so evaluation
scores wins rather than shaping.

The potential is **the agent's own remaining travel**, normalised: the sum, over
its pieces, of the distance from each to the nearest field of its target zone,
divided by the 140 steps facing it at the opening and subtracted from 1. So it
runs 0 at the opening to exactly 1 once every piece is home — and, with 15 pieces
and 15 target fields, a remaining travel of zero *is* the win condition, so the
top of the scale coincides with winning rather than approximating it.

Two things it is deliberately not:

- **Not the lead over the opponent.** A lead can be held by obstructing as
  easily as by advancing, and an agent trained on the difference took exactly
  that route — it finished with none of its 15 pieces home while holding the
  opponent from 14 down to 12.
- **Not the bots' `advancedDistanceScore + homeBonusScore`.** That is what
  `heuristics/` ranks moves by, and it is a poor thing to shape with. Two thirds
  of it is `simpleDistanceScore`, the distance to the single *tip* field of the
  target triangle rather than to the triangle: measured over 235 moves it moved
  ten times further per move than the zone-distance term, so it was effectively
  the whole signal, and it aimed at one corner. It also never bottoms out — a won
  position still scored 1.25 against the opening's 7.69, leaving 16% of the
  shaping budget unreachable and paying pieces already home to shuffle towards
  the tip. Its distance term is averaged over the pieces still out, too, so a
  piece arriving shrank the numerator and the divisor together.

Nearest target field per piece, rather than a min-cost assignment of pieces to
target fields. The assignment is the exact remaining travel, but the two
correlate at 0.996 over real games and agree on which moves help, so it is not
worth an O(n³) matching — or a scipy dependency — on every step.

---

## Invariants you must not break

These are load-bearing and mostly non-obvious. Each is pinned by a test.

**1. `player.distanceScore` is maintained incrementally.**
`board.updatePlayerDistanceScore` adjusts only the terms that changed rather
than recomputing over all pieces. Two properties depend on it and are verified
in `tests/test_scores.py`: applying a move and then its reverse restores the
score exactly, and the incremental value matches a full recomputation. Both
`Strategy.bestMove` and `GamePlaybackController.backwardGame` rely on this.

**2. Reversing a move is a true undo.**
`applyMoveForPlayer((end, start), player)` must restore the board *and* the
player's derived sets (`positions`, `nonArrived`, `openEndPositions`). Playback
and bot search both depend on it.

**3. `Move.jumpedOvers is None` means "not reconstructed yet".**
It must **not** be initialised to `[]` — `needsReconstruction()` reports on
exactly that `None`, so an empty list would make every move look already
reconstructed. `fullStepsList()` raises a `RuntimeError` if called first.

**4. Reversing nests correctly across players.**
`LookaheadStrategy` applies its own move and then the opponent's reply inside
it. Each `moveApplied` block undoes exactly its own move and they unwind in
order, so both players' `positions`, `nonArrived`, `openEndPositions` and
`distanceScore` come back unchanged. Pinned in `tests/test_strategy.py`.

**5. Derived player sets are updated on every move.**
`nonArrived` and `openEndPositions` are kept in sync by
`updatePositionWithMove` so the heuristics never have to recompute them. Code
that moves pieces by writing `field.playerID` directly (as some tests do
deliberately) bypasses this and leaves the player object stale.

**6. `gameLength()` is the position's version, and `HalmaEnv` caches on it.**
Every real move goes through `playMove`, which appends to the move list, and
`currentPlayer()` is derived from that count — so the move count identifies the
position within an episode. `HalmaEnv._legalActions` memoises on it, because
generating moves is the most expensive thing the environment does and one step
used to ask five times over (the caller's `action_masks()`, `step`'s legality
check, the opponent's search, the mobility scalar, and `action_masks()` again
inside `_info`) for what are only two distinct positions. Scoring a candidate
does *not* bump the count — `moveApplied` goes straight to the board — which is
sound only because scoring never calls `_legalActions`. Hand-placing pieces
does not bump it either, so tests that do that must not then read the mask.
`reset` clears the entry, since a new game restarts the count.

---

## How a move actually happens

**A bot move**

```
HalmaGame.playNextMove(player)
  └─ getNextMove          → board.allValidMovesWithWay(player)   (full paths)
       └─ player.chooseMove → Strategy.bestMove
            └─ for each candidate: apply → score → reverse
  └─ playMove             → board.applyMoveForPlayer  (moves piece, updates
                             player sets + distanceScore)
                          → append Move to history
```

**A human move**

Click a piece, then a destination. `HumanInputHandler` collects the two clicks,
validates the pair against `allValidMovesWithWay`, plays it through the same
`playMove`, and advances the playback cursor so rendering stays in sync.

**Stepping through history**

`GamePlaybackController` replays or un-applies recorded moves on the *live*
board — there are no board snapshots. `gameStateAt(n)` rewinds to the start and
replays `n` moves.

---

## Design decisions worth knowing

**camelCase throughout.** Un-pythonic but consistent. Ruff's `N` rules are
deliberately disabled (`ruff.toml`) because enabling them reports ~300
violations and would invite a rename touching every file for no functional gain.
Match the surrounding style in new code.

**The engine never prints.** `HalmaGame.play()` returns the winner's identifier
(or `None` on a draw by exhaustion) instead of writing to stdout, because
self-play training runs it thousands of times. `printBoard()` is the one
deliberate exception — printing is its purpose.

**Distances are precomputed.** `calculateDistanceMatrix` fills a 121×121 table
once at setup so the heuristics can look up any pair in O(1).

**Board dimensions are derived, never hardcoded.** The field count comes from
`len(board.fields)` and the piece count from `len(player.positions)` /
`len(player.endPositions)`, including in the RL action encoding
(`HalmaEnv.fieldCount`). The numbers 121 and 15 should not appear as literals.

---

## Current state

| Area | State |
|---|---|
| `game/` | Refactored, documented, characterization-tested |
| `heuristics/` | Working; five bots, strength measured against each other |
| `visual/` | Working; refactored into focused modules. No tests |
| `env/` | Satisfies the Gymnasium API, masked and shaped |
| Agent | Cloned from a bot, then fine-tuned by PPO to 99% against `advancedDistScore` |

`env/` passes `gymnasium.utils.env_checker.check_env` and has action masking, a
canonical observation and reproducible seeding.

`scripts/baseline.py` plays every pairing of the bots and is the yardstick a
trained agent has to clear. The number that shaped the plan: **a random agent
wins none of 700 games**, so an untrained policy sees the same reward in
essentially every episode. That is why shaping came before training.

**Shaping was not enough on its own.** With the reward shaped, the geometry
given to a CNN and the action head factored, PPO from scratch still ended a
quarter hour of training at 0 wins and single-digit percent of pieces home. The
signal is there, but the region of policy space where Halma is played is not
somewhere random exploration arrives.

**What worked was copying a bot first.** `scripts/pretrain.py` plays games with
`bottleneck` on the agent's seat, records its choice in every position, and fits
the policy to those choices by cross-entropy over the masked distribution. On
150k positions and 12 epochs the policy agrees with the bot on 73% of positions
and goes from 0% wins to 86% against `advancedDistScore` (argmax, 50 games).
Scored over the same games, the teacher itself takes 68% — but it wins 64% against
`sparsityScore` where the clone takes 54%, so the clone is at roughly teacher
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
take it from 82% to **99% against `advancedDistScore`** (198W 2L of 200 games,
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
against `advancedDistScore` but 90-92% against `random`, where the clone was at
96%. Beating one bot decisively is not the same as playing Halma well, so the
next measurement worth having is against opponents it never saw.

From here: self-play, which is the only route that does not inherit a teacher's
ceiling, and search. One thing agreed for later is making position evaluation
parallelisable, which today's `moveApplied` prevents.

After that, replace the pygame front-end with a browser-based one — a backend
around the unchanged engine plus a canvas front-end.
