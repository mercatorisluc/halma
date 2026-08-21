from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from heuristics.calibration import calibrator

if TYPE_CHECKING:
    from game.board import HalmaBoard
    from game.boardTypes import MovePath
    from game.player import HalmaPlayer

# Which board primitive each of `calibrated`'s weights multiplies.
CALIBRATED_TERMS: dict[str, str] = {
    "distance": "openTargetDistanceScore",
    "home": "unfilledTargetScore",
    "clustering": "clusteringScore",
    "stragglerLag": "stragglerLagScore",
    "jumpPotential": "jumpPotentialScore",
}


# One family of bots, one scoring function: a member is nothing but a
# weight vector, and a term is switched off by weighting it 0. They exist to
# be *different* rather than uniformly strong -- `env/` trains against an
# opponent pool, and a pool of near-identical bots teaches an agent to beat
# one opponent. Each vector is fitted by `scripts/fitWeights.py`; the
# starting values here are `shaped`'s weights transplanted onto the
# calibrated terms, which makes plain `calibrated` the honest "same bot,
# comparable scales" control for the fit.
CALIBRATED_VARIANTS: dict[str, dict[str, float]] = {
    "calibrated": {
        "distance": 1.0,
        "home": 3.885,
        "clustering": 0.052,
        "stragglerLag": 0.166,
        "jumpPotential": 0.04,
    },
    # `calibratedPlain` -- distance and home only, no shape terms -- was removed
    # on 2026-08-21 and is not coming back. It agreed with `distance` on 91.4%
    # of midgame positions and 97.5% of early ones, and won 48.0% against it,
    # so both measures said the same thing: a bigger `home` weight is not a
    # different bot. Its slot went to `calibratedClusterJump` below. Restore it
    # from git if a lean control is ever wanted, but not for a pool.
    #
    # One shape term each, so the three play visibly differently.
    "calibratedCluster": {
        "distance": 1.0,
        "home": 1.0,
        "clustering": 0.13,
        "stragglerLag": 0.0,
        "jumpPotential": 0.0,
    },
    "calibratedLag": {
        "distance": 1.0,
        "home": 2.753,
        "clustering": 0.0,
        "stragglerLag": 0.125,
        "jumpPotential": 0.0,
    },
    "calibratedJump": {
        "distance": 1.0,
        "home": 2.803,
        "clustering": 0.0,
        "stragglerLag": 0.0,
        "jumpPotential": 0.316,
    },
    # The one pair, replacing `calibratedPlain`, which the 2026-08-21 agreement
    # measurement found to be `distance` with a bigger `home` weight (91.4% of
    # midgame positions) and therefore worth nothing to a pool that can hold
    # `distance` itself. `clustering` and `jumpPotential` were picked as the two
    # axes furthest apart in the panel.
    #
    # **These are not the weights the fit ranked first**, and that is deliberate.
    # Its three finalists were statistically level -- 60.3 +/- 5.5, 56.3 +/- 5.6,
    # 56.3 +/- 5.6 over 300 games -- so win rate did not separate them, and win
    # rate is not what this variant is for. Move agreement did separate them, in
    # the opposite order: the fit's winner agreed with `calibrated` on 83.0% of
    # midgame positions, this one on 60.0%, and its highest agreement with
    # anything is 63.2%. Adopting the strongest would have rebuilt the duplicate
    # the variant exists to replace.
    #
    # The fit drove `jumpPotential` to 0.040 -- the same near-off value it chose
    # for `calibrated` -- so this bot is cluster-driven in practice; the name
    # records the term set that was searched, not two live terms. What makes it
    # different from `calibratedCluster` is the weights, not the terms: 2.4x the
    # clustering and 3.9x the home pull, which is enough that the two agree on
    # only 29.9% of midgame positions.
    "calibratedClusterJump": {
        "distance": 1.0,
        "home": 3.886,
        "clustering": 0.316,
        "stragglerLag": 0.0,
        "jumpPotential": 0.040,
    },
}


class Strategy:
    """Scores candidate moves for a bot, by a named combination of the board's
    heuristics. Lower is better throughout.

    ``SCORERS`` maps the public strategy name to the method implementing it and
    is the single source of truth for which strategies exist -- construction
    validates against it and ``scripts/baseline.py`` enumerates it, so no second
    list can drift out of step.
    """

    SCORERS: ClassVar[dict[str, str]] = {
        "distance": "plainDistance",
        "tipDistance": "tipDistance",
        "shaped": "shaped",
        "straggler": "straggler",
        "random": "chooseRandom",
        **dict.fromkeys(CALIBRATED_VARIANTS, "calibrated"),
    }

    # The names these bots went by until 2026-08-14, still accepted so that
    # every command line, script and training recipe written down before then
    # keeps working. They resolve to the canonical name at construction, so
    # nothing downstream ever sees the old spelling. Deliberately temporary --
    # drop them once the recorded recipes have been moved over.
    ALIASES: ClassVar[dict[str, str]] = {
        "advancedDistScore": "distance",
        "simpleDistScore": "tipDistance",
        "sparsityScore": "shaped",
        "bottleneck": "straggler",
    }

    # How heavily stragglerTravelScore counts next to the distance term. Measured
    # over 150 games against `distance`: 0.02 -> 81%, 0.05 -> 84%,
    # 0.1 -> 82%, 0.3 and 1.0 -> 81%. The exact value barely matters, because
    # the term mostly reorders moves the distance score leaves tied.
    STRAGGLER_WEIGHT = 0.05

    # How heavily `shaped`'s three shape terms count against its distance
    # term. Unlike STRAGGLER_WEIGHT this one matters a great deal and is not
    # smooth -- measured over 200-300 games against the previous formulation:
    # 0.02 -> 38.5%, 0.05 -> 52.5%, 0.08 -> 72.0%, 0.10 -> 62.0%, 0.13 -> 81.3%,
    # 0.16 -> 79.3%, 0.20 -> 23.7%, 0.30 -> 4.5%. The cliff above 0.16 is the
    # endgame failure its docstring describes, and the dip at 0.10 is real
    # rather than noise: these bots are deterministic bar tie-breaks, so a small
    # reweighting reorders whole games. Do not tune this by reasoning about it.
    SHAPE_WEIGHT = 0.13

    # `calibrated`'s terms, in the order its feature vector lists them. The
    # first is pinned to 1.0 by convention: scoring only ever compares
    # candidates within one position, so multiplying every weight by a positive
    # constant changes nothing, and leaving that freedom in would give the fit
    # in `scripts/fitWeights.py` a direction it could wander along forever.
    WEIGHT_ORDER: ClassVar[list[str]] = [
        "distance",
        "home",
        "clustering",
        "stragglerLag",
        "jumpPotential",
    ]

    def __init__(self, strategyName: str) -> None:
        strategyName = self.ALIASES.get(strategyName, strategyName)
        # Fail here rather than at the first scoring call, which used to raise a
        # bare KeyError somewhere deep inside a game.
        if strategyName not in self.SCORERS:
            raise ValueError(f"unknown strategy {strategyName!r}; known: {sorted(self.SCORERS)}")
        self.strategyName = strategyName
        variant = CALIBRATED_VARIANTS.get(strategyName)
        self.weights = dict(variant) if variant else {}
        # Held as closures rather than looked up per call; only built for the
        # strategies that use them, since every Strategy runs this.
        self.calibrators = (
            {name: calibrator(name) for name in CALIBRATED_TERMS.values()} if variant else {}
        )

    def plainDistance(self, board: HalmaBoard, player: HalmaPlayer) -> float:
        """Remaining travel to the open targets, plus how much of the target is
        still empty.

        The distance term used to be blended half-and-half with
        `tipDistanceScore`, the static distance to the tip of the target
        triangle. Dropping that half is both stronger and cheaper, measured over
        400 games with seats swapped: 60.5% (+/- 4.8) head to head, and better
        against every third party too -- 20.2% against `straggler` where the
        blend scored 18.5%, 21.8% against `shaped` where it scored 19.2%,
        92.0% against `tipDistance` where it scored 88.8%. What it gains is
        not a better distance measure but a bigger share for `unfilledTargetScore`:
        keeping the distance term's old weight and only swapping the measure
        scores 34.0%.

        Cheaper because `openTargetDistanceScore` is O(1) -- it reads the
        incrementally maintained `player.distanceScore` -- where the static term
        sums over all 15 pieces: 0.081 us against 0.422 us for the blend, and
        0.224 us against 0.583 us for this scorer, a factor of 2.6. Dispatch
        through `scoringFunction` adds a flat 0.097 us on top of either. That is
        the number that matters for `scripts/pretrain.py`, which generates its
        samples with this bot.
        """
        distanceScore = board.openTargetDistanceScore(player)
        homeBonus = board.unfilledTargetScore(player)
        return (distanceScore + homeBonus) / 2

    def tipDistance(self, board: HalmaBoard, player: HalmaPlayer) -> float:
        distanceScore = board.tipDistanceScore(player)
        homeBonus = board.unfilledTargetScore(player)
        return (distanceScore + homeBonus) / 2

    def chooseRandom(self, board: HalmaBoard, player: HalmaPlayer) -> float:
        # Every move scores the same, so bestMove's tie-break picks at random.
        return 1

    def shaped(self, board: HalmaBoard, player: HalmaPlayer) -> float:
        """Distance and home progress, shaped by three terms that fade out.

        The three shape terms -- clustering, group cohesion, jump potential --
        are opening advice. Weighted equally with the progress terms they used
        to outvote them in the endgame and the bot could not finish: measured
        in a stuck position, moving a piece into the target improved distance
        by -0.052 and home bonus by -0.067 while the shape terms objected by
        +0.143, so the move scored worse and was never played. Two pieces stayed
        out forever, 37 of 40 self-play games ended in the move limit.

        Multiplying them by `unfilledTargetScore` makes them fade as pieces
        arrive and vanish once everything is home, leaving only progress to
        decide the endgame. That single change removed every draw (0 of 40) and
        took the bot from 22.5% against `distance` to 78.3%.

        The distance term is `openTargetDistanceScore`, the same one
        `plainDistance` uses. It used to be a blend of that and the static tip
        distance, which is roughly 7.5x larger, and the shape terms were
        weighted against *that* magnitude implicitly -- swapping the measure
        without touching the weights therefore handed them the vote and the bot
        won 0.0% of 800 games. Making the weight explicit and re-measuring it
        (see SHAPE_WEIGHT) is what let the blend go, and it bought a much
        stronger bot rather than merely an equal one. Over 400 games with seats
        swapped it wins 81.3% against the version it replaced, and against the
        rest of the panel, with that version's score for comparison: 58.2%
        against `straggler` (was 38.0%), 92.0% against `distance` (was 74.5%),
        99.8% against `tipDistance` (was 96.8%). That makes it the strongest of
        the one-ply bots.
        """
        home = board.unfilledTargetScore(player)
        shape = (
            board.clusteringScore(player)
            + board.stragglerLagScore(player)
            + board.jumpPotentialScore(player)
        )
        return board.openTargetDistanceScore(player) + home + self.SHAPE_WEIGHT * home * shape

    def calibratedFeatures(self, board: HalmaBoard, player: HalmaPlayer) -> list[float]:
        """`calibrated`'s five terms, in `WEIGHT_ORDER`, before weighting.

        Split out from the scorer so that `scripts/fitWeights.py` optimises the
        exact quantity the bot goes on to play. The score is a dot product of
        this and the weights, which is what makes the fit cheap: the features of
        a candidate do not depend on the weights, so a whole position's
        candidates can be reduced to one small matrix and every weight vector
        after that costs a multiply.

        `unfilledTargetScore` appears twice and only once calibrated. As a
        summand it is a score like any other and belongs on the common scale. As
        the factor that fades the shape terms out it is **not** a score at all
        -- it is the fraction of the target still empty, and the endgame depends
        on it reaching 0 exactly, which a calibrated version never does (its
        lowest knot maps to 0.0065). That fade is what took `shaped` from 37
        stuck games in 40 to none; calibrating it would quietly undo that.
        """
        home = board.unfilledTargetScore(player)
        calibrate = self.calibrators
        shape = [
            calibrate[CALIBRATED_TERMS[name]](float(getattr(board, CALIBRATED_TERMS[name])(player)))
            for name in ("clustering", "stragglerLag", "jumpPotential")
        ]
        return [
            calibrate["openTargetDistanceScore"](board.openTargetDistanceScore(player)),
            calibrate["unfilledTargetScore"](home),
            *[home * term for term in shape],
        ]

    def calibrated(self, board: HalmaBoard, player: HalmaPlayer) -> float:
        """`shaped` with every summand on a common scale and one weight each.

        Same terms as `shaped` and the same fade, but the primitives go through
        `heuristics/calibration.py` first, so the weights are the whole story
        rather than sharing the job with whatever divisor each primitive
        happened to carry. That is what makes `SHAPE_WEIGHT` -- one number
        covering three terms of very different spreads -- splittable into five.
        """
        features = self.calibratedFeatures(board, player)
        return sum(
            self.weights[name] * feature
            for name, feature in zip(self.WEIGHT_ORDER, features, strict=True)
        )

    def straggler(self, board: HalmaBoard, player: HalmaPlayer) -> float:
        """`plainDistance`, plus a penalty for the piece left furthest behind.

        Advancing the pack is not enough to win -- the last piece home ends the
        game. Adding that straggler's remaining distance beats `plainDistance`
        on its own by a wide margin (84% over 150 games).
        """
        return self.plainDistance(
            board, player
        ) + self.STRAGGLER_WEIGHT * board.stragglerTravelScore(player)

    def scoringFunction(self, board: HalmaBoard, player: HalmaPlayer) -> float:
        scorer = getattr(self, self.SCORERS[self.strategyName])
        return float(scorer(board, player))

    def bestMove(self, moves: list[MovePath], board: HalmaBoard, player: HalmaPlayer) -> MovePath:
        """Pick a lowest-scoring move, breaking ties at random.

        Each candidate is scored on the board as it would look after the move;
        ``moveApplied`` puts it back afterwards.
        """
        scored = []
        for move in moves:
            with board.moveApplied(move, player):
                scored.append((self.scoringFunction(board, player), move))
        return self.pickLowest(scored, player)

    @staticmethod
    def pickLowest(scored: list[tuple[float, MovePath]], player: HalmaPlayer) -> MovePath:
        minValue = min(score for score, _ in scored)
        return player.rng.choice([move for score, move in scored if score == minValue])


class LookaheadStrategy(Strategy):
    """Two-ply search: assume the opponent answers with its own best reply.

    Opt-in and deliberately not part of the RL pipeline. Measured at roughly
    23ms per move against 0.28ms for one-ply -- some 80x slower, about 3s per
    game -- which is fine for playing a human but far too slow for generating
    training data.

    It is the strongest bot here: 90% (+/- 9.3 over 40 games) against
    ``straggler``, which is itself 84% against ``distance``. Worst
    observed move takes 132ms, which is a natural-feeling pause in a
    turn-based game but far too slow to generate training data with.

    This is also the only place where scoring the opponent pays off. At one ply
    the opponent's position is identical across all of the mover's candidates,
    so subtracting its score shifts every candidate equally and cannot change
    which move wins; verified by measurement. Here the replies differ, so the
    difference carries information.

    Known imprecision: an opponent's ``openEndPositions`` does not account for
    the mover's own pieces occupying its target fields, so blocking is not
    valued accurately.
    """

    NAME = "lookahead2"

    def __init__(self) -> None:
        # straggler is the leaf evaluation; the public name is its own, since
        # this is a search rather than one of the scoring functions.
        super().__init__("straggler")
        self.strategyName = self.NAME

    def evaluate(self, board: HalmaBoard, player: HalmaPlayer) -> float:
        """Own progress minus the opposition's. Lower is better."""
        own = self.straggler(board, player)
        return own - sum(self.straggler(board, other) for other in player.opponents)

    def bestMove(self, moves: list[MovePath], board: HalmaBoard, player: HalmaPlayer) -> MovePath:
        scored = []
        for move in moves:
            with board.moveApplied(move, player):
                scored.append((self.worstCaseReply(board, player), move))
        return self.pickLowest(scored, player)

    def worstCaseReply(self, board: HalmaBoard, player: HalmaPlayer) -> float:
        """Value of the position after the opponent plays its best answer.

        "Best" for the opponent means lowest for its own one-ply score, which
        is what the opposing bots actually do -- searching every reply against
        our own evaluation would cost another factor of the branching factor.
        """
        if not player.opponents:
            return self.evaluate(board, player)
        opponent = player.opponents[0]
        replies = board.allValidMovesWithWay(opponent)
        if not replies:
            return self.evaluate(board, player)
        best, bestValue = replies[0], None
        for reply in replies:
            with board.moveApplied(reply, opponent):
                value = self.straggler(board, opponent)
            if bestValue is None or value < bestValue:
                best, bestValue = reply, value
        with board.moveApplied(best, opponent):
            return self.evaluate(board, player)


# Every strategy a Computer can be given. The scoring functions come from
# Strategy.SCORERS, the search adds itself; scripts/baseline.py reads this so
# there is no second list to keep in step.
STRATEGY_NAMES = [*Strategy.SCORERS, LookaheadStrategy.NAME]


def makeStrategy(strategyName: str) -> Strategy:
    """Build the strategy behind a name, search or scoring function alike."""
    if strategyName == LookaheadStrategy.NAME:
        return LookaheadStrategy()
    return Strategy(strategyName)
