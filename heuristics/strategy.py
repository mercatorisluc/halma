from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from game.board import HalmaBoard
    from game.boardTypes import MovePath
    from game.player import HalmaPlayer


class Strategy:
    """Scores candidate moves for a bot, by a named combination of the board's
    heuristics. Lower is better throughout.

    ``SCORERS`` maps the public strategy name to the method implementing it and
    is the single source of truth for which strategies exist -- construction
    validates against it and ``scripts/baseline.py`` enumerates it, so no second
    list can drift out of step.
    """

    SCORERS: ClassVar[dict[str, str]] = {
        "advancedDistScore": "advancedDist",
        "simpleDistScore": "simpleDist",
        "sparsityScore": "sparsity",
        "bottleneck": "bottleneck",
        "random": "chooseRandom",
    }

    # How heavily bottleneckScore counts next to the distance term. Measured
    # over 150 games against advancedDistScore: 0.02 -> 81%, 0.05 -> 84%,
    # 0.1 -> 82%, 0.3 and 1.0 -> 81%. The exact value barely matters, because
    # the term mostly reorders moves the distance score leaves tied.
    BOTTLENECK_WEIGHT = 0.05

    def __init__(self, strategyName: str) -> None:
        # Fail here rather than at the first scoring call, which used to raise a
        # bare KeyError somewhere deep inside a game.
        if strategyName not in self.SCORERS:
            raise ValueError(f"unknown strategy {strategyName!r}; known: {sorted(self.SCORERS)}")
        self.strategyName = strategyName

    def advancedDist(self, board: HalmaBoard, player: HalmaPlayer) -> float:
        distanceScore = board.advancedDistanceScore(player)
        homeBonus = board.homeBonusScore(player)
        return (distanceScore + homeBonus) / 2

    def simpleDist(self, board: HalmaBoard, player: HalmaPlayer) -> float:
        distanceScore = board.simpleDistanceScore(player)
        homeBonus = board.homeBonusScore(player)
        return (distanceScore + homeBonus) / 2

    def chooseRandom(self, board: HalmaBoard, player: HalmaPlayer) -> float:
        # Every move scores the same, so bestMove's tie-break picks at random.
        return 1

    def sparsity(self, board: HalmaBoard, player: HalmaPlayer) -> float:
        """Distance and home progress, shaped by three terms that fade out.

        The three shape terms -- clustering, group cohesion, jump potential --
        are opening advice. Weighted equally with the progress terms they used
        to outvote them in the endgame and the bot could not finish: measured
        in a stuck position, moving a piece into the target improved distance
        by -0.052 and home bonus by -0.067 while the shape terms objected by
        +0.143, so the move scored worse and was never played. Two pieces stayed
        out forever, 37 of 40 self-play games ended in the move limit.

        Multiplying them by homeBonusScore makes them fade as pieces arrive and
        vanish once everything is home, leaving only progress to decide the
        endgame. That single change removed every draw (0 of 40) and took the
        bot from 22.5% against advancedDistScore to 78.3%, level with
        bottleneck.
        """
        home = board.homeBonusScore(player)
        shape = (
            board.sparsityScore(player)
            + board.playerSparsityScore(player)
            + board.potentialJumpScore(player)
        )
        return board.advancedDistanceScore(player) + home + home * shape

    def bottleneck(self, board: HalmaBoard, player: HalmaPlayer) -> float:
        """advancedDist, plus a penalty for the piece left furthest behind.

        Advancing the pack is not enough to win -- the last piece home ends the
        game. Adding that straggler's remaining distance beats plain
        advancedDist by a wide margin (84% over 150 games).
        """
        return self.advancedDist(board, player) + self.BOTTLENECK_WEIGHT * board.bottleneckScore(
            player
        )

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
    ``bottleneck``, which is itself 84% against ``advancedDistScore``. Worst
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
        # bottleneck is the leaf evaluation; the public name is its own, since
        # this is a search rather than one of the scoring functions.
        super().__init__("bottleneck")
        self.strategyName = self.NAME

    def evaluate(self, board: HalmaBoard, player: HalmaPlayer) -> float:
        """Own progress minus the opposition's. Lower is better."""
        own = self.bottleneck(board, player)
        return own - sum(self.bottleneck(board, other) for other in player.opponents)

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
                value = self.bottleneck(board, opponent)
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
