from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from game.board import HalmaBoard
    from game.boardTypes import MovePath
    from game.player import HalmaPlayer


class Strategy:
    """Scores candidate moves for a bot, by a named combination of the board's
    heuristics. Lower is better throughout."""

    def __init__(self, strategyName: str) -> None:
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
        distanceScore = board.advancedDistanceScore(player)
        sparsityScore = board.sparsityScore(player)
        playerSparsityScore = board.playerSparsityScore(player)
        jumpScore = board.potentialJumpScore(player)
        homeBonus = board.homeBonusScore(player)
        return distanceScore + sparsityScore + playerSparsityScore + jumpScore + homeBonus

    def scoringFunction(self, board: HalmaBoard, player: HalmaPlayer) -> float:
        strategyMap = {
            "advancedDistScore": self.advancedDist,
            "sparsityScore": self.sparsity,
            "simpleDistScore": self.simpleDist,
            "random": self.chooseRandom,
        }
        score = strategyMap[self.strategyName]
        return score(board, player)

    def bestMove(self, moves: list[MovePath], board: HalmaBoard, player: HalmaPlayer) -> MovePath:
        """Pick a lowest-scoring move, breaking ties at random.

        Each candidate is scored on the board as it would look after the move;
        ``moveApplied`` puts it back afterwards.
        """
        scored = []
        for move in moves:
            with board.moveApplied(move, player):
                scored.append((self.scoringFunction(board, player), move))
        minValue = min(score for score, _ in scored)
        return player.rng.choice([move for score, move in scored if score == minValue])
