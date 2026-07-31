from __future__ import annotations

import random
from typing import TYPE_CHECKING

from game.boardTypes import AnyMove, FieldId, MovePath, PlayerId
from heuristics.strategy import makeStrategy

if TYPE_CHECKING:
    from game.board import HalmaBoard


class HalmaPlayer:
    """A player's pieces and goal, plus the derived sets the bots score on.

    ``positions`` are the pieces' current fields; ``endPositions`` the target
    base. ``nonArrived`` (pieces not yet home) and ``openEndPositions`` (target
    fields still empty) are kept in sync on every move so the heuristics don't
    have to recompute them.
    """

    def __init__(self, identifier: PlayerId) -> None:
        self.identifier = identifier
        self.positions: set[FieldId] = set()
        self.startPositions: set[FieldId] = set()
        self.endPositions: set[FieldId] = set()
        self.openEndPositions: set[FieldId] = set()
        self.nonArrived: set[FieldId] = set()
        self.homeBase: FieldId | None = None
        self.distanceScore = 0
        # Replaced with the game's generator when seated, so one seed
        # reproduces a whole game.
        self.rng = random.Random()
        # Filled in by HalmaGame.initPlayers. Only searching strategies need it.
        self.opponents: list[HalmaPlayer] = []

    def setHomeBase(self, position: FieldId) -> None:
        self.homeBase = position

    def prepareForGameStart(self, board: HalmaBoard) -> None:
        for id in self.startPositions:
            board.fields[id].playerID = self.identifier
        self.positions.update(self.startPositions)
        self.nonArrived = self.positions - self.endPositions
        self.openEndPositions = self.endPositions - self.positions
        self.distanceScore = board.calculatePlayerDistanceScore(self)

    def updatePositionWithMove(self, move: AnyMove) -> None:
        start, end = move[0], move[-1]
        self.positions.remove(start)
        self.positions.add(end)
        self.openEndPositions.discard(end)
        self.openEndPositions.add(start)
        self.openEndPositions &= self.endPositions
        self.nonArrived.add(end)
        self.nonArrived.discard(start)
        self.nonArrived -= self.endPositions

    def isWinning(self) -> bool:
        return self.positions == self.endPositions

    def isHuman(self) -> bool:
        """Whether moves come from the UI rather than from the player itself."""
        raise NotImplementedError

    def chooseMove(self, moves: list[MovePath], board: HalmaBoard) -> MovePath:
        """Pick one of ``moves``. Only bots can: a human's move arrives through
        the visualization, so :meth:`HalmaGame.getNextMove` is never reached for
        them."""
        raise NotImplementedError

    def setStartPositions(self, positions: list[FieldId]) -> None:
        self.startPositions = set(positions)

    def setEndPositions(self, positions: list[FieldId]) -> None:
        self.endPositions = set(positions)


class HumanPlayer(HalmaPlayer):
    """A player whose moves come from the visualization's click handling."""

    def isHuman(self) -> bool:
        return True


class Computer(HalmaPlayer):
    """A bot player that picks moves via a named heuristic strategy."""

    def __init__(self, identifier: PlayerId, strategyName: str) -> None:
        super().__init__(identifier)
        self.strategy = makeStrategy(strategyName)

    def isHuman(self) -> bool:
        return False

    def chooseMove(self, moves: list[MovePath], board: HalmaBoard) -> MovePath:
        # Delegate the choice to the configured heuristic strategy.
        return self.strategy.bestMove(moves, board, self)
