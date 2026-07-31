from __future__ import annotations

from collections import deque
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np

from game.boardTypes import AnyMove, Coord, FieldId, MoveEndpoints, MovePath, PlayerId
from game.field import HalmaField

if TYPE_CHECKING:
    from game.player import HalmaPlayer


class HalmaBoard:
    """The 121-field star board: piece placement, move generation and the
    heuristic scoring functions used by the bots.

    Fields are addressed by integer id (0-120) and are stored in id order, so
    ``fields[id]`` is the field itself and ``coordFromId`` needs no lookup.
    ``idFromCoord`` is the reverse index, for the callers that start from a
    coordinate. ``distanceMatrix`` caches the board distance between every pair
    of fields.
    """

    def __init__(self) -> None:
        self.fields: list[HalmaField] = []
        self.idByCoord: dict[Coord, FieldId] = {}
        self.distanceMatrix: list[list[int]] = []

    def setFields(self, fields: list[HalmaField]) -> None:
        # fields must be ordered by id (index i holds the field with id i).
        self.fields = fields
        self.idByCoord = {field.coord: field.id for field in fields}

    def coordFromId(self, id: FieldId) -> Coord:
        return self.fields[id].coord

    def idFromCoord(self, coord: Coord) -> FieldId:
        return self.idByCoord[coord]

    def calculateDistanceMatrix(self) -> None:
        # fields are ordered by id, so index == id and no lookup is needed.
        coords = [field.coord for field in self.fields]
        self.distanceMatrix = [[self.distance(a, b) for b in coords] for a in coords]

    def placePiece(self, id: FieldId, playerID: PlayerId) -> None:
        self.fields[id].playerID = playerID

    def removePiece(self, id: FieldId) -> None:
        assert self.fields[id].playerID != 0
        self.fields[id].removePlayer()

    def applyMoveForPlayer(self, move: AnyMove, player: HalmaPlayer) -> None:
        start, end = move[0], move[-1]
        self.placePiece(end, self.fields[start].playerID)
        self.removePiece(start)
        player.updatePositionWithMove(move)
        self.updatePlayerDistanceScore(player, move)

    @contextmanager
    def moveApplied(self, move: AnyMove, player: HalmaPlayer) -> Generator[HalmaBoard]:
        """Apply ``move`` for the duration of the block, then take it back.

        Judging a candidate move means looking at the board as it would be
        afterwards. Rather than copy the board, it is mutated and restored --
        which is sound because reversing a move is an exact inverse, including
        the incrementally maintained ``player.distanceScore`` (both properties
        are pinned in ``tests/test_scores.py``).

        The undo runs in a ``finally``: without it, a scoring function that
        raises would leave the move applied to the real game board.
        """
        self.applyMoveForPlayer(move, player)
        try:
            yield self
        finally:
            self.applyMoveForPlayer((move[-1], move[0]), player)

    def getValidNeighbourFields(self, id: FieldId) -> list[FieldId]:
        """Empty fields one step away — the destinations of a single step."""
        return [n for n in self.fields[id].neighbours if self.fields[n].isEmpty()]

    def getValidJumpFields(self, id: FieldId) -> list[FieldId]:
        """Landing fields of a single jump: something to hop over, empty behind."""
        return [
            landing
            for jumpedOver, landing in self.fields[id].jumpNeighbours.items()
            if not self.fields[jumpedOver].isEmpty() and self.fields[landing].isEmpty()
        ]

    def isJumpMove(self, start: FieldId, end: FieldId) -> bool:
        return end not in self.fields[start].neighbours

    def allValidMoves(self, player: HalmaPlayer) -> list[MoveEndpoints]:
        """All legal moves for ``player`` as ``(start, end)`` id pairs.

        The endpoints view of :meth:`allValidMovesWithWay` — the same moves
        without the intermediate jump landings, which only the visualization
        and :class:`~game.move.Move` need.
        """
        return [(way[0], way[-1]) for way in self.allValidMovesWithWay(player)]

    def allValidMovesWithWay(self, player: HalmaPlayer) -> list[MovePath]:
        """All legal moves for ``player`` as full paths ``[start, ..., end]``.

        A single step lands on an empty neighbour; jumps hop over an occupied
        field onto the empty one beyond and chain, so the path records every
        landing along the way. Only moves whose destination the player is
        allowed to occupy are kept.

        ``reachable`` is seeded with the piece's own field and its single-step
        neighbours, which does two things: a destination is only ever offered
        once, and a step is never chained into a jump — only the piece's own
        field is queued, so jumps start from there.
        """
        allMoves = []
        for start in player.positions:
            steps = self.getValidNeighbourFields(start)
            moves = [[start, end] for end in steps]
            reachable = {start, *steps}
            queue = deque([[start]])
            while queue:
                way = queue.popleft()
                for landing in self.getValidJumpFields(way[-1]):
                    if landing not in reachable:
                        reachable.add(landing)
                        moves.append([*way, landing])
                        queue.append([*way, landing])
            allMoves.extend(moves)
        return [move for move in allMoves if self.fields[move[-1]].allows(player)]

    def distance(self, coordA: Coord, coordB: Coord) -> int:
        """Board distance between two axial hex coordinates.

        On this grid a diagonal counts as one step when the two axes move in
        opposite directions (hence the sign check); otherwise the moves add up.
        """
        distanceScore = 0
        verticalDist = coordA[0] - coordB[0]
        horizontalDist = coordA[1] - coordB[1]
        if (verticalDist * horizontalDist) < 0:
            distanceScore += max(abs(verticalDist), abs(horizontalDist))
        else:
            distanceScore += abs(verticalDist) + abs(horizontalDist)
        return distanceScore

    def simpleDistanceScore(self, player: HalmaPlayer) -> float:
        # Total distance of the player's pieces from their own home base;
        # lower is better (pieces have advanced further). Scaled by 16.
        homeBase = player.homeBase
        assert homeBase is not None, "homeBase is set during game setup"
        score = 0
        for id in player.positions:
            score += self.distanceMatrix[id][homeBase]
        # scaling factor of 16
        return score / 16

    def calculatePlayerDistanceScore(self, player: HalmaPlayer) -> int:
        score = 0
        for pieceId in player.nonArrived:
            score += sum(
                self.distanceMatrix[pieceId][targetId] for targetId in player.openEndPositions
            )
        return score

    def playerDistanceScore(self, player: HalmaPlayer) -> float:
        score = player.distanceScore
        score /= max((len(player.nonArrived) * len(player.openEndPositions)), 1)
        # scaling factor of 12
        return score / 12

    def advancedDistanceScore(self, player: HalmaPlayer) -> float:
        # Blend of the cached distance-to-open-targets score and the simple
        # distance-to-home score; lower is better.
        return (self.playerDistanceScore(player) + self.simpleDistanceScore(player)) / 2

    def updatePlayerDistanceScore(self, player: HalmaPlayer, move: AnyMove) -> None:
        # Incrementally maintain player.distanceScore after a move instead of
        # recomputing over all pieces: adjust only the terms that changed as a
        # piece left `start` and arrived at `end` (with corrections for moves
        # into or out of the target end positions).
        start, end = move[0], move[-1]
        toAdd, toSubtract = (0, 0)
        if start not in player.endPositions:
            toSubtract += sum(self.distanceMatrix[id][start] for id in player.openEndPositions)
        else:
            toAdd += sum(self.distanceMatrix[id][start] for id in player.nonArrived)
        if end in player.endPositions:
            toSubtract += sum(self.distanceMatrix[id][end] for id in player.nonArrived)
        else:
            toAdd += sum(self.distanceMatrix[id][end] for id in player.openEndPositions)
        # missed
        if (start in player.endPositions) and (end not in player.endPositions):
            toSubtract += self.distanceMatrix[start][end]
        # missed
        if (start not in player.endPositions) and (end in player.endPositions):
            toSubtract += self.distanceMatrix[start][end]
        player.distanceScore += toAdd - toSubtract

    def sparsityScore(self, player: HalmaPlayer) -> float:
        # Rewards keeping pieces loosely clustered: penalises each piece by how
        # far its share of occupied neighbours is from an ideal 0.75. Lower is
        # better. Averaged over the player's pieces.
        score = 0
        for id in player.positions:
            neighbours = set(self.fields[id].neighbours)
            idScore = 0
            for neighbour in neighbours:
                if not self.fields[neighbour].isEmpty():
                    idScore += 1
            score += 4 / 3 * abs(0.75 - (idScore / len(neighbours)))
        return score / len(player.positions)

    def playerSparsityScore(self, player: HalmaPlayer) -> float:
        # How far the most trailing piece lags behind the group: the largest
        # deviation of any piece's distance-from-home above the group mean.
        # Lower is better (pieces advance together). Scaled by 12.
        homeBase = player.homeBase
        assert homeBase is not None, "homeBase is set during game setup"
        distances = [self.distanceMatrix[p][homeBase] for p in player.positions]
        meanDist = np.mean(distances)
        maxDeviation = max(d - meanDist for d in distances)
        return maxDeviation / 12

    def potentialJumpScore(self, player: HalmaPlayer) -> float:
        # Rewards positions that have available jumps (a piece to hop over onto
        # an empty landing); lower score = more jump potential. Averaged over
        # the player's pieces.
        score = 0
        for id in player.positions:
            idScore = 0
            jumpNeighbours = self.fields[id].jumpNeighbours
            for jumpOver, jumpOn in jumpNeighbours.items():
                if (not self.fields[jumpOver].isEmpty()) and (self.fields[jumpOn].isEmpty()):
                    idScore += 1
            score += 1 - idScore / len(jumpNeighbours)
        return score / len(player.positions)

    def homeBonusScore(self, player: HalmaPlayer) -> float:
        # Fraction of target fields NOT yet occupied by the player; lower is
        # better (0 once every piece is home).
        return 1 - len(player.positions & player.endPositions) / len(player.endPositions)

    def boardState(self) -> np.ndarray:
        return np.array([field.playerID for field in self.fields])
