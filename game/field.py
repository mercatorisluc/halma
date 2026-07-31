from __future__ import annotations

from typing import TYPE_CHECKING

from game.boardTypes import Coord, FieldId, PlayerId

if TYPE_CHECKING:
    from game.player import HalmaPlayer


class HalmaField:
    """One field of the board: who stands on it, what it borders, who may enter.

    Addressed by ``id`` — that is what ``neighbours``, ``jumpNeighbours`` and
    every move are expressed in. ``coord`` is kept because geometry is still
    needed at play time: drawing and click hit-testing, the board distances,
    the RL symmetry permutations, and finding the field a jump passed over.

    There is deliberately no ``fieldNumber``. That third addressing scheme
    exists only to derive adjacency while the board is being built, and
    ``Initializer`` computes it from ``coord`` where it needs it.
    """

    def __init__(self, coord: Coord, id: FieldId) -> None:
        self.coord = coord
        self.id = id
        # 0 means empty; otherwise the identifier of the player standing here.
        self.playerID: PlayerId = 0
        self.neighbours: list[FieldId] = []
        # jumped-over field -> the field landed on behind it
        self.jumpNeighbours: dict[FieldId, FieldId] = {}
        self.permissions: list[PlayerId] = []

    def addNeighbour(self, id: FieldId) -> None:
        self.neighbours.append(id)

    def addJumpNeighbour(self, neighbour: FieldId, jumpNeighbour: FieldId) -> None:
        self.jumpNeighbours[neighbour] = jumpNeighbour

    def removePlayer(self) -> None:
        self.playerID = 0

    def isEmpty(self) -> bool:
        return self.playerID == 0

    def setPermissions(self, players: list[HalmaPlayer]) -> None:
        for player in players:
            self.permissions.append(player.identifier)

    def allows(self, player: HalmaPlayer) -> bool:
        return player.identifier in self.permissions
