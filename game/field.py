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

    def __init__(self, coord, id):
        self.coord = coord
        self.id = id
        self.playerID = 0
        self.neighbours = []
        self.jumpNeighbours = {}
        self.permissions = []

    def addNeighbour(self, id):
        self.neighbours.append(id)

    def addJumpNeighbour(self, neighbour, jumpNeighbour):
        self.jumpNeighbours[neighbour] = jumpNeighbour

    def removePlayer(self):
        self.playerID = 0

    def isEmpty(self):
        return self.playerID == 0

    def setPermissions(self, players):
        for player in players:
            self.permissions.append(player.identifier)

    def allows(self, player):
        return player.identifier in self.permissions
