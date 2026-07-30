from game.field import HalmaField


class Initializer:
    """Builds the star board and the players' starting layout.

    Fields are addressed three ways: an axial ``coord`` (x, y), a stable
    sequential ``id`` (0-120), and a ``fieldNumber`` that embeds the coord in a
    17x17 grid (``fieldNumberFromCoord``) so neighbours can be found by simple
    offset arithmetic (see ``DIRECTIONS`` / ``directionMapper``).

    The three are tied together by one rule: **a field's id is its rank in
    fieldNumber order**. Building the fields in that order (see ``initNodes``)
    therefore produces the ids directly, and ``idByFieldNumber`` / ``idByCoord``
    make every later translation a dict lookup.
    """

    # The six hex-grid directions used to wire up field adjacency. A tuple so
    # the shared class attribute cannot be mutated by accident.
    DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1), (-1, 1), (1, -1))

    def __init__(self):
        self.fields = []
        self.idByFieldNumber = {}
        self.idByCoord = {}


    def initializeBoard(self, board):
        """Create every field, wire up neighbours/jumps, and cache distances."""
        self.initNodes()
        board.setFieldPositionsMapper(self.fields)
        board.setFields(self.fields)
        self.initEdges(board)
        board.calculateDistanceMatrix()


    def player1Positions(self):
        startPositions, endPositions = [], []
        for i in range(5):
            for j in range(i+1):
                startPositions.append(self.identifierFromCoord((i, -4-j)))
                endPositions.append(self.identifierFromCoord((-i, 4+j)))
        homeBase = self.identifierFromCoord((-4, 8))
        return (startPositions, endPositions, homeBase)


    def player2Positions(self):
        startPositions, endPositions = [], []
        for i in range(5):
            for j in range(i+1):
                startPositions.append(self.identifierFromCoord((-4-j, i)))
                endPositions.append(self.identifierFromCoord((4+j, -i)))
        homeBase = self.identifierFromCoord((8, -4))
        return (startPositions, endPositions, homeBase)


    def player3Positions(self):
        startPositions, endPositions = [], []
        for i in range(5):
            for j in range(i+1):
                startPositions.append(self.identifierFromCoord((4-j, i)))
                endPositions.append(self.identifierFromCoord((j-4, -i)))
        homeBase = self.identifierFromCoord((-4, -4))
        return (startPositions, endPositions, homeBase)


    def initNodes(self):
        """Create every field, already ordered by id.

        The board is a central 9x9 rhombus plus four outward triangles. A
        field's ``id`` is by definition its rank once all fields are ordered by
        ``fieldNumber``, so sorting the coordinates once makes the id fall out
        of ``enumerate`` — no second pass to assign ids, and no ranking scan.

        Everything is rebuilt from scratch: an ``Initializer`` is reused across
        ``reset()``, so carrying anything over would corrupt every id.
        """
        coords = [(i, j) for i in range(-4, 5) for j in range(-4, 5)]
        for i in range(1, 5):
            for j in range(1, i+1):
                coords.extend([(-4-j, i), (4+j, -i), (-i, 4+j), (i, -4-j)])
        coords.sort(key=self.fieldNumberFromCoord)
        self.fields = [
            HalmaField(coord, id, self.fieldNumberFromCoord(coord))
            for id, coord in enumerate(coords)
        ]
        self.idByFieldNumber = {field.fieldNumber: field.id for field in self.fields}
        self.idByCoord = {field.coord: field.id for field in self.fields}


    def initEdges(self, board):
        # A neighbour is one direction step away on the 17x17 grid, the jump
        # landing two. Both only exist if that fieldNumber is on the board,
        # which idByFieldNumber answers directly.
        for field in board.fields:
            for (di, dj) in self.DIRECTIONS:
                neighbour = field.fieldNumber + self.directionMapper(di, dj)
                jumpNeighbour = field.fieldNumber + self.directionMapper(2*di, 2*dj)
                if neighbour in self.idByFieldNumber:
                    field.addNeighbour(self.idByFieldNumber[neighbour])
                    if jumpNeighbour in self.idByFieldNumber:
                        field.addJumpNeighbour(self.idByFieldNumber[neighbour],
                                               self.idByFieldNumber[jumpNeighbour])


    def initPermissions(self, board, players):
        # A field surrounded entirely by one player's start/end cells belongs
        # exclusively to that player (their home triangles); every other field
        # is open to all players.
        for player in players:
            positions = player.positions | player.endPositions
            for id in positions:
                explicitPermission = True
                neighbours = board.fields[id].neighbours
                for neighbourID in neighbours:
                    if neighbourID not in positions:
                        explicitPermission = False
                if explicitPermission:
                    board.fields[id].setPermissions([player])
        for field in board.fields:
            if not field.permissions:
                field.setPermissions(players)


    def fieldNumberFromCoord(self, coord):
        x, y = coord
        return (x+8) + (y+8)*17


    def identifierFromFieldNumber(self, fieldNumber):
        return self.idByFieldNumber[fieldNumber]


    def identifierFromCoord(self, coord):
        return self.idByCoord[coord]


    def directionMapper(self, di, dj):
        return di + 17 * dj
