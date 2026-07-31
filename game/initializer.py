from game.field import HalmaField


class Initializer:
    """Builds the star board and the players' starting layout.

    Stateless on purpose: it constructs things and hands them over, so nothing
    it derives can go stale or be read after the fact. Everything it needs to
    look up afterwards it asks the board for, which owns that data.

    A field carries two addresses, an axial ``coord`` (x, y) and a stable
    sequential ``id`` (0-120). This class uses a third internally: a
    ``fieldNumber`` that embeds the coord in a 17x17 grid
    (``fieldNumberFromCoord``) so neighbours can be found by simple offset
    arithmetic (see ``DIRECTIONS`` / ``directionMapper``). It ties them
    together by one rule: **a field's id is its rank in fieldNumber order**, so
    building the fields in that order (see ``buildFields``) produces the ids
    directly.

    ``fieldNumber`` never leaves this class — it is derived from the coordinate
    where needed and is not stored on the fields, because nothing after
    construction has any use for it.
    """

    # The six hex-grid directions used to wire up field adjacency. A tuple so
    # the shared class attribute cannot be mutated by accident.
    DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1), (-1, 1), (1, -1))

    def initializeBoard(self, board):
        """Create every field, wire up neighbours/jumps, and cache distances."""
        board.setFields(self.buildFields())
        self.initEdges(board)
        board.calculateDistanceMatrix()

    def player1Positions(self, board):
        startPositions, endPositions = [], []
        for i in range(5):
            for j in range(i + 1):
                startPositions.append(board.idFromCoord((i, -4 - j)))
                endPositions.append(board.idFromCoord((-i, 4 + j)))
        homeBase = board.idFromCoord((-4, 8))
        return (startPositions, endPositions, homeBase)

    def player2Positions(self, board):
        startPositions, endPositions = [], []
        for i in range(5):
            for j in range(i + 1):
                startPositions.append(board.idFromCoord((-4 - j, i)))
                endPositions.append(board.idFromCoord((4 + j, -i)))
        homeBase = board.idFromCoord((8, -4))
        return (startPositions, endPositions, homeBase)

    def player3Positions(self, board):
        startPositions, endPositions = [], []
        for i in range(5):
            for j in range(i + 1):
                startPositions.append(board.idFromCoord((4 - j, i)))
                endPositions.append(board.idFromCoord((j - 4, -i)))
        homeBase = board.idFromCoord((-4, -4))
        return (startPositions, endPositions, homeBase)

    def buildFields(self):
        """Return every field, already ordered by id.

        The board is a central 9x9 rhombus plus four outward triangles. A
        field's ``id`` is by definition its rank once all fields are ordered by
        ``fieldNumber``, so sorting the coordinates once makes the id fall out
        of ``enumerate`` — no second pass to assign ids, and no ranking scan.
        """
        coords = [(i, j) for i in range(-4, 5) for j in range(-4, 5)]
        for i in range(1, 5):
            for j in range(1, i + 1):
                coords.extend([(-4 - j, i), (4 + j, -i), (-i, 4 + j), (i, -4 - j)])
        coords.sort(key=self.fieldNumberFromCoord)
        return [HalmaField(coord, id) for id, coord in enumerate(coords)]

    def initEdges(self, board):
        # A neighbour is one direction step away on the 17x17 grid, the jump
        # landing two. Both only exist if that fieldNumber is on the board.
        # fieldNumber lives only in this method: derived from the coordinate,
        # used to wire adjacency, never stored on the field or anywhere else.
        fieldNumbers = {field.id: self.fieldNumberFromCoord(field.coord) for field in board.fields}
        idByFieldNumber = {number: id for id, number in fieldNumbers.items()}
        for field in board.fields:
            for di, dj in self.DIRECTIONS:
                neighbour = fieldNumbers[field.id] + self.directionMapper(di, dj)
                jumpNeighbour = fieldNumbers[field.id] + self.directionMapper(2 * di, 2 * dj)
                if neighbour in idByFieldNumber:
                    field.addNeighbour(idByFieldNumber[neighbour])
                    if jumpNeighbour in idByFieldNumber:
                        field.addJumpNeighbour(
                            idByFieldNumber[neighbour], idByFieldNumber[jumpNeighbour]
                        )

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
        return (x + 8) + (y + 8) * 17

    def directionMapper(self, di, dj):
        return di + 17 * dj
