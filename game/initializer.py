from game.field import HalmaField


class Initializer:
    """Builds the star board and the players' starting layout.

    Fields are addressed three ways: an axial ``coord`` (x, y), a stable
    sequential ``id`` (0-120), and a ``fieldNumber`` that embeds the coord in a
    17x17 grid (``fieldNumberFromCoord``) so neighbours can be found by simple
    offset arithmetic (see ``DIRECTIONS`` / ``directionMapper``).
    """

    # The six hex-grid directions used to wire up field adjacency.
    DIRECTIONS = [(1, 0), (0, 1), (-1, 0), (0, -1), (-1, 1), (1, -1)]

    def __init__(self):
        self.fields = []
    
    
    def initBoard(self, board):
        """Create every field, wire up neighbours/jumps, and cache distances."""
        self.initNodes()
        board.setFieldPositionsMapper(self.fields)
        board.setFields([
            HalmaField(f["coord"], f["id"], f["fieldNumber"])
            for f in sorted(self.fields, key=lambda f: f["id"])
        ])
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
        # The board is a central 9x9 diamond plus the four outward triangles
        # (the star's points); assign each resulting field a stable id.
        self.fields = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                self.fields.append({'coord': (i, j), 'fieldNumber': self.fieldNumberFromCoord((i, j))})
        for i in range(1, 5):
            for j in range(1, i+1):
                for k in [(-4-j, i), (4+j, -i), (-i, 4+j), (i, -4-j)]:
                    self.fields.append({'coord': k, 'fieldNumber': self.fieldNumberFromCoord(k)})
        for field in self.fields:
            field['id'] = self.identifierFromCoord(field['coord'])
                                            
                        
    def initEdges(self, board):
        for field in board.fields:
            fieldNumber = field.fieldNumber
            for (di, dj) in self.DIRECTIONS: 
                neighbour = fieldNumber + self.directionMapper(di, dj)
                jumpNeighbour = fieldNumber + self.directionMapper(2*di, 2*dj)
                if (neighbour in board.allFieldNumbers()):
                    field.addNeighbour(self.identifierFromFieldNumber(neighbour))
                    if (jumpNeighbour in board.allFieldNumbers()):
                        n = self.identifierFromFieldNumber(neighbour)
                        nj = self.identifierFromFieldNumber(jumpNeighbour)
                        field.addJumpNeighbour(n, nj) 
                            

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
        return len([x for x in self.allFieldNumbers() if x < fieldNumber])
    
    
    def identifierFromCoord(self, coord):
        return self.identifierFromFieldNumber(self.fieldNumberFromCoord(coord))


    def directionMapper(self, di, dj):
        return di + 17 * dj 
    
    
    def allFieldNumbers(self):
        return sorted([x['fieldNumber'] for x in self.fields])
