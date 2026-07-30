import math


class BoardProjector:
    """Maps between board coordinates and screen pixels, and owns rotation.

    This is the single source of truth for the current view rotation
    (``flipAngle``). Both rendering and click hit-testing go through the same
    ``coordToPos`` / ``posToCoord`` tables, so a rotated board always projects
    and un-projects consistently.
    """

    def __init__(self, scale=40, centerX=300, centerY=300):
        self.scale = scale
        self.centerX = centerX
        self.centerY = centerY
        self.flipAngle = 0
        self.edges = []
        self.coordToPos = {}
        self.posToCoord = {}

    def computeMappers(self, fields):
        flippedCoords = [f.coord for f in fields]
        for a in (0, 60, 120, 180, 240, 300):
            self.coordToPos[a] = {}
            self.posToCoord[a] = {}
            for i, coord in enumerate([f.coord for f in fields]):
                coordProjected = self.xyPositions(flippedCoords[i])
                self.coordToPos[a][coord] = coordProjected
                self.posToCoord[a][coordProjected] = coord
            flippedCoords = self.flipCoords60Deg(flippedCoords)

    def computeEdges(self, board):
        self.edges = []
        idEdges = []
        for field in board.fields:
            idA = field.id
            idEdges.extend([(idA, idB) for idB in field.neighbours if (idB, idA) not in idEdges])
        for edge in idEdges:
            coordA = board.coordFromId(edge[0])
            coordB = board.coordFromId(edge[1])
            self.edges.append((self.visualPosition(coordA), self.visualPosition(coordB)))

    def xyPositions(self, coords):
        x, y = coords
        projectedX = self.centerX + (y * self.scale / 2) + (x * self.scale)
        projectedY = self.centerY - (y * self.scale * math.sqrt(3) / 2)
        return (projectedX, projectedY)

    def visualPosition(self, coord):
        return self.coordToPos[self.flipAngle][coord]

    def visualCoord(self, pos):
        return self.posToCoord[self.flipAngle][pos]

    def fieldAtPixel(self, board, clickPos, radius):
        """Return the field whose projected centre is within ``radius`` of the
        click, or ``None``."""
        for pos in self.coordToPos[self.flipAngle].values():
            dist = math.hypot(pos[0] - clickPos[0], pos[1] - clickPos[1])
            if dist <= radius:
                id = board.idFromCoord(self.visualCoord(pos))
                return board.fields[id]

    def flipBoardByDegree(self, degrees):
        self.flipAngle = (self.flipAngle + degrees) % 360

    def flipCoords60Deg(self, coords):
        # took me some time to realize
        return [(-y, x + y) for x, y in coords]
