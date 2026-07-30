class HalmaField:

    def __init__(self, coord, id=0, fieldNumber=None):
        self.coord = coord
        self.id = id
        self.fieldNumber = fieldNumber
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


