from heuristics.strategy import Strategy

class HalmaPlayer:
    """A player's pieces and goal, plus the derived sets the bots score on.

    ``positions`` are the pieces' current fields; ``endPositions`` the target
    base. ``nonArrived`` (pieces not yet home) and ``openEndPositions`` (target
    fields still empty) are kept in sync on every move so the heuristics don't
    have to recompute them.
    """

    def __init__(self, identifier):
        self.identifier = identifier
        self.positions = set()
        self.startPositions = set()
        self.endPositions = set()
        self.openEndPositions = set()
        self.nonArrived = set()
        self.homeBase = None
        self.distanceScore = 0
        
    
    def setHomeBase(self, position):
        self.homeBase = position
                               
        
    def prepareForGameStart(self, board):
        for id in self.startPositions:
            board.fields[id].playerID = self.identifier
        self.positions.update(self.startPositions)
        self.nonArrived = self.positions - self.endPositions
        self.openEndPositions = self.endPositions - self.positions
        self.distanceScore = board.calculatePlayerDistanceScore(self)


    def updatePositionWithMove(self, move):
        start, end = move[0], move[-1]
        self.positions.remove(start)
        self.positions.add(end)
        self.openEndPositions.discard(end)
        self.openEndPositions.add(start)
        self.openEndPositions &= self.endPositions
        self.nonArrived.add(end)
        self.nonArrived.discard(start)
        self.nonArrived -= self.endPositions
               
    
    def isWinning(self):
        return (self.positions == self.endPositions)
        
        
    def setStartPositions(self, positions):
        self.startPositions = set(positions)
        
        
    def setEndPositions(self, positions):
        self.endPositions = set(positions)
        
        
    
class HumanPlayer(HalmaPlayer):
    """A player whose moves come from the visualization's click handling."""

    def __init__(self, identifier):
        super().__init__(identifier)

    def isHuman(self):
        return True


class Computer(HalmaPlayer):
    """A bot player that picks moves via a named heuristic strategy."""

    def __init__(self, identifier, strategyName):
        super().__init__(identifier)
        self.strategy = Strategy(strategyName)

    def isHuman(self):
        return False

    def chooseMove(self, moves, board):
        # Delegate the choice to the configured heuristic strategy.
        return self.strategy.bestMove(moves, board, self)
