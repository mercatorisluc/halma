from heuristics.strategy import Strategy

class HalmaPlayer: 
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
    
    def __init__(self, identifier):
        super().__init__(identifier)
        
    
    def isHuman(self):
        return True
    
        

class Computer(HalmaPlayer): 
    
    def __init__(self, identifier, strategyName):
        super().__init__(identifier)
        self.strategy = Strategy(strategyName)
        
        
    def isHuman(self):
        return False
    
    
    def chooseMove(self, moves, board):
        return self.strategy.bestMove(moves, board, self)
    
        

    
        
    
    