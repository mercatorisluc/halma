from collections import deque

class Move: 
    
    def __init__(self, idList, player):
        self.steps = idList
        self.start = self.steps[0]
        self.end = self.steps[-1]
        self.player = player
        self.jumpedOvers = None
                    
    
    def partMoves(self):
        return [(self.steps[i], self.steps[i+1]) for i in range(len(self.steps)-1)]
    
    
    def needsReconstruction(self):
        return self.jumpedOvers is None
    
    
    def reconstructFullMove(self, board):
        if board.fields[self.start].isEmpty():
            startField, endField = (self.end, self.start)
        else:
            startField, endField = (self.start, self.end)
        if not board.isJumpMove(startField, endField):
            self.jumpedOvers = []
            return
        else:
            queue = deque([[startField]])
            validMoves = []
            visited = set()
            visited.add(startField)
            while queue:
                tempWay = queue.popleft()
                jumpPositions = board.getValidJumpFields(tempWay[-1])
                for jumpPosition in jumpPositions:
                    if jumpPosition == endField:
                        validMoves.append(tempWay + [endField])
                    elif jumpPosition not in visited: 
                        visited.add(jumpPosition)
                        queue.append(tempWay + [jumpPosition])
            assert len(validMoves) > 0
            move = min(validMoves, key=len)
            if board.fields[self.start].isEmpty():
                move.reverse()
            self.steps = move
            self.calculateJumpedOverFields(board)
    
    
    def calculateJumpedOverFields(self, board):
        self.jumpedOvers = []
        for partMove in self.partMoves():
            start, end = partMove
            if board.isJumpMove(start, end):
                coordX = (board.fields[start].coord[0] + board.fields[end].coord[0]) / 2
                coordY = (board.fields[start].coord[1] + board.fields[end].coord[1]) / 2
                id = board.fieldPositionsMapper.idFromCoord((coordX, coordY))
                self.jumpedOvers.append(id)
    
    def fullStepsList(self):
        fullStepsList = []
        fullStepsList.append(self.start)
        substeps = list(self.steps)
        substeps.pop(0)
        substeps.pop(-1)
        for i in range(len(self.jumpedOvers)):
            fullStepsList.append(self.jumpedOvers[i])
            if i < len(substeps):
                fullStepsList.append(substeps[i])
        fullStepsList.append(self.end)
        return fullStepsList
        
      
    
