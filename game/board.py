from collections import deque
from game.field import HalmaField
from game.fieldPositionsMapper import FieldPositionsMapper
import numpy as np

class HalmaBoard:
    """The 121-field star board: piece placement, move generation and the
    heuristic scoring functions used by the bots.

    Fields are addressed by integer id (0-120); ``fieldPositionsMapper`` maps
    between ids and axial coordinates. ``distanceMatrix`` caches the board
    distance between every pair of fields.
    """

    def __init__(self):
        self.fields = [None] * 121
        self.fieldPositionsMapper = FieldPositionsMapper()
        self.distanceMatrix = [[0 for _ in range(121)] for _ in range(121)]
        
        
    def setFieldPositionsMapper(self, fieldPositionsList):
        for fieldPosition in fieldPositionsList:
            self.fieldPositionsMapper.addFieldPosition(fieldPosition)
        
        
    def setField(self, coord, id, fieldNumber):
        self.fields[id] = HalmaField(coord, id, fieldNumber)
        
        
    def calculateDistanceMatrix(self):
        for i in range(121):
            coordA = self.fieldPositionsMapper.idDict[i]["coord"]
            for j in range(121):
                coordB = self.fieldPositionsMapper.idDict[j]["coord"]
                self.distanceMatrix[i][j] = self.distance(coordA, coordB)
                
                                 
    def placePiece(self, id, playerID):
        self.fields[id].playerID = playerID
            
            
    def removePiece(self, id):
        assert self.fields[id].playerID != 0
        self.fields[id].removePlayer()
            
            
    def applyMoveForPlayer(self, move, player):
        start, end = move[0], move[-1]
        self.placePiece(end, self.fields[start].playerID)
        self.removePiece(start)
        player.updatePositionWithMove(move)
        self.updatePlayerDistanceScore(player, move) 

                               
    def getValidNeighbourFields(self, id):
        movePositions = []
        for neighbour in self.fields[id].neighbours:
            if self.fields[neighbour].playerID == 0: 
                movePositions.append(neighbour)
        return movePositions
    
    
    def getValidJumpFields(self, id):
        movePositions = []
        for neighbour in self.fields[id].jumpNeighbours:
            if (self.fields[neighbour].playerID != 0): 
                jumpNeighbour = self.fields[id].jumpNeighbours[neighbour]
                if self.fields[jumpNeighbour].playerID == 0:
                    movePositions.append(jumpNeighbour)   
        return movePositions
    
    
    def isJumpMove(self, start, end):
        return (end not in self.fields[start].neighbours)
    
    
    def allValidMoves(self, player):
        """All legal moves for ``player`` as ``(start, end)`` id pairs.

        A single step lands on an empty neighbour; jumps chain over occupied
        fields (BFS over reachable jump landings). Only moves whose destination
        the player is allowed to occupy are kept.
        """
        moves = []
        for start in player.positions:
            queue = deque([(start)])
            visited = set()
            visited.add(start)
            visited.update(self.getValidNeighbourFields(start))          
            while queue:
                curr_position = queue.popleft()
                jump_positions = self.getValidJumpFields(curr_position)
                for jump_position in jump_positions:
                    if jump_position not in visited:                  
                        visited.add(jump_position)
                        queue.append(jump_position)
            visited.remove(start)
            moves.extend([(start, end) for end in visited])
        return [move for move in moves if self.fields[move[-1]].allows(player)]
        
    
    def allValidMovesWithWay(self, player):
        """Like :meth:`allValidMoves`, but each move carries its full path
        ``[start, ..., end]`` instead of just the endpoints.

        The path (the sequence of jump landings) is what the visualization
        needs to draw a multi-hop jump and what :class:`~game.move.Move` uses to
        reconstruct the intermediate steps.
        """
        allMoves = []
        for position in player.positions:
            queue = deque([[position]])
            moves = []
            visited = set()
            moves.extend([[position, x] for x in self.getValidNeighbourFields(position)]) 
            visited.update(self.getValidNeighbourFields(position))
            visited.add(position)
            while queue:
                tempWay = queue.popleft()
                jumpPositions = self.getValidJumpFields(tempWay[-1])
                for jumpPosition in jumpPositions:
                    if jumpPosition not in visited: 
                        visited.add(jumpPosition)
                        moves.append(tempWay + [jumpPosition])
                        queue.append(tempWay + [jumpPosition])
            allMoves.extend(moves)
        validMoves = [move for move in allMoves if self.fields[move[-1]].allows(player)]
        return validMoves
        
    
    def allFieldNumbers(self):
        return np.array([x.fieldNumber for x in self.fields])
    
    
    def distance(self, coordA, coordB):
        """Board distance between two axial hex coordinates.

        On this grid a diagonal counts as one step when the two axes move in
        opposite directions (hence the sign check); otherwise the moves add up.
        """
        distanceScore = 0
        verticalDist = coordA[0] - coordB[0]
        horizontalDist = coordA[1] - coordB[1]
        if (verticalDist * horizontalDist) < 0:
            distanceScore += max(abs(verticalDist), abs(horizontalDist))
        else: 
            distanceScore += abs(verticalDist) + abs(horizontalDist)
        return distanceScore
    
    
    def simpleDistanceScore(self, player):
        # Total distance of the player's pieces from their own home base;
        # lower is better (pieces have advanced further). Scaled by 16.
        hb = player.homeBase
        score = 0
        for id in player.positions:  
            score += self.distanceMatrix[id][hb]
        #scaling factor of 16
        return score/16
    
    
    def calculatePlayerDistanceScore(self, player):
        score = 0
        for x in player.nonArrived:
            score += sum([self.distanceMatrix[x][y] for y in player.openEndPositions])
        return score
    
    
    def playerDistanceScore(self, player):
        score = player.distanceScore
        score /=  max((len(player.nonArrived) * len(player.openEndPositions)), 1)
        #scaling factor of 12
        return score/12
    
    
    def advancedDistanceScore(self, player):
        # Blend of the cached distance-to-open-targets score and the simple
        # distance-to-home score; lower is better.
        return ((self.playerDistanceScore(player)) + (self.simpleDistanceScore(player))) / 2
    
    
    def updatePlayerDistanceScore(self, player, move):
        # Incrementally maintain player.distanceScore after a move instead of
        # recomputing over all pieces: adjust only the terms that changed as a
        # piece left `start` and arrived at `end` (with corrections for moves
        # into or out of the target end positions).
        start, end = move[0], move[-1]
        toAdd, toSubtract = (0, 0)
        if start not in player.endPositions: 
            toSubtract += sum([self.distanceMatrix[x][start] for x in player.openEndPositions])
        else: 
            toAdd += sum([self.distanceMatrix[x][start] for x in player.nonArrived])
        if end in player.endPositions:
            toSubtract += sum([self.distanceMatrix[x][end] for x in player.nonArrived])
        else:
            toAdd += sum([self.distanceMatrix[x][end] for x in player.openEndPositions])
        #missed
        if (start in player.endPositions) and (end not in player.endPositions): 
            toSubtract += self.distanceMatrix[start][end]
        #missed
        if (start not in player.endPositions) and (end in player.endPositions):
            toSubtract += self.distanceMatrix[start][end]
        player.distanceScore +=  (toAdd - toSubtract)
            
    
    def sparsityScore(self, player):
        # Rewards keeping pieces loosely clustered: penalises each piece by how
        # far its share of occupied neighbours is from an ideal 0.75. Lower is
        # better. Scaled by 15 (pieces per player).
        score = 0
        for id in player.positions:
            neighbours = set(self.fields[id].neighbours)
            idScore = 0
            for neighbour in neighbours:
                if not self.fields[neighbour].isEmpty():
                    idScore += 1
            score += 4/3 * abs(0.75 - (idScore / len(neighbours)))
        return score/15
    
    
    def playerSparsityScore(self, player):
        sparsityScore = 0
        centerDist = np.mean([self.distanceMatrix[p][player.homeBase] for p in player.positions])
        maxDist = max([(self.distanceMatrix[p][player.homeBase] - centerDist) for p in player.positions])
        return maxDist/12
    
    
    def potentialJumpScore(self, player):
        # Rewards positions that have available jumps (a piece to hop over onto
        # an empty landing); lower score = more jump potential. Scaled by 15.
        score = 0
        for id in player.positions:
            idScore = 0
            dicts = self.fields[id].jumpNeighbours.items()
            for jumpOver, jumpOn in self.fields[id].jumpNeighbours.items():
                if (not self.fields[jumpOver].isEmpty()) and (self.fields[jumpOn].isEmpty()):
                    idScore += 1
            score += (1 - idScore/len(dicts))
        return score/15
    
    
    def homeBonusScore(self, player):
        # Fraction of pieces NOT yet on a target field; lower is better
        # (0 when all 15 pieces are home).
        return 1 - len(player.positions & player.endPositions)/15
    
    
    def boardState(self):
        return np.array([field.playerID for field in self.fields])

                    
