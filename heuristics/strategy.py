import random

class Strategy: 
    def __init__(self, strategyName):
        self.strategyName = strategyName
        
              
    def advancedDist(self, board, player):
        distanceScore = board.advancedDistanceScore(player)
        homeBonus = board.homeBonusScore(player)
        return (distanceScore + homeBonus) / 2
    
    
    def simpleDist(self, board, player):
        distanceScore = board.simpleDistanceScore(player)
        homeBonus = board.homeBonusScore(player)
        return (distanceScore + homeBonus) / 2
    
    
    def chooseRandom(self, board, player):
        return 1
    
    
    def sparsity(self, board, player):
        distanceScore = board.advancedDistanceScore(player)
        sparsityScore = board.sparsityScore(player)
        playerSparsityScore = board.playerSparsityScore(player)
        jumpScore = board.potentialJumpScore(player)
        homeBonus = board.homeBonusScore(player)
        return distanceScore + sparsityScore + playerSparsityScore + jumpScore + homeBonus  
            
        
    def scoringFunction(self, board, player):
        strategy_map = {
            "advancedDistScore": self.advancedDist, 
            "sparsityScore": self.sparsity, 
            "simpleDistScore": self.simpleDist, 
            "random": self.chooseRandom
        }
        score = strategy_map.get(self.strategyName)
        return score(board, player)
    
    
    def bestMove(self, moves, board, player):
        moveValuePairs = {}
        for move in moves:
            board.applyMoveForPlayer(move, player)
            moveValuePairs[tuple(move)] = self.scoringFunction(board, player)
            board.applyMoveForPlayer((move[-1], move[0]), player)
        minValue = min(moveValuePairs.values())
        possibleMoves = [key for key, value in moveValuePairs.items() if value == minValue]
        chosenMove = random.choice(possibleMoves)
        return chosenMove