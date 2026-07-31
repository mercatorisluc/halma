import numpy as np


class Normalizer:
    def __init__(self, board):
        self.boardStructure = np.array(
            [(x.coord[0], x.coord[1], x.id) for x in board.fields],
            dtype=[("x", int), ("y", int), ("id", int)],
        )
        self.permutations = self.initPermutations()
        self.coordsX = self.initCoordsX()

    def initPermutations(self):
        permutations = {}
        permutations["player1WithoutFlip"] = np.arange(len(self.boardStructure))
        permutations["player1WithFlip"] = self.turnAndFlipBoardForPlayer1()
        permutations["player2WithoutFlip"] = self.turnBoardForPlayer2()
        permutations["player2WithFlip"] = self.turnAndFlipBoardForPlayer2()
        permutations["player1WithoutFlipInv"] = np.argsort(permutations["player1WithoutFlip"])
        permutations["player1WithFlipInv"] = np.argsort(permutations["player1WithFlip"])
        permutations["player2WithoutFlipInv"] = np.argsort(permutations["player2WithoutFlip"])
        permutations["player2WithFlipInv"] = np.argsort(permutations["player2WithFlip"])
        return permutations

    def initCoordsX(self):
        return [x for x, _, _ in self.boardStructure]

    def turnBoard120DegreesPermutation(self):
        permutation = [(-x - y, x, _) for x, y, _ in self.boardStructure]
        permutation.sort(key=lambda x: (x[1], x[0]))
        return np.array([x[2] for x in permutation])

    def turnBoard60DegreesPermutation(self):
        permutation = [(-y, x + y, _) for x, y, _ in self.boardStructure]
        permutation.sort(key=lambda x: (x[1], x[0]))
        return np.array([x[2] for x in permutation])

    def flipAlongYAxisPermutation(self):
        permutation = [(-x, x + y, _) for (x, y, _) in self.boardStructure]
        permutation.sort(key=lambda x: (x[1], x[0]))
        return np.array([x[2] for x in permutation])

    def turnAndFlipBoardForPlayer1(self):
        flippedAlongYAxis = self.boardStructure[self.flipAlongYAxisPermutation()]
        permutation = flippedAlongYAxis[self.turnBoard60DegreesPermutation()]
        return np.argsort([x[2] for x in permutation])

    def turnBoardForPlayer2(self):
        permutation = self.boardStructure[self.turnBoard120DegreesPermutation()]
        return np.argsort([x[2] for x in permutation])

    def turnAndFlipBoardForPlayer2(self):
        turn120Degrees = self.boardStructure[self.turnBoard120DegreesPermutation()]
        flippedAlongYAxis = turn120Degrees[self.flipAlongYAxisPermutation()]
        permutation = flippedAlongYAxis[self.turnBoard60DegreesPermutation()]
        return np.argsort([x[2] for x in permutation])

    def permute(self, observation, permutationKey):
        obs = np.array(observation)
        perm = np.array(self.permutations[permutationKey])
        return obs[perm]

    def permuteMoves(self, moves, permutationKey):
        permutation = self.permutations[permutationKey]
        return [(permutation[start], permutation[end]) for start, end in moves]

    def inverseMove(self, move, permutationKey):
        inversePermutation = self.permutations[permutationKey + "Inv"]
        return inversePermutation[move[0]], inversePermutation[move[1]]

    def sumCoordsX(self, ids):
        return [self.coordsX[id] for id in ids]
