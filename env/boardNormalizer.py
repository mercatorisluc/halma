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

    def turnBoard240DegreesPermutation(self):
        """Player 2's actual corner, not player 3's.

        The three home corners sit 120 degrees apart, in the order player1 (0
        degrees) -> player3 (120) -> player2 (240) -- verified against
        Initializer's actual coordinates, not assumed. ``(x, y) -> (y, -x-y)``
        is ``(x, y) -> (-x-y, x)`` composed with itself.
        """
        permutation = [(y, -x - y, _) for x, y, _ in self.boardStructure]
        permutation.sort(key=lambda x: (x[1], x[0]))
        return np.array([x[2] for x in permutation])

    def turnBoard60DegreesPermutation(self):
        permutation = [(-y, x + y, _) for x, y, _ in self.boardStructure]
        permutation.sort(key=lambda x: (x[1], x[0]))
        return np.array([x[2] for x in permutation])

    def turnBoard300DegreesPermutation(self):
        """``(x, y) -> (x+y, -x)``: 60 degrees composed with 240."""
        permutation = [(x + y, -x, _) for x, y, _ in self.boardStructure]
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
        permutation = self.boardStructure[self.turnBoard240DegreesPermutation()]
        return np.argsort([x[2] for x in permutation])

    def turnAndFlipBoardForPlayer2(self):
        """Player 1's own flip is ``rot60`` composed with a Y-axis flip,
        chosen because it maps player 1's corner onto itself -- the same
        corner canonical ids 0-14 always mean, whichever pullback is used.
        Player 2's pullback has to land player 1's *actual* corner (still
        where ids 0-14 point) onto player 2's, the way ``turnBoardForPlayer2``
        already does with a plain 240-degree turn. Composing that after
        player 1's flip does it: ``rot240 . (rot60 . flip) = rot300 . flip``.
        Verified directly against Initializer's coordinates, not assumed.
        """
        flippedAlongYAxis = self.boardStructure[self.flipAlongYAxisPermutation()]
        permutation = flippedAlongYAxis[self.turnBoard300DegreesPermutation()]
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
