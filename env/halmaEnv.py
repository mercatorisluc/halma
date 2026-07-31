import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.boardNormalizer import Normalizer
from game.gameManager import ComputedGame


class HalmaEnv(gym.Env):
    def __init__(self):
        self.game = ComputedGame().reset()
        self.fieldCount = len(self.game.board.fields)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(2, self.fieldCount), dtype=np.float32
        )
        self.reward_space = spaces.Discrete(self.fieldCount**2)
        self.normalizer = Normalizer(self.game.board)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        obs = self.getObservation()
        return obs, {}

    def step(self, player, permutationKey, action: int):
        move = self.decodeAction(action)
        inverseNormalizedMove = self.normalizer.inverseMove(move, permutationKey)
        self.game.playMove(player, inverseNormalizedMove)
        done = self.game.winner() == player
        reward = 1.0 if done else 0.0
        obs = self.getObservation()
        return obs, reward, done, {}

    def getPermutation(self, boardState, playerID):
        """Key of the permutation that maps this player's view to the canonical
        one. Only the two-player game is normalised."""
        if playerID not in (1, 2):
            return "Undefined"
        flip = "WithFlip" if self.needsFlip(boardState, playerID) else "WithoutFlip"
        return f"player{playerID}{flip}"

    def needsFlip(self, boardState, playerID):
        xCoords = np.where(boardState == playerID)[0]
        return np.sum(self.normalizer.sumCoordsX(xCoords)) < 0

    def normalizeBoardState(self, observation, playerID, permutationKey):
        normalized = self.normalizer.permute(observation, permutationKey)
        obs = np.zeros((2, self.fieldCount), dtype=np.float32)
        obs[0] = (normalized == playerID).astype(np.float32)
        obs[1] = ((normalized != playerID) & (normalized != 0)).astype(np.float32)
        return obs

    def legalMoves(self):
        player = self.game.currentPlayer()
        return [self.encodeAction(x) for x in self.game.board.allValidMoves(player)]

    def dummyNN(self, moveIDs):
        return random.choice(moveIDs)

    def render(self):
        pass

    def getObservation(self):
        boardState, moves, player = self.game.gameObservation()
        permutationKey = self.getPermutation(boardState, player.identifier)
        normalizedBoardState = self.normalizeBoardState(
            boardState, player.identifier, permutationKey
        )
        normalizedMoves = self.normalizer.permuteMoves(moves, permutationKey)
        return normalizedBoardState, normalizedMoves, player, permutationKey

    def playNextMove(self):
        _, validMoves, player, permutationKey = self.getObservation()
        moveIDs = [self.encodeAction(validMove) for validMove in validMoves]
        action = self.dummyNN(moveIDs)
        self.step(player, permutationKey, action)

    def decodeAction(self, action: int):
        startID = action // self.fieldCount
        endID = action % self.fieldCount
        return (startID, endID)

    def encodeAction(self, move):
        return move[0] * self.fieldCount + move[-1]
