from game.gameManager import ComputedGame
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from env.boardNormalizer import Normalizer

class HalmaEnv(gym.Env):
    def __init__(self):
        self.game = ComputedGame().reset()
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2, 121), dtype=np.float32)
        self.reward_space = spaces.Discrete(121**2)
        self.normalizer = Normalizer(self.game.board)

    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        obs = self.get_observation()
        return obs, {}

    
    def step(self, player, permutationKey, action: int):
        move = self.decode_action(action)
        inverseNormalizedMove = self.normalizer.inverseMove(move, permutationKey)
        self.game.playMove(player, inverseNormalizedMove)
        done = self.game.winner() == player
        reward = 1.0 if done else 0.0
        obs = self.get_observation()
        return obs, reward, done, {}
    
    
    def getPermutation(self, boardState, playerID):
        if playerID == 1:
            if self.needsFlip(boardState, playerID):
                return "player1WithFlip"
            else:
                return "player1WithoutFlip"
        elif playerID == 2:
            if self.needsFlip(boardState, playerID):
                return "player2WithFlip"
            else:
                return "player2WithoutFlip"
        return "Undefined"
            
            
    def needsFlip(self, boardState, playerID):
        xCoords = np.where(boardState == playerID)[0]
        return np.sum(self.normalizer.sumCoordsX(xCoords)) < 0
    
    
    def normalizeBoardState(self, observation, playerID, permutationKey):
        normalized = self.normalizer.permute(observation, permutationKey)
        obs = np.zeros((2, 121), dtype=np.float32)
        obs[0] = (normalized == playerID).astype(np.float32)
        obs[1] = ((normalized != playerID) & (normalized != 0)).astype(np.float32)
        return obs
    
    
    def legalMoves(self):
        player = self.game.currentPlayer()
        return [self.encode_action(x) for x in self.game.board.allValidMoves(player)]
    
    
    def dummyNN(self, moveIDs):
        return random.choice(moveIDs)
    
    
    def render(self):
        pass
    
    
    def get_observation(self):
        boardState, moves, player = self.game.gameObservation()
        permutationKey = self.getPermutation(boardState, player.identifier)
        normalizedBoardState = self.normalizeBoardState(boardState, player.identifier, permutationKey)
        normalizedMoves = self.normalizer.permuteMoves(moves, permutationKey)
        return normalizedBoardState, normalizedMoves, player, permutationKey
    
    
    def playNextMove(self):
        boardState, validMoves, player, permutationKey = self.get_observation()
        moveIDs = [self.encode_action(validMove) for validMove in validMoves]
        action = self.dummyNN(moveIDs)
        self.step(player, permutationKey, action)
        
    
    def decode_action(self, action:int):
        startID = action // 121
        endID = action % 121
        return (startID, endID)
    
    
    def encode_action(self, move):
        return move[0]*121 + move[-1]
    

