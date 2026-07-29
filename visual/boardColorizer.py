import pygame
import math
from collections import defaultdict

class BoardColorizer():
    
    def __init__(self):
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.VIOLET = (150, 0, 255)
        self.GREY = (230, 230, 230)
        self.YELLOW = (238, 255, 65)
        self.playerColorsDict = defaultdict(lambda: self.GREY)
        self.backgroundColors = [(239, 154, 154), (129, 212, 250), (165, 214, 167)]

        
    def setPlayerWithColors(self, playerID, color):
        self.playerColorsDict[playerID] = color
        
    
    def playerColor(self, playerID):
        return self.playerColorsDict[playerID]
    
    
    def colorForFlag(self, flag):
        if flag == 'M':
            return self.moveHighlighting()
        elif (flag == 'P') or (flag == 'C'):
            return self.clickHighlighting()
    
    
    def moveHighlighting(self):
        return self.VIOLET
    
    
    def clickHighlighting(self):
        return self.YELLOW
    
    
    def playerColor(self, playerID):
        return self.playerColorsDict[playerID]
    
    
    def black(self):
        return self.BLACK
    
    
    def white(self):
        return self.WHITE
    
