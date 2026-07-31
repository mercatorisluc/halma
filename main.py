from game.gameManager import InteractiveGame
from visual.gameVisualization import GameVisualization

if __name__ == "__main__":
    gameManager = InteractiveGame()
    gameManager.initStandardGame()
    visualization = GameVisualization(gameManager)
    visualization.visualizeGame()
