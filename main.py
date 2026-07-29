from game.gameManager import InteractiveGame
from visual.gameVisualization import GameVisualization


if __name__ == "__main__":
    game = InteractiveGame()
    game.initStandardGame()
    visualization = GameVisualization(game)
    visualization.visualizeGame()
