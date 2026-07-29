from env.halmaEnv import HalmaEnv
from visual.gameVisualization import GameVisualization
from game.gameManager import InteractiveGame, ComputedGame

if __name__ == "__main__":
    game = InteractiveGame()
    game.initStandardGame()
    visualGame = GameVisualization(game)
    visualGame.visualizeGame()
    
    
    
    
"""    env = HalmaEnv()
    print('start')
    for i in range(1): 
        print('start move ' + str(i+1))
        env.game.printBoard()
        env.playNextMove()
    game = env.game
    GameVisualization = GameVisualization(game)
    GameVisualization.visualizeGame()"""
    