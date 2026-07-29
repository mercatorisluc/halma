import pygame

from visual.boardColorizer import BoardColorizer
from visual.boardProjector import BoardProjector
from visual.boardRenderer import BoardRenderer
from visual.gamePlaybackController import GamePlaybackController
from visual.humanInputHandler import HumanInputHandler


class GameVisualization:
    """pygame front-end for a Halma game.

    Wires together the projector (coordinate maths + rotation), renderer
    (drawing), playback controller (move-history navigation) and input handler
    (human interaction), and runs the main loop. Public entry point is
    ``visualizeGame()``.
    """

    def __init__(self, game):
        pygame.init()
        self.WIDTH, self.HEIGHT = 600, 600
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()

        self.game = game
        self.colorizer = BoardColorizer()
        self.projector = BoardProjector(scale=40, centerX=300, centerY=300)
        self.renderer = BoardRenderer(self.screen, self.colorizer, self.projector, self.game)
        self.playback = GamePlaybackController(self.game)
        self.input = HumanInputHandler(self.playback)

    def initializePlayerColors(self, players):
        # red
        player1Colors = (244, 67, 54)
        self.colorizer.setPlayerWithColors(players[0].identifier, player1Colors)
        # green
        player2Colors = (76, 175, 80)
        self.colorizer.setPlayerWithColors(players[1].identifier, player2Colors)
        # blue
        if len(players) > 2:
            player3Colors = (51, 153, 255)
            self.colorizer.setPlayerWithColors(players[2].identifier, player3Colors)

    def prepareGameVisualization(self):
        self.playback.goToStartingPosition()
        self.projector.computeMappers(self.game.board.fields)
        self.projector.computeEdges(self.game.board)
        self.initializePlayerColors(self.game.players)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and self.playback.moveTraveler > 0:
                    self.playback.backwardGame()
                if event.key == pygame.K_RIGHT:
                    self.playback.playNextMove()
                if event.key == pygame.K_UP:
                    self.projector.flipBoardByDegree(60)
                if event.key == pygame.K_DOWN:
                    self.projector.flipBoardByDegree(-60)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                clicked = self.projector.fieldAtPixel(self.game.board, event.pos, 11)
                self.input.handleClickedField(clicked)
        return True

    def runGame(self, halmaGame):
        running = True
        while running:
            self.renderer.drawBoard(halmaGame.board)
            self.renderer.drawLastMove(self.playback.moveTraveler)
            self.renderer.drawInteractiveElements(
                self.input.clickedField,
                self.input.waitingForHumanMove,
                self.input.validHumanMoves,
            )
            running = self.handle_events()
            self.input.adaptToHumanInteraction(halmaGame)
            if self.input.waitingForHumanMove:
                self.input.handleHumanMove(halmaGame)
            elif self.playback.moveTraveler == len(halmaGame.moves):
                # Bot's turn at the live front of the game: play automatically.
                # While the history is being reviewed (cursor not at the front)
                # this is skipped, so back/forward stepping is unaffected.
                self.playback.playNextMove()
            if halmaGame.winner() is not None:
                running = False
            pygame.display.flip()
            self.clock.tick(10)

    def visualizeGame(self):
        self.prepareGameVisualization()
        self.runGame(self.game)
        pygame.quit()
