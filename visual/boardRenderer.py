import pygame


class BoardRenderer:
    """Draws the board, pieces, home bases and move highlights with pygame.

    Holds no game state of its own: it reads the game/board through the shared
    references and projects coordinates via the ``BoardProjector``. State that
    varies per frame (the playback position, the current interaction) is passed
    in by the orchestrator.
    """

    def __init__(self, screen, colorizer, projector, game):
        self.screen = screen
        self.colorizer = colorizer
        self.projector = projector
        self.game = game

    def drawBoard(self, board):
        self.screen.fill(self.colorizer.white())
        self.drawHomeBases()
        self.drawEdges()
        self.drawFields(board.fields, 'S')

    def drawEdges(self):
        for edge in self.projector.edges:
            pygame.draw.line(self.screen, self.colorizer.black(), edge[0], edge[1], 2)

    def drawFields(self, fields, flag):
        for field in fields:
            self.drawField(field, flag)

    def drawTriangle(self, polygon, color):
        projectedPoly = [self.projector.visualPosition(coord) for coord in polygon]
        pygame.draw.polygon(self.screen, color, projectedPoly)

    def drawHomeBases(self):
        p1Polygons = [[(0, -4), (4, -4), (4, -8)], [(0, 4), (-4, 4), (-4, 8)]]
        p2Polygons = [[(0, -4), (-4, 0), (-4, -4)], [(4, 0), (4, 4), (0, 4)]]
        p3Polygons = [[(-4, 0), (-4, 4), (-8, 4)], [(4, -4), (4, 0), (8, -4)]]
        for g in p1Polygons:
            self.drawTriangle(g, self.colorizer.backgroundColors[0])
        for r in p2Polygons:
            self.drawTriangle(r, self.colorizer.backgroundColors[1])
        for b in p3Polygons:
            self.drawTriangle(b, self.colorizer.backgroundColors[2])

    def drawField(self, field, flag):
        x, y = self.projector.visualPosition(field.coord)
        fillColor = self.colorizer.playerColorsDict[field.playerID]
        if flag == 'S':
            self.drawCircle((x, y), fillColor, 11, True)
        elif flag == 'M':
            self.drawCircle((x, y), self.colorizer.colorForFlag(flag), 13, False)
            self.drawCircle((x, y), fillColor, 11, True)
        elif flag == 'C':
            self.drawCircle((x, y), self.colorizer.colorForFlag(flag), 14, False)
            self.drawCircle((x, y), fillColor, 11, True)
        elif flag == 'P':
            self.drawCircle((x, y), self.colorizer.colorForFlag(flag), 13, False)
            self.drawCircle((x, y), fillColor, 11, True)

    def drawCircle(self, pos, fillColor, size, hasBorder):
        if hasBorder:
            pygame.draw.circle(self.screen, self.colorizer.black(), pos, size, 0)
            pygame.draw.circle(self.screen, fillColor, pos, size - 2, 0)
        else:
            pygame.draw.circle(self.screen, fillColor, pos, size, 0)

    def drawMove(self, move, flag):
        board = self.game.board
        proj = [self.projector.visualPosition(board.fields[id].coord) for id in move]
        for i in range(len(proj) - 1):
            pygame.draw.line(
                self.screen, self.colorizer.colorForFlag(flag), proj[i], proj[i + 1], 6)
        self.drawFields([board.fields[id] for id in move], flag)

    def drawValidMoves(self, start, validHumanMoves):
        for move in validHumanMoves:
            if move[0] == start:
                move = self.game.createMoveForPlayer(move, self.game.currentPlayer())
                move.reconstructFullMove(self.game.board)
                self.drawMove(move.fullStepsList(), 'P')

    def drawLastMove(self, moveTraveler):
        if moveTraveler >= 1:
            if self.game.moves[moveTraveler - 1].needsReconstruction():
                self.game.moves[moveTraveler - 1].reconstructFullMove(self.game.board)
            move = self.game.moves[moveTraveler - 1]
            self.drawMove(move.fullStepsList(), 'M')

    def drawInteractiveElements(self, clickedField, waitingForHumanMove, validHumanMoves):
        if clickedField is not None:
            self.drawField(clickedField, 'C')
            if waitingForHumanMove:
                self.drawValidMoves(clickedField.id, validHumanMoves)
