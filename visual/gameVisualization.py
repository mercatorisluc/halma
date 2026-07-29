import pygame
import math
from collections import defaultdict
from visual.boardColorizer import BoardColorizer

class GameVisualization():
    def __init__(self, game):
        pygame.init()
        self.WIDTH, self.HEIGHT = 600, 600
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.scale, self.centerX, self.centerY = (40, 300, 300)
        self.clock = pygame.time.Clock()
        self.colorizer = BoardColorizer()
        self.flipAngle = 0
        
        self.game = game
        self.moveTraveler = 0
        self.edges = []
        self.coordToPos = {}
        self.posToCoord = {}
        
        self.clickedField = None
        self.waitingForHumanMove = False
        self.humanMove = None
        self.validHumanMoves = None
        
    
    def computeEdges(self, board):
        self.edges = []
        idEdges = []
        for field in board.fields: 
            idA = field.id
            idEdges.extend([(idA, idB) for idB in field.neighbours if (idB, idA) not in idEdges])
        for edge in idEdges:
            coordA = board.fieldPositionsMapper.coordFromId(edge[0])
            coordB = board.fieldPositionsMapper.coordFromId(edge[1])
            self.edges.append((self.visualPosition(coordA), self.visualPosition(coordB)))
                    
            
    def computeMappers(self, fields):            
        flippedCoords = [f.coord for f in fields]
        for a in (0, 60, 120, 180, 240, 300):
            self.coordToPos[a] = {}
            self.posToCoord[a] = {}
            for i, coord in enumerate([f.coord for f in fields]):
                coordProjected = self.xyPositions(flippedCoords[i])
                self.coordToPos[a][coord] = coordProjected
                self.posToCoord[a][coordProjected] = coord
            flippedCoords = self.flipCoords60Deg(flippedCoords)
            
    
    def xyPositions(self, coords):
        x, y = coords
        projectedX = self.centerX + (y * self.scale/2) + (x * self.scale)
        projectedY = self.centerY - (y * self.scale * math.sqrt(3) / 2)        
        return (projectedX, projectedY)
    
    
    def visualPosition(self, coord):
        return self.coordToPos[self.flipAngle][coord]
    
    
    def visualCoord(self, pos):
        return self.posToCoord[self.flipAngle][pos]
        
    
    def drawEdges(self):
        for edge in self.edges:
            pygame.draw.line(self.screen, self.colorizer.black(), edge[0], edge[1], 2)
            
            
    def drawFields(self, fields, flag):
        for field in fields:
            self.drawField(field, flag)
            
            
    def drawTriangle(self, polygon, color):
        projectedPoly = [self.visualPosition(coord) for coord in polygon]
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
        x, y = self.visualPosition(field.coord)
        fillColor = self.colorizer.playerColorsDict[field.playerID]
        if flag == 'S':
            self.drawCircle((x, y), fillColor, 11, True)
        elif flag == 'M':
            self.drawCircle((x,y), self.colorizer.colorForFlag(flag), 13, False)
            self.drawCircle((x, y), fillColor, 11, True)
        elif flag == 'C':
            self.drawCircle((x,y), self.colorizer.colorForFlag(flag), 14, False)
            self.drawCircle((x, y), fillColor, 11, True)
        elif flag == 'P':
            self.drawCircle((x,y), self.colorizer.colorForFlag(flag), 13, False)
            self.drawCircle((x, y), fillColor, 11, True)
     
            
    def drawCircle(self, pos, fillColor, size, hasBorder):
        if hasBorder:
            pygame.draw.circle(self.screen, self.colorizer.black(), pos, size, 0)
            pygame.draw.circle(self.screen, fillColor, pos, size-2, 0)
        else: 
            pygame.draw.circle(self.screen, fillColor, pos, size, 0)
                
    
    def drawMove(self, move, flag):
        board = self.game.board
        proj = [self.visualPosition(board.fields[id].coord) for id in move]
        for i in range(len(proj) - 1):
            pygame.draw.line(self.screen, self.colorizer.colorForFlag(flag), proj[i], proj[i+1], 6)
        self.drawFields([board.fields[id] for id in move], flag)
        
            
    def drawBoard(self, board):
        self.screen.fill(self.colorizer.white())
        self.drawHomeBases()
        self.drawEdges()
        self.drawFields(board.fields, 'S')
        
        
    def drawValidMoves(self, start):
        for move in self.validHumanMoves: 
            if move[0] == start:
                move = self.game.createMoveForPlayer(move, self.game.currentPlayer())
                move.reconstructFullMove(self.game.board)
                self.drawMove(move.fullStepsList(), 'P')
                
                
    def drawLastMove(self, halmaGame):
        if self.moveTraveler >= 1:
            if halmaGame.moves[self.moveTraveler - 1].needsReconstruction():
                halmaGame.moves[self.moveTraveler - 1].reconstructFullMove(halmaGame.board)
            move = halmaGame.moves[self.moveTraveler - 1]
            self.drawMove(move.fullStepsList(), 'M') 
            
            
    def drawInteractiveElements(self):
        if self.clickedField is not None:
            self.drawField(self.clickedField, 'C') 
            if self.waitingForHumanMove:
                self.drawValidMoves(self.clickedField.id)

    
    def initializePlayerColors(self, players):
        #green
        player1Colors = ((244, 67, 54))
        self.colorizer.setPlayerWithColors(players[0].identifier, player1Colors)
        #red
        player2Colors = (76, 175, 80)
        self.colorizer.setPlayerWithColors(players[1].identifier, player2Colors)
        #blue
        if (len(players) > 2):
            player3Colors = (51, 153, 255)
            self.colorizer.setPlayerWithColors(players[2].identifier, player3Colors)
        

    def getClickedField(self, clickPos, radius):
        for pos in self.coordToPos[self.flipAngle].values():
            dist = math.hypot(pos[0]-clickPos[0], pos[1]-clickPos[1])
            if dist <= radius:
                id = self.game.board.fieldPositionsMapper.idFromCoord(self.visualCoord(pos))
                return self.game.board.fields[id]
            
    
    def humanMoveIsOkay(self):
        start, end = self.humanMove
        for move in self.validHumanMoves: 
            if (start == move[0]) and (end == move[-1]):
                return move
        return None
    
    
    def adaptToHumanInteraction(self, halmaGame):
        player = halmaGame.currentPlayer()
        self.waitingForHumanMove = player.isHuman()
        if self.waitingForHumanMove:
            if self.validHumanMoves is None:   
                self.validHumanMoves = halmaGame.board.allValidMovesWithWay(player)          
        else:
            self.validHumanMoves = None
    
    
    def handleHumanMove(self, halmaGame):
        move = self.humanMove
        if move:
            chosenMove = self.humanMoveIsOkay()
            if chosenMove:
                player = halmaGame.currentPlayer()
                halmaGame.playMove(player, chosenMove)
                self.moveTraveler += 1
                self.humanMove = None
                self.clickedField = None
            else: 
                self.humanMove = None
    
    
    def handleClickedField(self, clicked):
        if clicked is None:
            self.clickedField = None
            return
        if self.clickedField is None:
            self.clickedField = clicked
        else:
            if self.waitingForHumanMove:
                start = self.clickedField.id
                end = clicked.id
                self.humanMove = (start, end)
                self.clickedField = None
            else:
                self.clickedField = clicked


    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and self.moveTraveler > 0:
                    self.backwardGame()
                if event.key == pygame.K_RIGHT:
                    self.playNextMove()
                if event.key == pygame.K_UP:
                    self.flipBoardByDegree(60)
                if event.key == pygame.K_DOWN:
                    self.flipBoardByDegree(-60)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                clicked = self.getClickedField(event.pos, 11)
                self.handleClickedField(clicked)
        return True


    def playNextMove(self):
        if (self.moveTraveler < len(self.game.moves)):
            self.forwardGame()
        else:
            player = self.game.currentPlayer()
            if not player.isHuman():
                self.game.playNextMove(player)
                self.moveTraveler += 1
    
    
    def gameStateAt(self, moveTravelerPosition):
        assert 0 <= moveTravlerPosition < len(self.game.moves)
        self.goToStartingPosition()
        for _ in range():
            self.forwardGame()
            
            
    def goToStartingPosition(self):
        self.moveTraveler = len(self.game.moves)
        while self.moveTraveler > 0:
            self.backwardGame()

                    
    def forwardGame(self):
        move = self.game.moves[self.moveTraveler]
        self.game.board.applyMoveForPlayer((move.start, move.end), move.player)
        self.moveTraveler += 1
    
        
    def backwardGame(self):
        move = self.game.moves[self.moveTraveler - 1]
        self.game.board.applyMoveForPlayer((move.end, move.start), move.player)
        self.moveTraveler -= 1
    
    
    def prepareGameVisualization(self):
        self.goToStartingPosition()
        self.computeMappers(self.game.board.fields)
        self.computeEdges(self.game.board)
        self.initializePlayerColors(self.game.players)
        
    
    def runGame(self, halmaGame):
        running = True
        while running:
            self.drawBoard(halmaGame.board)
            self.drawLastMove(halmaGame)
            self.drawInteractiveElements()
            running = self.handle_events()
            self.adaptToHumanInteraction(halmaGame)
            if self.waitingForHumanMove:
                self.handleHumanMove(halmaGame)
            if halmaGame.winner() is not None:
               running = False                                    
            pygame.display.flip()
            self.clock.tick(10)
        
        
    def visualizeGame(self):
        self.prepareGameVisualization()
        self.runGame(self.game)
        pygame.quit()
        
        
    def flipBoardByDegree(self, degrees):
        self.flipAngle = (self.flipAngle + degrees) % 360
        
            
    def flipCoords60Deg(self, coords):
        #took me some time to realize
        return [(-y, x+y) for x, y in coords]
    
    