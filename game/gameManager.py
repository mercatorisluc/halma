from __future__ import annotations

import random

from game.board import HalmaBoard
from game.boardTypes import AnyMove, FieldId, MovePath, PlayerId
from game.initializer import Initializer
from game.move import Move
from game.player import Computer, HalmaPlayer, HumanPlayer


class HalmaGame:
    """Base game engine: owns the board, the players, the turn order and the
    recorded move history, and enforces the rules (turn rotation, win check).

    Subclasses (:class:`ComputedGame`, :class:`InteractiveGame`) only differ in
    which players they seat.
    """

    # Cut a bot-vs-bot game off after this many moves so a stalemate between
    # two heuristics cannot loop forever.
    MAX_MOVES = 250

    def __init__(self) -> None:
        self.initializer = Initializer()
        self.board: HalmaBoard = HalmaBoard()
        self.players: list[HalmaPlayer] = []
        self.playOrder: list[HalmaPlayer] = []
        self.moves: list[Move] = []

    def initGame(self, players: list[HalmaPlayer]) -> None:
        self.initializer.initializeBoard(self.board)
        self.initPlayers(players)
        self.initializer.initPermissions(self.board, self.players)
        self.moves = []

    def initStandardGame(self) -> None:
        # Subclasses seat their own players (bots and/or a human).
        raise NotImplementedError

    def reset(self) -> HalmaGame:
        self.initStandardGame()
        return self

    def initPlayers(self, players: list[HalmaPlayer]) -> None:
        self.players = []
        if len(players) > 0:
            self.setPlayerPositions(players[0], self.initializer.player1Positions(self.board))
        if len(players) > 1:
            self.setPlayerPositions(players[1], self.initializer.player2Positions(self.board))
        if len(players) > 2:
            self.setPlayerPositions(players[2], self.initializer.player3Positions(self.board))
        for player in players:
            self.players.append(player)
            player.prepareForGameStart(self.board)
        self.computePlayersOrder()

    def setPlayerPositions(
        self, player: HalmaPlayer, positionsTriplet: tuple[list[FieldId], list[FieldId], FieldId]
    ) -> None:
        startPositions, endPositions, homeBase = positionsTriplet
        player.setStartPositions(startPositions)
        player.setEndPositions(endPositions)
        player.setHomeBase(homeBase)

    def computePlayersOrder(self) -> None:
        self.playOrder = random.sample(self.players, k=len(self.players))

    def gameLength(self) -> int:
        return len(self.moves)

    def play(self) -> PlayerId | None:
        """Play the game out to a winner and return that player's identifier.

        Returns ``None`` if no one has won within ``MAX_MOVES`` (a draw by
        exhaustion). Deliberately silent: this runs thousands of times during
        bot simulation, so the caller decides whether to report anything.
        """
        for _ in range(self.MAX_MOVES):
            player = self.currentPlayer()
            self.playNextMove(player)
            if self.winner() is not None:
                return self.winner()
        return None

    def playNextMove(self, player: HalmaPlayer) -> None:
        chosenMove = self.getNextMove(player)
        self.playMove(player, chosenMove)

    def createMoveForPlayer(self, move: AnyMove, player: HalmaPlayer) -> Move:
        return Move(move, player)

    def playMove(self, player: HalmaPlayer, move: AnyMove) -> None:
        self.board.applyMoveForPlayer(move, player)
        self.moves.append(self.createMoveForPlayer(move, player))

    def getNextMove(self, player: HalmaPlayer) -> MovePath:
        validMoves = self.board.allValidMovesWithWay(player)
        return player.chooseMove(validMoves, self.board)

    def winner(self) -> PlayerId | None:
        """Return the identifier of the winning player, or ``None``.

        A player wins either by getting all their pieces onto their target
        home base, or when their target base is completely full (which can
        include opponent pieces blocking it — see
        :meth:`playerIsWinningByBlockedFields`).
        """
        for player in self.players:
            if player.isWinning() or (self.playerIsWinningByBlockedFields(player)):
                return player.identifier
        return None

    def playerIsWinningByBlockedFields(self, player: HalmaPlayer) -> bool:
        # If every field of the target base is occupied (by anyone), the player
        # can no longer be blocked out and counts as home.
        targetFields = [self.board.fields[i] for i in player.endPositions]
        occupied = [field for field in targetFields if not field.isEmpty()]
        return len(occupied) == len(player.endPositions)

    def currentPlayer(self) -> HalmaPlayer:
        # Turn order rotates by move count over the (randomised) play order.
        index = self.gameLength() % len(self.playOrder)
        return self.playOrder[index]

    def gameObservation(self):
        player = self.currentPlayer()
        boardState = self.board.boardState()
        validMoves = self.board.allValidMoves(player)
        return boardState, validMoves, player

    def printBoard(self) -> None:
        boardMatrix = [[" " for _ in range(26)] for _ in range(18)]
        for field in self.board.fields:
            x, y = field.coord
            projectedX = 13 + y + (x * 2)
            projectedY = 9 - y
            boardMatrix[projectedY][projectedX] = field.playerID
        for row in boardMatrix:
            print(" ".join(str(x) for x in row))


class ComputedGame(HalmaGame):
    """A game played entirely by bots (used for simulation and the RL env)."""

    def __init__(self):
        super().__init__()

    def initStandardGame(self) -> None:
        bot1 = Computer(1, "advancedDistScore")
        bot2 = Computer(2, "sparsityScore")
        super().initGame([bot1, bot2])

    def init3PlayerGame(self) -> None:
        bot1 = Computer(1, "advancedDistScore")
        bot2 = Computer(2, "sparsityScore")
        bot3 = Computer(3, "advancedDistScore")
        super().initGame([bot1, bot2, bot3])

    def isHumanGame(self) -> bool:
        return False


class InteractiveGame(HalmaGame):
    """A game with a human player, driven through the visualization."""

    def __init__(self):
        super().__init__()

    def initStandardGame(self) -> None:
        human = HumanPlayer("HumanPlayer")
        bot = Computer("Bot2", "sparsityScore")
        super().initGame([human, bot])

    def init3PlayerGame(self) -> None:
        bot1 = Computer("Bot1", "advancedDistScore")
        bot2 = Computer("Bot2", "sparsityScore")
        human = HumanPlayer("HumanPlayer")
        super().initGame([bot1, bot2, human])

    def isHumanGame(self) -> bool:
        return True
