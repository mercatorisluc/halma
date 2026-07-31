class HumanInputHandler:
    """Tracks the human player's in-progress interaction and turns a pair of
    clicks (from field, to field) into a validated move.

    Owns the interaction state: the currently clicked field, whether we are
    waiting for a human move, the pending (start, end) selection, and the list
    of valid moves the human may choose from. When a valid move is played it
    advances the playback cursor so rendering stays in sync.
    """

    def __init__(self, playback):
        self.playback = playback
        self.clickedField = None
        self.waitingForHumanMove = False
        self.humanMove = None
        self.validHumanMoves = None

    def matchingValidMove(self):
        """The full-path move whose endpoints match the clicked pair, if legal.

        The click gives only ``(start, end)``; playing it needs the whole jump
        path, so the pair is looked up among the legal moves rather than
        merely validated.
        """
        start, end = self.humanMove
        for move in self.validHumanMoves:
            if (start == move[0]) and (end == move[-1]):
                return move
        return None

    def adaptToHumanInteraction(self, game):
        player = game.currentPlayer()
        self.waitingForHumanMove = player.isHuman()
        if self.waitingForHumanMove:
            if self.validHumanMoves is None:
                self.validHumanMoves = game.board.allValidMovesWithWay(player)
        else:
            self.validHumanMoves = None

    def handleHumanMove(self, game):
        if not self.humanMove:
            return
        chosenMove = self.matchingValidMove()
        self.humanMove = None
        if chosenMove:
            game.playMove(game.currentPlayer(), chosenMove)
            self.playback.moveTraveler += 1
            self.clickedField = None

    def handleClickedField(self, clicked):
        if clicked is None:
            self.clickedField = None
            return
        if self.clickedField is None:
            self.clickedField = clicked
        elif self.waitingForHumanMove:
            # Second click completes the pair: from the held field to this one.
            self.humanMove = (self.clickedField.id, clicked.id)
            self.clickedField = None
        else:
            self.clickedField = clicked
