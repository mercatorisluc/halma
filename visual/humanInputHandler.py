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
        merely validated. ``None`` if nothing is pending or nothing matches.
        """
        if self.humanMove is None or self.validHumanMoves is None:
            return None
        start, end = self.humanMove
        for move in self.validHumanMoves:
            if (start == move[0]) and (end == move[-1]):
                return move
        return None

    def atLiveFront(self):
        """Whether the board shows the current position rather than history.

        The arrow keys rewind the board through recorded moves. While that
        cursor is behind the front, the position on screen is a past one: the
        human's legal moves are not the ones being drawn, and a move played
        into it would be appended to a history it does not follow from.
        """
        return self.playback.moveTraveler == len(self.playback.game.moves)

    def adaptToHumanInteraction(self, game):
        player = game.currentPlayer()
        self.waitingForHumanMove = player.isHuman() and self.atLiveFront()
        if self.waitingForHumanMove:
            # Recomputed every frame rather than cached for the turn. The board
            # is mutated underneath this by the playback cursor, and a cached
            # jump path that the board no longer supports made
            # Move.reconstructFullMove fail its assertion while merely *drawing*
            # the move -- the game died on a keypress plus a click.
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
