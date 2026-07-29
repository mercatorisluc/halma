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
                self.playback.moveTraveler += 1
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
