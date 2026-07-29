class GamePlaybackController:
    """Owns the move-history cursor (``moveTraveler``) and steps the game
    forwards/backwards through its recorded moves.

    Moving forward past the end of the recorded history asks the current
    (non-human) player for its next move; a human player's move is fed in by
    the input handler instead.
    """

    def __init__(self, game):
        self.game = game
        self.moveTraveler = 0

    def playNextMove(self):
        if self.moveTraveler < len(self.game.moves):
            self.forwardGame()
        else:
            player = self.game.currentPlayer()
            if not player.isHuman():
                self.game.playNextMove(player)
                self.moveTraveler += 1

    def gameStateAt(self, moveTravelerPosition):
        """Set the board to the state after ``moveTravelerPosition`` recorded
        moves have been applied (0 = starting position)."""
        assert 0 <= moveTravelerPosition <= len(self.game.moves)
        self.goToStartingPosition()
        for _ in range(moveTravelerPosition):
            self.forwardGame()

    def goToStartingPosition(self):
        # Rewind from wherever the cursor currently is; the cursor and the
        # board are kept in sync as moves are played/replayed.
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
