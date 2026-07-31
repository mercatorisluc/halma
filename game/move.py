from collections import deque


class Move:
    """A single recorded move by a player.

    Stored compactly as its landing steps (``[start, ..., end]``). Moves are
    often recorded as just ``[start, end]``; :meth:`reconstructFullMove` fills
    in the intermediate jump landings and the fields jumped over, which the
    visualization needs to draw the full path.
    """

    def __init__(self, idList, player):
        self.steps = idList
        self.start = self.steps[0]
        self.end = self.steps[-1]
        self.player = player
        # Only known once reconstructFullMove has run; ``None`` is what
        # needsReconstruction reports on, so it must not start as [].
        self.jumpedOvers: list[int] | None = None

    def partMoves(self):
        return [(self.steps[i], self.steps[i + 1]) for i in range(len(self.steps) - 1)]

    def needsReconstruction(self):
        return self.jumpedOvers is None

    def reconstructFullMove(self, board):
        """Recover the full jump path for a move stored as just start/end.

        Works in either direction (the piece may already have been moved, so
        ``start`` can be the empty field), finds the shortest chain of jumps
        between the two endpoints via BFS, and records the jumped-over fields.
        A plain single step yields an empty ``jumpedOvers``.
        """
        if board.fields[self.start].isEmpty():
            startField, endField = (self.end, self.start)
        else:
            startField, endField = (self.start, self.end)
        if not board.isJumpMove(startField, endField):
            self.jumpedOvers = []
            return

        queue = deque([[startField]])
        validMoves = []
        visited = {startField}
        while queue:
            tempWay = queue.popleft()
            for jumpPosition in board.getValidJumpFields(tempWay[-1]):
                if jumpPosition == endField:
                    validMoves.append([*tempWay, endField])
                elif jumpPosition not in visited:
                    visited.add(jumpPosition)
                    queue.append([*tempWay, jumpPosition])
        assert len(validMoves) > 0
        move = min(validMoves, key=len)
        if board.fields[self.start].isEmpty():
            move.reverse()
        self.steps = move
        self.calculateJumpedOverFields(board)

    def calculateJumpedOverFields(self, board):
        self.jumpedOvers = []
        for partMove in self.partMoves():
            start, end = partMove
            if board.isJumpMove(start, end):
                coordX = (board.fields[start].coord[0] + board.fields[end].coord[0]) / 2
                coordY = (board.fields[start].coord[1] + board.fields[end].coord[1]) / 2
                id = board.idFromCoord((coordX, coordY))
                self.jumpedOvers.append(id)

    def fullStepsList(self):
        """Flatten the move into an alternating list
        ``[start, over, land, over, land, ..., end]`` for drawing the path.

        :meth:`reconstructFullMove` must have run first — the jumped-over
        fields are not known before that.
        """
        if self.jumpedOvers is None:
            raise RuntimeError("fullStepsList() needs reconstructFullMove() to have run first")
        fullStepsList = [self.start]
        substeps = list(self.steps)[1:-1]
        for i, jumpedOver in enumerate(self.jumpedOvers):
            fullStepsList.append(jumpedOver)
            if i < len(substeps):
                fullStepsList.append(substeps[i])
        fullStepsList.append(self.end)
        return fullStepsList
