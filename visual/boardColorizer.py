from collections import defaultdict


class BoardColorizer:
    """Colour palette for the visualization: named base colours, per-player
    piece colours and home-base background tints.

    Drawing is driven by a single-letter flag saying *why* a field is being
    drawn:

    - ``'S'`` — plain, every field of the board
    - ``'M'`` — part of the last move played
    - ``'C'`` — the field the human just clicked
    - ``'P'`` — preview of a move the human could make from there

    Only the last three are highlighted; ``'S'`` fields are drawn in the
    occupying player's colour and never reach :meth:`colorForFlag`.
    """

    def __init__(self):
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.VIOLET = (150, 0, 255)
        self.GREY = (230, 230, 230)
        self.YELLOW = (238, 255, 65)
        self.playerColorsDict = defaultdict(lambda: self.GREY)
        self.backgroundColors = [(239, 154, 154), (129, 212, 250), (165, 214, 167)]

    def setPlayerWithColors(self, playerID, color):
        self.playerColorsDict[playerID] = color

    def playerColor(self, playerID):
        return self.playerColorsDict[playerID]

    def colorForFlag(self, flag):
        """Highlight colour for a draw flag, or ``None`` for one that needs no
        highlight."""
        if flag == "M":
            return self.moveHighlighting()
        if flag in ("P", "C"):
            return self.clickHighlighting()
        return None

    def moveHighlighting(self):
        return self.VIOLET

    def clickHighlighting(self):
        return self.YELLOW

    def black(self):
        return self.BLACK

    def white(self):
        return self.WHITE
