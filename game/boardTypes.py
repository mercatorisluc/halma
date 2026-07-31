"""Type aliases for the values passed around the engine.

These exist to make two distinctions visible to the type checker that are
otherwise only described in prose:

- **`MoveEndpoints` vs `MovePath`** — a move is either just its two endpoints
  or the full sequence of landings. Picking the wrong one is the classic
  mistake in this codebase (see ARCHITECTURE.md). `tuple` and `list` are
  distinguishable, so annotating them actually catches it.
- **`FieldId` vs `Coord`** — an integer 0-120 versus an axial (x, y) pair. Both
  address a field; only one indexes `board.fields`.
"""

FieldId = int

# Axial hex coordinate. Halves occur transiently in Move's midpoint arithmetic.
Coord = tuple[int, int]

# A player's seat: 1, 2 or 3. Also what lands in HalmaField.playerID, where 0
# additionally means "empty" -- so identifiers must never be 0.
#
# This is an int on purpose. board.boardState() collects playerIDs into a numpy
# array, and the RL code does arithmetic on it (``normalized != 0`` to find
# occupied fields, ``== playerID`` to find one player's pieces). A string
# identifier turns that array into a string array, where every empty field
# compares unequal to 0 and reads as an opponent piece.
PlayerId = int

# A move as its two endpoints, e.g. from board.allValidMoves.
MoveEndpoints = tuple[FieldId, FieldId]

# A move as its full path [start, landing, ..., end], e.g. from
# board.allValidMovesWithWay. A single step is [start, end].
MovePath = list[FieldId]

# For the code that only ever reads move[0] and move[-1] and so takes either.
AnyMove = MoveEndpoints | MovePath
