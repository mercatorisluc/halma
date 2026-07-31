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

# Whatever a player was constructed with: ints for bot games, names for the
# interactive one. Also what lands in HalmaField.playerID.
PlayerId = int | str

# A move as its two endpoints, e.g. from board.allValidMoves.
MoveEndpoints = tuple[FieldId, FieldId]

# A move as its full path [start, landing, ..., end], e.g. from
# board.allValidMovesWithWay. A single step is [start, end].
MovePath = list[FieldId]

# For the code that only ever reads move[0] and move[-1] and so takes either.
AnyMove = MoveEndpoints | MovePath
