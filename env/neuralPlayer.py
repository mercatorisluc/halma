"""A trained policy seated as an ordinary player, so a human can face it.

``heuristics/`` cannot host this: it would drag torch into the engine's only
dependency-free consumer. So it lives here, where the observation encoding
already does.

The point of this module is that the network sees *exactly* what it saw while
training. Rebuilding the observation by hand would be the obvious approach and
the wrong one -- the two copies would drift, and a network fed a subtly
different board plays worse for reasons no one can see. Instead a ``HalmaEnv``
is kept purely as an encoder and pointed at the live game, so there is one
implementation of the observation and one of the action encoding.

Each ``NeuralComputer`` carries its own encoder, told at construction which
seat is its own (``HalmaEnv(selfSeat=identifier)``), so the observation is
always built from *this* player's viewpoint even when another ``NeuralComputer``
occupies the other seat -- that is what lets two checkpoints play each other.
Only seats 1 and 2 are valid: the normalizer has no permutation for a third,
and the constructor refuses rather than letting that fail quietly deep inside
an observation.
"""

from __future__ import annotations

from sb3_contrib import MaskablePPO

from env.halmaEnv import HalmaEnv
from game.board import HalmaBoard
from game.boardTypes import MovePath, PlayerId
from game.gameManager import HalmaGame
from game.player import HalmaPlayer


class NeuralComputer(HalmaPlayer):
    """A player whose moves come from a trained MaskablePPO checkpoint."""

    def __init__(self, identifier: PlayerId, checkpoint: str, deterministic: bool = True) -> None:
        if identifier not in (HalmaEnv.AGENT_SEAT, HalmaEnv.OPPONENT_SEAT):
            raise ValueError(
                f"a policy plays seat {HalmaEnv.AGENT_SEAT} or {HalmaEnv.OPPONENT_SEAT}, "
                f"not {identifier}: the normalizer has no permutation for a third seat"
            )
        super().__init__(identifier)
        self.model = MaskablePPO.load(checkpoint)
        # An encoder, not a game. Its own board is discarded by attachTo; the
        # geometry it derived in __init__ (raster layout, normaliser) is the
        # same for every standard board, which is what makes the swap safe.
        # selfSeat matches this player's own seat, so the observation is
        # built from its own viewpoint whichever side it plays.
        self.encoder = HalmaEnv(selfSeat=identifier)
        # Argmax by default -- the policy's actual best move. Sampling makes it
        # play its distribution instead, which is a weaker but more varied
        # opponent, and is how the evaluation numbers labelled "sampled" arise.
        self.deterministic = deterministic

    def isHuman(self) -> bool:
        return False

    def attachTo(self, game: HalmaGame) -> None:
        """Point the encoder at the game actually being played."""
        self.encoder.game = game
        self.encoder._legalCache = None

    def chooseMove(self, moves: list[MovePath], board: HalmaBoard) -> MovePath:
        # The cache is keyed on the move count, and a human game advances that
        # through the same playMove, so it stays honest. Cleared anyway,
        # because a fresh game restarts the count.
        self.encoder._legalCache = None
        action, _ = self.model.predict(
            self.encoder._observation(),
            action_masks=self.encoder.action_masks(),
            deterministic=self.deterministic,
        )
        start, end = self.encoder.normalizer.inverseMove(
            self.encoder.decodeAction(int(action)), self.encoder._permutationKey(self)
        )
        for move in moves:
            if move[0] == start and move[-1] == end:
                return move
        # Masking makes this unreachable; if it ever fires, the encoder and the
        # live game have come apart, and picking some other move would hide it.
        raise RuntimeError(f"the policy chose {(start, end)}, which is not a legal move")
