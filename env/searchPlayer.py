"""A trained policy that looks a move ahead instead of trusting its instinct.

``NeuralComputer`` plays whatever one forward pass ranks highest. This plays the
move that still looks best *after* the opponent has answered it -- the smallest
useful search, and the first thing here to use the critic that every checkpoint
has carried, unused, since training.

Three things it is built out of, none of them new:

* **The policy orders the moves.** A position has some 65 legal moves and most
  are obviously poor; searching all of them would spend the budget on moves the
  policy already rejects. Only the top ``candidates`` are expanded, and the same
  ranking picks the opponent's ``replies``. Good move ordering is what makes a
  shallow search worth anything, and a trained policy is an excellent orderer.

* **The critic evaluates the leaf** -- with a correction that is easy to miss.
  PPO trained it on the *shaped* return, and potential-based shaping shifts the
  value function by exactly the potential: ``V'(s) = V(s) - w * phi(s)``. That
  offset depends on the position, so it does **not** cancel when two sibling
  leaves are compared -- a raw critic comparison silently ranks by "value minus
  remaining travel". ``_value`` adds ``w * phi(s)`` back.

* **Repetition is a rule here, not something learned.** Two deterministic
  policies deadlock (RESULTS.md: 2% of the opening census, 12.8% off the
  beaten track), and no amount of training addresses it, because PPO samples
  its own actions and the cycle never arises in training. A search can simply
  decline to re-enter a position it has already been in, which is why this
  keeps ``seen``.

Sequential, and knowingly so. Each node is one forward pass on its own
(``candidates * (1 + replies) + 1`` per move at depth 2), where a batched
evaluator would score a whole ply in one pass. Batching is blocked by
``board.moveApplied``: candidates are scored by mutating the real board and
undoing it, so two hypothetical positions cannot exist at once. That refactor
is worth doing when the search grows deeper -- it is not worth doing to find
out whether depth 2 helps at all.

The searched position is a hypothetical, and two details of the encoding follow
from that. ``_legalCache`` is keyed on the game's move count, which
``moveApplied`` deliberately does not bump, so it is cleared before every
lookup rather than trusted. And the observation's move-progress scalar reads
the real game length, so a leaf is presented as one or two plies earlier than
it is -- 2 of 250, below the resolution of anything the scalar feeds.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

import numpy as np
import torch

from env.halmaEnv import HalmaEnv
from env.neuralPlayer import NeuralComputer
from game.board import HalmaBoard
from game.boardTypes import AnyMove, MovePath, PlayerId
from game.gameManager import HalmaGame
from game.player import HalmaPlayer

# The root ranks full jump paths and the reply search ranks endpoint pairs;
# both go through _rank, which must hand back what it was given.
MoveT = TypeVar("MoveT", bound=AnyMove)


class SearchingComputer(NeuralComputer):
    """A ``NeuralComputer`` that expands the policy's best moves one ply."""

    # Beyond anything the critic can return, so a forced win outranks every
    # evaluated position rather than competing with one.
    WIN_VALUE = 1e6

    def __init__(
        self,
        identifier: PlayerId,
        checkpoint: str,
        candidates: int = 6,
        replies: int = 3,
        deterministic: bool = True,
    ) -> None:
        super().__init__(identifier, checkpoint, deterministic)
        otherSeat = (
            HalmaEnv.OPPONENT_SEAT if identifier == HalmaEnv.AGENT_SEAT else HalmaEnv.AGENT_SEAT
        )
        # A second encoder, seated on the other side: the reply is chosen from
        # the opponent's viewpoint, by the same network. "The opponent plays as
        # I would" is an assumption, and the cheap one -- the alternative is
        # minimising our own critic over every reply, which costs a forward
        # pass per legal move instead of one per node.
        self.opponentEncoder = HalmaEnv(selfSeat=otherSeat)
        self.candidates = candidates
        self.replies = replies
        # Board states this player has already been on turn in. A cycle needs
        # both sides to repeat, so seeing only our own turns is enough to break
        # one.
        self.seen: set[bytes] = set()

    def attachTo(self, game: HalmaGame) -> None:
        super().attachTo(game)
        self.opponentEncoder.game = game
        self.opponentEncoder._legalCache = None
        # A fresh game has no history to avoid, and these players are reused
        # across games by every script that runs a match.
        self.seen = set()

    def chooseMove(self, moves: list[MovePath], board: HalmaBoard) -> MovePath:
        self.seen.add(board.boardState().tobytes())
        if not self.opponents:
            return super().chooseMove(moves, board)
        opponent = self.opponents[0]

        best: MovePath | None = None
        bestValue = -np.inf
        # Kept separately so a position where *every* candidate repeats still
        # produces a move: declining to move is not an option.
        repeat: MovePath | None = None
        repeatValue = -np.inf

        for move in self._rank(self.encoder, self, moves)[: self.candidates]:
            with board.moveApplied(move, self):
                repeats = board.boardState().tobytes() in self.seen
                value = self._valueAfterReply(board, opponent)
            if repeats:
                if value > repeatValue:
                    repeat, repeatValue = move, value
            elif value > bestValue:
                best, bestValue = move, value

        if best is not None:
            return best
        return repeat if repeat is not None else moves[0]

    def _valueAfterReply(self, board: HalmaBoard, opponent: HalmaPlayer) -> float:
        """What the position is worth once the opponent has answered.

        The minimum over the replies considered, because the opponent picks
        them: a candidate is only as good as its worst likely answer.

        ``replies = 0`` stops before the opponent moves and evaluates our own
        move directly. That is not a search worth playing -- it is the
        diagnostic that separates the two ways this can fail. The opponent ply
        rests on the critic *and* on "the opponent plays as I would"; without
        it only the critic is left, so an arm at 0 says which of the two is
        doing the damage.
        """
        if self.isWinning():
            return self.WIN_VALUE
        replies = board.allValidMoves(opponent)
        if not replies or not self.replies:
            return self._value()

        worst = np.inf
        for reply in self._rank(self.opponentEncoder, opponent, replies)[: self.replies]:
            with board.moveApplied(reply, opponent):
                value = -self.WIN_VALUE if opponent.isWinning() else self._value()
            worst = min(worst, value)
        return float(worst)

    def _value(self) -> float:
        """The critic's estimate of the current board, from this player's seat.

        Plus the potential the shaping subtracted -- see the module docstring.
        """
        self.encoder._legalCache = None
        tensor, _ = self.model.policy.obs_to_tensor(self.encoder._observation())
        with torch.no_grad():
            value = float(self.model.policy.predict_values(tensor)[0].item())
        return value + self.encoder.shapingWeight * self.encoder._potential()

    def _rank(self, encoder: HalmaEnv, player: HalmaPlayer, moves: Sequence[MoveT]) -> list[MoveT]:
        """``moves``, most probable first, as the policy sees them from ``player``.

        The mask is built from ``moves`` rather than read off the encoder:
        ``action_masks()`` would answer for whoever the *game* thinks is on
        turn, and inside the search that is nobody's turn in particular.
        """
        encoder._legalCache = None
        actions = self._encode(encoder, player, moves)
        mask = np.zeros(encoder.actionCount, dtype=bool)
        mask[actions] = True
        tensor, _ = self.model.policy.obs_to_tensor(encoder._observation())
        with torch.no_grad():
            distribution = self.model.policy.get_distribution(tensor, action_masks=mask)
            probabilities = distribution.distribution.probs[0].numpy()  # pyright: ignore[reportAttributeAccessIssue]
        order = sorted(
            range(len(moves)), key=lambda i: float(probabilities[actions[i]]), reverse=True
        )
        return [moves[i] for i in order]

    @staticmethod
    def _encode(encoder: HalmaEnv, player: HalmaPlayer, moves: Sequence[AnyMove]) -> list[int]:
        """Encoded action ids for ``moves``, in ``player``'s canonical frame.

        Only the endpoints matter: the engine applies and scores a jump by its
        first and last field, and so does the action encoding.
        """
        endpoints: list[AnyMove] = [(move[0], move[-1]) for move in moves]
        normalized = encoder.normalizer.permuteMoves(endpoints, encoder._permutationKey(player))
        return [encoder.encodeAction(move) for move in normalized]
