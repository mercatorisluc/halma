from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.boardNormalizer import Normalizer
from game.board import HalmaBoard
from game.boardTypes import MoveEndpoints
from game.gameManager import ComputedGame
from game.player import Computer, HalmaPlayer

PIECES_PER_PLAYER = 15


class HalmaEnv(gym.Env):
    """Two-player Halma as a single-agent Gymnasium environment.

    Gymnasium models one agent against an environment, while Halma has two
    players. So the agent owns one seat and the opponent is a heuristic bot
    that moves *inside* ``step``: one env step is the agent's move plus the
    reply. From the agent's side the opponent is simply part of the world.

    Everything the agent sees is expressed in the canonical frame that
    :class:`~env.boardNormalizer.Normalizer` maps every viewpoint onto, so the
    policy only ever learns one orientation. Actions are in that frame too and
    are mapped back before being played.

    Two players only: the normalizer has no permutation for a third seat, and
    three-player Halma is not zero-sum. See ARCHITECTURE.md.
    """

    AGENT_SEAT = 1
    OPPONENT_SEAT = 2

    def __init__(
        self,
        agentStrategy: str = "advancedDistScore",
        opponentStrategy: str = "sparsityScore",
        shapingWeight: float = 1.0,
        gamma: float = 0.99,
    ) -> None:
        super().__init__()
        # The agent's own seat is a Computer only so the game object is
        # well-formed; its strategy is never consulted, because moves arrive
        # through step().
        self.agentStrategy = agentStrategy
        self.opponentStrategy = opponentStrategy
        # gamma has to be the discount the agent is trained with, or the
        # shaping stops being policy-invariant. shapingWeight = 0 turns shaping
        # off, which is how to measure whether it is earning its keep.
        self.shapingWeight = shapingWeight
        self.gamma = gamma
        self.previousPotential = 0.0

        self.game = ComputedGame()
        self._seatPlayers()
        self.fieldCount = len(self.game.board.fields)
        self.normalizer = Normalizer(self.game.board)

        self.actionCount = self.fieldCount**2
        self.action_space = spaces.Discrete(self.actionCount)
        # Two binary planes (own pieces, opponent pieces) followed by the
        # scalars described in _observation, all scaled into [0, 1].
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(2 * self.fieldCount + 4,), dtype=np.float32
        )

    # ------------------------------------------------------------------ setup

    def _seatPlayers(self) -> None:
        agent = Computer(self.AGENT_SEAT, self.agentStrategy)
        opponent = Computer(self.OPPONENT_SEAT, self.opponentStrategy)
        self.game.initGame([agent, opponent])

    @property
    def board(self) -> HalmaBoard:
        return self.game.board

    def _player(self, seat: int) -> HalmaPlayer:
        return next(p for p in self.game.players if p.identifier == seat)

    # ------------------------------------------------------------------- gym

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        # The engine has its own generator; seeding it is what makes a whole
        # episode reproducible, since seat order and the opponent's tie-breaks
        # both draw from it.
        self.game.seed(seed)
        self._seatPlayers()
        # Play order is randomised, so the opponent may be on move first.
        self._playOpponentUntilAgentsTurn()
        self.previousPotential = self._potential()
        return self._observation(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self._isAgentsTurn():
            raise RuntimeError("step() called when it is not the agent's turn")

        if int(action) not in self._legalActions():
            # Cannot happen while the caller respects action_masks(), but the
            # environment must stay total: playing an illegal move would
            # corrupt the game state silently. Forfeit instead, and flag it so
            # a masking bug shows up as something other than a bad policy.
            return self._observation(), -1.0, True, False, self._info(illegalAction=True)

        self._playAgentMove(action)
        if self.game.winner() is None:
            self._playOpponentUntilAgentsTurn()

        winner = self.game.winner()
        terminated = winner is not None
        truncated = not terminated and self.game.gameLength() >= self.game.MAX_MOVES
        outcome = 0.0
        if terminated:
            outcome = 1.0 if winner == self.AGENT_SEAT else -1.0
        reward = outcome + self._shaping(terminated)
        return (
            self._observation(),
            reward,
            terminated,
            truncated,
            self._info(outcome=outcome),
        )

    def render(self) -> None:
        self.game.printBoard()

    # --------------------------------------------------------------- shaping

    def _progress(self, player: HalmaPlayer) -> float:
        """How far this player still has to go. Lower is closer to winning."""
        return self.board.advancedDistanceScore(player) + self.board.homeBonusScore(player)

    def _potential(self) -> float:
        """How good the position is for the agent. Higher is better.

        The lead over the opponent, not the agent's own progress alone --
        falling behind has to lower it, or the agent is rewarded for advancing
        while losing. Being a difference between two comparable players it
        stays small, roughly -0.3 to +0.7, and starts at 0 from an even board.
        """
        return -(
            self._progress(self._player(self.AGENT_SEAT))
            - self._progress(self._player(self.OPPONENT_SEAT))
        )

    def _shaping(self, terminated: bool) -> float:
        """Potential-based shaping: ``weight * (gamma * phi(s') - phi(s))``.

        Winning is the only real reward, and it arrives once per roughly 69
        decisions -- a random agent never sees it at all, measured over 700
        games. That is not enough to learn from, so progress is rewarded every
        step instead.

        This particular form (Ng, Harada & Russell 1999) is the one that does
        not change which policy is optimal: the added terms telescope, so over
        an episode they sum to a constant that no policy can influence. The
        agent is hurried along, not redirected. Two things it depends on --
        ``gamma`` matching the training discount, and the terminal potential
        being zero, which is why ``terminated`` is passed in rather than read
        off the board.

        Verified: over 32 games that ended in a win or loss, the discounted
        return shifts by exactly ``-phi(s0)`` as the theory says, to machine
        precision.

        Note ``terminated`` and not "the episode stopped". A game cut off at
        ``MAX_MOVES`` is truncated, not over, and its potential is deliberately
        *not* zeroed -- the agent should bootstrap from that state's value, as
        it would from any other. The exact relation above therefore does not
        hold for truncated episodes, which is a property of time limits rather
        than of the shaping.
        """
        potential = 0.0 if terminated else self._potential()
        shaping = self.shapingWeight * (self.gamma * potential - self.previousPotential)
        self.previousPotential = potential
        return shaping

    # --------------------------------------------------------------- actions

    def action_masks(self) -> np.ndarray:
        """Boolean mask over the action space, as MaskablePPO expects.

        Only ~65 of 14641 actions are legal in a typical position, so without
        this the policy would spend its capacity learning which actions are
        illegal instead of which are good.
        """
        mask = np.zeros(self.actionCount, dtype=bool)
        for move in self._legalActions():
            mask[move] = True
        return mask

    def _legalActions(self) -> list[int]:
        player = self.game.currentPlayer()
        moves = self.board.allValidMoves(player)
        normalized = self.normalizer.permuteMoves(moves, self._permutationKey(player))
        return [self.encodeAction(move) for move in normalized]

    def encodeAction(self, move: MoveEndpoints) -> int:
        return int(move[0]) * self.fieldCount + int(move[-1])

    def decodeAction(self, action: int) -> MoveEndpoints:
        return (action // self.fieldCount, action % self.fieldCount)

    # ------------------------------------------------------------ game steps

    def _isAgentsTurn(self) -> bool:
        return self.game.currentPlayer().identifier == self.AGENT_SEAT

    def _playAgentMove(self, action: int) -> None:
        player = self.game.currentPlayer()
        normalizedMove = self.decodeAction(int(action))
        move = self.normalizer.inverseMove(normalizedMove, self._permutationKey(player))
        self.game.playMove(player, (int(move[0]), int(move[1])))

    def _playOpponentUntilAgentsTurn(self) -> None:
        while (
            not self._isAgentsTurn()
            and self.game.winner() is None
            and self.game.gameLength() < self.game.MAX_MOVES
        ):
            opponent = self.game.currentPlayer()
            self.game.playNextMove(opponent)

    # ---------------------------------------------------------- observations

    def _permutationKey(self, player: HalmaPlayer) -> str:
        """Key of the permutation taking ``player``'s view to the canonical one."""
        seat = player.identifier
        flip = "WithFlip" if self._needsFlip(seat) else "WithoutFlip"
        return f"player{seat}{flip}"

    def _needsFlip(self, seat: int) -> bool:
        occupied = np.where(self.board.boardState() == seat)[0]
        return bool(np.sum(self.normalizer.sumCoordsX(occupied)) < 0)

    def _observation(self) -> np.ndarray:
        """Board planes plus scalars, from the agent's point of view.

        Layout: ``[own pieces (121), opponent pieces (121), progress, own
        pieces home, opponent pieces home, mobility]``. The zones themselves
        are deliberately absent -- after normalisation the start zone is always
        fields 0-14 and the target 102-120, so they are constant and carry no
        information.
        """
        agent = self._player(self.AGENT_SEAT)
        opponent = self._player(self.OPPONENT_SEAT)
        state = self.normalizer.permute(self.board.boardState(), self._permutationKey(agent))

        own = (state == self.AGENT_SEAT).astype(np.float32)
        other = (state == self.OPPONENT_SEAT).astype(np.float32)
        scalars = np.array(
            [
                self.game.gameLength() / self.game.MAX_MOVES,
                len(agent.positions & agent.endPositions) / PIECES_PER_PLAYER,
                len(opponent.positions & opponent.endPositions) / PIECES_PER_PLAYER,
                min(len(self.board.allValidMoves(agent)), self.fieldCount) / self.fieldCount,
            ],
            dtype=np.float32,
        )
        return np.concatenate([own, other, scalars])

    def _info(self, illegalAction: bool = False, outcome: float = 0.0) -> dict[str, Any]:
        # outcome is the unshaped result, +1/-1/0. Evaluation must read this
        # rather than the reward, or shaping would be scored as if it were
        # winning.
        return {
            "action_mask": self.action_masks(),
            "illegalAction": illegalAction,
            "outcome": outcome,
        }
