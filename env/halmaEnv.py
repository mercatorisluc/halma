from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.boardNormalizer import Normalizer
from game.board import HalmaBoard
from game.boardTypes import MoveEndpoints
from game.gameManager import ComputedGame, HalmaGame
from game.player import Computer, HalmaPlayer

if TYPE_CHECKING:
    # Only for typing: env.neuralPlayer imports HalmaEnv itself, so the real
    # import has to happen lazily inside __init__ -- see the comment there.
    from env.neuralPlayer import NeuralComputer

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

    ``selfSeat`` defaults to ``AGENT_SEAT`` -- training always wants the agent
    on that seat, and every existing caller relies on it. It exists as a
    parameter because the permutation the observation is built with
    (``_permutationKey``) already keyed itself off a player's own identifier
    rather than the ``AGENT_SEAT`` constant, for every use except building the
    observation itself -- so a policy seated on ``OPPONENT_SEAT`` needed only
    that one piece filled in, not a second encoding. That is what lets two
    ``NeuralComputer``s face each other: each keeps its own encoder, seated on
    its own actual seat.

    ``opponentModel``, if given, seats a frozen ``NeuralComputer`` -- loaded
    from that checkpoint -- as the opponent instead of a heuristic, and takes
    priority over ``opponentStrategy`` and ``opponentModelPool`` both. It is
    loaded once in ``__init__`` and reused for every episode via ``attachTo``,
    not reconstructed per reset, which would reload the checkpoint from disk
    every game. The checkpoint's weights never change: this is a fixed
    sparring partner for PPO, not self-play, which would additionally need
    the opponent's own weights kept in step with training.

    ``opponentModelPool``, if given, extends the per-episode draw that
    ``opponentPool`` already does for heuristics to a set of frozen
    checkpoints as well -- reset() draws from the combined pool, so a run can
    mix heuristic and tuned-model opponents rather than being limited to one
    or the other. Every checkpoint is loaded once, here, for the same reason
    ``opponentModel`` is.
    """

    AGENT_SEAT = 1
    OPPONENT_SEAT = 2

    def __init__(
        self,
        agentStrategy: str = "advancedDistScore",
        opponentStrategy: str | Sequence[str] = "sparsityScore",
        opponentModel: str | None = None,
        opponentModelPool: Sequence[str] | None = None,
        shapingWeight: float = 1.0,
        gamma: float = 0.99,
        selfSeat: int = AGENT_SEAT,
    ) -> None:
        super().__init__()
        # Which seat's viewpoint the observation is built from. Everything
        # else in this class already keyed off a player's own identifier
        # rather than this constant; only the observation and the few things
        # that mean "my own seat" needed to learn to read it.
        self.selfSeat = selfSeat
        self.otherSeat = self.OPPONENT_SEAT if selfSeat == self.AGENT_SEAT else self.AGENT_SEAT
        # The agent's own seat is a Computer only so the game object is
        # well-formed; its strategy is never consulted, because moves arrive
        # through step().
        self.agentStrategy = agentStrategy
        # A single name is the common case and keeps every existing caller
        # unchanged; a sequence is a pool that reset() redraws from each
        # episode, so a fine-tune does not sharpen against one opponent's
        # blind spots at the cost of every other. self.opponentStrategy is
        # what _seatPlayers() actually reads, so picking the initial entry
        # here (before the first reset()'s draw) still leaves the env
        # well-formed if it is ever used before reset() is called.
        self.opponentPool = (
            (opponentStrategy,) if isinstance(opponentStrategy, str) else tuple(opponentStrategy)
        )
        # Empty is legal, but only for a pure self-play run, where every
        # opponent comes from opponentModelPool and no heuristic should ever
        # be drawn. The constructor checks that below, once both pools exist.
        self.opponentStrategy = self.opponentPool[0] if self.opponentPool else ""
        # Loaded once, here, and reused by every _seatPlayers() call below --
        # constructing a NeuralComputer loads a checkpoint from disk, which a
        # per-episode rebuild would repeat every single game.
        self._opponentNeural: NeuralComputer | None = None
        # A fixed opponentModel is never redrawn in reset() below, unlike
        # opponentPool and _opponentModelPool -- this flag is what tells
        # reset() to leave self._opponentNeural alone rather than overwrite
        # it with a pool draw.
        self._opponentModelFixed = opponentModel is not None
        if opponentModel is not None:
            # env.neuralPlayer imports HalmaEnv (for AGENT_SEAT/OPPONENT_SEAT
            # and to build its own encoder), so importing it at module level
            # here would be circular; deferred to first use instead.
            from env.neuralPlayer import NeuralComputer

            self._opponentNeural = NeuralComputer(self.otherSeat, opponentModel)

        # Frozen checkpoints reset() can draw into alongside opponentPool's
        # heuristics -- see the class docstring. Loaded once here rather than
        # per draw, same reasoning as opponentModel above.
        self._opponentModelPool: list[NeuralComputer] = []
        if opponentModelPool:
            from env.neuralPlayer import NeuralComputer

            self._opponentModelPool = [
                NeuralComputer(self.otherSeat, path) for path in opponentModelPool
            ]
        # With no heuristics to fall back on, _seatPlayers() has nothing to
        # build an opponent from until reset() draws one, so seat a checkpoint
        # now -- __init__ calls _seatPlayers() before any reset happens.
        if not self.opponentPool and self._opponentNeural is None:
            if not self._opponentModelPool:
                raise ValueError(
                    "an empty opponentStrategy needs opponentModel or opponentModelPool to "
                    "supply the opponent"
                )
            self._opponentNeural = self._opponentModelPool[0]
        # gamma has to be the discount the agent is trained with, or the
        # shaping stops being policy-invariant. shapingWeight = 0 turns shaping
        # off, which is how to measure whether it is earning its keep.
        self.shapingWeight = shapingWeight
        self.gamma = gamma
        self.previousPotential = 0.0
        # (position version, encoded legal moves) -- see _legalActions.
        self._legalCache: tuple[int, list[int]] | None = None

        # Declared as the base class because only its API is used here, which
        # is what lets env/neuralPlayer.py point the same encoding at an
        # InteractiveGame. Training makes a ComputedGame: it is the fast one.
        self.game: HalmaGame = ComputedGame()
        self._seatPlayers()
        self.fieldCount = len(self.game.board.fields)
        self.normalizer = Normalizer(self.game.board)
        # Field id -> steps from there to the nearest target field. The target
        # zone is fixed per seat, so this is a constant vector rather than
        # something to recompute while scoring.
        self.distanceToTarget = self._targetDistances(self._player(self.selfSeat))
        # Fixed scale for the potential. The opening is identical every game,
        # so this is a constant, not per-episode state -- the shaping would
        # stop telescoping if it moved during a game.
        self.openingProgress = self._progress(self._player(self.selfSeat))

        self.actionCount = self.fieldCount**2
        self.action_space = spaces.Discrete(self.actionCount)
        # Field id -> (row, col) on the 17x17 raster, in the canonical frame.
        # fieldNumber already embeds the hex board there, which is why the
        # board can be handed to a convolution at all.
        self.rasterIndex = np.array(
            [self._rasterCell(field.coord) for field in self.game.board.fields]
        )
        self.observation_space = spaces.Dict(
            {
                # Own pieces, opponent pieces, and which cells of the 17x17
                # square are real fields at all -- 168 of 289 are not, and
                # without that plane a convolution cannot tell an empty field
                # from the void outside the star.
                "board": spaces.Box(low=0.0, high=1.0, shape=(3, 17, 17), dtype=np.float32),
                "scalars": spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32),
            }
        )
        self.boardMask = np.zeros((17, 17), dtype=np.float32)
        self.boardMask[self.rasterIndex[:, 0], self.rasterIndex[:, 1]] = 1.0

    # ------------------------------------------------------------------ setup

    def _seatPlayers(self) -> None:
        agent = Computer(self.selfSeat, self.agentStrategy)
        opponent: HalmaPlayer = self._opponentNeural or Computer(
            self.otherSeat, self.opponentStrategy
        )
        # HalmaGame.initPlayers hands out home corners by list position, not
        # by a player's own identifier -- players[0] always gets
        # player1Positions -- so the two have to go in seat order regardless
        # of which one is "self", or selfSeat=OPPONENT_SEAT would seat the
        # agent on the wrong player's corner.
        ordered: list[HalmaPlayer] = (
            [agent, opponent] if self.selfSeat < self.otherSeat else [opponent, agent]
        )
        self.game.initGame(ordered)
        if self._opponentNeural is not None:
            # Points its encoder at *this* game object, and clears its own
            # legal-action cache -- otherwise it would keep answering from
            # whatever game it last played.
            self._opponentNeural.attachTo(self.game)

    @property
    def board(self) -> HalmaBoard:
        return self.game.board

    def _player(self, seat: int) -> HalmaPlayer:
        return next(p for p in self.game.players if p.identifier == seat)

    # ------------------------------------------------------------------- gym

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        # Redraw the opponent before seating, so a pool of more than one
        # strategy -- heuristic, tuned checkpoint, or both -- actually
        # rotates; a fixed self.opponentStrategy/self._opponentNeural would
        # otherwise stick to whichever entry __init__ happened to pick.
        # self.np_random is seeded by super().reset() above, so this draw is
        # reproducible along with everything else the episode does. Skipped
        # entirely when opponentModel pinned a single fixed opponent -- that
        # takes priority over both pools, see the class docstring.
        if not self._opponentModelFixed and (len(self.opponentPool) > 1 or self._opponentModelPool):
            draw = self.np_random.integers(len(self.opponentPool) + len(self._opponentModelPool))
            if draw < len(self.opponentPool):
                self.opponentStrategy = self.opponentPool[draw]
                self._opponentNeural = None
            else:
                self._opponentNeural = self._opponentModelPool[draw - len(self.opponentPool)]
        # The engine has its own generator; seeding it is what makes a whole
        # episode reproducible, since seat order and the opponent's tie-breaks
        # both draw from it.
        self.game.seed(seed)
        self._seatPlayers()
        # A new game restarts the move count, so last episode's entry would
        # look current.
        self._legalCache = None
        # Play order is randomised, so the opponent may be on move first.
        self._playOpponentUntilAgentsTurn()
        self.previousPotential = self._potential()
        return self._observation(), self._info()

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
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
        outOfMoves = winner is None and self.game.gameLength() >= self.game.MAX_MOVES
        # The move cap is one of the game's own rules, not a harness time limit,
        # so running it out ends the episode rather than cutting it short --
        # there is nothing left to bootstrap from.
        terminated = winner is not None or outOfMoves

        if winner is not None:
            outcome = 1.0 if winner == self.selfSeat else -1.0
        elif outOfMoves:
            # Priced as a loss. Scoring 0 against -1 for losing made stalling
            # strictly the better play, and an agent duly found it: it ended
            # with 0 of 15 pieces home while holding the opponent back from 14
            # to 12. Nobody reaching the target is not a better result than
            # losing.
            outcome = -1.0
        else:
            outcome = 0.0

        reward = outcome + self._shaping(terminated)
        return (
            self._observation(),
            reward,
            terminated,
            False,
            self._info(outcome=outcome),
        )

    def render(self) -> None:
        self.game.printBoard()

    # --------------------------------------------------------------- shaping

    def _targetDistances(self, player: HalmaPlayer) -> np.ndarray:
        """For every field, the steps from it to the nearest target field."""
        distances = self.board.distanceMatrix
        targets = sorted(player.endPositions)
        return np.array(
            [
                min(distances[field][target] for target in targets)
                for field in range(self.fieldCount)
            ],
            dtype=np.float64,
        )

    def _progress(self, player: HalmaPlayer) -> float:
        """Travel this player still has to do. Lower is closer to winning.

        The distance every piece not yet home still has to cover to reach the
        target zone, summed. Pieces already home contribute nothing, so this is
        zero exactly when the game is won -- there are 15 target fields and 15
        pieces, so a sum of zero means each piece stands on one.

        It replaces ``advancedDistanceScore + homeBonusScore``, which the bots
        score on and which is the wrong objective to *shape* with. Two thirds of
        that measure is ``simpleDistanceScore``, the distance to the single tip
        field of the target triangle rather than to the zone: measured over 235
        moves it moved 10x further per move than the zone-distance term, so it
        was effectively the whole signal, and it pulled pieces at one corner
        instead of into the target. It also does not bottom out -- a won
        position still scores 1.25 of the opening's 7.69, leaving 16% of the
        shaping budget unreachable and paying pieces already home to shuffle
        towards the tip. This measure ends at exactly 0.

        Summed, deliberately, not averaged. The old distance term divided by the
        number of pieces still out, so a piece arriving both shrank the sum and
        shrank the divisor, and the average could hold still on real progress.

        Nearest target field per piece, rather than a min-cost assignment of
        pieces to target fields. The assignment is the exact remaining travel,
        but the two correlate at 0.996 over real games and agree on which moves
        help, so it is not worth an O(n^3) matching -- or a scipy dependency --
        on every step.
        """
        return float(sum(self.distanceToTarget[piece] for piece in player.positions))

    def _potential(self) -> float:
        """How good the position is for the agent, in roughly [-1, 0].

        The agent's own progress, and deliberately not its lead over the
        opponent. A lead can be held just as well by holding the opponent back
        as by advancing, and an agent trained on the difference took exactly
        that route: it finished with none of its 15 pieces home -- fewer than a
        random player -- while keeping the opponent from 14 down to 12. Only
        real progress of its own moves this.

        Ground covered, in [0, 1]: 0 at the opening, exactly 1 with everything
        home. The fraction of the opening's remaining travel that the agent has
        already walked off -- see :meth:`_progress` for what is measured.

        The sign is load-bearing and easy to get backwards -- an earlier version
        measured the ground *remaining*, putting the potential in [-1, 0], and
        that quietly paid the agent to do nothing. With a discount below 1 the
        shaping term for an unchanged position is ``(gamma - 1) * phi``, which
        for a negative potential is *positive*: 0.99*(-1) - (-1) = +0.01 every
        step, whatever the move. Over a 125-step game that is +1.25 against -1
        for losing, so stalling paid better than winning, and three training
        runs duly learned to stall. Measured from zero upwards the same term is
        (gamma - 1) * phi <= 0: standing still earns nothing, and dawdling near
        the goal costs a little.

        Also divided by the travel facing the agent at the opening, so an
        episode's shaping sums to about 1, the same order as the +/-1 for the
        result. Unnormalised the opening is 140 steps of travel, which would
        drown the result out entirely.
        """
        remaining = self._progress(self._player(self.selfSeat))
        return 1.0 - remaining / self.openingProgress

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

        Running out of moves counts as terminating here, because the cap is one
        of the game's rules rather than a harness time limit: the game really is
        over, so the potential is zeroed like any other ending.
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
        """Encoded legal moves for whoever is on turn, memoised per position.

        Generating them is the single most expensive thing the environment
        does, and one env step used to ask for them five times: the caller's
        ``action_masks()``, ``step``'s legality check, the opponent's own
        search, the mobility scalar in ``_observation``, and ``action_masks()``
        again inside ``_info``. Only two of those are separate positions.

        ``gameLength()`` is the position's version: every real move goes
        through ``playMove``, which appends to the move list, and
        ``currentPlayer()`` is itself derived from that count. Scoring a
        candidate move does *not* bump it -- ``moveApplied`` goes straight to
        the board -- but scoring never calls this either. The one thing that
        would fool the cache is hand-placing pieces without playing a move,
        which only tests do.
        """
        version = self.game.gameLength()
        if self._legalCache is not None and self._legalCache[0] == version:
            return self._legalCache[1]
        player = self.game.currentPlayer()
        moves = self.board.allValidMoves(player)
        normalized = self.normalizer.permuteMoves(moves, self._permutationKey(player))
        actions = [self.encodeAction(move) for move in normalized]
        self._legalCache = (version, actions)
        return actions

    def _agentMobility(self) -> int:
        """How many moves the agent has. Free while the agent is on turn.

        Which is the normal case: the observation is built at the end of a
        step, with the opponent's reply already played. Only a terminal
        position can leave someone else on turn, and there it is worth one
        generation rather than complicating the scalar's meaning.
        """
        if self._isAgentsTurn():
            return len(self._legalActions())
        return len(self.board.allValidMoves(self._player(self.selfSeat)))

    def encodeAction(self, move: MoveEndpoints) -> int:
        return int(move[0]) * self.fieldCount + int(move[-1])

    def decodeAction(self, action: int) -> MoveEndpoints:
        return (action // self.fieldCount, action % self.fieldCount)

    # ------------------------------------------------------------ game steps

    def _isAgentsTurn(self) -> bool:
        return self.game.currentPlayer().identifier == self.selfSeat

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
        """Whether this seat's home corner sits on the negative-x side.

        Fixed per seat, not recomputed from *current* piece positions: a
        player's pieces drift away from their home corner over the game and,
        for a seat whose corner sits near the coordinate origin, the sum of
        their x-coordinates can hover near zero and cross it back and forth
        almost every ply as individual pieces move. Each crossing swapped the
        whole canonical frame the observation was built in -- a discontinuity
        training never produced, because seat 1's own corner never crosses
        that threshold in practice, so this always resolved to one constant
        answer for the only seat that was ever trained on. Measured: a
        checkpoint at 99% against a heuristic from seat 1 lost every one of
        20 games from seat 2 with the *current*-position version, and split
        roughly evenly once this used the fixed home corner instead.
        """
        startPositions = self._player(seat).startPositions
        return bool(np.sum(self.normalizer.sumCoordsX(startPositions)) < 0)

    @staticmethod
    def _rasterCell(coord: tuple[int, int]) -> tuple[int, int]:
        """Where a field sits on the 17x17 square, as (row, col).

        This is ``fieldNumber`` split back into its two halves -- that scheme
        embeds the hex board in a square grid, which is exactly what a
        convolution needs.
        """
        x, y = coord
        return y + 8, x + 8

    def _observation(self) -> dict[str, np.ndarray]:
        """The board as a 17x17 picture plus a few scalars, agent's view.

        Laid out spatially rather than as a flat vector so a convolution can
        see that neighbouring fields are neighbours. A flat vector hides that,
        leaving the network to learn adjacency from data it does not have.

        Planes: own pieces, opponent pieces, and which cells are real fields --
        168 of the 289 squares are outside the star, and without that plane an
        empty field and the void look identical.

        Scalars: progress through the move budget, own pieces home, opponent
        pieces home, mobility. The zones are deliberately absent: after
        normalisation the start is always fields 0-14 and the target 102-120,
        so they are constant and carry nothing.
        """
        agent = self._player(self.selfSeat)
        opponent = self._player(self.otherSeat)
        state = self.normalizer.permute(self.board.boardState(), self._permutationKey(agent))

        board = np.zeros((3, 17, 17), dtype=np.float32)
        rows, cols = self.rasterIndex[:, 0], self.rasterIndex[:, 1]
        board[0, rows, cols] = (state == self.selfSeat).astype(np.float32)
        board[1, rows, cols] = (state == self.otherSeat).astype(np.float32)
        board[2] = self.boardMask

        scalars = np.array(
            [
                self.game.gameLength() / self.game.MAX_MOVES,
                len(agent.positions & agent.endPositions) / PIECES_PER_PLAYER,
                len(opponent.positions & opponent.endPositions) / PIECES_PER_PLAYER,
                min(self._agentMobility(), self.fieldCount) / self.fieldCount,
            ],
            dtype=np.float32,
        )
        return {"board": board, "scalars": scalars}

    def _info(self, illegalAction: bool = False, outcome: float = 0.0) -> dict[str, Any]:
        # outcome is the unshaped result, +1/-1/0. Evaluation must read this
        # rather than the reward, or shaping would be scored as if it were
        # winning.
        return {
            "action_mask": self.action_masks(),
            "illegalAction": illegalAction,
            "outcome": outcome,
        }
