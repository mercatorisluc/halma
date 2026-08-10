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

    ``opponentSampling`` is the fraction of episodes in which a *neural*
    opponent plays its own distribution instead of its argmax move. A frozen
    checkpoint answers a given position identically every time, so a run
    against one revisits a narrow band of the game tree however many episodes
    it plays; sampling widens that band with moves the opponent itself
    considers plausible, which is the difference between this and
    ``randomOpeningPlies`` -- that varies the position with uniformly random
    moves, this varies the opponent with its own. The draw is per episode
    rather than per move, so a game is played against one coherent opponent,
    and the argmax opponent still supplies ``1 - opponentSampling`` of the
    episodes: a mixture on purpose, because the one run that replaced the
    standard case outright rather than mixing it lost measurably on the
    distribution it stopped seeing (ARCHITECTURE.md). Heuristics have no
    distribution to sample, so a value above zero without any checkpoint to
    apply it to is refused rather than silently ignored.

    ``randomOpeningPlies`` plays that many uniformly random legal moves before
    handing control to the agent, so an episode starts from a varied position
    rather than the one opening the game always has. It exists because two
    argmax policies facing each other have only two possible games -- see
    ``scripts/compareCheckpoints.py`` -- and a frozen checkpoint opponent is
    close enough to that for training to see the same handful of positions
    over and over. The plies are counted in total, not per side, and are
    played by whoever is on turn -- so an even count gives both sides the same
    number of them and an odd one hands the extra ply to whoever the play
    order put first. They are part of the reset, not of the episode: the agent
    is never asked for them and is never trained on them,
    and ``previousPotential`` is taken afterwards so the shaping still
    telescopes from wherever the random opening left the board.

    ``parityWeight`` is how hard the potential charges for pieces that still
    owe a parity change -- see ``_parityMismatch``. Jumps cannot leave the
    class ``(x mod 2, y mod 2)`` they start in, so a straggler in a class with
    no open target field left must spend at least one single step to get out
    of it, and remaining *distance* cannot see that at all. 0 restores the
    travel-only potential every result recorded before 2026-08-07 was measured
    with, and is the control this should be measured against.
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
        opponentSampling: float = 0.0,
        randomOpeningPlies: int = 0,
        parityWeight: float = 0.25,
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
        # Refused rather than ignored: with no checkpoint anywhere in the
        # draw there is nothing whose distribution could be sampled, so the
        # setting would do nothing at all and the run would look like it had
        # varied opponents when it had not.
        if not 0.0 <= opponentSampling <= 1.0:
            raise ValueError("opponentSampling is a probability, so it must lie in [0, 1]")
        if opponentSampling > 0.0 and opponentModel is None and not opponentModelPool:
            raise ValueError(
                "opponentSampling needs opponentModel or opponentModelPool: a heuristic has "
                "no distribution to sample"
            )
        self.opponentSampling = opponentSampling
        # gamma has to be the discount the agent is trained with, or the
        # shaping stops being policy-invariant. shapingWeight = 0 turns shaping
        # off, which is how to measure whether it is earning its keep.
        self.shapingWeight = shapingWeight
        self.gamma = gamma
        if randomOpeningPlies < 0:
            raise ValueError("randomOpeningPlies cannot be negative")
        self.randomOpeningPlies = randomOpeningPlies
        if parityWeight < 0.0:
            raise ValueError("parityWeight is a penalty weight, so it cannot be negative")
        # How hard the potential penalises pieces that still owe a parity
        # change -- see _parityMismatch. 0 restores the travel-only potential
        # every result recorded before 2026-08-07 was measured with, and is
        # the control this should be measured against.
        self.parityWeight = parityWeight
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
        self.boardMask = np.zeros((17, 17), dtype=np.float32)
        self.boardMask[self.rasterIndex[:, 0], self.rasterIndex[:, 1]] = 1.0
        # The permutation taking this seat's view to the canonical one is
        # fixed for the env's lifetime -- _needsFlip keys on the home corner,
        # which never moves -- so the geometry planes below can be painted
        # once here rather than rebuilt on every observation.
        self._selfKey = self._permutationKey(self._player(self.selfSeat))
        # Which jump-invariant class each field belongs to -- see
        # _parityClasses. Needed before the geometry planes, one of which is
        # derived from it, and before any potential is taken.
        self.parityClass = self._parityClasses()
        self.geometryPlanes = self._buildGeometryPlanes()
        self.boardPlanes = 3 + int(self.geometryPlanes.shape[0])
        self.observation_space = spaces.Dict(
            {
                # Own pieces, opponent pieces, which cells of the 17x17 square
                # are real fields at all -- 168 of 289 are not, and without
                # that plane a convolution cannot tell an empty field from the
                # void outside the star -- and then the geometry planes built
                # by _buildGeometryPlanes, which document their own layout.
                "board": spaces.Box(
                    low=0.0, high=1.0, shape=(self.boardPlanes, 17, 17), dtype=np.float32
                ),
                "scalars": spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32),
            }
        )

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

    def _buildGeometryPlanes(self) -> np.ndarray:
        """The board's fixed geometry, as planes the convolutions can read.

        Eleven planes, in this order::

            0  own target zone         6-9  jump class, one-hot over 4 classes
            1  own start zone           10  closeness to a *same-class* own
            2  opponent target zone         target
            3  opponent start zone
            4  closeness to own target
            5  closeness to the opponent's target

        Planes 6-10 are the jump-parity structure -- see
        :meth:`_parityClasses`. The one-hot group is the class itself, which a
        convolution reads off per cell; :meth:`_classPlanes` explains why it is
        one-hot over the whole raster rather than a compact code. Plane 10 is
        the same distance map as plane 4 but restricted to targets of the
        field's own class, which answers "how far can this piece get without
        ever taking a single step". It differs from plane 4 on 56 of the 121
        fields, so it is a second map rather than a rescaling of the first --
        and it, rather than the class planes, is what carries the *long-range*
        comparison, since two 3x3 convolutions see only 5x5 and a piece is
        usually nowhere near the target zone.

        All eleven are **constant**: the four zones are fixed sets of fields,
        the three maps are derived from the distance matrix, the class of a
        cell is a property of its coordinates, and the canonical frame does not
        move during a game. So they carry no per-position information
        at all, and the reason to feed them anyway is specific -- a convolution
        shares its weights across the board and is therefore translation
        invariant, so the conv stack cannot know *where* a pattern it has found
        is sitting. Nothing in the three piece planes says which end of the
        star a group of pieces is at. These planes are the positional encoding
        that breaks that invariance, which is why they are worth their
        parameters even though a Kolmogorov-minimal encoding would omit them.
        The gain is confined to the conv stack: the flatten hands the head
        every cell separately, so the head already had position.

        Four distinct zones, not two. The two seats do not face each other
        across the star -- ``Initializer`` puts seat 1 on the bottom corner
        travelling to the top and seat 2 on the left corner travelling to the
        right, so own start, own target, opponent start and opponent target are
        four different corners of the six, sharing only the single field where
        two of them touch.

        The distance maps are the one thing here that is not binary, and
        deliberately: they are ``1 - d / max(d)``, so a cell carries how *close*
        it is to the zone rather than merely whether it is in it. A binary zone
        plane tells the convolution where the goal is; the map tells it which
        way is forwards from anywhere on the board, which is the same signal
        ``_progress`` shapes the reward with. Cells outside the star stay 0,
        which the mask plane already distinguishes from a genuinely distant
        field.
        """
        agent = self._player(self.selfSeat)
        opponent = self._player(self.otherSeat)
        zones = (
            agent.endPositions,
            agent.startPositions,
            opponent.endPositions,
            opponent.startPositions,
        )
        planes = [self._rasterPlane(self._zoneVector(zone)) for zone in zones]
        planes.append(self._rasterPlane(self._closenessVector(self.distanceToTarget)))
        planes.append(self._rasterPlane(self._closenessVector(self._targetDistances(opponent))))
        planes.extend(self._classPlanes())
        planes.append(self._rasterPlane(self._closenessVector(self._sameClassDistances(agent))))
        return np.stack(planes).astype(np.float32)

    def _parityClasses(self) -> np.ndarray:
        """Field id -> which of the four jump-invariant classes it belongs to.

        **Every jump preserves ``(x mod 2, y mod 2)``.** The six jump deltas on
        this board are ``(+-2, 0)``, ``(0, +-2)`` and ``(+-2, -+2)`` -- all of
        them even in both coordinates -- so a jump, and therefore a whole
        jump chain however long, can never leave the class it started in. Only
        a single step changes it, and a single step covers one field where a
        jump covers two.

        That splits the 121 fields into four classes of 37/28/28/28, and it is
        the sharper invariant than the parity of the *distance*: a delta of
        ``(1, 1)`` has hex distance 2, which an even-distance argument would
        call jump-reachable, while the class correctly says it is not.

        Verified against the board rather than assumed -- ``jumpNeighbours``
        was enumerated over all 121 fields and every delta came out even.
        """
        return np.array(
            [(field.coord[0] % 2) * 2 + (field.coord[1] % 2) for field in self.board.fields]
        )

    def _parityMismatch(self, player: HalmaPlayer) -> int:
        """Pieces that must still change class, which costs a single step each.

        The pieces not yet home have to end up on the target fields not yet
        filled. Within a class that is free -- jumps alone can do it -- but a
        class holding more stragglers than it has open target fields must send
        the surplus elsewhere, and every one of those crossings costs at least
        one single step. Summed over the classes, that surplus is a **lower
        bound on the single steps still owed**, no matter how good the jump
        chains are.

        It is worth having because nothing else in the observation or the
        reward expresses it. ``_progress`` counts board distance, and distance
        is blind to this: two positions with identical remaining travel can
        differ by several forced steps.

        Zero at the opening, and that is not a coincidence -- start and target
        zone have the same class distribution (6/3/3/3), so the opening is
        already perfectly matched and a jump-only solution is not ruled out.
        Zero again once every piece is home, since both sets are then empty.
        So this measures a detour the middle game can wander into and back out
        of, which is exactly the shape a shaping term should have.

        Counted on the class *labels* only, so it does not matter that the
        canonical frame permutes which class is which -- a rotation maps the
        four classes onto each other bijectively, and a sum over all four is
        invariant under that relabelling.
        """
        stragglers = player.positions - player.endPositions
        openTargets = player.endPositions - player.positions
        pieces = np.bincount(self.parityClass[sorted(stragglers)], minlength=4)
        targets = np.bincount(self.parityClass[sorted(openTargets)], minlength=4)
        return int(np.maximum(pieces - targets, 0).sum())

    def _sameClassDistances(self, player: HalmaPlayer) -> np.ndarray:
        """Steps to the nearest target field **of the field's own class**.

        The jump-only reach, where ``_targetDistances`` gives the reach with
        steps allowed. The two differ on 56 of the 121 fields -- by one step on
        50 of them and by two on 6 -- so this is not a rescaling of the map it
        sits beside, and the difference is precisely the fields whose nearest
        target is across a class boundary.
        """
        distances = self.board.distanceMatrix
        targets = sorted(player.endPositions)
        return np.array(
            [
                min(
                    distances[field][target]
                    for target in targets
                    if self.parityClass[target] == self.parityClass[field]
                )
                for field in range(self.fieldCount)
            ],
            dtype=np.float64,
        )

    def _classPlanes(self) -> list[np.ndarray]:
        """The jump class as four one-hot planes over the whole raster.

        **One-hot rather than a number, because the four classes have no
        order.** Every step delta is odd in at least one coordinate -- the
        three of them are ``(1,0)``, ``(0,1)`` and ``(1,1)`` mod 2 -- so from
        any class a single step reaches *all three* others. Measured on the
        board: the classes form a complete graph, every pair exactly one step
        apart. There are no near and far classes.

        So any encoding carrying a metric misstates the geometry. A single
        plane holding 1..4 would tell a weighted sum that class 1 and class 4
        are far apart and 1 and 2 are close, and both are wrong. A two-bit
        ``(x mod 2, y mod 2)`` code -- which is what this replaced -- makes a
        weaker version of the same mistake: it puts ``(0,0)`` two bits from
        ``(1,1)`` and one bit from ``(1,0)``, where the board puts both at one
        step. Four mutually equidistant indicators state exactly what is true
        and nothing else. The cost is 1,152 parameters in the first
        convolution, out of some two million.

        **Painted over all 289 cells, not only the 121 real fields.** Parity is
        a property of the raster, not of the board: ``row % 2`` is defined
        everywhere, and stopping it at the edge of the star would break the
        checkerboard exactly where a 3x3 kernel straddles the boundary,
        inventing a discontinuity the geometry does not have. This is the one
        plane group where filling the void is reading the raster rather than
        making data up -- a distance map has nothing to say about a cell no
        piece can occupy, which is why those stay 0 and lean on the mask.

        The labels are canonical-frame ones (``rasterCell`` is ``(y + 8,
        x + 8)`` and 8 is even, so the raster's parity is the coordinate's).
        They need not agree with ``self.parityClass``, whose labels come from
        the raw frame: nothing cross-references the two, and each is internally
        consistent. A rotation permutes the labels bijectively, which is also
        why ``_parityMismatch`` can sum over all four and ignore the question.
        """
        rows = np.arange(17).reshape(17, 1)
        cols = np.arange(17).reshape(1, 17)
        classes = (cols % 2) * 2 + (rows % 2)
        return [(classes == index).astype(np.float32) for index in range(4)]

    def _zoneVector(self, zone: set[int]) -> np.ndarray:
        vector = np.zeros(self.fieldCount, dtype=np.float64)
        vector[sorted(zone)] = 1.0
        return vector

    @staticmethod
    def _closenessVector(distances: np.ndarray) -> np.ndarray:
        """Distances turned into ``1`` at the zone and ``0`` at the far corner."""
        return 1.0 - distances / float(distances.max())

    def _rasterPlane(self, values: np.ndarray) -> np.ndarray:
        """A per-field vector in raw id order, painted onto the canonical raster.

        Permuted with exactly the call ``_observation`` makes for the board
        state, so a plane built here cannot come out of alignment with the
        piece planes -- both are indexed by canonical field id by construction
        rather than by an argument about which direction the permutation runs.
        """
        plane = np.zeros((17, 17), dtype=np.float32)
        canonical = self.normalizer.permute(values, self._selfKey)
        plane[self.rasterIndex[:, 0], self.rasterIndex[:, 1]] = canonical
        return plane

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
        # How the drawn checkpoint will play this episode: its best move, or
        # its distribution. Drawn after the opponent itself, since it is that
        # opponent's flag being set, and only when asked for -- guarding the
        # draw keeps the random stream of every run that does not use this
        # byte-identical to before it existed, so earlier results stay
        # reproducible from their seeds.
        if self._opponentNeural is not None and self.opponentSampling > 0.0:
            self._opponentNeural.deterministic = self.np_random.random() >= self.opponentSampling
        # The engine has its own generator; seeding it is what makes a whole
        # episode reproducible, since seat order and the opponent's tie-breaks
        # both draw from it.
        self.game.seed(seed)
        self._seatPlayers()
        # A new game restarts the move count, so last episode's entry would
        # look current.
        self._legalCache = None
        # Before anyone's policy is consulted: a random opening, if asked for.
        # Played first so the play order still decides who makes the first of
        # those moves, exactly as it decides who makes the first real one.
        self._playRandomOpening()
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

        **Travel is not the whole story, and the parity penalty is the rest.**
        Distance is blind to the jump structure: a position can have exactly
        the remaining travel of another and still owe several single steps that
        the other does not, because its stragglers sit in classes with no open
        target field left (:meth:`_parityMismatch`). Those steps are real and
        forced, so the potential charges for them.

        Both ends of the scale survive it. The mismatch is 0 at the opening --
        start and target zone have the same class distribution -- and 0 again
        once every piece is home, so the potential still runs from exactly 0 to
        exactly 1 and the penalty only bites in between. It is a detour that
        can be wandered into and back out of, not a shift of the scale.

        **The penalty scales the travel term rather than being subtracted from
        it**, and that is not cosmetic. Subtracting was tried first and breaks
        two properties at once. Travel starts at 0, so an early ply with any
        mismatch at all goes negative -- measured, 5.7% of plies down to -0.024
        at weight 0.25 -- and a negative potential is the exact bug that taught
        three training runs to stall. Clamping that at zero fixes the sign and
        then breaks the other property instead: two consecutive clamped plies
        both have ``phi = 0``, so the shaping between them is exactly 0 and the
        signal-on-every-step guarantee is gone, which the test suite caught.

        Multiplying has neither failure. ``covered`` is in [0, 1] and the
        factor is too, so the product cannot leave [0, 1] or go flat while the
        agent is making progress; both terms stay monotone, so advancing always
        helps and clearing parity debt always helps. What it means is that the
        penalty charges a *fraction of the ground already covered* -- parity
        debt is cheap while there is little to lose and expensive near the end,
        which is also when it is genuinely harder to fix.
        """
        agent = self._player(self.selfSeat)
        covered = 1.0 - self._progress(agent) / self.openingProgress
        if self.parityWeight <= 0.0:
            return covered
        owed = self.parityWeight * self._parityMismatch(agent) / PIECES_PER_PLAYER
        # max() only matters for a weight above 1, which nothing uses; it is
        # here so no setting can put the factor -- and with it the potential --
        # below zero.
        return covered * max(0.0, 1.0 - owed)

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

    def _playRandomOpening(self) -> None:
        """Open with ``randomOpeningPlies`` uniformly random legal moves.

        Drawn from ``self.np_random``, which ``reset()`` has just seeded, so
        the opening is part of what an episode seed reproduces rather than a
        second source of randomness beside it. Both sides play theirs: the
        point is to move the *position* off the one fixed opening, and only
        randomising the agent's own moves would leave the opponent replying
        from its usual book.

        Nobody can have won this early -- the shortest game here is far longer
        than any opening worth randomising -- but the winner check is kept
        anyway, so a larger ``randomOpeningPlies`` cannot walk past the end of
        a game.
        """
        for _ in range(self.randomOpeningPlies):
            if self.game.winner() is not None:
                break
            player = self.game.currentPlayer()
            moves = self.board.allValidMovesWithWay(player)
            self.game.playMove(player, moves[int(self.np_random.integers(len(moves)))])

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
        empty field and the void look identical -- followed by the six fixed
        geometry planes :meth:`_buildGeometryPlanes` describes. Only the first
        three change during a game; the geometry is painted once and copied in.

        The geometry used to be left out on the grounds that it is constant
        after normalisation and therefore carries nothing. That is true of the
        information content and false of what a convolution can use: shared
        weights are translation invariant, so without those planes the conv
        stack cannot tell which end of the board it is looking at. See
        :meth:`_buildGeometryPlanes`.

        Scalars: progress through the move budget, own pieces home, opponent
        pieces home, mobility, and the parity mismatch -- the single steps the
        agent still owes no matter how its jump chains fall
        (:meth:`_parityMismatch`).
        """
        agent = self._player(self.selfSeat)
        opponent = self._player(self.otherSeat)
        state = self.normalizer.permute(self.board.boardState(), self._permutationKey(agent))

        board = np.zeros((self.boardPlanes, 17, 17), dtype=np.float32)
        rows, cols = self.rasterIndex[:, 0], self.rasterIndex[:, 1]
        board[0, rows, cols] = (state == self.selfSeat).astype(np.float32)
        board[1, rows, cols] = (state == self.otherSeat).astype(np.float32)
        board[2] = self.boardMask
        board[3:] = self.geometryPlanes

        scalars = np.array(
            [
                self.game.gameLength() / self.game.MAX_MOVES,
                len(agent.positions & agent.endPositions) / PIECES_PER_PLAYER,
                len(opponent.positions & opponent.endPositions) / PIECES_PER_PLAYER,
                min(self._agentMobility(), self.fieldCount) / self.fieldCount,
                self._parityMismatch(agent) / PIECES_PER_PLAYER,
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
