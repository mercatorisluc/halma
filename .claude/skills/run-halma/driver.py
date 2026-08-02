#!/usr/bin/env python
"""Programmatic handle on the Halma app -- launch it, click it, screenshot it.

``main.py`` opens a pygame window and blocks in an infinite loop, which is the
human path and useless to an agent. This exposes the same front-end one frame
at a time, over a line-based REPL on stdin, with synthetic clicks and keys and
PNG screenshots. It runs headless (``SDL_VIDEODRIVER=dummy``) so no window
appears and nothing needs a display.

Modes (first argv):

    repl      (default) interactive REPL, see COMMANDS below
    smoke     scripted end-to-end run: GUI move, bot reply, screenshots
    engine    headless bot-vs-bot game, no pygame -- game/ + heuristics/
    env       steps HalmaEnv with random legal actions -- env/
    policy    loads a checkpoint and plays it against a bot -- env/neuralPlayer

The REPL, not the modes, is the point: the modes are one canned path each,
while the REPL lets you reach an arbitrary position and look at it.
"""

from __future__ import annotations

import os
import random
import sys

# Running a script by path puts the script's own directory on sys.path, not the
# repo root, so `import game` would fail from anywhere. The root is three levels
# up: <root>/.claude/skills/run-halma/driver.py.
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)

# Must precede any pygame display use: picks the null video backend, so
# set_mode() gives a real drawable Surface with no window and no display server.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from game.gameManager import ComputedGame, InteractiveGame
from game.player import HumanPlayer
from visual.gameVisualization import GameVisualization

COMMANDS = """\
new [seed=N] [agent=models/tunedEnt001] [sampled]
                    start a game. Without agent= the human (seat 1) faces
                    lookahead2, as main.py does. With agent= a policy takes
                    seat 1 and the human seat 2, as scripts/playAgainstAgent
                    does -- the observation is built for seat 1 only.
tick [n=1]          run n iterations of the front-end's main loop. A bot on
                    move plays one move per tick; a human on move idles.
state               seat on move, human?, move count, pieces home, winner
moves [n=10]        legal (start,end) moves for the player on move
move START END      click START then END and tick -- plays a human move
click FIELD_ID      one synthetic click at that field's pixel
key LEFT|RIGHT|UP|DOWN
                    history back / forward, rotate +60 / -60
auto [n=1]          play n human moves by picking a legal one at random,
                    ticking the bot's reply after each
ss PATH             screenshot the current frame to PATH (.png)
quit
"""


def fail(message: str) -> None:
    print(f"error: {message}", flush=True)


class Driver:
    """One live front-end plus the poking tools the main loop does not expose."""

    def __init__(self) -> None:
        self.viz: GameVisualization | None = None
        self.rng = random.Random(0)

    # -- launch ------------------------------------------------------------

    def new(self, args: list[str]) -> None:
        seed = None
        agentPath = None
        sampled = False
        for arg in args:
            if arg.startswith("seed="):
                seed = int(arg.split("=", 1)[1])
            elif arg.startswith("agent="):
                agentPath = arg.split("=", 1)[1]
            elif arg == "sampled":
                sampled = True
            else:
                return fail(f"unknown argument {arg!r}")

        game = InteractiveGame()
        game.seed(seed)
        if agentPath is None:
            game.initStandardGame()
        else:
            # Imported lazily: it pulls in torch (~530 MB) and takes seconds,
            # which a plain GUI run should not pay for.
            from env.neuralPlayer import NeuralComputer

            agent = NeuralComputer(1, agentPath, deterministic=not sampled)
            game.initGame([agent, HumanPlayer(2)])
            agent.attachTo(game)

        if self.viz is not None:
            pygame.display.quit()
        self.viz = GameVisualization(game)
        self.viz.prepareGameVisualization()
        self.render()  # draw the opening frame, without letting anyone move
        self.state([])

    def require(self) -> GameVisualization | None:
        if self.viz is None:
            fail("no game -- run `new` first")
        return self.viz

    # -- the main loop, one frame at a time --------------------------------

    def tick(self, args: list[str]) -> None:
        viz = self.require()
        if viz is None:
            return
        count = int(args[0]) if args else 1
        game = viz.game
        # The body of GameVisualization.runGame, minus the `while` and the
        # 10 fps clock. Kept in step with that method by hand -- it is ten
        # lines and has not changed in the life of this repo.
        for _ in range(max(count, 1)):
            self.render()
            viz.handleEvents()
            viz.input.adaptToHumanInteraction(game)
            if viz.input.waitingForHumanMove:
                viz.input.handleHumanMove(game)
            elif viz.playback.moveTraveler == len(game.moves):
                viz.playback.playNextMove()
            if game.winner() is not None:
                break
        self.state([])

    def render(self) -> None:
        """Draw one frame from the current state, without advancing anything.

        The real loop draws *before* it handles input, so the buffer always
        lags the state by a frame. `ss` calls this first so a screenshot shows
        what is true now rather than what was true before the last click.
        """
        viz = self.viz
        assert viz is not None
        viz.renderer.drawBoard(viz.game.board)
        viz.renderer.drawLastMove(viz.playback.moveTraveler)
        viz.renderer.drawInteractiveElements(
            viz.input.clickedField,
            viz.input.waitingForHumanMove,
            viz.input.validHumanMoves,
        )
        pygame.display.flip()

    # -- observation -------------------------------------------------------

    def state(self, args: list[str]) -> None:
        viz = self.require()
        if viz is None:
            return
        game = viz.game
        player = game.currentPlayer()
        home = {p.identifier: len(p.positions) - len(p.nonArrived) for p in game.players}
        print(
            f"seat {player.identifier} on move (human={player.isHuman()})  "
            f"moves={len(game.moves)} cursor={viz.playback.moveTraveler}  "
            f"home={home}  winner={game.winner()}",
            flush=True,
        )

    def moves(self, args: list[str]) -> None:
        viz = self.require()
        if viz is None:
            return
        limit = int(args[0]) if args else 10
        legal = viz.game.board.allValidMoves(viz.game.currentPlayer())
        print(f"{len(legal)} legal: {legal[:limit]}", flush=True)

    # -- input -------------------------------------------------------------

    def pixelOf(self, fieldId: int) -> tuple[float, float]:
        viz = self.viz
        assert viz is not None
        # Through the projector, so a rotated board is clicked where it is
        # actually drawn -- the same table hit-testing uses.
        return viz.projector.visualPosition(viz.game.board.coordFromId(fieldId))

    def click(self, args: list[str]) -> None:
        viz = self.require()
        if viz is None:
            return
        if not args:
            return fail("usage: click FIELD_ID")
        pos = self.pixelOf(int(args[0]))
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=1))
        print(f"clicked field {args[0]} at {pos}", flush=True)

    def move(self, args: list[str]) -> None:
        viz = self.require()
        if viz is None:
            return
        if len(args) != 2:
            return fail("usage: move START END")
        if not viz.game.currentPlayer().isHuman():
            return fail("not the human's turn -- `tick` until it is")
        before = len(viz.game.moves)
        self.clickPair(int(args[0]), int(args[1]))
        if len(viz.game.moves) == before:
            return fail(f"move {args[0]}->{args[1]} rejected -- not legal from here")
        self.tickQuiet(1)  # the bot's reply
        self.state([])

    def clickPair(self, start: int, end: int) -> None:
        """Two clicks, one per frame -- see the one-frame lag in `render`.

        Both in a single frame does not work: the second click is only read as
        the end of a pair when ``waitingForHumanMove`` is already set, and that
        flag is refreshed *after* the events are drained.
        """
        self.click([str(start)])
        self.tickQuiet(1)
        self.click([str(end)])
        self.tickQuiet(1)

    def key(self, args: list[str]) -> None:
        viz = self.require()
        if viz is None:
            return
        names = {
            "LEFT": pygame.K_LEFT,
            "RIGHT": pygame.K_RIGHT,
            "UP": pygame.K_UP,
            "DOWN": pygame.K_DOWN,
        }
        if not args or args[0].upper() not in names:
            return fail(f"usage: key {'|'.join(names)}")
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=names[args[0].upper()]))
        self.tickQuiet(1)
        print(
            f"key {args[0].upper()}  flipAngle={viz.projector.flipAngle} "
            f"cursor={viz.playback.moveTraveler}/{len(viz.game.moves)}",
            flush=True,
        )

    def auto(self, args: list[str]) -> None:
        viz = self.require()
        if viz is None:
            return
        for _ in range(int(args[0]) if args else 1):
            if viz.game.winner() is not None:
                break
            if not viz.game.currentPlayer().isHuman():
                self.tickQuiet(1)
                continue
            legal = viz.game.board.allValidMoves(viz.game.currentPlayer())
            start, end = self.rng.choice(legal)
            before = len(viz.game.moves)
            self.clickPair(start, end)
            if len(viz.game.moves) == before:
                return fail(f"auto: the front-end rejected legal move {start}->{end}")
            self.tickQuiet(1)  # the bot's reply
        self.state([])

    def tickQuiet(self, count: int) -> None:
        out = sys.stdout
        sys.stdout = open(os.devnull, "w")  # noqa: SIM115
        try:
            self.tick([str(count)])
        finally:
            sys.stdout.close()
            sys.stdout = out

    # -- output ------------------------------------------------------------

    def ss(self, args: list[str]) -> None:
        viz = self.require()
        if viz is None:
            return
        if not args:
            return fail("usage: ss PATH")
        path = args[0]
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        self.render()
        pygame.image.save(viz.screen, path)
        print(f"wrote {path} ({os.path.getsize(path)} bytes)", flush=True)


def repl() -> int:
    driver = Driver()
    dispatch = {
        "new": driver.new,
        "tick": driver.tick,
        "state": driver.state,
        "moves": driver.moves,
        "move": driver.move,
        "click": driver.click,
        "key": driver.key,
        "auto": driver.auto,
        "ss": driver.ss,
    }
    print("halma driver ready. `help` for commands.", flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, *args = line.split()
        if name in ("quit", "exit"):
            break
        if name == "help":
            print(COMMANDS, flush=True)
            continue
        handler = dispatch.get(name)
        if handler is None:
            fail(f"unknown command {name!r} -- try `help`")
            continue
        try:
            handler(args)
        except Exception as exc:  # keep the session alive on a bad command
            fail(f"{type(exc).__name__}: {exc}")
    pygame.quit()
    return 0


# -- non-REPL modes --------------------------------------------------------


def engineMode(argv: list[str]) -> int:
    """game/ + heuristics/ with no pygame and no torch: one bot-vs-bot game."""
    game = ComputedGame()
    game.seed(int(argv[0]) if argv else 1)
    game.initStandardGame()
    winner = game.play()
    print(f"winner={winner} after {len(game.moves)} moves", flush=True)
    return 0 if winner is not None else 1


def envMode(argv: list[str]) -> int:
    """env/: reset, then random legal actions until the episode ends."""
    import numpy as np

    from env.halmaEnv import HalmaEnv

    steps = int(argv[0]) if argv else 40
    env = HalmaEnv()
    obs, info = env.reset(seed=3)
    print(f"obs keys={sorted(obs)} shapes={ {k: v.shape for k, v in obs.items()} }", flush=True)
    rng = np.random.default_rng(0)
    total = 0.0
    for i in range(steps):
        legal = np.flatnonzero(env.action_masks())
        obs, reward, terminated, truncated, info = env.step(int(rng.choice(legal)))
        total += reward
        if terminated or truncated:
            print(f"episode ended at step {i} outcome={info.get('outcome')}", flush=True)
            break
    print(f"{steps} steps ok, return={total:.3f}, last info keys={sorted(info)}", flush=True)
    return 0


def policyMode(argv: list[str]) -> int:
    """env/neuralPlayer: seat a checkpoint against a bot and play it out."""
    from env.neuralPlayer import NeuralComputer
    from game.player import Computer

    model = argv[0] if argv else "models/tunedEnt001"
    game = ComputedGame()
    game.seed(5)
    agent = NeuralComputer(1, model)
    game.initGame([agent, Computer(2, "advancedDistScore")])
    agent.attachTo(game)
    winner = game.play()
    print(f"{model} vs advancedDistScore: winner={winner} in {len(game.moves)} moves", flush=True)
    return 0


def smokeMode(argv: list[str]) -> int:
    """The full path an agent cares about: launch, look, move, verify, look."""
    outDir = argv[0] if argv else "/tmp/halma-driver"
    os.makedirs(outDir, exist_ok=True)
    driver = Driver()
    driver.new(["seed=7"])
    viz = driver.viz
    assert viz is not None
    driver.ss([f"{outDir}/01-opening.png"])

    # Seat 2 (the bot) may be on move first -- play order is randomised.
    driver.tickQuiet(2)
    assert viz.game.currentPlayer().isHuman(), "human never got the move"

    legal = viz.game.board.allValidMoves(viz.game.currentPlayer())
    start, end = legal[0]
    before = len(viz.game.moves)
    driver.clickPair(start, end)
    assert len(viz.game.moves) > before, "the synthetic clicks did not produce a move"
    driver.tickQuiet(1)  # the bot's reply
    driver.ss([f"{outDir}/02-after-human-move.png"])

    driver.auto(["20"])
    driver.ss([f"{outDir}/03-midgame.png"])

    driver.key(["UP"])
    assert viz.projector.flipAngle == 60, "UP did not rotate the board"
    driver.ss([f"{outDir}/04-rotated.png"])
    driver.key(["DOWN"])

    driver.key(["LEFT"])
    assert viz.playback.moveTraveler < len(viz.game.moves), "LEFT did not step back"
    driver.key(["RIGHT"])

    print(f"SMOKE OK -- screenshots in {outDir}", flush=True)
    pygame.quit()
    return 0


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "repl"
    rest = sys.argv[2:]
    modes = {
        "repl": lambda _: repl(),
        "smoke": smokeMode,
        "engine": engineMode,
        "env": envMode,
        "policy": policyMode,
    }
    if mode not in modes:
        print(f"usage: driver.py [{'|'.join(modes)}] ...", file=sys.stderr)
        return 2
    return modes[mode](rest)


if __name__ == "__main__":
    raise SystemExit(main())
