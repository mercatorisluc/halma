"""Play Halma against a trained policy, in the pygame window.

    python -m scripts.playAgainstAgent
    python -m scripts.playAgainstAgent --model models/tunedEnt001 --sampled

``main.py`` seats ``lookahead2``, the strongest bot. This seats a network
instead, so its play can be watched rather than only scored -- a win rate says
that it wins, not what it does, and the specialisation the evaluation found
(99% against the bot it trained on, ~90% against random) is the kind of thing
that shows itself in the moves.

The human plays seat 2 because the policy has to play seat 1: its observation
is built for ``HalmaEnv.AGENT_SEAT``. Seats differ only in which triangle they
start from, and play order is randomised as in every other game here, so this
costs the human nothing.
"""

from __future__ import annotations

import argparse

from env.neuralPlayer import NeuralComputer
from game.gameManager import InteractiveGame
from game.player import HumanPlayer
from visual.gameVisualization import GameVisualization


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/tunedEnt000", help="checkpoint to play")
    parser.add_argument(
        "--sampled",
        action="store_true",
        help="play the policy's distribution instead of its best move",
    )
    parser.add_argument("--seed", type=int, default=None, help="fixes who moves first")
    args = parser.parse_args()

    agent = NeuralComputer(1, args.model, deterministic=not args.sampled)
    game = InteractiveGame()
    game.seed(args.seed)
    game.initGame([agent, HumanPlayer(2)])
    agent.attachTo(game)

    print(f"{args.model} on seat 1, you on seat 2")
    GameVisualization(game).visualizeGame()


if __name__ == "__main__":
    main()
