"""Score trained checkpoints against the heuristics, the yardstick.

    python -m scripts.evaluateAgainstBots models/Talos1.0 models/Talos1.1
    python -m scripts.evaluateAgainstBots models/Talos1.1 --games 60 --bots lookahead2

The checkpoints come first because both they and ``--bots`` take a list: with
the flag first, argparse hands it every remaining word and the positional is
left empty.

``scripts/baseline.py`` measures the heuristics against each other and
``scripts/compareCheckpoints.py`` measures checkpoints against each other. The
third side of that triangle -- a checkpoint against the bots -- existed only as
the final report of a training run, so re-measuring a checkpoint that already
exists meant training something. This is that report on its own.

Both argmax and sampled are printed. Argmax is the policy's actual play and is
what the strength claim should rest on; sampled is its distribution, weaker but
varied, and is the honest measure while a policy is still mid-training. Play
order is drawn per game, as in ``scripts/baseline.py``, so the first-mover
advantage averages out rather than being folded into the number invisibly.

``lookahead2`` is roughly twenty times slower per game than the scoring bots,
which is why ``--bots`` exists: it is worth including in a final measurement and
not in a quick check.
"""

from __future__ import annotations

import argparse

from sb3_contrib import MaskablePPO

from heuristics.strategy import STRATEGY_NAMES
from scripts.train import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", help="model paths to score")
    parser.add_argument("--games", type=int, default=100, help="games per bot per mode")
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument(
        "--bots",
        nargs="+",
        default=None,
        help=f"bots to play (default: all of {', '.join(STRATEGY_NAMES)})",
    )
    args = parser.parse_args()

    bots = args.bots or STRATEGY_NAMES
    width = max(len(path) for path in args.checkpoints)

    print(f"{args.games} games per bot per mode, seed {args.seed}\n")
    header = (
        f"{'checkpoint':<{width}} {'bot':<18} {'argmax win%':>14}  "
        f"{'sampled win%':>14}  {'argmax home%':>12}"
    )
    print(header)
    print("-" * len(header))
    for path in args.checkpoints:
        # Loaded once per checkpoint rather than per bot: the load is a network
        # deserialisation and the games against a scoring bot are not.
        model = MaskablePPO.load(path)
        for bot in bots:
            argmax = evaluate(model, bot, args.games, seed=args.seed)
            sampled = evaluate(model, bot, args.games, seed=args.seed, deterministic=False)
            argmaxRate = f"{argmax['winRate'] * 100:5.1f} +/- {argmax['marginOfError'] * 100:4.1f}"
            sampledRate = (
                f"{sampled['winRate'] * 100:5.1f} +/- {sampled['marginOfError'] * 100:4.1f}"
            )
            print(
                f"{path:<{width}} {bot:<18} {argmaxRate:>14}  {sampledRate:>14}  "
                f"{argmax['homeFraction'] * 100:11.1f}%"
            )


if __name__ == "__main__":
    main()
