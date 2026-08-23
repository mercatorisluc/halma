"""Measure how the existing bots do against each other.

Run before training anything:

    python -m scripts.baseline --games 200

These numbers are the yardstick. A trained agent is only worth keeping if it
beats them, and without them "the bot seems better" is an opinion. Play order
is randomised per game, so first-move advantage averages out.
"""

from __future__ import annotations

import argparse
from functools import partial
from itertools import combinations
from multiprocessing import Pool

from heuristics.strategy import STRATEGY_NAMES
from scripts.matchStats import Bot, Result, playMatch


def playPair(strategyA: str, strategyB: str, games: int, seed: int) -> Result:
    """One matchup, as a picklable top-level function for the worker pool."""
    return playMatch(Bot(strategyA), Bot(strategyB), games, seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100, help="games per matchup")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--bots",
        nargs="+",
        default=STRATEGY_NAMES,
        help="restrict to these bots; the full panel by default",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="matchups to play in parallel; each has its own board, so they do not interfere",
    )
    args = parser.parse_args()

    unknown = sorted(set(args.bots) - set(STRATEGY_NAMES))
    if unknown:
        raise SystemExit(f"unknown bots {unknown}; known: {STRATEGY_NAMES}")

    # Every pairing, so the ranking is complete rather than inferred. Seats do
    # not confer an advantage -- play order is shuffled per game -- so each
    # unordered pair is played once. random against itself is kept as a sanity
    # check: two aimless players should never finish.
    matchups = list(combinations(args.bots, 2))
    if "random" in args.bots:
        matchups.insert(0, ("random", "random"))

    print(f"{args.games} games per matchup, seed {args.seed}\n")
    header = f"{'seat 1':<18} {'seat 2':<18} {'win%':>14}  {'draws':>6}  {'avg moves':>9}"
    print(header)
    print("-" * len(header))
    play = partial(playPair, games=args.games, seed=args.seed)
    if args.jobs > 1:
        with Pool(args.jobs) as pool:
            results = pool.starmap(play, matchups)
    else:
        results = [play(a, b) for a, b in matchups]
    for (a, b), r in zip(matchups, results, strict=True):
        winRate = f"{r.winRate * 100:5.1f} +/- {r.marginOfError * 100:4.1f}"
        print(f"{a:<18} {b:<18} {winRate:>14}  {r.draws:>6}  {r.moves / r.games:>9.0f}")


if __name__ == "__main__":
    main()
