"""How much of each teacher did a clone actually learn?

    python -m scripts.teacherAgreement models/teacherSingle models/teacherMixed \\
        --teachers calibrated calibratedCluster calibratedJump calibratedClusterJump straggler

`scripts/pretrain.py` reports agreement with its teachers as one number averaged
over all of them, and that average cannot answer the question a mixed teacher
set is meant to answer. A clone that copies one teacher perfectly and ignores
the other four scores the same average as one that covers all five moderately --
the first is narrow, the second is broad, and breadth is the entire reason to
mix teachers at all. So this reports the agreement **per teacher**.

How to read it:

- **high on one, low on the rest** -- the mixture collapsed. The clone found the
  teacher easiest to imitate and the rest became noise it learned to ignore.
- **low on all** -- blurred rather than broad. Averaging incompatible teachers
  move by move can land the policy between them, playing something none of them
  would play. This is the failure mode a mixed teacher is usually accused of.
- **moderate on all, and the *minimum* well above the single-teacher clone's
  minimum** -- broad, which is the outcome worth having.

The comparison is only meaningful on a shared position set, so positions are
collected once and every checkpoint is scored on the same ones. They come from
teacher-driven play in a real `HalmaEnv`, which is what makes them the positions
a clone would actually face: the driver rotates through the teachers so no
single one's trajectory dominates the sample.

Note that a clone agreeing *less* with a teacher is not automatically worse. The
number is imitation, not strength, and `scripts/evaluateAgainstBots.py` remains
the place strength claims come from.
"""

from __future__ import annotations

import argparse

import numpy as np
from sb3_contrib import MaskablePPO

from env.halmaEnv import HalmaEnv
from heuristics.strategy import Strategy

# Plies between recorded positions. Consecutive ones are near-duplicates and
# would mostly measure the board not moving, the same reason
# `scripts/variantAgreement.py` strides.
STRIDE = 5


def makeStrategy(name: str) -> Strategy:
    return Strategy(name)


def collect(
    teachers: list[str], opponent: str, positions: int, seed: int
) -> tuple[list[dict[str, np.ndarray]], list[np.ndarray], dict[str, list[int]]]:
    """Positions from teacher-driven play, with every teacher's move on each.

    Returns the observations, their action masks, and per teacher the action it
    would have played. The driver rotates so the sample is not one teacher's
    trajectory scored by the others.
    """
    strategies = {name: makeStrategy(name) for name in teachers}
    observations: list[dict[str, np.ndarray]] = []
    masks: list[np.ndarray] = []
    choices: dict[str, list[int]] = {name: [] for name in teachers}
    games = 0
    while len(observations) < positions:
        env = HalmaEnv(opponentStrategy=opponent)
        obs, _ = env.reset(seed=seed + games)
        driver = strategies[teachers[games % len(teachers)]]
        games += 1
        ply = 0
        while True:
            player = env.game.currentPlayer()
            moves = env.board.allValidMovesWithWay(player)
            mask = env.action_masks()
            key = env._permutationKey(player)

            def encode(move: list[int], key: object = key, env: HalmaEnv = env) -> int:
                normalized = env.normalizer.permuteMoves([(move[0], move[-1])], key)[0]
                return env.encodeAction(normalized)

            played = encode(driver.bestMove(moves, env.board, player))
            if ply % STRIDE == 0 and len(moves) > 1:
                observations.append({k: v.copy() for k, v in obs.items()})
                masks.append(mask.copy())
                for name, strategy in strategies.items():
                    choices[name].append(encode(strategy.bestMove(moves, env.board, player)))
            ply += 1

            obs, _, terminated, truncated, _ = env.step(played)
            if terminated or truncated or len(observations) >= positions:
                break
    print(f"  {len(observations):,} positions from {games} games")
    return observations, masks, choices


def predict(
    path: str, observations: list[dict[str, np.ndarray]], masks: list[np.ndarray]
) -> list[int]:
    """The checkpoint's argmax action in every position."""
    model = MaskablePPO.load(path, device="cpu")
    actions = []
    for obs, mask in zip(observations, masks, strict=True):
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        actions.append(int(action))
    return actions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", help="model paths to score")
    parser.add_argument(
        "--teachers",
        nargs="+",
        required=True,
        help="bots to measure agreement against, whether or not they taught this clone",
    )
    parser.add_argument("--opponent", default="distance")
    parser.add_argument("--positions", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=70_000)
    args = parser.parse_args()

    print(f"collecting {args.positions} positions, driver rotating over {args.teachers}")
    observations, masks, choices = collect(args.teachers, args.opponent, args.positions, args.seed)

    # Teachers as rows: their names are long and several share a prefix, so a
    # column per teacher either truncates them into ambiguity or wraps the line.
    rates = {}
    for path in args.checkpoints:
        actions = np.array(predict(path, observations, masks))
        rates[path] = [float(np.mean(actions == np.array(choices[name]))) for name in args.teachers]

    width = max(len(name) for name in args.teachers) + 2
    columns = [path.rsplit("/", 1)[-1] for path in args.checkpoints]
    print(f"\nagreement with each teacher, argmax over {len(observations)} positions")
    print(f"{'teacher':<{width}}" + "".join(f"{column:>22}" for column in columns))
    print("-" * (width + 22 * len(columns)))
    for row, name in enumerate(args.teachers):
        cells = "".join(f"{rates[path][row] * 100:21.1f}%" for path in args.checkpoints)
        print(f"{name:<{width}}{cells}")
    print("-" * (width + 22 * len(columns)))
    for label, reduce in (("mean", np.mean), ("min", min)):
        cells = "".join(f"{reduce(rates[path]) * 100:21.1f}%" for path in args.checkpoints)
        print(f"{label:<{width}}{cells}")

    print(
        "\nBreadth is the minimum, not the mean: a clone that copied one teacher"
        "\nand ignored the rest scores the same mean as one that covers them all."
    )


if __name__ == "__main__":
    main()
