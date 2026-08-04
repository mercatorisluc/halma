"""Progressive self-play training: six rounds against a growing checkpoint pool.

Every round trains against frozen checkpoints only -- no heuristic is ever
drawn as a training opponent. Round 1 plays Talos1.0 against itself; each
later round adds the round just finished to the pool, so the agent has to
beat every earlier version of itself rather than only the newest one.

    Round 1:  50k steps, init Talos1.0, pool [Talos1.0]
    Round 2:  75k steps, init round1,   pool [Talos1.0, round1]
    Round 3: 100k steps, init round2,   pool [Talos1.0, round1, round2]
    Round 4: 125k steps, init round3,   pool [Talos1.0, round1..round3]
    Round 5: 150k steps, init round4,   pool [Talos1.0, round1..round4]
    Round 6: 175k steps, init round5,   pool [Talos1.0, round1..round5]

Total: 675k steps. Step count grows with the pool because a larger pool
spreads the same budget over more distinct opponents.

Heuristics still appear in the progress reports and the final evaluation --
they are the yardstick the run is measured on -- but never in training.
"""

import subprocess
import sys
from pathlib import Path

MODELS = Path(__file__).resolve().parent.parent / "models"

NUM_ROUNDS = 6
INIT = "models/Talos1.0"
BASENAME = "Talos1.0_progressive"
ENTROPY = 0.03
SEED = 42
# 3e-4 throws a sharp cloned policy out of the region cloning found; the
# opponent pool makes that worse, since the gradient direction changes with
# the draw.
LEARNING_RATE = 1e-4
# The load-bearing setting, not an optional guard. Without it, round 2 of the
# 2026-08-04 run drove approx_kl to 0.064 and collapsed argmax strength against
# advancedDistScore from 97% to 46% -- losing 14% of games even to random.
# Rerunning that same round with targetKl 0.02 and nothing else changed
# restored it to 100%. Measured at both entropy 0.03 and 0.01, so this is the
# fix; entropy was a symptom (it stops running away once updates are capped).
TARGET_KL = 0.02

STEPS_PER_ROUND = {1: 50_000, 2: 75_000, 3: 100_000, 4: 125_000, 5: 150_000, 6: 175_000}


def checkpointPath(roundNum: int) -> str:
    return str(MODELS / f"{BASENAME}_round{roundNum}")


def runRound(roundNum: int, initModel: str, pool: list[str]) -> str:
    print(f"\n{'=' * 70}")
    print(f"Round {roundNum}/{NUM_ROUNDS}: {STEPS_PER_ROUND[roundNum]:,} steps")
    print(f"  init: {initModel}")
    print(f"  opponent pool ({len(pool)}): {', '.join(pool)}")
    print(f"{'=' * 70}", flush=True)

    cmd = [
        "python",
        "-m",
        "scripts.train",
        "--steps",
        str(STEPS_PER_ROUND[roundNum]),
        "--init",
        initModel,
        "--entropy",
        str(ENTROPY),
        "--seed",
        str(SEED),
        "--lr",
        str(LEARNING_RATE),
        "--targetKl",
        str(TARGET_KL),
        "--noHeuristicOpponents",
        "--opponentModelPool",
        *pool,
        "--name",
        f"{BASENAME}_round{roundNum}",
    ]
    print(f"\n$ {' '.join(cmd)}\n", flush=True)

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\nRound {roundNum} failed (exit {result.returncode}); stopping.")
        sys.exit(result.returncode)

    return checkpointPath(roundNum)


def main() -> None:
    total = sum(STEPS_PER_ROUND.values())
    print("\n" + "=" * 70)
    print("Progressive self-play for Talos1.0")
    print("=" * 70)
    for i in range(1, NUM_ROUNDS + 1):
        print(f"  Round {i}: {STEPS_PER_ROUND[i]:>7,} steps")
    print(f"  Total:   {total:>7,} steps")
    print(f"  lr {LEARNING_RATE}, entropy {ENTROPY}, targetKl {TARGET_KL}, seed {SEED}")
    print("  training opponents: checkpoints only, no heuristics")
    print("=" * 70, flush=True)

    pool = [INIT]
    initModel = INIT

    for roundNum in range(1, NUM_ROUNDS + 1):
        checkpoint = runRound(roundNum, initModel, list(pool))
        pool.append(checkpoint)
        initModel = checkpoint

    print("\n" + "=" * 70)
    print(f"All {NUM_ROUNDS} rounds complete.")
    print("=" * 70)
    print(f"\nFinal model: {initModel}")
    print("\nFull checkpoint pool:")
    for entry in pool:
        print(f"  {entry}")
    print("\nCompare it against the version it started from:")
    print(f"  python -m scripts.compareCheckpoints {INIT} {initModel}")


if __name__ == "__main__":
    main()
