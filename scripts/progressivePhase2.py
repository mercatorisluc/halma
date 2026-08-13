"""Five progressive rounds of 300k from Talos1.0, producing Talos1.2.

    Round 1: 300k steps, init Talos1.0, pool [Talos1.0]
    Round 2: 300k steps, init round1,   pool [Talos1.0, round1]
    Round 3: 300k steps, init round2,   pool [Talos1.0, round1, round2]
    Round 4: 300k steps, init round3,   pool [Talos1.0, round1..round3]
    Round 5: 300k steps, init round4,   pool [Talos1.0, round1..round4]

Total: 1.5M steps, against the 675k that produced Talos1.1.

The structure is ``progressivePhase1.py``'s, and that script is left alone
because it documents how Talos1.1 was built and is the only reproducible
member of that lineage. What differs here:

* **Rounds are 300k and flat**, rather than 50k growing to 175k. Phase A
  (ARCHITECTURE.md) measured what a single 300k round from Talos1.0 against a
  frozen Talos1.0 is worth: it lands level with Talos1.1 on random positions
  and just under it on the opening sweep. So 300k is roughly the budget in
  which a round has its effect, and the earlier schedule's opening rounds were
  too short to spend the pool they were given.

* **Entropy 0.01 rather than 0.03.** The bonus was blamed for the collapse
  ``--targetKl`` actually fixed, and the run that raised it to 0.03 while
  training away from the standard opening lost measurably on the sampled
  distribution. 0.01 is what Phase A's six runs used, so it is also the value
  the round-length claim above was measured at.

* **``--opponentSampling 0.5``, on a judgement call rather than on the
  measurement.** Phase A came back neutral: +2.4 points of breadth over three
  seeds against a between-seed spread of ten points, so a rank test cannot
  separate the arms, and nothing here should be read as evidence that the knob
  works. It is in this run because the argument for it survives the null
  result -- a frozen opponent that answers every position identically is a
  narrow teacher whatever three seeds happened to show, and Phase A gave it
  only one sparring partner and one round, which is the setting where it has
  least to offer. With a pool that grows each round, the same knob varies
  every member of it. The one thing in the data that points the same way is
  weak and is recorded as such: the sampled arm's three seeds spread over 3.8
  points where the control's spread over 10.2, which would mean more reliable
  runs rather than better ones.

  The cost of being wrong is that Talos1.2 carries an unmeasured change, so
  that this paragraph is the mitigation: if a later result is puzzling, this
  is the first thing to switch off and re-run.

Talos1.1 is deliberately **not** in the pool. It is the fixed reference every
recorded number is expressed against, and a checkpoint trained against it
would score higher without being stronger -- round 3 of the old league beat
pool members 95% while beating Talos1.0 only 65%. Talos1.2's ancestry is
therefore Talos1.0 alone, which also means the two can be compared without an
asterisk.
"""

import subprocess
import sys
from pathlib import Path

MODELS = Path(__file__).resolve().parent.parent / "models"

NUM_ROUNDS = 5
INIT = "models/Talos1.0"
BASENAME = "Talos1.2_round"
ENTROPY = 0.01
SEED = 42
LEARNING_RATE = 1e-4
# The load-bearing setting, unchanged from phase 1: without it, a round drove
# approx_kl to 0.064 and collapsed argmax strength against distance
# from 97% to 46%.
TARGET_KL = 0.02
STEPS_PER_ROUND = 300_000
# Half the episodes face the pool's distribution rather than its best move --
# see the docstring, which records that this is a bet and not a finding.
OPPONENT_SAMPLING = 0.5
# Steps between intermediate checkpoints within a round.
CHECKPOINT_EVERY = 100_000
# The bots are the wrong yardstick now -- every checkpoint since Talos1.0 has
# been at 98-100% against them -- so the inline evaluation is kept small and
# the real measurement is scripts/randomPositionSweep.py and
# scripts/openingSweep.py against Talos1.1, run per round once this finishes.
EVAL_GAMES = 20


def checkpointPath(roundNum: int) -> str:
    return str(MODELS / f"{BASENAME}{roundNum}")


def runRound(roundNum: int, initModel: str, pool: list[str]) -> str:
    print(f"\n{'=' * 70}")
    print(f"Round {roundNum}/{NUM_ROUNDS}: {STEPS_PER_ROUND:,} steps")
    print(f"  init: {initModel}")
    print(f"  opponent pool ({len(pool)}): {', '.join(pool)}")
    print(f"{'=' * 70}", flush=True)

    cmd = [
        "python",
        "-m",
        "scripts.train",
        "--steps",
        str(STEPS_PER_ROUND),
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
        "--opponentSampling",
        str(OPPONENT_SAMPLING),
        # Intermediate checkpoints, so the run can be swept while it is still
        # running rather than only at the round boundaries. The inline
        # progress column is not a substitute: it swung 25-55% across a run
        # whose true strength barely moved (ARCHITECTURE.md).
        "--saveCheckpointsEvery",
        str(CHECKPOINT_EVERY),
        "--checkpointBasename",
        f"{BASENAME}{roundNum}_step",
        "--games",
        str(EVAL_GAMES),
        "--noHeuristicOpponents",
        "--opponentModelPool",
        *pool,
        "--name",
        f"{BASENAME}{roundNum}",
    ]
    print(f"\n$ {' '.join(cmd)}\n", flush=True)

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\nRound {roundNum} failed (exit {result.returncode}); stopping.")
        sys.exit(result.returncode)

    return checkpointPath(roundNum)


def main() -> None:
    print("\n" + "=" * 70)
    print("Progressive self-play for Talos1.2")
    print("=" * 70)
    print(f"  {NUM_ROUNDS} rounds x {STEPS_PER_ROUND:,} steps = {NUM_ROUNDS * STEPS_PER_ROUND:,}")
    print(f"  lr {LEARNING_RATE}, entropy {ENTROPY}, targetKl {TARGET_KL}, seed {SEED}")
    print(f"  opponent sampling: {OPPONENT_SAMPLING:.0%} of episodes")
    print("  training opponents: checkpoints only, no heuristics, no Talos1.1")
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
    print("\nMeasure every round against the reference, and against the anchor:")
    for roundNum in range(1, NUM_ROUNDS + 1):
        print(f"  python -m scripts.randomPositionSweep {checkpointPath(roundNum)} models/Talos1.1")
    print(f"  python -m scripts.openingSweep {checkpointPath(NUM_ROUNDS)} models/Talos1.0")


if __name__ == "__main__":
    main()
