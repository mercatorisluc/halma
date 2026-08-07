"""Four progressive rounds of 300k from Talos1.2, the candidate for Talos1.3.

    Round 1: 300k steps, init Talos1.2, pool [Talos1.2]
    Round 2: 300k steps, init round1,   pool [Talos1.2, round1]
    Round 3: 300k steps, init round2,   pool [Talos1.2, round1, round2]
    Round 4: 300k steps, init round3,   pool [Talos1.2, round1..round3]

Total: 1.2M steps, against the 1.5M that produced Talos1.2.

The schedule is ``progressivePhase2.py``'s and the training settings are
unchanged from it, deliberately: that run produced the largest generation step
measured here (69.4% over Talos1.1 on the opening census), and nothing in its
result points at a setting to change. ``progressivePhase2.py`` is left alone
because it documents how Talos1.2 was built. What differs here is the starting
point, the length, and above all how the run is measured.

* **The reference rolls.** Phase 2 scored every round against a frozen
  Talos1.1 and rounds 3-4 came back flat (63.7% then 63.1% breadth), which
  read on its own would have said the league had run out. It had not: round 4
  beats round 3 head to head 57.4%, and round 5 beats round 4 72.9%. Once a
  candidate takes ~70% of a census the reference has no resolution left, so
  from here the measurement of record is **the previous round**, and each
  round prints its own sweep command on completion so it can be scored while
  the next round trains. Talos1.1 and Talos1.0 survive only as the bridge run
  at the end, which is what keeps this run's numbers attached to everything
  recorded in ARCHITECTURE.md.

* **Four rounds, not five.** Phase 2's gains did not taper -- its last round
  was its largest -- so the round count is a budget decision rather than a
  prediction, and the intermediate checkpoints are kept so the run can be cut
  short or extended on what the rolling sweeps actually show.

* **Talos1.0 is not in the pool.** Phase 2 carried it because it was that
  run's starting point; here it would be an opponent some 80 points weaker
  than the learner, taking a fifth of the episodes to teach it nothing.
  Trimming the pool from the bottom would start to matter again past round 4:
  with four rounds the pool never exceeds four members, so no rule for it is
  written here rather than one being guessed at.

Talos1.1 stays out of the training pool for the reason ``progressivePhase2.py``
gives -- it is the fixed reference the recorded numbers are expressed against,
and a checkpoint trained against its own yardstick scores higher without being
stronger.

The final checkpoint is ``Talos1.3_round4`` and keeps that name. Renaming it to
``Talos1.3`` is a separate decision that belongs to the measurement, not to the
run: a generation step here has meant ~65-70% over the previous release, and
anything meaningfully short of that is a ``_selfplayN`` iteration instead.
"""

import subprocess
import sys
from pathlib import Path

MODELS = Path(__file__).resolve().parent.parent / "models"

NUM_ROUNDS = 4
INIT = "models/Talos1.2"
BASENAME = "Talos1.3_round"
# The reference the recorded lineage is expressed against. Never a training
# opponent -- only the bridge measurement at the end.
BRIDGE = ["models/Talos1.2", "models/Talos1.1"]

ENTROPY = 0.01
SEED = 42
LEARNING_RATE = 1e-4
# The load-bearing setting, unchanged since phase 1: without it, a round drove
# approx_kl to 0.064 and collapsed argmax strength against advancedDistScore
# from 97% to 46%.
TARGET_KL = 0.02
STEPS_PER_ROUND = 300_000
# Carried over from phase 2, where it was recorded as a bet rather than a
# finding (Phase A measured it neutral over three seeds). It stays because
# changing it now would confound this run with that question: Talos1.2 was
# built with it, and a phase 3 without it would not be a clean continuation.
# If a result here is puzzling, this is still the first thing to switch off.
OPPONENT_SAMPLING = 0.5
# Steps between intermediate checkpoints within a round.
CHECKPOINT_EVERY = 100_000
# The bots are the wrong yardstick for this lineage -- everything since
# Talos1.0 scores 98-100% argmax against them -- so the inline evaluation is
# kept small and the real measurement is the sweeps printed after each round.
EVAL_GAMES = 20


def checkpointPath(roundNum: int) -> str:
    return str(MODELS / f"{BASENAME}{roundNum}")


def previousCheckpoint(roundNum: int) -> str:
    """The rolling reference: the round before this one, or the run's start."""
    return INIT if roundNum == 1 else checkpointPath(roundNum - 1)


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


def printRollingSweep(roundNum: int) -> None:
    """The measurement of record for this round, against the previous one.

    Printed rather than run, so it can be scored on another core while the
    next round trains -- a sweep is 800 games and a round is 300k steps.
    """
    candidate = checkpointPath(roundNum)
    reference = previousCheckpoint(roundNum)
    print(f"\nMeasure round {roundNum} against the round it started from:")
    print(f"  python -m scripts.openingSweep {candidate} {reference}")
    print(f"  python -m scripts.randomPositionSweep {candidate} {reference}", flush=True)


def main() -> None:
    print("\n" + "=" * 70)
    print("Progressive self-play from Talos1.2")
    print("=" * 70)
    print(f"  {NUM_ROUNDS} rounds x {STEPS_PER_ROUND:,} steps = {NUM_ROUNDS * STEPS_PER_ROUND:,}")
    print(f"  lr {LEARNING_RATE}, entropy {ENTROPY}, targetKl {TARGET_KL}, seed {SEED}")
    print(f"  opponent sampling: {OPPONENT_SAMPLING:.0%} of episodes")
    print("  training opponents: checkpoints only, no heuristics, no Talos1.1")
    print("  measurement: the previous round, not a fixed reference")
    print("=" * 70, flush=True)

    pool = [INIT]
    initModel = INIT

    for roundNum in range(1, NUM_ROUNDS + 1):
        checkpoint = runRound(roundNum, initModel, list(pool))
        printRollingSweep(roundNum)
        pool.append(checkpoint)
        initModel = checkpoint

    print("\n" + "=" * 70)
    print(f"All {NUM_ROUNDS} rounds complete.")
    print("=" * 70)
    print(f"\nFinal model: {initModel}")
    print("\nBridge the run back to the recorded lineage:")
    for reference in BRIDGE:
        print(f"  python -m scripts.openingSweep {initModel} {reference}")
        print(f"  python -m scripts.randomPositionSweep {initModel} {reference}")
    print("\nAnd the sampled heuristic column, the one that still moves:")
    print(f"  python -m scripts.evaluateAgainstBots {initModel}")


if __name__ == "__main__":
    main()
