"""Stage 3 of the generation-2 recipe: five progressive rounds of self-play.

The recipe a ``.0`` model is built from, end to end -- ARCHITECTURE.md records
what each stage measured:

    Stage 1  clone      scripts/pretrain.py, 500k samples, three teachers
    Stage 2  heuristics scripts/train.py, 300k + 500k against bots and pools
    Stage 3  self-play  this script, 5 x 300k, checkpoint-only opponents

This script is stage 3:

    Round 1: 300k steps, init stage2, pool [stage2]
    Round 2: 300k steps, init round1, pool [stage2, round1]
    Round 3: 300k steps, init round2, pool [stage2, round1, round2]
    Round 4: 300k steps, init round3, pool [stage2, round1..round3]
    Round 5: 300k steps, init round4, pool [stage2, round1..round4]

``progressivePhase2.py``'s recipe, re-pointed at the new generation. That
script is left alone for the same reason ``progressivePhase1.py`` is: it
documents how Talos1.2 was built, and its ``INIT`` no longer loads.

What differs from phase 2, and why:

* **The anchor is ``models/talos2_stage6``**, the end of stage 2 -- 500k steps
  against the five heuristics *and* a model pool, which is what step 6 of the
  old lineage did to produce Talos1.0. The 300k stage before it
  (``models/talos2_parity025``) is not the right anchor: the league is what the
  old generation ran *after* Talos1.0, and starting it two stages early would
  be a different experiment. Stage 2 beats that earlier checkpoint 64.1% over
  800 openings, so the extra 500k are not free.

* **``--parity 0.25`` is passed explicitly** rather than inherited from
  ``scripts/train.py``'s default. The anchor was trained with it, so the league
  keeps it; writing it out means a later change of that default cannot silently
  reinterpret this run. The control arm ``models/talos2_parity0`` exists and
  measured level everywhere except sampled play against ``lookahead2``, where
  it was much better (45.0% against 18.3%, 60 games). That gap is unexplained
  and is the first thing to suspect if this league disappoints -- see
  ARCHITECTURE.md.

Everything else is phase 2 unchanged: flat 300k rounds, entropy 0.01,
``--targetKl 0.02`` (load-bearing -- without it a round drove approx_kl to
0.064 and collapsed argmax strength from 97% to 46%), ``--opponentSampling
0.5``, checkpoint-only opponents with no heuristic in the draw.

There is no held-out reference in this generation the way Talos1.1 was one for
phase 2: every earlier checkpoint is unloadable. The bots are therefore the
only yardstick that bridges, and among them only ``lookahead2`` still moves.

A league stopped part-way resumes with ``--startRound N``: rounds before ``N``
are read from the ``Talos2.0_roundN`` files on disk and go back into the pool,
so the resumed rounds see the same opponents an uninterrupted run would have
given them. Without it the script restarts at round 1 and overwrites what is
there.

The rounds are named ``Talos2.0_roundN``. The name is a claim about the
lineage, not yet about strength: whichever round earns it becomes ``Talos2.0``,
and a league that ends level with its own anchor produces no ``.0`` at all.
"""

import argparse
import subprocess
import sys
from pathlib import Path

MODELS = Path(__file__).resolve().parent.parent / "models"

NUM_ROUNDS = 5
INIT = "models/talos2_stage6"
BASENAME = "Talos2.0_round"
ENTROPY = 0.01
SEED = 42
LEARNING_RATE = 1e-4
TARGET_KL = 0.02
PARITY = 0.25
STEPS_PER_ROUND = 300_000
OPPONENT_SAMPLING = 0.5
# Steps between intermediate checkpoints within a round, or None for round
# boundaries only. Phase 2 saved every 100k so a round could be swept while it
# was still running; here that would be 15 files of 23.5 MB against the 5 the
# league actually needs, and a lost round costs ~12 minutes to redo.
CHECKPOINT_EVERY = None
# The bots are saturated -- 98-100% on all three of the training heuristics
# already at the anchor -- so the inline evaluation stays small and the real
# measurement is scripts/openingSweep.py and scripts/randomPositionSweep.py
# between rounds, plus scripts/evaluateAgainstBots.py on lookahead2.
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
        # sys.executable rather than "python": this script is normally started
        # from the venv, and a bare "python" would hand the round to whatever
        # interpreter the subprocess environment happens to resolve.
        sys.executable,
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
        "--parity",
        str(PARITY),
        "--opponentSampling",
        str(OPPONENT_SAMPLING),
        "--games",
        str(EVAL_GAMES),
        "--noHeuristicOpponents",
        "--opponentModelPool",
        *pool,
        # scripts/train.py prefixes --name with models/, so this must not.
        "--name",
        f"{BASENAME}{roundNum}",
    ]
    if CHECKPOINT_EVERY is not None:
        cmd += [
            "--saveCheckpointsEvery",
            str(CHECKPOINT_EVERY),
            "--checkpointBasename",
            f"{BASENAME}{roundNum}_step",
        ]
    print(f"\n$ {' '.join(cmd)}\n", flush=True)

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\nRound {roundNum} failed (exit {result.returncode}); stopping.")
        sys.exit(result.returncode)

    return checkpointPath(roundNum)


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 3 of the generation-2 recipe: five progressive rounds of self-play."
    )
    parser.add_argument(
        "--startRound",
        type=int,
        default=1,
        help=(
            "resume at this round instead of 1, rebuilding the opponent pool from the "
            f"{BASENAME}N checkpoints already on disk"
        ),
    )
    return parser.parse_args()


def resumeState(startRound: int) -> tuple[str, list[str]]:
    """The ``--init`` and opponent pool a round would have been given in a full run.

    Rebuilt from disk rather than re-run, so that a league stopped part-way
    continues with the same pool it would have had.
    """
    if not 1 <= startRound <= NUM_ROUNDS:
        sys.exit(f"--startRound must be between 1 and {NUM_ROUNDS}, got {startRound}")

    pool = [INIT]
    initModel = INIT
    for roundNum in range(1, startRound):
        checkpoint = checkpointPath(roundNum)
        if not Path(f"{checkpoint}.zip").exists():
            sys.exit(f"cannot resume at round {startRound}: {checkpoint}.zip is missing")
        pool.append(checkpoint)
        initModel = checkpoint
    return initModel, pool


def main() -> None:
    args = parseArgs()

    print("\n" + "=" * 70)
    print("Progressive self-play, generation 2")
    print("=" * 70)
    print(f"  {NUM_ROUNDS} rounds x {STEPS_PER_ROUND:,} steps = {NUM_ROUNDS * STEPS_PER_ROUND:,}")
    print(f"  anchor: {INIT}")
    print(f"  lr {LEARNING_RATE}, entropy {ENTROPY}, targetKl {TARGET_KL}, seed {SEED}")
    print(f"  parity {PARITY}, opponent sampling {OPPONENT_SAMPLING:.0%} of episodes")
    print("  training opponents: checkpoints only, no heuristics")
    if args.startRound > 1:
        print(
            f"  resuming at round {args.startRound}; rounds 1-{args.startRound - 1} read from disk"
        )
    print("=" * 70, flush=True)

    initModel, pool = resumeState(args.startRound)

    for roundNum in range(args.startRound, NUM_ROUNDS + 1):
        checkpoint = runRound(roundNum, initModel, list(pool))
        pool.append(checkpoint)
        initModel = checkpoint

    print("\n" + "=" * 70)
    print(f"All {NUM_ROUNDS} rounds complete.")
    print("=" * 70)
    print(f"\nFinal model: {initModel}")
    print("\nMeasure every round against the anchor, and the last one on the bots:")
    for roundNum in range(1, NUM_ROUNDS + 1):
        print(f"  python -m scripts.randomPositionSweep {checkpointPath(roundNum)} {INIT}")
    print(f"  python -m scripts.openingSweep {checkpointPath(NUM_ROUNDS)} {INIT}")
    print(f"  python -m scripts.evaluateAgainstBots {checkpointPath(NUM_ROUNDS)} --bots lookahead2")


if __name__ == "__main__":
    main()
