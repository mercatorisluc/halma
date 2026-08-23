"""Stage 3 of the generation-2 recipe: five progressive rounds of self-play.

The recipe a ``.0`` model is built from, end to end -- RESULTS.md records
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

The recipe that built ``Talos1.2``, re-pointed at the new generation. That
script -- ``progressivePhase2.py`` -- and its two siblings were deleted on
2026-08-23: their ``INIT`` checkpoints stopped loading on 2026-08-07, so all
three had become unrunnable. This is the living copy of the schedule; git at
``d194c5d`` has the originals if the exact old wording is ever wanted.

What differs from that generation, and why:

* **The anchor is ``models/talos2_stage6``**, the end of stage 2 -- 500k steps
  against the five heuristics *and* a model pool, which is what step 6 of the
  old lineage did to produce Talos1.0. The 300k stage before it
  (``models/talos2_parity025``) is not the right anchor: the league is what the
  old generation ran *after* Talos1.0, and starting it two stages early would
  be a different experiment. Stage 2 clearly beats that earlier checkpoint
  over the 800-opening census, so the extra 500k are not free -- the figure is
  in RESULTS.md under "Stage 2's two numbers".

* **``--parity 0.25`` is passed explicitly** rather than inherited from
  ``scripts/train.py``'s default. The anchor was trained with it, so the league
  keeps it; writing it out means a later change of that default cannot silently
  reinterpret this run. The control arm ``models/talos2_parity0`` exists and
  measured level everywhere except sampled play against ``lookahead2``, where
  it was *much* better. That gap is unexplained and is the first thing to
  suspect if this league disappoints -- see RESULTS.md, "Stage 2's two
  numbers".

Everything else is phase 2 unchanged: flat 300k rounds, entropy 0.01,
``--targetKl 0.02`` (load-bearing -- without it a round drove approx_kl an
order of magnitude too high and roughly halved argmax strength; never drop
it), ``--opponentSampling
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

# The recipe that produced Talos2.0_round1..5, as the defaults below. Every one
# of them is a flag, and that is deliberate: the parity-0 replication in
# TODO.md needs a second league that differs in exactly one setting, and
# editing constants in place would leave the recorded league no longer
# reproducible from what is on disk. Change a default only when the recipe
# itself changes.
DEFAULTS = {
    "rounds": 5,
    "init": "models/talos2_stage6",
    "basename": "Talos2.0_round",
    "steps": 300_000,
    "entropy": 0.01,
    "seed": 42,
    "lr": 1e-4,
    "targetKl": 0.02,
    "parity": 0.25,
    "opponentSampling": 0.5,
    # The bots are saturated -- 98-100% on all three training heuristics
    # already at the anchor -- so the inline evaluation stays small and the
    # real measurement is scripts/openingSweep.py and
    # scripts/randomPositionSweep.py between rounds, plus
    # scripts/evaluateAgainstBots.py on lookahead2.
    "evalGames": 20,
}


def checkpointPath(basename: str, roundNum: int) -> str:
    return str(MODELS / f"{basename}{roundNum}")


def opponentPool(anchor: str, finished: list[str], cap: int | None) -> list[str]:
    """The checkpoints a round trains against: the anchor plus recent rounds.

    ``cap`` limits how many *finished rounds* are drawn alongside the anchor,
    keeping the most recent ones. Uncapped, the pool grows every round and the
    draw in ``HalmaEnv.reset`` is uniform over it, so round 5 already spent
    four fifths of its episodes re-beating history and a sixth would spend five
    sixths. The anchor keeps its slot whatever the cap: it is the fixed
    reference every round is measured against, so dropping it would leave the
    league with no common yardstick.
    """
    if cap is None:
        return [anchor, *finished]
    return [anchor, *finished[-cap:]] if cap else [anchor]


def runRound(args: argparse.Namespace, roundNum: int, initModel: str, pool: list[str]) -> str:
    print(f"\n{'=' * 70}")
    print(f"Round {roundNum}/{args.rounds}: {args.steps:,} steps")
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
        str(args.steps),
        "--init",
        initModel,
        "--entropy",
        str(args.entropy),
        "--seed",
        str(args.seed),
        "--lr",
        str(args.lr),
        "--targetKl",
        str(args.targetKl),
        "--parity",
        str(args.parity),
        "--opponentSampling",
        str(args.opponentSampling),
        "--games",
        str(args.evalGames),
        "--noHeuristicOpponents",
        "--opponentModelPool",
        *pool,
        # scripts/train.py prefixes --name with models/, so this must not.
        "--name",
        f"{args.basename}{roundNum}",
    ]
    if args.checkpointEvery is not None:
        cmd += [
            "--saveCheckpointsEvery",
            str(args.checkpointEvery),
            "--checkpointBasename",
            f"{args.basename}{roundNum}_step",
        ]
    print(f"\n$ {' '.join(cmd)}\n", flush=True)

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\nRound {roundNum} failed (exit {result.returncode}); stopping.")
        sys.exit(result.returncode)

    return checkpointPath(args.basename, roundNum)


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 3 of the generation-2 recipe: progressive rounds of self-play."
    )
    parser.add_argument("--rounds", type=int, default=DEFAULTS["rounds"])
    parser.add_argument("--init", default=DEFAULTS["init"], help="the anchor, and round 1's start")
    parser.add_argument(
        "--basename",
        default=DEFAULTS["basename"],
        help="rounds are named <basename><N>; give a run that changes the recipe its own",
    )
    parser.add_argument("--steps", type=int, default=DEFAULTS["steps"])
    parser.add_argument("--entropy", type=float, default=DEFAULTS["entropy"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--targetKl", type=float, default=DEFAULTS["targetKl"])
    parser.add_argument(
        "--parity",
        type=float,
        default=DEFAULTS["parity"],
        help="parity penalty weight; 0 is the control arm of the replication in TODO.md",
    )
    parser.add_argument("--opponentSampling", type=float, default=DEFAULTS["opponentSampling"])
    parser.add_argument("--evalGames", type=int, default=DEFAULTS["evalGames"])
    parser.add_argument(
        "--poolCap",
        type=int,
        default=None,
        help=(
            "draw from the anchor plus only this many most recent rounds "
            "(default: every round so far -- see opponentPool)"
        ),
    )
    parser.add_argument(
        "--checkpointEvery",
        type=int,
        default=None,
        help=(
            "steps between intermediate checkpoints within a round. Off by default: it would be "
            "15 files of 23.5 MB against the 5 the league needs, and a lost round costs ~12 min"
        ),
    )
    parser.add_argument(
        "--startRound",
        type=int,
        default=1,
        help=(
            "resume at this round instead of 1, rebuilding the opponent pool from the "
            "<basename>N checkpoints already on disk"
        ),
    )
    return parser.parse_args()


def resumeState(args: argparse.Namespace) -> tuple[str, list[str]]:
    """The ``--init`` and finished rounds a run would have reached by ``startRound``.

    Rebuilt from disk rather than re-run, so that a league stopped part-way
    continues with the pool it would have had.
    """
    if not 1 <= args.startRound <= args.rounds:
        sys.exit(f"--startRound must be between 1 and {args.rounds}, got {args.startRound}")

    finished: list[str] = []
    initModel = args.init
    for roundNum in range(1, args.startRound):
        checkpoint = checkpointPath(args.basename, roundNum)
        if not Path(f"{checkpoint}.zip").exists():
            sys.exit(f"cannot resume at round {args.startRound}: {checkpoint}.zip is missing")
        finished.append(checkpoint)
        initModel = checkpoint
    return initModel, finished


def main() -> None:
    args = parseArgs()

    print("\n" + "=" * 70)
    print("Progressive self-play, generation 2")
    print("=" * 70)
    print(f"  {args.rounds} rounds x {args.steps:,} steps = {args.rounds * args.steps:,}")
    print(f"  anchor: {args.init}")
    print(f"  lr {args.lr}, entropy {args.entropy}, targetKl {args.targetKl}, seed {args.seed}")
    print(f"  parity {args.parity}, opponent sampling {args.opponentSampling:.0%} of episodes")
    print("  training opponents: checkpoints only, no heuristics")
    if args.poolCap is not None:
        print(f"  opponent pool capped at the anchor plus the last {args.poolCap} round(s)")
    if args.startRound > 1:
        print(
            f"  resuming at round {args.startRound}; rounds 1-{args.startRound - 1} read from disk"
        )
    print("=" * 70, flush=True)

    initModel, finished = resumeState(args)

    for roundNum in range(args.startRound, args.rounds + 1):
        pool = opponentPool(args.init, finished, args.poolCap)
        checkpoint = runRound(args, roundNum, initModel, pool)
        finished.append(checkpoint)
        initModel = checkpoint

    last = checkpointPath(args.basename, args.rounds)
    print("\n" + "=" * 70)
    print(f"All {args.rounds} rounds complete.")
    print("=" * 70)
    print(f"\nFinal model: {initModel}")
    print("\nMeasure every round against the anchor, and the last one on the bots:")
    for roundNum in range(1, args.rounds + 1):
        print(
            f"  python -m scripts.randomPositionSweep "
            f"{checkpointPath(args.basename, roundNum)} {args.init}"
        )
    print(f"  python -m scripts.openingSweep {last} {args.init}")
    print(f"  python -m scripts.evaluateAgainstBots {last} --bots lookahead2")


if __name__ == "__main__":
    main()
