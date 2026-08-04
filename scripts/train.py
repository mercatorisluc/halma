"""Train a masked PPO agent on Halma and report whether it beat the baseline.

    python -m scripts.train --steps 200000

Masking is not optional here: only ~65 of 14641 encoded actions are legal in a
typical position, so an unmasked policy would spend itself learning which
actions are illegal rather than which are good. ``MaskablePPO`` picks up
``HalmaEnv.action_masks`` on its own.

Success is measured on ``info["outcome"]`` -- the unshaped win or loss -- never
on the reward, which includes shaping and would flatter the agent.

The defaults are set for throughput, because training is bound by the network:
the environment alone steps at ~1650/s, the environment with PPO in the loop at
~120/s. A squeezed feature extractor and four PPO epochs instead of ten run at
319 steps/s against 120. Measured over a fixed quarter hour each, same seed and
opponent, the cheap configuration got through 288,769 steps against 108,545 and
came out ahead on every measure -- pieces home against ``advancedDistScore``
8.5% against 1.7% by argmax, and better on all six pairings. So the extra steps
more than pay for the smaller network and the fewer passes. Neither won a game.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import cast

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from env.features import HalmaFeatures
from env.halmaEnv import HalmaEnv
from env.policy import FactoredMaskablePolicy

MODELS = Path(__file__).resolve().parent.parent / "models"


class ProgressReport(BaseCallback):
    """Print the honest win rate every ``every`` steps, so a run is watchable.

    Stable-Baselines' own log shows ``ep_rew_mean``, which includes shaping and
    therefore rises when the agent merely advances. That is worth seeing, but it
    is not the thing being optimised for, so this measures actual games as well.

    ``candidates`` is a list of ``(label, opponentModel)`` pairs: a heuristic
    entry has ``opponentModel=None`` and ``label`` names the strategy passed
    as ``opponent``; a tuned-checkpoint entry has ``opponentModel`` set to its
    path and ``label`` only names the printed column, since ``opponentModel``
    overrides whatever ``opponent`` string ``evaluate()`` is given. One line
    is printed per report with one win rate per candidate, so a pooled run
    against several opponent kinds shows progress against each rather than
    just the one ``--opponent`` tracks.
    """

    def __init__(
        self,
        candidates: list[tuple[str, str | None]],
        every: int = 25_000,
        games: int = 20,
    ) -> None:
        super().__init__()
        self.candidates = candidates
        self.every = every
        self.games = games
        self.nextAt = every

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.nextAt:
            self.nextAt += self.every
            # self.model is typed as the base algorithm; here it is always the
            # MaskablePPO being trained, whose predict() takes action_masks.
            model = cast("MaskablePPO", self.model)
            columns = []
            for label, opponentModel in self.candidates:
                # Sampled, not argmax: the argmax of a half-trained policy is a
                # degenerate fixed rule and says nothing about progress.
                # Measured on the same model: argmax 0.5% pieces home, sampled
                # 3.7%.
                result = evaluate(
                    model,
                    label,
                    self.games,
                    seed=90_000,
                    deterministic=False,
                    opponentModel=opponentModel,
                )
                columns.append(f"{label} {result['winRate'] * 100:5.1f}%")
            print(f"  [{self.num_timesteps:>7,} steps]  " + "   ".join(columns), flush=True)
        return True


class CheckpointSaver(BaseCallback):
    """Save intermediate checkpoints every ``every`` steps for self-play pool."""

    def __init__(self, every: int = 50_000, basename: str = "checkpoint") -> None:
        super().__init__()
        self.every = every
        self.basename = basename
        self.nextAt = every
        self.checkpointNum = 0

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.nextAt:
            self.nextAt += self.every
            self.checkpointNum += 1
            model = cast("MaskablePPO", self.model)
            # The .zip is spelled out because sb3 only appends it when
            # Path.suffix is empty, and a name like "Talos1.0_sp_001" already
            # has a "suffix" of ".0_sp_001" as far as pathlib is concerned --
            # so the checkpoint would silently land without an extension.
            path = MODELS / f"{self.basename}_{self.checkpointNum:03d}.zip"
            model.save(path)
            msg = f"  saved checkpoint {self.checkpointNum} at {self.num_timesteps:,} steps"
            print(msg, flush=True)
        return True


def evaluate(
    model: MaskablePPO | None,
    opponent: str,
    games: int,
    seed: int = 10_000,
    deterministic: bool = True,
    opponentModel: str | None = None,
) -> dict:
    """Play out games and count wins by outcome, not by reward.

    ``model=None`` plays uniformly at random, which is the floor to clear: a
    random agent wins none of 700 games against any of the bots.

    ``opponentModel``, if given, is a frozen checkpoint seated as the
    opponent instead of the heuristic named by ``opponent`` -- see
    ``HalmaEnv``. Built once, outside the game loop below, and reused for
    every game: a fresh ``HalmaEnv`` per game would reload that checkpoint
    from disk every game, which is fine for a heuristic but not for a model.
    """
    wins = losses = draws = 0
    steps = 0
    homeFractions = []
    env = HalmaEnv(opponentStrategy=opponent, opponentModel=opponentModel)
    for i in range(games):
        obs, _ = env.reset(seed=seed + i)
        rng = np.random.default_rng(seed + i)
        while True:
            if model is None:
                legal = np.flatnonzero(env.action_masks())
                action = int(rng.choice(legal))
            else:
                action, _ = model.predict(
                    obs, action_masks=get_action_masks(env), deterministic=deterministic
                )
                action = int(action)
            obs, _, terminated, truncated, info = env.step(action)
            steps += 1
            if terminated or truncated:
                if info["outcome"] > 0:
                    wins += 1
                elif info["outcome"] < 0:
                    losses += 1
                else:
                    draws += 1
                agent = env._player(env.AGENT_SEAT)
                homeFractions.append(
                    len(agent.positions & agent.endPositions) / len(agent.endPositions)
                )
                break
    rate = wins / games
    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "winRate": rate,
        "marginOfError": 1.96 * math.sqrt(max(rate * (1 - rate), 1e-9) / games),
        "avgSteps": steps / games,
        # Pieces home when the game ended. The win rate is expected to sit at
        # zero for a long while, so this is what shows whether the agent is
        # learning to advance at all -- it moves long before wins appear.
        "homeFraction": float(np.mean(homeFractions)),
    }


def report(label: str, result: dict) -> None:
    print(
        f"  {label:<24} {result['winRate'] * 100:5.1f} % +/- {result['marginOfError'] * 100:4.1f}"
        f"   pieces home {result['homeFraction'] * 100:4.1f} %"
        f"   ({result['wins']}W {result['losses']}L {result['draws']}D,"
        f" {result['avgSteps']:.0f} steps)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100_000, help="training timesteps")
    parser.add_argument(
        "--opponent",
        default="advancedDistScore",
        help="opponent tracked in progress reports and the final results",
    )
    parser.add_argument(
        "--opponentPool",
        nargs="+",
        default=None,
        help=(
            "bots to train against, one drawn at random each episode "
            "(default: --opponent alone, today's fixed-opponent behaviour)"
        ),
    )
    parser.add_argument(
        "--evalOpponents",
        nargs="+",
        default=None,
        help="bots to report results against (default: --opponent, sparsityScore, random)",
    )
    parser.add_argument("--games", type=int, default=100, help="evaluation games")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--shaping", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    # sb3 defaults this to 0. With a per-step signal this small the policy
    # can collapse onto an arbitrary subset of moves before the reward has
    # said anything, which would end below random -- as it did.
    # For self-play training with checkpoint pools, higher entropy (0.02-0.05)
    # promotes exploration and prevents overfitting to one version of the opponent.
    parser.add_argument("--entropy", type=float, default=0.01)
    # sb3 defaults this to 10. Four gradient passes over each rollout instead
    # of ten is the single biggest lever on throughput, because the update is
    # ~95% of the network time and the network is ~95% of a training step.
    # Fewer passes learn less per sample, so this is only worth it if the extra
    # steps more than pay it back -- measured, they do; see --steps below.
    parser.add_argument("--epochs", type=int, default=4, help="PPO passes per rollout")
    # Defaults to 1 because parallel environments measured no real gain here:
    # 114 steps/s on one against 123 on eight, on a 10-core machine. MaskablePPO
    # fetches the action mask from every worker on every step, and that IPC
    # eats what the parallelism wins. The flag stays because the picture would
    # change with a heavier network or a cheaper mask.
    parser.add_argument("--envs", type=int, default=1, help="parallel environments")
    # sb3 defaults this to 3e-4, which suits a policy starting from noise: it is
    # diffuse, so a large step costs nothing. A cloned policy is already sharp,
    # and the same step throws it out of the region cloning found. Measured on
    # this clone at 3e-4, approx_kl ran 0.06-0.37 against a healthy ~0.01 and a
    # fifth to a third of samples hit the clipping limit from the first update.
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    # The backstop for the same problem: sb3 abandons the rest of a rollout's
    # passes once the update has moved the policy this far. None is sb3's
    # default, meaning all --epochs passes run however far they drag it.
    parser.add_argument("--targetKl", type=float, default=None, help="abort an update past this KL")
    parser.add_argument("--name", default="maskedPPO")
    # A checkpoint to start from, normally one of scripts/pretrain.py's clones.
    # PPO from noise has nothing to learn from here; from a clone of a bot it
    # starts inside a policy that already plays.
    parser.add_argument("--init", default=None, help="checkpoint to fine-tune")
    # A frozen checkpoint seated as the opponent instead of any heuristic --
    # takes priority over --opponent/--opponentPool for seating, though
    # --opponent still labels the progress reports and evalOpponents default.
    # Its weights never change during this run: a fixed sparring partner, not
    # self-play, which would also need to update the opponent's own weights.
    parser.add_argument("--opponentModel", default=None, help="frozen checkpoint to train against")
    # Extends --opponentPool's per-episode draw to tuned checkpoints as well
    # as heuristics -- see HalmaEnv's opponentModelPool. Ignored if
    # --opponentModel pins a single fixed opponent, same as --opponentPool.
    parser.add_argument(
        "--opponentModelPool",
        nargs="+",
        default=None,
        help="tuned checkpoints to train against, drawn alongside --opponentPool",
    )
    # Pure self-play: drop heuristics from the training draw entirely, so every
    # episode is played against a checkpoint from --opponentModelPool. Without
    # this the draw always includes --opponent, which is a heuristic.
    parser.add_argument(
        "--noHeuristicOpponents",
        action="store_true",
        help="train only against --opponentModelPool, never a heuristic",
    )
    parser.add_argument(
        "--reportEvery", type=int, default=25_000, help="steps between progress reports"
    )
    parser.add_argument(
        "--saveCheckpointsEvery",
        type=int,
        default=None,
        help="save intermediate checkpoints every N steps (for self-play pool training)",
    )
    parser.add_argument(
        "--checkpointBasename",
        default="checkpoint",
        help="basename for saved checkpoints (e.g., 'Talos1.0' -> Talos1.0_001, Talos1.0_002, ...)",
    )
    args = parser.parse_args()

    # Training opponents: a pool if given, else --opponent alone -- the
    # existing fixed-opponent behaviour. Progress reports and the final
    # results still track --opponent specifically, whether or not it is in
    # the pool, so a pool run stays comparable to a fixed-opponent one.
    trainingOpponents = [] if args.noHeuristicOpponents else (args.opponentPool or [args.opponent])
    if args.noHeuristicOpponents and not (args.opponentModelPool or args.opponentModel):
        parser.error("--noHeuristicOpponents needs --opponentModelPool or --opponentModel")

    # The env's shaping discount has to be the one PPO trains with, or the
    # shaping stops being policy-invariant.
    def makeEnv(rank: int):
        def build() -> HalmaEnv:
            return HalmaEnv(
                opponentStrategy=trainingOpponents,
                opponentModel=args.opponentModel,
                opponentModelPool=args.opponentModelPool,
                shapingWeight=args.shaping,
                gamma=args.gamma,
            )

        return build

    env = (
        SubprocVecEnv([makeEnv(i) for i in range(args.envs)])
        if args.envs > 1
        else HalmaEnv(
            opponentStrategy=trainingOpponents,
            opponentModel=args.opponentModel,
            opponentModelPool=args.opponentModelPool,
            shapingWeight=args.shaping,
            gamma=args.gamma,
        )
    )

    # Evaluated against the bot progress is tracked on and against the weaker
    # ones: progress is likely to show against a weak opponent well before it
    # shows against the one it is being beaten by. --evalOpponents overrides
    # this, e.g. to check a pooled run against every bot in heuristics/.
    opponents = args.evalOpponents or [args.opponent, "sparsityScore", "random"]
    opponents = list(dict.fromkeys(opponents))

    if args.opponentModel:
        print(f"opponent model {args.opponentModel}, {args.steps} steps, gamma {args.gamma}\n")
    elif args.opponentModelPool:
        print(
            f"training pool {trainingOpponents} + models {args.opponentModelPool}, "
            f"{args.steps} steps, gamma {args.gamma}\n"
        )
    elif len(trainingOpponents) > 1:
        print(f"training pool {trainingOpponents}, {args.steps} steps, gamma {args.gamma}\n")
    else:
        print(f"opponent {args.opponent}, {args.steps} steps, gamma {args.gamma}\n")
    print("before training (random agent):")
    for opponent in opponents:
        report(f"vs {opponent}", evaluate(None, opponent, args.games))
    if args.opponentModel:
        report(
            f"vs model {args.opponentModel}",
            evaluate(None, args.opponent, args.games, opponentModel=args.opponentModel),
        )

    print("\ntraining (ep_rew_mean includes shaping; the wins line does not):", flush=True)
    settings = {
        "gamma": args.gamma,
        "ent_coef": args.entropy,
        "n_epochs": args.epochs,
        "learning_rate": args.lr,
        "target_kl": args.targetKl,
        "seed": args.seed,
        "verbose": 1,
    }
    if args.init:
        print(f"  starting from {args.init}", flush=True)
        # The clone carries the policy weights; everything else is this run's.
        # Its value head is untrained -- pretraining fits only the action
        # distribution -- so the first rollouts have bad advantages until the
        # critic catches up.
        model = MaskablePPO.load(args.init, env=env, custom_objects=settings)
        report(
            "  the clone, before PPO",
            evaluate(model, args.opponent, args.games, opponentModel=args.opponentModel),
        )
    else:
        model = MaskablePPO(
            FactoredMaskablePolicy,
            env,
            policy_kwargs={"features_extractor_class": HalmaFeatures},
            **settings,
        )
    # What the periodic progress reports track: the fixed opponent model
    # alone if one is pinned, else --opponent plus one column per tuned
    # checkpoint in the pool, so a mixed run shows progress against each
    # kind of opponent rather than just the heuristic named by --opponent.
    # cast, not a `str | None`-annotated variable, because a plain assignment
    # of the `None` literal narrows to `None` regardless of the declared
    # type, and list's invariance then refuses to widen the literal below to
    # `list[tuple[str, str | None]]`.
    noOpponentModel = cast("str | None", None)
    reportCandidates: list[tuple[str, str | None]]
    if args.opponentModel:
        reportCandidates = [(args.opponent, args.opponentModel)]
    elif args.opponentModelPool:
        reportCandidates = [(args.opponent, noOpponentModel)] + [
            (Path(checkpoint).stem, checkpoint) for checkpoint in args.opponentModelPool
        ]
    else:
        reportCandidates = [(args.opponent, noOpponentModel)]

    callbacks: list[BaseCallback] = [ProgressReport(reportCandidates, every=args.reportEvery)]
    if args.saveCheckpointsEvery:
        callbacks.append(
            CheckpointSaver(every=args.saveCheckpointsEvery, basename=args.checkpointBasename)
        )

    model.learn(
        total_timesteps=args.steps,
        progress_bar=False,
        callback=callbacks,
    )

    MODELS.mkdir(exist_ok=True)
    # Explicit .zip -- see CheckpointSaver above for why sb3 cannot be trusted
    # to add it for names carrying a version number.
    path = MODELS / f"{args.name}.zip"
    model.save(path)

    print("\nafter training (argmax, then sampled):")
    for opponent in opponents:
        report(f"vs {opponent}", evaluate(model, opponent, args.games))
        report(
            "   sampled",
            evaluate(model, opponent, args.games, deterministic=False),
        )
    if args.opponentModel:
        report(
            f"vs model {args.opponentModel}",
            evaluate(model, args.opponent, args.games, opponentModel=args.opponentModel),
        )
        report(
            "   sampled",
            evaluate(
                model,
                args.opponent,
                args.games,
                deterministic=False,
                opponentModel=args.opponentModel,
            ),
        )
    print(f"\nsaved to {path}")


if __name__ == "__main__":
    main()
