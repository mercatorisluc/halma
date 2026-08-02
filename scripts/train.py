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
    """

    def __init__(self, opponent: str, every: int = 25_000, games: int = 20) -> None:
        super().__init__()
        self.opponent = opponent
        self.every = every
        self.games = games
        self.nextAt = every

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.nextAt:
            self.nextAt += self.every
            # self.model is typed as the base algorithm; here it is always the
            # MaskablePPO being trained, whose predict() takes action_masks.
            model = cast("MaskablePPO", self.model)
            # Sampled, not argmax: the argmax of a half-trained policy is a
            # degenerate fixed rule and says nothing about progress. Measured
            # on the same model: argmax 0.5% pieces home, sampled 3.7%.
            result = evaluate(model, self.opponent, self.games, seed=90_000, deterministic=False)
            print(
                f"  [{self.num_timesteps:>7,} steps]"
                f"  wins {result['winRate'] * 100:5.1f} %"
                f"  pieces home {result['homeFraction'] * 100:5.1f} %"
                f"  {result['avgSteps']:.0f} steps/game",
                flush=True,
            )
        return True


def evaluate(
    model: MaskablePPO | None,
    opponent: str,
    games: int,
    seed: int = 10_000,
    deterministic: bool = True,
) -> dict:
    """Play out games and count wins by outcome, not by reward.

    ``model=None`` plays uniformly at random, which is the floor to clear: a
    random agent wins none of 700 games against any of the bots.
    """
    wins = losses = draws = 0
    steps = 0
    homeFractions = []
    for i in range(games):
        env = HalmaEnv(opponentStrategy=opponent)
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
    parser.add_argument("--opponent", default="advancedDistScore")
    parser.add_argument("--games", type=int, default=100, help="evaluation games")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--shaping", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    # sb3 defaults this to 0. With a per-step signal this small the policy
    # can collapse onto an arbitrary subset of moves before the reward has
    # said anything, which would end below random -- as it did.
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
    parser.add_argument("--name", default="maskedPPO")
    parser.add_argument(
        "--reportEvery", type=int, default=25_000, help="steps between progress reports"
    )
    args = parser.parse_args()

    # The env's shaping discount has to be the one PPO trains with, or the
    # shaping stops being policy-invariant.
    def makeEnv(rank: int):
        def build() -> HalmaEnv:
            return HalmaEnv(
                opponentStrategy=args.opponent, shapingWeight=args.shaping, gamma=args.gamma
            )

        return build

    env = (
        SubprocVecEnv([makeEnv(i) for i in range(args.envs)])
        if args.envs > 1
        else HalmaEnv(opponentStrategy=args.opponent, shapingWeight=args.shaping, gamma=args.gamma)
    )

    # Evaluated against the bot it trained on and against the weaker ones:
    # progress is likely to show against a weak opponent well before it shows
    # against the one it is being beaten by.
    opponents = [args.opponent, "sparsityScore", "random"]
    opponents = list(dict.fromkeys(opponents))

    print(f"opponent {args.opponent}, {args.steps} steps, gamma {args.gamma}\n")
    print("before training (random agent):")
    for opponent in opponents:
        report(f"vs {opponent}", evaluate(None, opponent, args.games))

    print("\ntraining (ep_rew_mean includes shaping; the wins line does not):", flush=True)
    model = MaskablePPO(
        FactoredMaskablePolicy,
        env,
        gamma=args.gamma,
        ent_coef=args.entropy,
        n_epochs=args.epochs,
        seed=args.seed,
        verbose=1,
        policy_kwargs={"features_extractor_class": HalmaFeatures},
    )
    model.learn(
        total_timesteps=args.steps,
        progress_bar=False,
        callback=ProgressReport(args.opponent, every=args.reportEvery),
    )

    MODELS.mkdir(exist_ok=True)
    path = MODELS / args.name
    model.save(path)

    print("\nafter training (argmax, then sampled):")
    for opponent in opponents:
        report(f"vs {opponent}", evaluate(model, opponent, args.games))
        report(
            "   sampled",
            evaluate(model, opponent, args.games, deterministic=False),
        )
    print(f"\nsaved to {path}.zip")


if __name__ == "__main__":
    main()
