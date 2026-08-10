"""Teach the policy to copy a heuristic bot, before PPO ever runs.

    python -m scripts.pretrain --samples 200000
    python -m scripts.train --steps 300000 --init models/cloned

PPO starts from noise, and from noise this game gives it nothing to learn from:
a random agent wins none of 700 games, and 300k steps of shaped training still
end at 0 wins and a few percent of pieces home. The bots, meanwhile, already
play it -- ``bottleneck`` beats ``advancedDistScore`` 84% of the time and costs
0.3ms a move. Copying one is a far cheaper way into the right region of policy
space than discovering it.

So this plays games with the bot on the agent's seat, records what it chose in
every position, and fits the policy to those choices by plain cross-entropy.
The result is a MaskablePPO checkpoint like any other, so ``train.py --init``
picks it up and fine-tunes it.

Why not mix the bot into PPO's own rollouts instead, playing the bot's move
some of the time? Because PPO is on-policy in a way that does not survive it:
``collect_rollouts`` stores the log-probability of the action the policy
*sampled*, and the update forms ``exp(log_prob - old_log_prob)`` against it.
Substituting a bot's action leaves that ratio measuring the wrong pair of
distributions, and nothing raises an error -- the run just optimises something
other than what it reports. Mixing the two policies during *collection* is a
real method (Ross et al.'s DAgger), but its labels come from the expert and its
loss is the supervised one below, not PPO's clipped objective. This is the
first half of it; --mix wires up the second.

What cloning cannot do is exceed its teacher, and it inherits the teacher's
blind spots. It is a starting point for PPO, not a replacement.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces
from sb3_contrib import MaskablePPO

from env.features import HalmaFeatures
from env.halmaEnv import HalmaEnv
from env.policy import FactoredMaskablePolicy
from heuristics.strategy import makeStrategy
from scripts.train import evaluate, report

MODELS = Path(__file__).resolve().parent.parent / "models"

# The observation planes that actually change during a game: own pieces and
# opponent pieces. Everything from index 2 on is constant -- see staticBlock.
DYNAMIC_PLANES = 2


def staticBlock(env: HalmaEnv) -> np.ndarray:
    """The planes of the observation that never change, as one (12, 17, 17) block.

    Index 2 onwards of the board: the field mask and the eleven geometry
    planes. ``collect`` stores only the two piece planes and ``fit`` glues this
    back on, so the constant part is held once instead of once per sample.
    """
    return np.concatenate([env.boardMask[None], env.geometryPlanes]).astype(np.float32)


def rebuildBoards(dynamic: np.ndarray, static: np.ndarray) -> np.ndarray:
    """Put a batch of stored piece planes back into full observation boards."""
    boards = np.empty((len(dynamic), DYNAMIC_PLANES + len(static), 17, 17), dtype=np.float32)
    boards[:, :DYNAMIC_PLANES] = dynamic
    boards[:, DYNAMIC_PLANES:] = static
    return boards


def collect(
    expert: str | Sequence[str],
    opponent: str,
    samples: int,
    seed: int,
    model: MaskablePPO | None = None,
    mix: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Play games and record (piece planes, scalars, packed mask, expert move).

    **Stored compactly, because the plain form does not fit.** A full
    observation is 14 planes of float32, but eleven of them are constant --
    the field mask and the geometry -- so keeping them per sample stores the
    same block half a million times. The action mask is worse: 14,641 entries
    of which some 65 are legal. At 500k samples the two together come to
    15.4 GB, which does not fit in memory; storing the two piece planes as
    uint8 and packing the mask to bits brings that to 1.2 GB.

    So this returns the *dynamic* half and ``fit`` reassembles each batch with
    :func:`rebuildBoards` and ``np.unpackbits``. A test pins that what comes
    back out is bit-for-bit the observation the environment produced -- a
    reconstruction that quietly disagreed would train the policy on something
    other than what it will be asked to play.

    ``mix`` is the probability that the *learner* picks the move actually
    played, while the expert still supplies the label -- DAgger's β. At 0 the
    states are the ones the expert visits, which is plain cloning and leaves
    the policy untested anywhere the expert never goes. Above 0 the states
    drift towards the ones the learner reaches, which is where its mistakes
    compound; the labels say what should have been done there.

    ``expert`` takes either one bot name or a sequence of them. A sequence
    draws one teacher per game, so the recorded moves -- and therefore the
    fitted policy -- are a blend rather than a single bot's blind spots.
    """
    experts = (expert,) if isinstance(expert, str) else tuple(expert)
    strategies = {name: makeStrategy(name) for name in set(experts)}
    dynamic, scalars, masks, actions = [], [], [], []
    rng = np.random.default_rng(seed)
    games = 0
    while len(actions) < samples:
        env = HalmaEnv(opponentStrategy=opponent)
        obs, _ = env.reset(seed=seed + games)
        strategy = strategies[experts[rng.integers(len(experts))]]
        games += 1
        while True:
            player = env.game.currentPlayer()
            moves = env.board.allValidMovesWithWay(player)
            best = strategy.bestMove(moves, env.board, player)
            key = env._permutationKey(player)
            normalized = env.normalizer.permuteMoves([(best[0], best[-1])], key)[0]
            expertAction = env.encodeAction(normalized)

            mask = env.action_masks()
            # Only the two piece planes are stored: everything from index 2 on
            # is constant, and packbits turns the 14,641-entry mask into 1,831
            # bytes. See the docstring -- at 500k samples the plain form is
            # 15.4 GB and this is 1.2 GB.
            dynamic.append(obs["board"][:DYNAMIC_PLANES].astype(np.uint8))
            scalars.append(obs["scalars"])
            masks.append(np.packbits(mask))
            actions.append(expertAction)

            played = expertAction
            if model is not None and mix > 0.0 and rng.random() < mix:
                # The learner drives, the expert still grades. Sampled rather
                # than argmax: an argmax learner revisits one narrow track and
                # the labels stop covering anything new.
                predicted, _ = model.predict(obs, action_masks=mask, deterministic=False)
                played = int(predicted)

            obs, _, terminated, truncated, _ = env.step(played)
            if terminated or truncated or len(actions) >= samples:
                break
    print(f"  {len(actions):,} positions from {games} games")
    return (
        np.array(dynamic, dtype=np.uint8),
        np.array(scalars, dtype=np.float32),
        np.array(masks, dtype=np.uint8),
        np.array(actions, dtype=np.int64),
    )


def fit(
    model: MaskablePPO,
    dynamic: np.ndarray,
    scalars: np.ndarray,
    masks: np.ndarray,
    actions: np.ndarray,
    static: np.ndarray,
    epochs: int,
    batch: int,
    learningRate: float,
    saveEvery: int = 0,
    checkpointName: str = "",
) -> None:
    """Maximise the log-probability the policy assigns to the expert's move.

    Cross-entropy over the *masked* distribution, which is the same
    distribution PPO will later sample from -- so the illegal actions carry no
    probability to begin with and the loss only ever ranks legal moves against
    each other.

    ``saveEvery`` writes a checkpoint every that many epochs, which ``main``
    otherwise does exactly once -- after the last epoch *and* after the closing
    evaluation. That single save is a long way to fall: a 500k run is some 80
    minutes of fitting, and anything that stops it before the end leaves
    nothing on disk at all. The counterpart on the PPO side is
    ``scripts/train.py --saveCheckpointsEvery``.

    The final epoch is checkpointed too rather than skipped as redundant. It is
    not redundant: ``main`` saves only after 300 evaluation games have run, so
    a failure in between would discard a finished fit.
    """
    policy = model.policy
    policy.set_training_mode(True)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learningRate)
    device = policy.device
    count = len(actions)
    actionSpace = model.action_space
    assert isinstance(actionSpace, spaces.Discrete)
    actionCount = int(actionSpace.n)
    actionTensor = torch.as_tensor(actions, device=device)

    for epoch in range(epochs):
        order = np.random.default_rng(epoch).permutation(count)
        totalLoss, correct = 0.0, 0
        for start in range(0, count, batch):
            index = order[start : start + batch]
            # Rebuilt here rather than held in memory -- see collect().
            batchObs = {
                "board": torch.as_tensor(rebuildBoards(dynamic[index], static), device=device),
                "scalars": torch.as_tensor(scalars[index], device=device),
            }
            batchMasks = np.unpackbits(masks[index], axis=1, count=actionCount).astype(bool)
            distribution = policy.get_distribution(batchObs, action_masks=batchMasks)
            expected = actionTensor[index]
            loss = -distribution.log_prob(expected).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totalLoss += loss.item() * len(index)
            # How often the policy's best legal move is the expert's move. The
            # honest read of progress -- the loss alone says little, since the
            # legal action count varies from position to position.
            with torch.no_grad():
                correct += int((distribution.get_actions(deterministic=True) == expected).sum())
        print(
            f"  epoch {epoch + 1}/{epochs}  loss {totalLoss / count:.4f}"
            f"  agrees with the bot {correct / count * 100:5.1f} %",
            flush=True,
        )
        if saveEvery and (epoch + 1) % saveEvery == 0:
            # .zip spelled out: sb3 only appends it when Path.suffix is empty,
            # and a name carrying a version number already looks suffixed to
            # pathlib -- the same trap scripts/train.py documents.
            path = MODELS / f"{checkpointName}_epoch{epoch + 1:02d}.zip"
            model.save(path)
            print(f"    saved {path.name}", flush=True)
            # save() flips the policy out of training mode.
            policy.set_training_mode(True)
    policy.set_training_mode(False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expert", nargs="+", default=["bottleneck"], help="bot(s) to copy, one drawn per game"
    )
    parser.add_argument("--opponent", default="advancedDistScore")
    parser.add_argument("--samples", type=int, default=100_000, help="positions to record")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--games", type=int, default=50, help="evaluation games")
    parser.add_argument("--seed", type=int, default=0)
    # DAgger rounds. 1 is plain cloning; more re-collects with the learner
    # driving a --mix share of the moves, so the labels start covering the
    # states its own mistakes lead to.
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument(
        "--mix", type=float, default=0.5, help="learner's share of moves after round 1"
    )
    parser.add_argument("--name", default="cloned")
    # Guards against losing a long fit: main() saves once, at the very end and
    # only after the closing evaluation. See fit().
    parser.add_argument(
        "--saveEvery",
        type=int,
        default=0,
        help="save a checkpoint every N epochs (0 = only the final model)",
    )
    args = parser.parse_args()

    print(f"cloning {' + '.join(args.expert)} against {args.opponent}\n")
    print("the teacher(s), for reference:")
    for expert in dict.fromkeys(args.expert):
        report(f"  {expert}", evaluateBot(expert, args.opponent, args.games))

    model = MaskablePPO(
        FactoredMaskablePolicy,
        HalmaEnv(opponentStrategy=args.opponent),
        seed=args.seed,
        verbose=0,
        policy_kwargs={"features_extractor_class": HalmaFeatures},
    )

    # Built once from a throwaway env: the constant planes are the same for
    # every position, which is the whole reason collect() does not store them.
    static = staticBlock(HalmaEnv())
    # Up front rather than beside the final save, because --saveEvery writes
    # into this directory long before then.
    MODELS.mkdir(exist_ok=True)

    for round in range(args.rounds):
        mix = 0.0 if round == 0 else args.mix
        label = "cloning" if round == 0 else f"DAgger round {round + 1}, learner drives {mix:.0%}"
        print(f"\n{label}: collecting", flush=True)
        started = time.perf_counter()
        dynamic, scalars, masks, actions = collect(
            args.expert,
            args.opponent,
            args.samples,
            args.seed + round * 1000,
            model=model if round else None,
            mix=mix,
        )
        print(f"  collected in {time.perf_counter() - started:.0f}s\nfitting:", flush=True)
        fit(
            model,
            dynamic,
            scalars,
            masks,
            actions,
            static,
            args.epochs,
            args.batch,
            args.lr,
            saveEvery=args.saveEvery,
            # DAgger refits the same model each round, so the round has to be
            # in the name or round 2 would overwrite round 1's checkpoints.
            checkpointName=args.name if args.rounds == 1 else f"{args.name}_r{round + 1}",
        )

        print("\nafter fitting:")
        for opponent in dict.fromkeys([args.opponent, "sparsityScore", "random"]):
            report(f"vs {opponent}", evaluate(model, opponent, args.games))
            report("   sampled", evaluate(model, opponent, args.games, deterministic=False))

    # Explicit .zip: sb3 only appends it when Path.suffix is empty, and a name
    # carrying a version number ("Talos1.0") already looks suffixed to pathlib.
    path = MODELS / f"{args.name}.zip"
    model.save(path)
    initHint = path.with_suffix("")
    print(f"\nsaved to {path}\n  fine-tune with: python -m scripts.train --init {initHint}")


def evaluateBot(strategy: str, opponent: str, games: int) -> dict:
    """The teacher's own score, so the clone has something to be measured against."""
    from game.gameManager import ComputedGame
    from game.player import Computer

    wins = losses = draws = 0
    homeFractions = []
    for seed in range(games):
        game = ComputedGame()
        game.seed(seed)
        teacher = Computer(HalmaEnv.AGENT_SEAT, strategy)
        game.initGame([teacher, Computer(HalmaEnv.OPPONENT_SEAT, opponent)])
        winner = game.play()
        if winner == HalmaEnv.AGENT_SEAT:
            wins += 1
        elif winner is None:
            draws += 1
        else:
            losses += 1
        homeFractions.append(
            len(teacher.positions & teacher.endPositions) / len(teacher.endPositions)
        )
    rate = wins / games
    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "winRate": rate,
        "marginOfError": 1.96 * float(np.sqrt(max(rate * (1 - rate), 1e-9) / games)),
        "avgSteps": 0.0,
        "homeFraction": float(np.mean(homeFractions)),
    }


if __name__ == "__main__":
    main()
