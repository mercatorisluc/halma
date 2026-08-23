"""The league's opponent-pool rule.

Worth pinning because it decides what a training round actually spends its
episodes on, and getting it wrong is invisible: the run still completes, it
just trains against the wrong mixture. `HalmaEnv.reset` draws from the pool
uniformly, so the pool's *composition* is the schedule.
"""

from scripts.talos2League import DEFAULTS, opponentPool

ANCHOR = "anchor"
ROUNDS = ["r1", "r2", "r3", "r4", "r5"]


def test_the_first_round_trains_against_the_anchor_alone():
    assert opponentPool(ANCHOR, [], None) == [ANCHOR]


def test_uncapped_keeps_every_finished_round():
    assert opponentPool(ANCHOR, ROUNDS, None) == [ANCHOR, *ROUNDS]


def test_a_cap_keeps_the_most_recent_rounds_not_the_oldest():
    assert opponentPool(ANCHOR, ROUNDS, 3) == [ANCHOR, "r3", "r4", "r5"]


def test_the_anchor_survives_any_cap():
    # It is the fixed reference every round is measured against, so a league
    # that dropped it would have no common yardstick left.
    assert opponentPool(ANCHOR, ROUNDS, 0) == [ANCHOR]
    assert ANCHOR in opponentPool(ANCHOR, ROUNDS, 1)


def test_a_cap_larger_than_the_history_is_harmless():
    assert opponentPool(ANCHOR, ["r1", "r2"], 3) == [ANCHOR, "r1", "r2"]


def test_the_recorded_recipe_is_still_the_default():
    # Talos2.0_round1..5 were produced with exactly these; changing one
    # silently would make the recorded league unreproducible.
    assert DEFAULTS["rounds"] == 5
    assert DEFAULTS["steps"] == 300_000
    assert DEFAULTS["parity"] == 0.25
    assert DEFAULTS["opponentSampling"] == 0.5
    assert DEFAULTS["seed"] == 42
