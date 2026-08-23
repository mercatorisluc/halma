"""The shared match harness six measurement scripts now run on.

What is locked here is the part that is easy to get wrong and expensive to
notice: that swapping seats swaps *instances*, that the reverse leg is scored
from the same side as the forward one, and that a self-match therefore comes
out at exactly 50%. Every recorded win rate in RESULTS.md rests on those
three things.
"""

import pytest

from game.player import Computer
from scripts.matchStats import (
    SEAT_ONE,
    SEAT_TWO,
    Bot,
    BotPool,
    Fixed,
    Result,
    Seated,
    marginOfError,
    playBothSeats,
    playMatch,
)


def test_result_counts_add_up():
    result = Result(wins=3, losses=2, draws=1)
    assert result.games == 6
    assert result.winRate == pytest.approx(0.5)


def test_result_of_no_games_does_not_divide_by_zero():
    empty = Result()
    assert empty.winRate == 0.0
    assert empty.marginOfError == 0.0


def test_flipping_a_result_swaps_wins_and_losses_but_not_draws():
    flipped = Result(wins=3, losses=2, draws=1, moves=10).flipped()
    assert (flipped.wins, flipped.losses, flipped.draws, flipped.moves) == (2, 3, 1, 10)


def test_results_add_componentwise():
    total = Result(wins=1, losses=2, draws=3, moves=4) + Result(
        wins=10, losses=20, draws=30, moves=40
    )
    assert (total.wins, total.losses, total.draws, total.moves) == (11, 22, 33, 44)


def test_margin_shrinks_as_games_grow():
    assert marginOfError(0.5, 100) > marginOfError(0.5, 400)


def test_bot_factory_seats_on_the_seat_it_is_asked_for():
    make = Bot("distance")
    assert make(SEAT_TWO, 0).identifier == SEAT_TWO


def test_bot_factory_builds_a_fresh_player_each_time():
    # baseline plays hundreds of games off one factory; a shared player would
    # carry the previous game's positions into the next.
    make = Bot("distance")
    assert make(SEAT_ONE, 0) is not make(SEAT_ONE, 1)


def test_bot_pool_cycles_by_game_index():
    make = BotPool(("distance", "shaped"))
    players = [make(SEAT_ONE, index) for index in range(4)]
    names = [player.strategy.strategyName for player in players if isinstance(player, Computer)]
    assert names == ["distance", "shaped", "distance", "shaped"]


def test_fixed_returns_the_same_player_every_game():
    player = Computer(SEAT_ONE, "distance")
    make = Fixed(player)
    assert make(SEAT_ONE, 0) is player
    assert make(SEAT_ONE, 9) is player


def test_seated_picks_the_instance_built_for_that_seat():
    # The one that matters: a player's seat is fixed at construction and
    # Initializer hands out home corners by list position, so the swapped leg
    # needs the other instance, not the same one on a different seat.
    bySeat = {SEAT_ONE: Computer(SEAT_ONE, "distance"), SEAT_TWO: Computer(SEAT_TWO, "distance")}
    make = Seated(bySeat)
    assert make(SEAT_ONE, 0) is bySeat[SEAT_ONE]
    assert make(SEAT_TWO, 0) is bySeat[SEAT_TWO]


def test_a_match_plays_the_games_it_was_asked_for():
    result = playMatch(Bot("distance"), Bot("shaped"), games=4, seed=0)
    assert result.games == 4
    assert result.moves > 0


def test_a_match_is_reproducible_from_its_seed():
    first = playMatch(Bot("distance"), Bot("shaped"), games=4, seed=7)
    again = playMatch(Bot("distance"), Bot("shaped"), games=4, seed=7)
    assert (first.wins, first.losses, first.draws, first.moves) == (
        again.wins,
        again.losses,
        again.draws,
        again.moves,
    )


def test_scoring_from_the_other_seat_inverts_the_result():
    forward = playMatch(Bot("shaped"), Bot("distance"), games=6, seed=0)
    fromSeatTwo = playMatch(Bot("shaped"), Bot("distance"), games=6, seed=0, scoredSeat=SEAT_TWO)
    assert fromSeatTwo.wins == forward.losses
    assert fromSeatTwo.losses == forward.wins


def test_both_seats_splits_a_self_match_exactly_evenly():
    # The sharpest control there is: the two legs of a self-match are mirror
    # images, so anything other than 50/50 means the swap is not a real swap.
    combined, onSeatOne, onSeatTwo = playBothSeats(Bot("shaped"), Bot("shaped"), games=8, seed=0)
    assert combined.winRate == pytest.approx(0.5)
    assert onSeatOne.wins + onSeatTwo.wins == onSeatOne.losses + onSeatTwo.losses


def test_both_seats_totals_its_two_legs():
    combined, onSeatOne, onSeatTwo = playBothSeats(Bot("shaped"), Bot("distance"), games=6, seed=0)
    assert combined.games == onSeatOne.games + onSeatTwo.games
    assert combined.wins == onSeatOne.wins + onSeatTwo.wins


def test_both_seats_scores_the_reverse_leg_from_the_same_side():
    # `shaped` beats `distance` by a wide margin, so a reverse leg scored from
    # the wrong side would drag the combined rate towards 50% or below.
    combined, _, onSeatTwo = playBothSeats(Bot("shaped"), Bot("distance"), games=10, seed=0)
    assert combined.winRate > 0.6
    assert onSeatTwo.winRate > 0.6
