"""Pytest configuration shared by the whole test suite.

Placing this file at the project root makes pytest add the root to ``sys.path``
(``prepend`` import mode), so tests can ``import game.board`` etc. the same way
``main.py`` does when run from the project root.
"""
import pytest

from game.gameManager import ComputedGame


@pytest.fixture
def game():
    """A freshly initialised standard 2-bot game."""
    g = ComputedGame()
    g.initStandardGame()
    return g


@pytest.fixture
def board(game):
    """The board of a freshly initialised standard game."""
    return game.board
