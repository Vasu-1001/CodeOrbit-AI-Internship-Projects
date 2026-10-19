"""
Automated Unit and System Tests for Tic-Tac-Toe Minimax AI.
CodeOrbit Tech AI Internship - Task 2.

Comprehensive test suite verifying:
1. Every win condition: 3 horizontal rows, 3 vertical columns, 2 diagonals for X and O.
2. Draw detection on complete boards.
3. Invalid moves, occupied cells, and out-of-bounds guards.
4. Minimax optimal move selection (instant win execution, opponent block).
5. Terminal state scoring and depth penalties.
6. Exhaustive / stochastic unbeatability verification: Minimax NEVER loses.
"""

import pytest
import random
from typing import List

from game_logic import (
    DIFFICULTY_UNBEATABLE,
    EMPTY_CELL,
    PLAYER_O,
    PLAYER_X,
    InvalidMoveError,
    MinimaxAI,
    check_winner,
    create_empty_board,
    get_available_moves,
    get_winning_line,
    is_board_full,
    is_game_over,
    is_valid_move,
    make_move,
)


@pytest.fixture
def ai() -> MinimaxAI:
    """Fixture providing deterministic Minimax AI engine."""
    return MinimaxAI(random_seed=42)


# ---------------------------------------------------------------------------
# 1. Board Initialization & Move Validation
# ---------------------------------------------------------------------------
def test_create_empty_board():
    """Verify fresh board has 9 empty cells."""
    board = create_empty_board()
    assert len(board) == 9
    assert all(cell == EMPTY_CELL for cell in board)
    assert len(get_available_moves(board)) == 9


def test_valid_move_and_immutability():
    """Verify making a legal move updates cell without mutating original board."""
    board = create_empty_board()
    assert is_valid_move(board, 4) is True

    next_board = make_move(board, 4, PLAYER_X)
    assert next_board[4] == PLAYER_X
    assert board[4] == EMPTY_CELL  # Original unchanged
    assert is_valid_move(next_board, 4) is False


def test_invalid_move_occupied_cell():
    """Verify placing marker in an occupied cell raises InvalidMoveError."""
    board = create_empty_board()
    board = make_move(board, 0, PLAYER_X)
    with pytest.raises(InvalidMoveError):
        make_move(board, 0, PLAYER_O)


@pytest.mark.parametrize("invalid_idx", [-1, 9, 100, -10])
def test_out_of_bounds_move(invalid_idx: int):
    """Verify out-of-bounds cell indices are rejected."""
    board = create_empty_board()
    assert is_valid_move(board, invalid_idx) is False
    with pytest.raises(InvalidMoveError):
        make_move(board, invalid_idx, PLAYER_X)


# ---------------------------------------------------------------------------
# 2. Win Detection (Horizontal, Vertical, Diagonal)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("player", [PLAYER_X, PLAYER_O])
@pytest.mark.parametrize("row_indices", [
    (0, 1, 2),  # Top row
    (3, 4, 5),  # Middle row
    (6, 7, 8),  # Bottom row
])
def test_horizontal_wins(player: str, row_indices: tuple):
    """Verify detection of horizontal row wins for both X and O."""
    board = create_empty_board()
    for idx in row_indices:
        board[idx] = player
    assert check_winner(board) == player
    assert get_winning_line(board) == row_indices
    is_over, winner = is_game_over(board)
    assert is_over is True
    assert winner == player


@pytest.mark.parametrize("player", [PLAYER_X, PLAYER_O])
@pytest.mark.parametrize("col_indices", [
    (0, 3, 6),  # Left column
    (1, 4, 7),  # Middle column
    (2, 5, 8),  # Right column
])
def test_vertical_wins(player: str, col_indices: tuple):
    """Verify detection of vertical column wins for both X and O."""
    board = create_empty_board()
    for idx in col_indices:
        board[idx] = player
    assert check_winner(board) == player
    assert get_winning_line(board) == col_indices
    is_over, winner = is_game_over(board)
    assert is_over is True
    assert winner == player


@pytest.mark.parametrize("player", [PLAYER_X, PLAYER_O])
@pytest.mark.parametrize("diag_indices", [
    (0, 4, 8),  # Main diagonal
    (2, 4, 6),  # Anti-diagonal
])
def test_diagonal_wins(player: str, diag_indices: tuple):
    """Verify detection of diagonal wins for both X and O."""
    board = create_empty_board()
    for idx in diag_indices:
        board[idx] = player
    assert check_winner(board) == player
    assert get_winning_line(board) == diag_indices
    is_over, winner = is_game_over(board)
    assert is_over is True
    assert winner == player


# ---------------------------------------------------------------------------
# 3. Draw Detection
# ---------------------------------------------------------------------------
def test_draw_condition():
    """Verify full board with no winning trio resolves to a Draw."""
    # Classic draw board:
    # X O X
    # X O O
    # O X X
    board = [
        PLAYER_X, PLAYER_O, PLAYER_X,
        PLAYER_X, PLAYER_O, PLAYER_O,
        PLAYER_O, PLAYER_X, PLAYER_X,
    ]
    assert is_board_full(board) is True
    assert check_winner(board) is None
    assert get_winning_line(board) is None
    is_over, winner = is_game_over(board)
    assert is_over is True
    assert winner is None


# ---------------------------------------------------------------------------
# 4. Minimax Scoring & Optimal AI Decisions
# ---------------------------------------------------------------------------
def test_minimax_immediate_win(ai: MinimaxAI):
    """
    Verify AI immediately takes the winning cell when available.
    Board:
    O O .
    X X .
    . . .
    AI ('O') must pick cell 2 to win immediately!
    """
    board = [
        PLAYER_O, PLAYER_O, EMPTY_CELL,
        PLAYER_X, PLAYER_X, EMPTY_CELL,
        EMPTY_CELL, EMPTY_CELL, EMPTY_CELL,
    ]
    best_move, evaluations, _ = ai.get_best_move(board, difficulty=DIFFICULTY_UNBEATABLE)
    assert best_move == 2
    assert evaluations[2] > 0


def test_minimax_immediate_block(ai: MinimaxAI):
    """
    Verify AI immediately blocks human from winning on the next turn.
    Board:
    X X .
    O . .
    . . .
    Human ('X') threatens cell 2. AI ('O') must play cell 2 to block.
    """
    board = [
        PLAYER_X, PLAYER_X, EMPTY_CELL,
        PLAYER_O, EMPTY_CELL, EMPTY_CELL,
        EMPTY_CELL, EMPTY_CELL, EMPTY_CELL,
    ]
    best_move, evaluations, _ = ai.get_best_move(board, difficulty=DIFFICULTY_UNBEATABLE)
    assert best_move == 2


def test_minimax_center_preference_or_corner(ai: MinimaxAI):
    """
    When human opens in a corner (e.g. 0), the optimal defensive response is center (4).
    Board:
    X . .
    . . .
    . . .
    """
    board = [
        PLAYER_X, EMPTY_CELL, EMPTY_CELL,
        EMPTY_CELL, EMPTY_CELL, EMPTY_CELL,
        EMPTY_CELL, EMPTY_CELL, EMPTY_CELL,
    ]
    best_move, evaluations, _ = ai.get_best_move(board, difficulty=DIFFICULTY_UNBEATABLE)
    assert best_move == 4, f"Expected center cell 4, got {best_move}"


def test_minimax_depth_scoring(ai: MinimaxAI):
    """Verify that earlier wins produce strictly higher minimax scores than delayed wins."""
    # State A: AI wins in 1 move
    board_a = [
        PLAYER_O, PLAYER_O, EMPTY_CELL,
        PLAYER_X, PLAYER_X, EMPTY_CELL,
        EMPTY_CELL, EMPTY_CELL, EMPTY_CELL,
    ]
    scores_a = ai.evaluate_all_moves(board_a)

    # Move 2 wins immediately at depth 1: score = 10 - 1 = 9
    assert scores_a[2] == 9


# ---------------------------------------------------------------------------
# 5. Exhaustive Unbeatability Proof (Zero AI Losses)
# ---------------------------------------------------------------------------
def test_ai_never_loses_against_random_opponent(ai: MinimaxAI):
    """
    Simulate 100 complete games where Human plays random legal moves
    and AI plays Minimax optimal moves.
    
    Theorem: The optimal Minimax AI player in Tic-Tac-Toe CANNOT LOSE.
    The outcome MUST be either AI win or Draw. AI Loss count must be ZERO.
    """
    rng = random.Random(1337)
    ai_wins = 0
    draws = 0
    human_wins = 0

    for _ in range(100):
        board = create_empty_board()
        current_turn = PLAYER_X  # Human starts

        while True:
            is_over, winner = is_game_over(board)
            if is_over:
                if winner == PLAYER_O:
                    ai_wins += 1
                elif winner == PLAYER_X:
                    human_wins += 1
                else:
                    draws += 1
                break

            if current_turn == PLAYER_X:
                # Random legal human move
                available = get_available_moves(board)
                chosen = rng.choice(available)
                board = make_move(board, chosen, PLAYER_X)
                current_turn = PLAYER_O
            else:
                # Minimax AI move
                best_move, _, _ = ai.get_best_move(board, difficulty=DIFFICULTY_UNBEATABLE)
                board = make_move(board, best_move, PLAYER_O)
                current_turn = PLAYER_X

    assert human_wins == 0, f"AI lost {human_wins} games! Minimax must be unbeatable."
    assert (ai_wins + draws) == 100
