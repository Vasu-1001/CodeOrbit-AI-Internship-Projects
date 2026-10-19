"""
Configuration and Constants for Task 2: Tic-Tac-Toe AI.
CodeOrbit Tech AI Internship - Task 2.

Defines board configurations, player markers, winning lines,
and UI styling metadata.
"""

from typing import List, Tuple

# ---------------------------------------------------------------------------
# Project Metadata
# ---------------------------------------------------------------------------
PROJECT_TITLE = "Tic-Tac-Toe AI Studio"
INTERNSHIP_TASK_NAME = "Task 2: Tic-Tac-Toe AI with Minimax"
INTERNSHIP_ORGANIZATION = "CodeOrbit Tech"
PROJECT_VERSION = "1.0.0"
AUTHOR = "AI Intern"

# ---------------------------------------------------------------------------
# Game Board Symbols & Players
# ---------------------------------------------------------------------------
PLAYER_X = "X"       # Human player
PLAYER_O = "O"       # Computer AI player
EMPTY_CELL = " "     # Unoccupied cell

# Board Dimensions
BOARD_SIZE = 3
TOTAL_CELLS = BOARD_SIZE * BOARD_SIZE  # 9 cells

# Winning Lines (3 horizontal, 3 vertical, 2 diagonal)
WINNING_COMBINATIONS: List[Tuple[int, int, int]] = [
    # Rows
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    # Columns
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    # Diagonals
    (0, 4, 8),
    (2, 4, 6),
]

# Minimax Base Terminal Scores
SCORE_AI_WIN = 10
SCORE_HUMAN_WIN = -10
SCORE_DRAW = 0

# AI Difficulty Presets
DIFFICULTY_UNBEATABLE = "Unbeatable (Minimax AI)"
DIFFICULTY_MEDIUM = "Intermediate (Tactical AI)"
DIFFICULTY_EASY = "Novice (Random AI)"
