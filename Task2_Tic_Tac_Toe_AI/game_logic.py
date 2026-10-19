"""
Tic-Tac-Toe Game Engine and Adversarial Minimax AI.
CodeOrbit Tech AI Internship - Task 2.

Implements pure Python 3x3 game board representation, move validation,
terminal win/draw detection, and optimal Minimax game-tree search.
"""

import math
import random
from typing import Dict, List, Optional, Tuple

try:
    from config_task2 import (
        DIFFICULTY_EASY,
        DIFFICULTY_MEDIUM,
        DIFFICULTY_UNBEATABLE,
        EMPTY_CELL,
        PLAYER_O,
        PLAYER_X,
        SCORE_AI_WIN,
        SCORE_DRAW,
        SCORE_HUMAN_WIN,
        TOTAL_CELLS,
        WINNING_COMBINATIONS,
    )
except ImportError:
    from .config_task2 import (  # type: ignore
        DIFFICULTY_EASY,
        DIFFICULTY_MEDIUM,
        DIFFICULTY_UNBEATABLE,
        EMPTY_CELL,
        PLAYER_O,
        PLAYER_X,
        SCORE_AI_WIN,
        SCORE_DRAW,
        SCORE_HUMAN_WIN,
        TOTAL_CELLS,
        WINNING_COMBINATIONS,
    )


class InvalidMoveError(Exception):
    """Raised when an illegal move is attempted (e.g. occupied cell or out of bounds)."""
    pass


def create_empty_board() -> List[str]:
    """Return a fresh 9-cell empty board represented as a list of strings."""
    return [EMPTY_CELL] * TOTAL_CELLS


def is_valid_move(board: List[str], index: int) -> bool:
    """Check if cell index is within [0, 8] and currently empty."""
    if not (0 <= index < TOTAL_CELLS):
        return False
    return board[index] == EMPTY_CELL


def make_move(board: List[str], index: int, player: str) -> List[str]:
    """
    Apply a player move onto the board.
    
    Returns a new board state list (functional, non-destructive to input).
    Raises InvalidMoveError if the target cell is already occupied or invalid.
    """
    if not (0 <= index < TOTAL_CELLS):
        raise InvalidMoveError(f"Move to cell {index} is illegal. Index is out of bounds [0..8].")
    if board[index] != EMPTY_CELL:
        raise InvalidMoveError(f"Move to cell {index} is illegal. Cell is already occupied by '{board[index]}'.")
    if player not in (PLAYER_X, PLAYER_O):
        raise ValueError(f"Invalid player '{player}'. Must be '{PLAYER_X}' or '{PLAYER_O}'.")

    new_board = list(board)
    new_board[index] = player
    return new_board


def get_available_moves(board: List[str]) -> List[int]:
    """Return list of integer indices [0..8] corresponding to empty cells."""
    return [i for i, cell in enumerate(board) if cell == EMPTY_CELL]


def check_winner(board: List[str]) -> Optional[str]:
    """
    Inspect all 8 winning combinations (3 rows, 3 columns, 2 diagonals).
    
    Returns 'X', 'O', or None if no winner has been achieved.
    """
    for a, b, c in WINNING_COMBINATIONS:
        if board[a] != EMPTY_CELL and board[a] == board[b] == board[c]:
            return board[a]
    return None


def get_winning_line(board: List[str]) -> Optional[Tuple[int, int, int]]:
    """Return the 3 cell indices forming the winning line, or None if no win."""
    for line in WINNING_COMBINATIONS:
        a, b, c = line
        if board[a] != EMPTY_CELL and board[a] == board[b] == board[c]:
            return line
    return None


def is_board_full(board: List[str]) -> bool:
    """Return True if all 9 cells are occupied, False otherwise."""
    return EMPTY_CELL not in board


def is_game_over(board: List[str]) -> Tuple[bool, Optional[str]]:
    """
    Check if the current board represents a terminal state.
    
    Returns:
        (is_over: bool, winner: Optional[str])
        - winner is 'X' or 'O' if won.
        - winner is None if game is a draw or still in progress.
    """
    winner = check_winner(board)
    if winner is not None:
        return True, winner
    if is_board_full(board):
        return True, None
    return False, None


# ---------------------------------------------------------------------------
# Minimax Adversarial Search Algorithm
# ---------------------------------------------------------------------------
class MinimaxAI:
    """
    Minimax adversarial search engine with Alpha-Beta pruning and depth scoring.
    
    Terminal evaluation:
    - AI ('O') win: +10 - depth  (rewards faster wins)
    - Human ('X') win: -10 + depth  (rewards delaying human wins)
    - Draw: 0
    """

    def __init__(self, random_seed: Optional[int] = None) -> None:
        self._rng = random.Random(random_seed)
        self.nodes_evaluated: int = 0

    def minimax(
        self,
        board: List[str],
        depth: int,
        is_maximizing: bool,
        alpha: float = -math.inf,
        beta: float = math.inf,
    ) -> int:
        """
        Recursive Minimax search with Alpha-Beta pruning.
        
        Args:
            board: Current 9-cell board state.
            depth: Current search tree recursion depth.
            is_maximizing: True if AI ('O') turn, False if Human ('X') turn.
            alpha: Best score achievable by maximizer along path.
            beta: Best score achievable by minimizer along path.
            
        Returns:
            Heuristic minimax score for the board state.
        """
        self.nodes_evaluated += 1

        winner = check_winner(board)
        if winner == PLAYER_O:
            return SCORE_AI_WIN - depth
        elif winner == PLAYER_X:
            return SCORE_HUMAN_WIN + depth
        elif is_board_full(board):
            return SCORE_DRAW

        available = get_available_moves(board)

        if is_maximizing:
            max_eval = -math.inf
            for move in available:
                board[move] = PLAYER_O
                eval_score = self.minimax(board, depth + 1, False, alpha, beta)
                board[move] = EMPTY_CELL
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return int(max_eval)
        else:
            min_eval = math.inf
            for move in available:
                board[move] = PLAYER_X
                eval_score = self.minimax(board, depth + 1, True, alpha, beta)
                board[move] = EMPTY_CELL
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return int(min_eval)

    def evaluate_all_moves(self, board: List[str]) -> Dict[int, int]:
        """
        Evaluate every legal move from the current board state using Minimax.
        
        Returns:
            Dictionary mapping cell_index -> minimax_score.
        """
        scores: Dict[int, int] = {}
        for move in get_available_moves(board):
            board[move] = PLAYER_O
            # Depth 1 because the candidate move represents 1 ply taken
            score = self.minimax(board, depth=1, is_maximizing=False)
            board[move] = EMPTY_CELL
            scores[move] = score
        return scores

    def get_best_move(
        self,
        board: List[str],
        difficulty: str = DIFFICULTY_UNBEATABLE,
    ) -> Tuple[int, Dict[int, int], int]:
        """
        Select the optimal move according to the configured difficulty mode.
        
        Returns:
            (best_move_index: int, evaluations_dict: Dict[int, int], total_nodes: int)
        """
        available = get_available_moves(board)
        if not available:
            raise InvalidMoveError("Cannot compute move on a full or terminal board.")

        self.nodes_evaluated = 0

        # Mode 1: Novice (Pure random legal move)
        if difficulty == DIFFICULTY_EASY:
            chosen = self._rng.choice(available)
            return chosen, {m: 0 for m in available}, 1

        # Mode 2: Tactical / Medium (Checks immediate win/block, else 60% optimal)
        if difficulty == DIFFICULTY_MEDIUM:
            # Check immediate win for AI
            for move in available:
                board[move] = PLAYER_O
                if check_winner(board) == PLAYER_O:
                    board[move] = EMPTY_CELL
                    return move, {move: 10}, 1
                board[move] = EMPTY_CELL

            # Check immediate block for Human
            for move in available:
                board[move] = PLAYER_X
                if check_winner(board) == PLAYER_X:
                    board[move] = EMPTY_CELL
                    return move, {move: -10}, 1
                board[move] = EMPTY_CELL

            # 40% chance of random tactical move
            if self._rng.random() < 0.40:
                chosen = self._rng.choice(available)
                return chosen, {m: 0 for m in available}, 1

        # Mode 3: Unbeatable Optimal Minimax
        move_scores = self.evaluate_all_moves(board)
        max_score = max(move_scores.values())

        # Filter all moves that achieve the maximum score
        best_candidates = [move for move, score in move_scores.items() if score == max_score]

        # Strategic preference among tied optimal moves:
        # 1. Center (cell 4)
        # 2. Corners (0, 2, 6, 8)
        # 3. Edges (1, 3, 5, 7)
        if 4 in best_candidates:
            chosen = 4
        else:
            corner_candidates = [c for c in best_candidates if c in (0, 2, 6, 8)]
            if corner_candidates:
                chosen = self._rng.choice(corner_candidates)
            else:
                chosen = self._rng.choice(best_candidates)

        return chosen, move_scores, self.nodes_evaluated
