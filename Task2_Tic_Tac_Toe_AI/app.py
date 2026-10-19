"""
Tic-Tac-Toe AI Studio - Web Application
CodeOrbit Tech AI Internship - Task 2: Tic-Tac-Toe with Simple AI (Minimax).

Interactive Streamlit interface featuring:
- Responsive 3x3 interactive game board with high-contrast X and O markers
- Unbeatable Minimax Adversarial AI with Alpha-Beta pruning
- Real-time decision explainability (candidate scores, nodes evaluated)
- Session scorecards tracking Human Wins, AI Wins, Draws, and Streaks
- Difficulty modes: Unbeatable (Minimax), Intermediate, and Novice
- Educational Viva guide on Game-Tree Search, Zero-Sum Games, and Minimax
"""

import math
from typing import Dict, List, Optional, Tuple
import streamlit as st

from config_task2 import (
    AUTHOR,
    BOARD_SIZE,
    DIFFICULTY_EASY,
    DIFFICULTY_MEDIUM,
    DIFFICULTY_UNBEATABLE,
    EMPTY_CELL,
    INTERNSHIP_ORGANIZATION,
    INTERNSHIP_TASK_NAME,
    PLAYER_O,
    PLAYER_X,
    PROJECT_TITLE,
    PROJECT_VERSION,
)
from game_logic import (
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

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=f"{PROJECT_TITLE} | {INTERNSHIP_ORGANIZATION}",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0284C7 100%);
        color: white;
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 20px;
        box-shadow: 0 4px 14px 0 rgba(0, 0, 0, 0.15);
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 800;
        margin: 0;
        color: #FFFFFF !important;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #BAE6FD;
        margin-top: 6px;
        margin-bottom: 0;
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-right: 6px;
        margin-top: 8px;
        background-color: rgba(255, 255, 255, 0.15);
        color: #F0F9FF;
    }

    /* Custom Board Button Styling */
    div[data-testid="stButton"] > button {
        height: 90px;
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        border-radius: 12px;
        border: 2px solid #CBD5E1;
        transition: all 0.2s ease-in-out;
    }
    div[data-testid="stButton"] > button:hover {
        border-color: #0284C7;
        background-color: #F0F9FF;
        transform: translateY(-2px);
    }

    .status-card {
        padding: 16px 20px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 1.15rem;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .status-turn {
        background: #F0F9FF;
        color: #0369A1;
        border: 1px solid #BAE6FD;
    }
    .status-win-human {
        background: #ECFDF5;
        color: #047857;
        border: 1px solid #6EE7B7;
    }
    .status-win-ai {
        background: #FEF2F2;
        color: #B91C1C;
        border: 1px solid #FCA5A5;
    }
    .status-draw {
        background: #FEF3C7;
        color: #B45309;
        border: 1px solid #FCD34D;
    }

    .tree-inspector-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #334155;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# AI Engine Singleton Initialization
# ---------------------------------------------------------------------------
@st.cache_resource
def get_ai_engine() -> MinimaxAI:
    return MinimaxAI(random_seed=42)

ai_engine = get_ai_engine()

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "board" not in st.session_state:
    st.session_state.board = create_empty_board()

if "game_over" not in st.session_state:
    st.session_state.game_over = False

if "winner" not in st.session_state:
    st.session_state.winner = None

if "winning_line" not in st.session_state:
    st.session_state.winning_line = None

if "last_ai_meta" not in st.session_state:
    st.session_state.last_ai_meta = {
        "move": None,
        "scores": {},
        "nodes": 0,
    }

if "move_history" not in st.session_state:
    st.session_state.move_history = []

# Persistent Statistics
if "stats" not in st.session_state:
    st.session_state.stats = {
        "human_wins": 0,
        "ai_wins": 0,
        "draws": 0,
        "total_games": 0,
    }

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def reset_current_game() -> None:
    """Reset board state for a fresh game while preserving session statistics."""
    st.session_state.board = create_empty_board()
    st.session_state.game_over = False
    st.session_state.winner = None
    st.session_state.winning_line = None
    st.session_state.last_ai_meta = {"move": None, "scores": {}, "nodes": 0}
    st.session_state.move_history = []


def reset_all_stats() -> None:
    """Clear session statistics and reset the board."""
    st.session_state.stats = {
        "human_wins": 0,
        "ai_wins": 0,
        "draws": 0,
        "total_games": 0,
    }
    reset_current_game()


def handle_cell_click(cell_idx: int, difficulty: str) -> None:
    """
    Execute human turn followed immediately by computer AI turn.
    """
    if st.session_state.game_over:
        return

    board = st.session_state.board

    if not is_valid_move(board, cell_idx):
        return

    # 1. Apply Human Move ('X')
    board = make_move(board, cell_idx, PLAYER_X)
    st.session_state.board = board
    st.session_state.move_history.append(f"Human (X) -> Cell {cell_idx}")

    # Check terminal state after human move
    is_over, winner = is_game_over(board)
    if is_over:
        st.session_state.game_over = True
        st.session_state.winner = winner
        st.session_state.winning_line = get_winning_line(board)
        st.session_state.stats["total_games"] += 1
        if winner == PLAYER_X:
            st.session_state.stats["human_wins"] += 1
        else:
            st.session_state.stats["draws"] += 1
        return

    # 2. Compute Computer AI Move ('O')
    best_move, move_scores, nodes_eval = ai_engine.get_best_move(board, difficulty=difficulty)

    board = make_move(board, best_move, PLAYER_O)
    st.session_state.board = board
    st.session_state.last_ai_meta = {
        "move": best_move,
        "scores": move_scores,
        "nodes": nodes_eval,
    }
    st.session_state.move_history.append(f"AI (O) -> Cell {best_move}")

    # Check terminal state after AI move
    is_over, winner = is_game_over(board)
    if is_over:
        st.session_state.game_over = True
        st.session_state.winner = winner
        st.session_state.winning_line = get_winning_line(board)
        st.session_state.stats["total_games"] += 1
        if winner == PLAYER_O:
            st.session_state.stats["ai_wins"] += 1
        else:
            st.session_state.stats["draws"] += 1


# ---------------------------------------------------------------------------
# Sidebar Settings & Statistics
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/game-controller.png", width=85)
    st.markdown(f"### {INTERNSHIP_TASK_NAME}")
    st.caption(f"{INTERNSHIP_ORGANIZATION} • Version {PROJECT_VERSION}")

    st.markdown("---")
    st.markdown("#### 🎯 AI Difficulty Setting")
    difficulty = st.selectbox(
        "Select Opponent AI Mode:",
        options=[DIFFICULTY_UNBEATABLE, DIFFICULTY_MEDIUM, DIFFICULTY_EASY],
        index=0,
        help="Unbeatable uses pure Minimax and is mathematically impossible to defeat.",
    )

    st.markdown("---")
    st.markdown("#### 🏆 Session Scorecard")
    st1, st2 = st.columns(2)
    with st1:
        st.metric("Human Wins (X)", st.session_state.stats["human_wins"])
        st.metric("Draws", st.session_state.stats["draws"])
    with st2:
        st.metric("AI Wins (O)", st.session_state.stats["ai_wins"])
        st.metric("Total Games", st.session_state.stats["total_games"])

    st.markdown("---")
    st.markdown("#### 🕹️ Game Controls")
    if st.button("🔄 New Game", use_container_width=True):
        reset_current_game()
        st.rerun()

    if st.button("🗑️ Reset All Scorecards", use_container_width=True):
        reset_all_stats()
        st.rerun()

    st.markdown("---")
    st.markdown("#### 📜 Match Turn History")
    if st.session_state.move_history:
        for item in reversed(st.session_state.move_history[-6:]):
            st.caption(f"• {item}")
    else:
        st.caption("No moves placed yet. Click any square on the board to start!")

# ---------------------------------------------------------------------------
# Main Layout
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero-banner">
        <h1 class="hero-title">⚔️ {PROJECT_TITLE}</h1>
        <p class="hero-subtitle">
            Adversarial Game-Tree Search • Optimal Minimax Algorithm with Alpha-Beta Pruning
        </p>
        <div>
            <span class="badge-pill">🏢 {INTERNSHIP_ORGANIZATION}</span>
            <span class="badge-pill">📌 {INTERNSHIP_TASK_NAME}</span>
            <span class="badge-pill">🛡️ Unbeatable Strategy</span>
            <span class="badge-pill">🧠 Deterministic Game Search (Not ML)</span>
            <span class="badge-pill">⚡ Sub-millisecond Latency</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_game, tab_explain, tab_viva = st.tabs(
    ["🎮 Play Game", "🔍 Minimax Decision Inspector", "📖 Algorithm Theory & Viva"]
)

# ---------------------------------------------------------------------------
# Tab 1: Interactive Play Game
# ---------------------------------------------------------------------------
with tab_game:
    # Status Banner
    if not st.session_state.game_over:
        st.markdown(
            """
            <div class="status-card status-turn">
                <span>🎮</span>
                <div>
                    <strong>Your Turn (Playing as X)</strong> — Click any unoccupied cell to make your move.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        winner = st.session_state.winner
        if winner == PLAYER_X:
            st.markdown(
                """
                <div class="status-card status-win-human">
                    <span>🎉</span>
                    <div>
                        <strong>VICTORY!</strong> You defeated the AI! (Remarkable achievement!)
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif winner == PLAYER_O:
            st.markdown(
                """
                <div class="status-card status-win-ai">
                    <span>🤖</span>
                    <div>
                        <strong>AI WINS!</strong> Minimax evaluated all prospective futures and claimed victory.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="status-card status-draw">
                    <span>🤝</span>
                    <div>
                        <strong>DRAW!</strong> Both players played defensively. No winning line was conceded.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # 3x3 Game Board Grid
    board_col, side_col = st.columns([1.8, 1.2])

    with board_col:
        st.markdown("#### 3×3 Game Arena")
        current_board = st.session_state.board
        winning_trio = st.session_state.winning_line or ()

        for row in range(BOARD_SIZE):
            c0, c1, c2 = st.columns(3)
            row_cols = [c0, c1, c2]

            for col in range(BOARD_SIZE):
                cell_index = row * BOARD_SIZE + col
                cell_val = current_board[cell_index]

                # Display symbol
                if cell_val == PLAYER_X:
                    label = "❌"
                elif cell_val == PLAYER_O:
                    label = "⭕"
                else:
                    label = " "

                # If winning line, prepend star
                if cell_index in winning_trio:
                    label = f"✨ {label}"

                # Render button
                is_disabled = (cell_val != EMPTY_CELL) or st.session_state.game_over

                with row_cols[col]:
                    if st.button(
                        label,
                        key=f"cell_{cell_index}",
                        disabled=is_disabled,
                        use_container_width=True,
                    ):
                        handle_cell_click(cell_index, difficulty)
                        st.rerun()

    with side_col:
        st.markdown("#### ℹ️ Current Game Insights")
        moves_left = len(get_available_moves(st.session_state.board))
        st.info(
            f"**Cells Remaining:** {moves_left} / 9  \n"
            f"**Difficulty Level:** `{difficulty}`  \n"
            f"**Your Marker:** `X` (First Move)  \n"
            f"**Computer Marker:** `O` (Second Move)"
        )

        if st.session_state.game_over:
            st.success("Round completed! Click **'Start New Round'** to play again.")
            if st.button("✨ Start New Round", use_container_width=True, type="primary"):
                reset_current_game()
                st.rerun()
        else:
            st.caption("Tip: Against **Unbeatable Minimax**, the best possible human result is a Draw!")

# ---------------------------------------------------------------------------
# Tab 2: Minimax Decision Inspector
# ---------------------------------------------------------------------------
with tab_explain:
    st.markdown("### 🔍 Real-Time Minimax Tree Search Inspector")
    st.write(
        "This tool visualizes the computer's exact thought process during its latest move. "
        "Tic-Tac-Toe is a **zero-sum perfect-information game**, meaning the entire future "
        "can be deterministically searched to find the optimal move."
    )

    last_meta = st.session_state.last_ai_meta

    if last_meta.get("move") is not None:
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Last AI Selected Cell", f"Cell #{last_meta['move']}")
            st.metric("Search Tree Nodes Evaluated", f"{last_meta['nodes']} states")

        with col_m2:
            st.markdown("#### Candidate Move Evaluations:")
            scores = last_meta.get("scores", {})
            if scores:
                for cell_idx, score in sorted(scores.items()):
                    verdict = "👑 Best Move" if cell_idx == last_meta["move"] else "Suboptimal"
                    st.markdown(f"- **Cell {cell_idx}:** Score = `{score:+d}` ({verdict})")
            else:
                st.write("Direct heuristic applied.")

        st.markdown("---")
        st.markdown("#### 🧠 What Do the Minimax Scores Mean?")
        st.markdown(
            """
            - **Positive Score ($> 0$):** Guarantees a forced **AI Win**. Higher scores (e.g. $+9$) mean faster wins.
            - **Zero Score ($= 0$):** Guarantees a forced **Draw** against perfect defense.
            - **Negative Score ($< 0$):** Concedes a **Human Win** if Human plays perfectly.
            
            Because the AI is the **Maximizer**, it selects $\\arg\\max_{m} \\text{Score}(m)$.
            """
        )
    else:
        st.info("Make your first move on the board to inspect the AI's search tree evaluations!")

# ---------------------------------------------------------------------------
# Tab 3: Algorithm Theory & Viva Guide
# ---------------------------------------------------------------------------
with tab_viva:
    st.markdown("### 🎓 Deep-Dive: Adversarial Search & Minimax Algorithm")

    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.markdown("#### 📌 Key Algorithm Characteristics")
        st.markdown(
            """
            1. **Not Machine Learning:** Minimax is **not** a neural network, regression model, or trained classifier. 
               It is an **adversarial game-tree search algorithm** derived from game theory.
            2. **Zero-Sum Game:** Any advantage gained by one player is an equal disadvantage to the other:
               $$\\text{Utility}(X) + \\text{Utility}(O) = 0$$
            3. **Perfect Information:** Both players can observe the exact, complete state of the board at all times.
            4. **Exhaustive Guarantee:** Because a 3×3 board has at most $9! = 362,880$ total move permutations 
               (and only ~255,168 valid legal game states), Minimax can search to the terminal leaves in milliseconds.
            """
        )

        st.markdown("#### ⚡ Alpha-Beta Pruning Optimization")
        st.markdown(
            """
            Alpha-Beta pruning skips branches in the game tree that are mathematically guaranteed 
            not to influence the final decision:
            - **$\\alpha$ (Alpha):** The minimum score the maximizing player is assured of.
            - **$\\beta$ (Beta):** The maximum score the minimizing player is assured of.
            - **Pruning condition:** Whenever $\\beta \\le \\alpha$, the remaining child nodes are pruned, 
              reducing search complexity while yielding identical optimal moves.
            """
        )

    with col_t2:
        st.markdown("#### 🔬 Viva Voce Quick Revision")

        with st.expander("Q1: Why does Minimax penalize deeper terminal states with depth?", expanded=True):
            st.write(
                "By returning `+10 - depth` for AI wins and `-10 + depth` for Human wins, the algorithm "
                "prioritizes winning in fewer moves (fastest path to victory) and delaying losses as long as possible."
            )

        with st.expander("Q2: Can the Minimax AI ever lose when playing optimally?"):
            st.write(
                "**No.** Tic-Tac-Toe is a mathematically solved game. With optimal play from both sides, "
                "the theoretical game-theoretic value is always a **Draw**. If the human makes a single suboptimal move, "
                "the AI will punish it and win. It cannot lose."
            )

        with st.expander("Q3: What is the time complexity of vanilla Minimax for Tic-Tac-Toe?"):
            st.write(
                "The branching factor $b$ starts at 9 and decreases by 1 each ply ($d \\le 9$). "
                "The worst-case theoretical complexity is $O(b^d) = O(9!)$ states. With alpha-beta pruning, "
                "the effective branching factor is dramatically reduced."
            )

        with st.expander("Q4: Why does the AI prefer the center cell on turn 1?"):
            st.write(
                "The center cell (cell 4) participates in 4 out of the 8 total winning lines (1 horizontal, "
                "1 vertical, and 2 diagonals), making it the most strategically dominant tile on the board."
            )
