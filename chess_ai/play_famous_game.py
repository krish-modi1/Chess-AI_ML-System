#!/usr/bin/env python3
"""
Plays through the Immortal Game (Anderssen vs Kieseritzky, London 1851) move by move.
At each position, the AI picks its preferred move, which is compared to the historical one.
The game always follows the historical moves so we stay in the original game's positions.

Usage:  cd chess_ai && python play_famous_game.py [--sims N]
"""
import sys, os, queue, time, argparse

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "game_engine"))

import torch
import chess

from game_engine.chess_env import ChessGame
from game_engine.mcts_worker_cpp import MCTSWorker
from game_engine.cnn import ChessCNN

# ---------------------------------------------------------------------------
# The Immortal Game — Anderssen vs Kieseritzky, London 1851 (UCI notation)
# 40 half-moves = 20 full moves
# ---------------------------------------------------------------------------
GAME_NAME = "The Immortal Game  —  Anderssen (White) vs Kieseritzky (Black), London 1851"
MOVES = [
    "e2e4", "e7e5",   #  1
    "f2f4", "e5f4",   #  2  King's Gambit accepted
    "f1c4", "d8h4",   #  3  Bishop to c4 / Queen check
    "e1f1", "b7b5",   #  4  King sidesteps / b5 pawn push
    "c4b5", "g8f6",   #  5  Bishop captures b5 / Knight out
    "g1f3", "h4h6",   #  6  Knight f3 / Queen retreats
    "d2d3", "f6h5",   #  7  d3 support / Knight hops to h5
    "f3h4", "h6g5",   #  8  Knight to h4 / Queen to g5
    "h4f5", "c7c6",   #  9  Knight to f5 / c6 attack
    "g2g4", "h5f6",   # 10  g4 pawn attack / Knight back
    "h1g1", "c6b5",   # 11  Rook to g1 / pawn captures bishop
    "h2h4", "g5g6",   # 12  h4 advance / Queen to g6
    "h4h5", "g6g5",   # 13  h5 push / Queen holds
    "d1f3", "f6g8",   # 14  Queen to f3 / Knight retreats
    "c1f4", "g5f6",   # 15  Bishop to f4 / Queen to f6
    "b1c3", "f8c5",   # 16  Knight out / Bishop to c5
    "c3d5", "f6b2",   # 17  Knight to d5 / Queen grabs b2
    "f4d6", "c5g1",   # 18  Bishop to d6 / Bishop takes rook!
    "e4e5", "b2a1",   # 19  Pawn to e5 / Queen takes rook!
    "f1e2", "b8a6",   # 20  King to e2 / Knight to a6
]


def board_to_string(board: chess.Board) -> str:
    """Return a compact 8-line ASCII board with rank/file labels."""
    rows = []
    for rank in range(7, -1, -1):
        row = f"  {rank + 1} "
        for file in range(8):
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            if piece is None:
                row += ". "
            else:
                row += piece.symbol() + " "
        rows.append(row)
    rows.append("    a b c d e f g h")
    return "\n".join(rows)


def move_to_san(board: chess.Board, uci: str) -> str:
    """Convert a UCI string to SAN notation for display."""
    try:
        move = chess.Move.from_uci(uci)
        return board.san(move)
    except Exception:
        return uci


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sims", type=int, default=200,
                        help="MCTS simulations per move (default 200)")
    parser.add_argument("--moves", type=int, default=40,
                        help="Number of half-moves to play through (default 40)")
    args = parser.parse_args()

    n_moves = min(args.moves, len(MOVES))
    moves = MOVES[:n_moves]

    # ── Load model ────────────────────────────────────────────────────────
    model_path = os.path.join(os.path.dirname(__file__), "game_engine/model/best_model.pth")
    if not os.path.exists(model_path):
        print(f"ERROR: model not found at {model_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nLoading model from {model_path} ... (device: {device})")
    model = ChessCNN()
    state = torch.load(model_path, map_location=device)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print("Model loaded.")

    # ── Create MCTS worker (search_direct doesn't use the queues) ────────
    worker = MCTSWorker(
        worker_id=0,
        input_queue=queue.Queue(),
        output_queue=queue.Queue(),
        simulations=args.sims,
        batch_size=8,
        seed=42,
    )

    # ── Play through the game ─────────────────────────────────────────────
    print(f"\n{'=' * 66}")
    print(f"  {GAME_NAME}")
    print(f"  {n_moves // 2} full moves  |  {args.sims} MCTS simulations per move")
    print(f"{'=' * 66}")

    game = ChessGame()
    board_history: list[chess.Board] = []   # newest first

    matches = 0
    total = 0

    for i, historical_uci in enumerate(moves):
        side = "White" if i % 2 == 0 else "Black"
        move_num = i // 2 + 1

        # Build history for to_tensor (up to 7 most recent boards, newest first)
        history = board_history[:7]

        # Ask the engine for its preferred move
        t0 = time.time()
        engine_uci, _ = worker.search_direct(
            game, model, temperature=0.1, history=history
        )
        elapsed = time.time() - t0

        # Convert moves to SAN for readability
        historical_san = move_to_san(game.board, historical_uci)
        engine_san = move_to_san(game.board, engine_uci) if engine_uci else "???"

        match = engine_uci == historical_uci
        if match:
            matches += 1
        total += 1
        marker = "✓" if match else " "

        print(f"\n  Move {move_num:>2} ({side})")
        print(board_to_string(game.board))
        print(f"\n  Historical : {historical_san:<10}  ({historical_uci})")
        print(f"  Engine     : {engine_san:<10}  ({engine_uci})  {marker}  [{elapsed:.1f}s]")

        # Advance the game with the historical move (stay in the famous game)
        board_history.insert(0, game.board.copy())
        game.push(historical_uci)
        worker.advance_root(historical_uci)

    # ── Summary ───────────────────────────────────────────────────────────
    pct = 100 * matches / total if total else 0
    print(f"\n{'=' * 66}")
    print(f"  Result: engine matched {matches}/{total} historical moves  ({pct:.0f}%)")
    print(f"  (Matching a ~1200 ELO model against a 2600+ masterpiece is hard —")
    print(f"   any agreement shows the move was strongly forced or principled.)")
    print(f"{'=' * 66}\n")


if __name__ == "__main__":
    main()
