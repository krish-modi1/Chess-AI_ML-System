#!/usr/bin/env python3
"""
Self-play: our AI plays both sides using the W2 C++ chess board.
Two independent MCTS workers (one per side) each build their own tree.

Usage:  cd chess_ai && python play_self.py [options]

Options:
  --sims N     MCTS simulations per move (default 200)
  --moves N    Max half-moves before draw (default 80)
  --temp F     Move temperature — higher = more exploration (default 0.5)
  --seed-w N   RNG seed for White (default 0)
  --seed-b N   RNG seed for Black (default 1)
"""
import sys, os, queue, time, argparse

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "game_engine"))

import torch
import chess

from game_engine.chess_env import ChessGame
from game_engine.mcts_worker_cpp import MCTSWorker
from game_engine.cnn import ChessCNN


def board_to_string(board: chess.Board) -> str:
    rows = []
    for rank in range(7, -1, -1):
        row = f"  {rank + 1} "
        for file in range(8):
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            row += (piece.symbol() if piece else ".") + " "
        rows.append(row)
    rows.append("    a b c d e f g h")
    return "\n".join(rows)


def uci_to_san(board: chess.Board, uci: str) -> str:
    try:
        return board.san(chess.Move.from_uci(uci))
    except Exception:
        return uci


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sims", type=int, default=200)
    parser.add_argument("--moves", type=int, default=80)
    parser.add_argument("--temp", type=float, default=0.5)
    parser.add_argument("--seed-w", type=int, default=0)
    parser.add_argument("--seed-b", type=int, default=1)
    args = parser.parse_args()

    # ── Load model (shared by both sides) ────────────────────────────────
    model_path = os.path.join(os.path.dirname(__file__), "game_engine/model/best_model.pth")
    if not os.path.exists(model_path):
        print(f"ERROR: model not found at {model_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model ... (device: {device})")
    model = ChessCNN()
    state = torch.load(model_path, map_location=device)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # ── Two independent MCTS workers, one per side ───────────────────────
    def make_worker(seed):
        return MCTSWorker(
            worker_id=seed,
            input_queue=queue.Queue(),
            output_queue=queue.Queue(),
            simulations=args.sims,
            batch_size=8,
            seed=seed,
        )

    workers = {
        chess.WHITE: make_worker(args.seed_w),
        chess.BLACK: make_worker(args.seed_b),
    }

    print(f"\n{'=' * 60}")
    print(f"  AI (White, seed={args.seed_w})  vs  AI (Black, seed={args.seed_b})")
    print(f"  {args.sims} sims/move  |  temperature {args.temp}  |  max {args.moves} half-moves")
    print(f"{'=' * 60}")

    game = ChessGame()
    board_history: list[chess.Board] = []
    move_count = 0
    pgn_moves = []

    while move_count < args.moves:
        board = game.board
        if board.is_game_over():
            break

        side = board.turn
        side_label = "White" if side == chess.WHITE else "Black"
        worker = workers[side]
        move_num = board.fullmove_number

        print(f"\n{'─' * 60}")
        print(f"  Move {move_num} — {side_label}")
        print(board_to_string(board))

        t0 = time.time()
        uci, _ = worker.search_direct(
            game, model,
            temperature=args.temp,
            history=board_history[:7],
        )
        elapsed = time.time() - t0

        san = uci_to_san(board, uci)
        print(f"\n  {side_label} plays: {san} ({uci})  [{elapsed:.1f}s]")
        pgn_moves.append(san)

        # Both workers advance their cached root with the played move
        board_history.insert(0, game.board.copy())
        game.push(uci)
        for w in workers.values():
            w.advance_root(uci)

        move_count += 1

    # ── Final board + result ──────────────────────────────────────────────
    board = game.board
    print(f"\n{'=' * 60}")
    print(board_to_string(board))

    if board.is_game_over():
        result = board.result()
        outcome = board.outcome()
        if outcome and outcome.winner is not None:
            winner = "White" if outcome.winner == chess.WHITE else "Black"
            verdict = f"{winner} wins  ({result})"
        else:
            verdict = f"Draw  ({result})"
    else:
        verdict = f"Move limit reached ({args.moves} half-moves) — adjudicated draw"
        result = "1/2-1/2"

    # Print PGN-style move list
    pgn = ""
    for i, move in enumerate(pgn_moves):
        if i % 2 == 0:
            pgn += f"{i // 2 + 1}. {move} "
        else:
            pgn += f"{move}  "
    print(f"\n  Result: {verdict}")
    print(f"\n  PGN:\n  {pgn.strip()}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
