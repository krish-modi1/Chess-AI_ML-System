#!/usr/bin/env python3
"""
Plays our AI (using the W2 C++ chess board) against Stockfish.

Usage:
  cd chess_ai && python play_vs_stockfish.py [options]

Options:
  --sims N        MCTS simulations per AI move (default 200)
  --elo N         Stockfish ELO limit (default 1200)
  --ai-color W|B  Which color our AI plays (default W)
  --moves N       Max half-moves before declaring draw (default 80)
  --stockfish P   Path to stockfish binary (auto-detected if omitted)
"""
import sys, os, queue, time, argparse, shutil

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "game_engine"))

import torch
import chess
import chess.engine

from game_engine.chess_env import ChessGame
from game_engine.mcts_worker_cpp import MCTSWorker
from game_engine.cnn import ChessCNN


def find_stockfish() -> str:
    # Common locations
    candidates = [
        shutil.which("stockfish"),
        "/usr/games/stockfish",
        "/usr/bin/stockfish",
        "/usr/local/bin/stockfish",
        os.path.join(os.path.dirname(__file__), "stockfish"),
    ]
    for p in candidates:
        if p and os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


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
    parser.add_argument("--elo", type=int, default=1320)
    parser.add_argument("--ai-color", choices=["W", "B"], default="W")
    parser.add_argument("--moves", type=int, default=80)
    parser.add_argument("--stockfish", type=str, default=None)
    args = parser.parse_args()

    # ── Locate Stockfish ─────────────────────────────────────────────────
    sf_path = args.stockfish or find_stockfish()
    if not sf_path:
        print("ERROR: Stockfish not found. Install it (apt install stockfish) or pass --stockfish PATH")
        sys.exit(1)
    print(f"Stockfish: {sf_path}")

    # ── Load our model ───────────────────────────────────────────────────
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

    # ── MCTS worker (search_direct doesn't use the queues) ───────────────
    worker = MCTSWorker(
        worker_id=0,
        input_queue=queue.Queue(),
        output_queue=queue.Queue(),
        simulations=args.sims,
        batch_size=8,
        seed=42,
    )

    # ── Stockfish engine ─────────────────────────────────────────────────
    sf_engine = chess.engine.SimpleEngine.popen_uci(sf_path)
    # Clamp ELO to the engine's supported range
    elo_option = sf_engine.options.get("UCI_Elo")
    elo = args.elo
    if elo_option:
        if elo_option.min is not None:
            elo = max(elo, elo_option.min)
        if elo_option.max is not None:
            elo = min(elo, elo_option.max)
    if elo != args.elo:
        print(f"Note: ELO clamped to {elo} (engine range {elo_option.min}–{elo_option.max})")
    sf_engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
    args.elo = elo  # update for display

    ai_is_white = (args.ai_color == "W")
    ai_label = f"Our AI ({args.sims} sims)"
    sf_label = f"Stockfish ELO {args.elo}"

    print(f"\n{'=' * 60}")
    print(f"  {'White':30s}  vs  {'Black':20s}")
    print(f"  {(ai_label if ai_is_white else sf_label):30s}  vs  {(sf_label if ai_is_white else ai_label):20s}")
    print(f"{'=' * 60}")

    game = ChessGame()
    board_history: list[chess.Board] = []
    move_count = 0

    try:
        while move_count < args.moves:
            board = game.board
            if board.is_game_over():
                break

            is_white_turn = (board.turn == chess.WHITE)
            ai_turn = (is_white_turn == ai_is_white)
            move_num = board.fullmove_number
            side = "White" if is_white_turn else "Black"

            print(f"\n{'─' * 60}")
            print(f"  Move {move_num} — {side}  ({'AI' if ai_turn else 'Stockfish'})")
            print(board_to_string(board))

            if ai_turn:
                # Our AI move via W2 C++ board
                t0 = time.time()
                uci, _ = worker.search_direct(
                    game, model,
                    temperature=0.1,
                    history=board_history[:7],
                )
                elapsed = time.time() - t0
                san = uci_to_san(board, uci)
                print(f"\n  AI plays: {san} ({uci})  [{elapsed:.1f}s]")
            else:
                # Stockfish move
                t0 = time.time()
                result = sf_engine.play(
                    board,
                    chess.engine.Limit(time=0.5),
                )
                elapsed = time.time() - t0
                uci = result.move.uci()
                san = uci_to_san(board, uci)
                print(f"\n  Stockfish plays: {san} ({uci})  [{elapsed:.1f}s]")

            # Apply move
            board_history.insert(0, game.board.copy())
            game.push(uci)
            worker.advance_root(uci)
            move_count += 1

        # ── Game over ─────────────────────────────────────────────────────
        print(f"\n{'=' * 60}")
        board = game.board
        print(board_to_string(board))

        if board.is_game_over():
            result = board.result()
            outcome = board.outcome()
            if outcome and outcome.winner is not None:
                winner_is_white = outcome.winner
                if winner_is_white == ai_is_white:
                    verdict = f"Our AI wins!  ({result})"
                else:
                    verdict = f"Stockfish wins.  ({result})"
            else:
                verdict = f"Draw.  ({result})"
        else:
            verdict = f"Move limit reached ({args.moves} half-moves) — position not resolved."

        print(f"\n  {verdict}")
        print(f"{'=' * 60}\n")

    finally:
        sf_engine.quit()


if __name__ == "__main__":
    main()
