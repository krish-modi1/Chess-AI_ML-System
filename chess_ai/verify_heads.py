"""
verify_heads.py — Sanity-check the policy and value heads of a trained ChessCNN.

Runs three checks against crafted positions (no Stockfish, no training needed):
  1. VALUE head  — WDL prediction (side-to-move perspective) on positions with a known
                   winner; confirms sign/direction and decisiveness.
  2. POLICY head — softmax mass on legal moves (vs uniform baseline) + top moves on the
                   start position; confirms the policy isn't degenerate/uniform.
  3. SELF-PLAY   — plays a short game via MCTS search_direct; confirms moves are legal and
                   prints the value trajectory.

Usage:
    python chess_ai/verify_heads.py --model chess_ai/game_engine/model/best_model.pth --sims 200 --moves 20
"""

import argparse, os, sys
import numpy as np
import chess
import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'game_engine'))

from game_engine.cnn import ChessCNN
from game_engine.chess_env import ChessGame
from game_engine.mcts_worker_cpp import MCTSWorker


def move_to_index(uci):
    """Python port of move_to_index in src/mcts_engine.h (AlphaZero 73x64 = 4672)."""
    sf, sr = ord(uci[0]) - 97, ord(uci[1]) - 49
    dfile, drank = ord(uci[2]) - 97, ord(uci[3]) - 49
    src = sf + sr * 8
    df, dr = dfile - sf, drank - sr
    if len(uci) == 5 and uci[4] != 'q':
        piece = {'n': 0, 'b': 1, 'r': 2}[uci[4]]
        d = 0 if df < 0 else (1 if df == 0 else 2)
        return src * 73 + 64 + d * 3 + piece
    adf, adr = abs(df), abs(dr)
    if (adf == 1 and adr == 2) or (adf == 2 and adr == 1):
        kdir = {(1, 2): 0, (2, 1): 1, (2, -1): 2, (1, -2): 3,
                (-1, -2): 4, (-2, -1): 5, (-2, 1): 6, (-1, 2): 7}[(df, dr)]
        return src * 73 + 56 + kdir
    if df == 0:
        d, dist = (0 if dr > 0 else 4), adr - 1
    elif dr == 0:
        d, dist = (2 if df > 0 else 6), adf - 1
    else:
        dist = adf - 1
        d = 1 if (df > 0 and dr > 0) else 7 if (df < 0 and dr > 0) else 3 if (df > 0 and dr < 0) else 5
    return src * 73 + d * 7 + dist


def load_model(path, device):
    model = ChessCNN()
    ckpt = torch.load(path, map_location=device, weights_only=True)
    state = ckpt.get('model_state_dict', ckpt)
    if any(k.startswith('_orig_mod.') for k in state):
        state = {k.removeprefix('_orig_mod.'): v for k, v in state.items()}
    model.load_state_dict(state)
    return model.eval().to(device)


@torch.no_grad()
def raw_eval(model, fen, device):
    """Return (policy_probs[4672], wdl_probs[3]) for the side to move."""
    game = ChessGame(fen=fen)
    t = torch.from_numpy(game.to_tensor()).unsqueeze(0).to(device)
    pol_logits, val_logits = model(t)
    pol = F.softmax(pol_logits.float(), dim=1)[0].cpu().numpy()
    wdl = F.softmax(val_logits.float(), dim=1)[0].cpu().numpy()
    return pol, wdl, game


def check_value(model, device):
    print("\n" + "=" * 70)
    print("1. VALUE HEAD  —  WDL = [P(win), P(draw), P(loss)] from side-to-move POV")
    print("=" * 70)
    cases = [
        ("Start position (balanced)",       "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "≈ balanced"),
        ("White K+Q vs lone K, White moves", "4k3/8/8/8/8/8/8/3QK3 w - - 0 1",                           "win >> loss"),
        ("Same, Black to move (lost)",       "4k3/8/8/8/8/8/8/3QK3 b - - 0 1",                           "loss >> win"),
        ("Black K+Q vs lone K, Black moves", "3qk3/8/8/8/8/8/8/4K3 b - - 0 1",                           "win >> loss"),
    ]
    for name, fen, expect in cases:
        _, wdl, _ = raw_eval(model, fen, device)
        verdict = "✓" if (("win >> loss" in expect and wdl[0] > wdl[2]) or
                          ("loss >> win" in expect and wdl[2] > wdl[0]) or
                          ("balanced" in expect and abs(wdl[0] - wdl[2]) < 0.4)) else "✗"
        print(f"  {verdict} {name:36s} W={wdl[0]:.2f} D={wdl[1]:.2f} L={wdl[2]:.2f}  (expect {expect})")


def check_policy(model, device):
    print("\n" + "=" * 70)
    print("2. POLICY HEAD  —  legal-move mass & top moves")
    print("=" * 70)
    fens = [
        ("Start position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("Italian (after 1.e4 e5 2.Nf3 Nc6 3.Bc4)", "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1"),
    ]
    for name, fen in fens:
        pol, _, game = raw_eval(model, fen, device)
        legal = game.legal_moves()
        legal_idx = [move_to_index(m) for m in legal]
        legal_mass = float(pol[legal_idx].sum())
        uniform = len(legal) / 4672.0
        # top-5 legal moves by policy prob
        ranked = sorted(zip(legal, pol[legal_idx]), key=lambda x: -x[1])[:5]
        board = chess.Board(fen)
        tops = ", ".join(f"{board.san(chess.Move.from_uci(m))}={p:.2f}" for m, p in ranked)
        verdict = "✓" if legal_mass > 0.5 else "✗"
        print(f"  {verdict} {name}")
        print(f"      legal-move mass = {legal_mass:.3f}  (uniform baseline = {uniform:.3f}, {len(legal)} legal)")
        print(f"      top: {tops}")


def play_game(model, device, sims, n_moves):
    print("\n" + "=" * 70)
    print(f"3. SELF-PLAY  —  {n_moves} half-moves via MCTS search_direct ({sims} sims)")
    print("=" * 70)
    worker = MCTSWorker(0, None, None, simulations=sims, batch_size=8)
    game = ChessGame()
    board = chess.Board()
    history = []
    print(f"  {'#':>3}  {'move':>7}  {'value(stm)':>10}")
    for ply in range(n_moves):
        if game.is_over:
            print(f"  game over: {game.result}")
            break
        mv, _ = worker.search_direct(game, model, temperature=0.0,
                                     use_dirichlet=False, history=list(history))
        # raw value of the position the model just moved from
        _, wdl, _ = raw_eval(model, board.fen(), device)
        v = wdl[0] - wdl[2]
        san = board.san(chess.Move.from_uci(mv))
        legal_ok = chess.Move.from_uci(mv) in board.legal_moves
        print(f"  {ply+1:>3}  {san:>7}  {v:+.3f}   {'' if legal_ok else '❌ ILLEGAL'}")
        worker.advance_root(mv)
        history.insert(0, board.copy()); history[:] = history[:7]
        board.push(chess.Move.from_uci(mv))
        game.push(mv)
    worker.reset_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='chess_ai/game_engine/model/best_model.pth')
    p.add_argument('--sims', type=int, default=200)
    p.add_argument('--moves', type=int, default=20)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Model: {args.model}")
    model = load_model(args.model, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded ChessCNN: {n_params/1e6:.1f}M params")

    check_value(model, device)
    check_policy(model, device)
    play_game(model, device, args.sims, args.moves)
    print("\nDone.")


if __name__ == '__main__':
    main()
