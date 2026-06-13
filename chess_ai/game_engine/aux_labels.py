#!/usr/bin/env python3
"""
Derive auxiliary training targets for the value-trunk regularizer heads (see
local/plans/auxiliary-targets.md). One definition, used by the self-play save path (main.py), the
one-time migration (migrate_aux.py), and the trainer's load (with on-the-fly fallback).

All targets are FORWARD-LOOKING (not already in the input planes), side-to-move POV:
  material : FINAL material margin at game end, clipped to [-1, 1]   (graded "who ends up ahead")
  plies    : plies remaining until game end, /100 clipped to [0, 1]  ("how close to decided")
  reply    : opponent's reply = argmax(policy of the NEXT position); last position = -1 (masked)

Plane layout (chess_board.cpp::fill_planes): current position uses planes 0-13 —
  0-5  = white pieces by type (PAWN,KNIGHT,BISHOP,ROOK,QUEEN,KING)
  6-11 = black pieces by type
  plane 117 (whole board = 1) = black to move.
"""
import numpy as np

PIECE_VALUES = np.array([1.0, 3.0, 3.0, 5.0, 9.0, 0.0], dtype=np.float32)  # P N B R Q K
MATERIAL_SCALE = 10.0   # clip(margin / scale, -1, 1) — ~a queen+pawn saturates the signal
PLIES_SCALE = 100.0


def derive_aux_labels(states, policies, values=None):
    """states (n,120,8,8) f32, policies (n,4672) f32 → (material f32[n], plies f32[n], reply i64[n])."""
    states = np.asarray(states)
    n = states.shape[0]

    # current material per position (white POV) from piece planes 0-11
    white_counts = states[:, 0:6].sum(axis=(2, 3))     # (n, 6)
    black_counts = states[:, 6:12].sum(axis=(2, 3))    # (n, 6)
    white_mat = white_counts @ PIECE_VALUES            # (n,)
    black_mat = black_counts @ PIECE_VALUES            # (n,)

    # FINAL margin (white POV) = material at the last stored (≈terminal) position
    final_margin_white = float(white_mat[-1] - black_mat[-1])

    stm_black = states[:, 117].reshape(n, -1).mean(axis=1) > 0.5    # (n,) bool
    material = np.where(stm_black, -final_margin_white, final_margin_white)
    material = np.clip(material / MATERIAL_SCALE, -1.0, 1.0).astype(np.float32)

    plies = np.arange(n - 1, -1, -1, dtype=np.float32)             # n-1-i remaining
    plies = np.clip(plies / PLIES_SCALE, 0.0, 1.0).astype(np.float32)

    moves = np.nan_to_num(np.asarray(policies), nan=0.0).argmax(axis=1).astype(np.int64)  # (n,)
    reply = np.empty(n, dtype=np.int64)
    reply[:-1] = moves[1:]      # opponent's actual reply = move chosen at the next position
    reply[-1] = -1              # last position has no reply → masked in the loss
    return material, plies, reply
