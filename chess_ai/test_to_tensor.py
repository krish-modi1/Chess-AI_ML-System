"""Cross-validate C++ to_tensor output against Python chess_env.py.
NOTE: Requires ChessBoard to be exposed to Python (Task 7). Run after Task 7.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import mcts_engine_cpp
from chess_env import ChessGame
import chess

def check(label, cpp_t, py_t):
    if not np.allclose(cpp_t, py_t, atol=1e-5):
        diff = np.abs(cpp_t - py_t)
        bad = list(zip(*np.where(diff > 1e-5)))
        print(f"FAIL {label}: {len(bad)} mismatched cells")
        for idx in bad[:5]:
            p, r, f = idx
            print(f"  plane {p} rank {r} file {f}: cpp={cpp_t[p,r,f]:.4f} py={py_t[p,r,f]:.4f}")
        return False
    print(f"PASS {label}")
    return True

ok = True

# Test 1: starting position, no history
cpp_b = mcts_engine_cpp.ChessBoard()
py_g  = ChessGame()
ok &= check("starting position", cpp_b.to_tensor(), py_g.to_tensor())

# Test 2: after 3 moves, no history
moves = ["e2e4", "e7e5", "d2d4"]
cpp_b2, py_g2 = mcts_engine_cpp.ChessBoard(), ChessGame()
for m in moves:
    cpp_b2.push(m); py_g2.push(m)
ok &= check("after 3 moves", cpp_b2.to_tensor(), py_g2.to_tensor())

# Test 3: with 1 history frame
cpp_b3, py_g3 = mcts_engine_cpp.ChessBoard(), ChessGame()
for m in moves:
    cpp_b3.push(m); py_g3.push(m)
cpp_hist = [mcts_engine_cpp.ChessBoard()]
py_hist  = [chess.Board()]
ok &= check("with 1 history frame", cpp_b3.to_tensor(cpp_hist), py_g3.to_tensor(py_hist))

# Test 4: after d5 (en passant square active)
cpp_b4, py_g4 = mcts_engine_cpp.ChessBoard(), ChessGame()
for m in ["e2e4", "d7d5"]:
    cpp_b4.push(m); py_g4.push(m)
ok &= check("en passant square", cpp_b4.to_tensor(), py_g4.to_tensor())

if ok:
    print("\nAll to_tensor tests passed!")
    sys.exit(0)
else:
    print("\nSome to_tensor tests FAILED")
    sys.exit(1)
