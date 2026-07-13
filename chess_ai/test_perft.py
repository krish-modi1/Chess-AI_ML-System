"""Perft test — validates C++ move generation against known node counts.
Run from the repo root: python3 chess_ai/test_perft.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "game_engine"))
import mcts_engine_cpp

def perft(board, depth):
    if depth == 0:
        return 1
    count = 0
    for move_uci in board.legal_moves():
        child = board.copy()
        child.push(move_uci)
        count += perft(child, depth - 1)
    return count

EXPECTED = {1: 20, 2: 400, 3: 8_902, 4: 197_281, 5: 4_865_609}

board = mcts_engine_cpp.ChessBoard()
all_pass = True
for depth in range(1, 6):
    got = perft(board, depth)
    exp = EXPECTED[depth]
    status = "PASS" if got == exp else "FAIL"
    print(f"  Depth {depth}: {got:>10,}  (expected {exp:>10,})  {status}")
    if got != exp:
        all_pass = False

if all_pass:
    print("\nAll perft tests passed!")
    sys.exit(0)
else:
    print("\nPerft FAILED — move generation is wrong.")
    sys.exit(1)
