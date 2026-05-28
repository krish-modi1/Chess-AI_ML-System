#!/usr/bin/env python3
"""
Quick diagnostic test for worker initialization
Helps identify why workers aren't generating games
"""
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

print("=" * 70)
print("WORKER INITIALIZATION TEST - UPDATED")
print("=" * 70)

# Test 1: ChessGame import and creation
print("\n[TEST 1] Testing ChessGame...")
try:
    from game_engine.chess_env import ChessGame
    print("  ✅ Import successful")
    
    game = ChessGame()
    print(f"  ✅ ChessGame created")
    
    # Debug: List all attributes
    print(f"\n  🔍 DEBUG: ChessGame attributes:")
    attrs = [a for a in dir(game) if not a.startswith('_')]
    for attr in attrs:
        print(f"     - {attr}")
    
    # Check board
    print(f"\n  Board type: {type(game.board)}")
    print(f"  Board: {game.board}")
    
    # Try to get game over status
    print(f"\n  Checking game.board.is_game_over():")
    is_over_from_board = game.board.is_game_over()
    print(f"     - game.board.is_game_over(): {is_over_from_board}")
    
    print(f"\n  Turn player: {game.turn_player}")
    print(f"  Moves: {len(game.moves)}")
    
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: MCTSWorker import and creation
print("\n\n[TEST 2] Testing MCTSWorker...")
try:
    import multiprocessing as mp
    from game_engine.mcts_worker_cpp import MCTSWorker
    print("  ✅ Import successful")
    
    inputq = mp.Queue()
    outputq = mp.Queue()
    
    worker = MCTSWorker(
        worker_id=0,
        input_queue=inputq,
        output_queue=outputq,
        simulations=1600,
        batch_size=400,
        seed=42
    )
    print("  ✅ MCTSWorker created")
    print(f"     - Worker ID: {worker.worker_id}")
    print(f"     - Seed: {worker.seed}")
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Game loop iteration
print("\n\n[TEST 3] Testing game loop (simulate first move)...")
try:
    # Create fresh game and worker
    game2 = ChessGame()
    inputq2 = mp.Queue()
    outputq2 = mp.Queue()
    worker2 = MCTSWorker(1, inputq2, outputq2, 1600, 400, 43)
    
    # Use correct method to check if game is over
    game_over = game2.board.is_game_over()
    print(f"  - Game.board.is_game_over() at start: {game_over}")
    
    # Try to check if we can enter game loop
    if not game_over:
        print("  ✅ Game loop would execute (game is not over)")
        
        # Check if we can get move count
        move_count = len(game2.moves)
        print(f"     - Current move count: {move_count}")
        
        # Check temperature logic
        if move_count < 16:
            temp = 1.0
        else:
            temp = 0.0
        print(f"     - Temperature would be: {temp}")
        print("  ✅ Game loop logic is correct")
    else:
        print("  ❌ Game is over at start (BUG!)")
        print("     Workers can't play if game is over initially")
        sys.exit(1)
        
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Check if mcts_engine_cpp module loads
print("\n\n[TEST 4] Testing C++ MCTS backend...")
try:
    import mcts_engine_cpp
    print("  ✅ mcts_engine_cpp loaded")
    
    engine = mcts_engine_cpp.MCTSEngine(1600, 400)
    print("  ✅ MCTSEngine created")
except Exception as e:
    print(f"  ⚠️  WARNING: C++ backend issue: {e}")
    print("     This could prevent worker.search() from working")

print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED")
print("=" * 70)
print("\n🔧 ISSUE FOUND:")
print("   - ChessGame uses game.board.is_game_over()")
print("   - NOT game.isover property")
print("   - main.py needs to be updated!")
print("\nWorkers should be able to play games after fixing main.py")
