#!/usr/bin/env python3
"""
Debug script to see ACTUAL BayesElo output
This will show you exactly what BayesElo is printing
"""

import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), 'game_engine'))

from bayeselo_runner import BayesEloRunner

print("="*70)
print("🔍 DEBUGGING BAYESELO OUTPUT")
print("="*70)
print()

try:
    runner = BayesEloRunner()
    pgn_path = "game_engine/evaluation/pgn/realistic_40games.pgn"
    
    if not os.path.exists(pgn_path):
        print(f"❌ PGN not found: {pgn_path}")
        print("Run realistic_test.py first to generate it")
        sys.exit(1)
    
    results = runner.run(pgn_path)
    
    if results and 'raw_output' in results:
        print("RAW BAYESELO OUTPUT (first 2000 chars):")
        print("-"*70)
        print(results['raw_output'][:2000])
        print("-"*70)
        print()
        print("="*70)
        print("PARSED RESULTS:")
        print("="*70)
        for key, value in results.items():
            if key != 'raw_output':
                print(f"{key:20} = {value}")
    else:
        print("❌ No results or raw_output")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
