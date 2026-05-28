#!/usr/bin/env python3
"""
Chess Data Preprocessor - 60/40 Split & Smooth Progress
=======================================================

Project: EE542 - AlphaZero Hybrid
Target: Scalable (Default 100M)
Output: Uncompressed Binary Memmaps

CHANGELOG:
1. Skew: 60% Expert (>1600), 40% Novice (<1600).
2. UI: Progress bar updates every 10k positions.
3. Disk: Flushes to disk every 100k positions.
4. Data: 13th Plane (Turn Indicator) included.

"""

import json
import logging
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import psutil

try:
    import chess
    import chess.pgn
except ImportError:
    print("ERROR: chess library required. Install: pip install python-chess")
    sys.exit(1)

# ============================================================================
# Configuration
# ============================================================================

class ProductionConfig:
    """Production extraction configuration."""
    
    # Target Total (Overridden by CLI argument)
    TARGET_POSITIONS_TOTAL = 100_000_000
    
    # Distribution Settings (UPDATED: 60/40 Split)
    ELO_CUTOFF = 1600
    HIGH_ELO_SHARE = 0.60  # 60% positions from > 1600
    LOW_ELO_SHARE = 0.40   # 40% positions from < 1600
    
    # Stratified Elo Buckets
    ELO_BUCKETS = [
        # Low Elo (< 1600)
        (800, 1000), (1000, 1200), (1200, 1400), (1400, 1600),
        # High Elo (>= 1600)
        (1600, 1800), (1800, 2000), (2000, 2200), (2200, 3000),
    ]
    
    # Filters
    time_control_min = 300
    game_length_min = 20
    game_length_max = 500
    
    # I/O Settings
    ui_update_interval = 1_000_000 # Update progress bar every 1k
    flush_interval = 500_000    # Write to disk every 250k
    validate_moves = True
    
    # Paths
    OUTPUT_DIR = Path('/home/krish/EE542-Project/FINAL/chess_ai/data/training_data')
    LOG_DIR = Path('/home/krish/EE542-Project/FINAL/chess_ai/data/processing_logs')
    
    def get_bucket_name(self, elo_min: int, elo_max: int) -> str:
        names = {
            (800, 1000): "Beginner", (1000, 1200): "Novice",
            (1200, 1400): "Intermediate", (1400, 1600): "Advanced",
            (1600, 1800): "Strong", (1800, 2000): "Master",
            (2000, 2200): "SuperGM", (2200, 3000): "Engine",
        }
        return names.get((elo_min, elo_max), f"Unknown({elo_min}-{elo_max})")

    def get_bucket_targets(self) -> dict:
        """Calculates exact position count required for each bucket based on skew."""
        targets = {}
        
        # Split buckets into High/Low groups
        low_buckets = [b for b in self.ELO_BUCKETS if b[1] <= self.ELO_CUTOFF]
        high_buckets = [b for b in self.ELO_BUCKETS if b[0] >= self.ELO_CUTOFF]
        
        # Calculate total slots for each group
        total_low_slots = int(self.TARGET_POSITIONS_TOTAL * self.LOW_ELO_SHARE)
        total_high_slots = int(self.TARGET_POSITIONS_TOTAL * self.HIGH_ELO_SHARE)
        
        # Distribute evenly within groups
        per_bucket_low = total_low_slots // len(low_buckets)
        per_bucket_high = total_high_slots // len(high_buckets)
        
        for b in low_buckets:
            targets[self.get_bucket_name(*b)] = per_bucket_low
            
        for b in high_buckets:
            targets[self.get_bucket_name(*b)] = per_bucket_high
            
        return targets

# ============================================================================
# Core Logic
# ============================================================================

class BoardEncoder:
    """Encode board to 13x8x8 tensor."""
    PIECE_TO_CHANNEL = {
        (chess.PAWN, chess.WHITE): 0, (chess.KNIGHT, chess.WHITE): 1,
        (chess.BISHOP, chess.WHITE): 2, (chess.ROOK, chess.WHITE): 3,
        (chess.QUEEN, chess.WHITE): 4, (chess.KING, chess.WHITE): 5,
        (chess.PAWN, chess.BLACK): 6, (chess.KNIGHT, chess.BLACK): 7,
        (chess.BISHOP, chess.BLACK): 8, (chess.ROOK, chess.BLACK): 9,
        (chess.QUEEN, chess.BLACK): 10, (chess.KING, chess.BLACK): 11,
    }
    
    @staticmethod
    def board_to_tensor(board: chess.Board) -> np.ndarray:
        tensor = np.zeros((13, 8, 8), dtype=np.float16)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                rank = square // 8
                file = square % 8
                channel = BoardEncoder.PIECE_TO_CHANNEL[(piece.piece_type, piece.color)]
                tensor[channel, rank, file] = 1.0
        
        # 13th Plane: Turn Indicator (0 for White, 1 for Black)
        if board.turn == chess.BLACK:
            tensor[12, :, :] = 1.0
        return tensor

class MoveEncoder:
    @staticmethod
    def move_to_index(move: chess.Move) -> int:
        from_sq = move.from_square
        to_sq = move.to_square
        if move.promotion:
            return from_sq * 64 + to_sq + 4096
        return from_sq * 64 + to_sq

# ============================================================================
# Streaming Extractor
# ============================================================================

class ProductionDataExtractor:
    def __init__(self, config: ProductionConfig, logger):
        self.config = config
        self.logger = logger
        self.encoder = BoardEncoder()
        
        # Stats
        self.total_positions = 0
        self.start_time = None
        self.bucket_counts = {config.get_bucket_name(l, h): 0 for l, h in config.ELO_BUCKETS}
        self.bucket_targets = config.get_bucket_targets()
        
        # Print Plan
        self.logger.info("-" * 40)
        self.logger.info(f"DISTRIBUTION PLAN (Total: {config.TARGET_POSITIONS_TOTAL:,})")
        self.logger.info("-" * 40)
        for bucket, target in self.bucket_targets.items():
            self.logger.info(f"{bucket:<15}: {target:,} positions")
        self.logger.info("-" * 40)
        
        # Buffers
        self.buf_pos = []
        self.buf_mov = []
        self.buf_res = []

    def extract_dataset(self, input_files: List[Path]):
        self.config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()
        
        # 1. Open File Handles (Append Mode)
        f_pos = open(self.config.OUTPUT_DIR / 'positions.bin', 'wb')
        f_mov = open(self.config.OUTPUT_DIR / 'moves.bin', 'wb')
        f_res = open(self.config.OUTPUT_DIR / 'results.bin', 'wb')
        
        self.logger.info("Starting Direct Stream Extraction...")
        
        try:
            for input_file in input_files:
                if self.total_positions >= self.config.TARGET_POSITIONS_TOTAL: break
                
                self.logger.info(f"\nProcessing: {input_file.name}")
                self._process_pgn_file(input_file, f_pos, f_mov, f_res)
                
        except KeyboardInterrupt:
            self.logger.info("\n\nStopping early... saving metadata.")
        finally:
            # Final Flush
            if self.buf_pos:
                self._flush_buffer(f_pos, f_mov, f_res)
            
            f_pos.close()
            f_mov.close()
            f_res.close()
            self._save_metadata()
            self.logger.info("\nExtraction Complete.")

    def _process_pgn_file(self, zst_file, f_pos, f_mov, f_res):
        try:
            process = subprocess.Popen(
                ['zstdcat', str(zst_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1024 * 1024,
                universal_newlines=True,
                text=True
            )
            
            while True:
                if self.total_positions >= self.config.TARGET_POSITIONS_TOTAL: break
                
                game = chess.pgn.read_game(process.stdout)
                if game is None: break
                
                # Filters
                headers = game.headers
                try:
                    w_elo = int(headers.get("WhiteElo", 0))
                    b_elo = int(headers.get("BlackElo", 0))
                    tc = headers.get("TimeControl", "0+0").split('+')[0]
                    base_time = int(tc)
                except: continue

                if base_time < self.config.time_control_min: continue
                
                bucket = self._get_bucket(w_elo, b_elo)
                if not bucket: continue
                
                # SKEW CHECK: Stop taking data from this bucket if it's full
                if self.bucket_counts[bucket] >= self.bucket_targets[bucket]: continue
                
                # Result
                res_str = headers.get("Result", "*")
                if res_str == "1-0": game_res = 1.0
                elif res_str == "0-1": game_res = -1.0
                else: game_res = 0.0 # Draw or unknown
                
                # Moves
                board = game.board()
                for move in game.mainline_moves():
                    if self.config.validate_moves and move not in board.legal_moves: break
                    
                    # Add to Buffer
                    self.buf_pos.append(self.encoder.board_to_tensor(board))
                    self.buf_mov.append(MoveEncoder.move_to_index(move))
                    self.buf_res.append(game_res if board.turn == chess.WHITE else -game_res)
                    
                    self.total_positions += 1
                    self.bucket_counts[bucket] += 1
                    board.push(move)
                    
                    # UPDATE UI every 10k
                    if self.total_positions % self.config.ui_update_interval == 0:
                        self._print_progress()

                    # FLUSH DISK every 100k
                    if len(self.buf_pos) >= self.config.flush_interval:
                        self._flush_buffer(f_pos, f_mov, f_res)
                        
        except Exception as e:
            self.logger.error(f"Stream error: {e}")

    def _flush_buffer(self, f_pos, f_mov, f_res):
        """Write buffer to disk immediately."""
        # Convert to numpy (fast C-contiguous)
        np_pos = np.array(self.buf_pos, dtype=np.float16)
        np_mov = np.array(self.buf_mov, dtype=np.int32)
        np_res = np.array(self.buf_res, dtype=np.float16)
        
        # Write raw bytes (append)
        f_pos.write(np_pos.tobytes())
        f_mov.write(np_mov.tobytes())
        f_res.write(np_res.tobytes())
        
        # Clear RAM
        self.buf_pos, self.buf_mov, self.buf_res = [], [], []
        
        # Update metadata so partial runs are valid
        self._save_metadata()

    def _save_metadata(self):
        meta = {
            'count': self.total_positions,
            'pos_shape': [self.total_positions, 13, 8, 8],
            'pos_dtype': 'float16',
            'mov_shape': [self.total_positions],
            'mov_dtype': 'int32',
            'res_shape': [self.total_positions],
            'res_dtype': 'float16',
            'bucket_stats': self.bucket_counts
        }
        with open(self.config.OUTPUT_DIR / 'metadata.json', 'w') as f:
            json.dump(meta, f, indent=2)

    def _get_bucket(self, w, b):
        avg = (w + b) // 2
        for l, h in self.config.ELO_BUCKETS:
            if l <= avg < h:
                return self.config.get_bucket_name(l, h)
        return None

    def _print_progress(self):
        """Prints a simple log line instead of a dynamic bar (better for log files)."""
        elapsed = time.time() - self.start_time
        if elapsed == 0: return
        rate = self.total_positions / elapsed
        percent = self.total_positions / self.config.TARGET_POSITIONS_TOTAL
        
        # Format: [2023-11-29 10:00:00] Progress: 15.5% | 15.5M / 100M | Speed: 9500 pos/s
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.logger.info(
            f"[{timestamp}] Progress: {percent:>6.1%} | "
            f"{self.total_positions/1e6:.1f}M / {self.config.TARGET_POSITIONS_TOTAL/1e6:.0f}M | "
            f"Speed: {rate:.0f} pos/s"
        )

def setup_logging(config):
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    return logging.getLogger(__name__)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, default='/home/krish/EE542-Project/FINAL/chess_ai/data/lichess_raw')
    parser.add_argument('--target', type=int, default=None, help='Total positions to extract')
    args = parser.parse_args()
    
    config = ProductionConfig()
    
    # CLI Override
    if args.target is not None:
        config.TARGET_POSITIONS_TOTAL = args.target
        
    logger = setup_logging(config)
    
    input_files = sorted(Path(args.input_dir).glob('*.pgn.zst'))
    if not input_files:
        print("No .pgn.zst files found!")
    else:
        extractor = ProductionDataExtractor(config, logger)
        extractor.extract_dataset(input_files)