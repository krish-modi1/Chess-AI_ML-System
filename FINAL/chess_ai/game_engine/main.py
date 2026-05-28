import multiprocessing as mp
import threading
import os
import sys
import time
import shutil
import gc
import torch
import numpy as np
import math
import chess 
import signal
import sys
import json

class TimeoutHandler:
    """Handle process timeouts to prevent deadlocks"""
    def __init__(self, timeout_seconds=9000):
        self.timeout_seconds = timeout_seconds
        self.start_time = None
    
    def start(self):
        self.start_time = time.time()
        signal.signal(signal.SIGALRM, self._timeout_handler)
        signal.alarm(self.timeout_seconds)
    
    def _timeout_handler(self, signum, frame):
        elapsed = int(time.time() - self.start_time)
        print(f"⚠️ TIMEOUT: No progress in {self.timeout_seconds}s ({elapsed}s elapsed)")
        print("⚠️ Likely deadlock detected. Shutting down gracefully...")
        sys.exit(1)
    
    def reset(self):
        """Reset timeout (call when iteration completes)"""
        if self.start_time:
            signal.alarm(self.timeout_seconds)

timeout_handler = TimeoutHandler(timeout_seconds=108000)

# Ensure project root is in path
sys.path.append(os.getcwd())

from game_engine.neural_net import InferenceServer
from game_engine.mcts_worker_cpp import MCTSWorker
from game_engine.chess_env import ChessGame
from game_engine.trainer import train_model
from game_engine.evaluation import Arena, StockfishEvaluator, MetricsLogger
from game_engine.cnn import ChessCNN

# ==========================================
class GracefulKiller:
    """Catches SIGTERM/SIGINT signals for graceful cloud shutdown"""
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        print("\n\n[Cloud Run] Received termination signal. Finishing current step...")
        self.kill_now = True

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(LOG_FILE, "a", buffering=1, encoding='utf-8')
    def write(self, message):
        try:
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
        except Exception:
            pass  # Ignore I/O errors to prevent logging from crashing the application 
    def flush(self):
        try:
            self.terminal.flush()
            self.log.flush()
        except Exception:
            pass  # Ignore I/O errors to prevent logging from crashing the application

def setup_child_logging():
    sys.stdout = Logger()
    sys.stderr = sys.stderr

def queue_monitor_thread(queue):
    while True:
        try:
            size = queue.qsize()
            if size > 500: 
                print(f"   [Server Monitor] High Load: {size} requests pending")
            time.sleep(2.0)
        except: break

def get_start_iteration(data_dir):
    if not os.path.exists(data_dir):
        return 1
    
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("iter_")]
    if not subdirs:
        return 1
        
    try:
        nums = [int(d.split("_")[1]) for d in subdirs]
        return max(nums) + 1
    except ValueError:
        return 1

def cleanup_memory():
    """Forces garbage collection and clears CUDA cache to prevent OOM"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def run_worker_batch(worker_id, input_queue, output_queue, game_limit, iteration):
    # Generate unique seed for this worker (same for python and C++ RNG)
    mcts_seed = int(time.time()) + worker_id
    np.random.seed(mcts_seed)  # Use same seed for numpy RNG
    
    if hasattr(os, 'sched_setaffinity'):
        try:
            os.sched_setaffinity(0, {worker_id % 44})
        except Exception:
            pass  # CPU affinity is optional; ignore if not supported
    
    setup_child_logging()
    
    iter_dir = os.path.join(DATA_DIR, f"iter_{iteration}")
    os.makedirs(iter_dir, exist_ok=True)
    
    # PASS SEED TO MCTSWorker for C++ RNG seeding
    worker = MCTSWorker(worker_id, input_queue, output_queue,
                        simulations=SIMULATIONS,
                        batch_size=WORKER_BATCH_SIZE,
                        seed=mcts_seed)
    
    for i in range(game_limit):
        print(f" [Worker {worker_id}] Starting Game {i+1}...")
        
        game_start = time.time()
        game = ChessGame()
        game_data = []
        forced_draw = False
        
        try:
            while not game.is_over:
                if len(game.moves) >= MAX_MOVES_PER_GAME:
                    forced_draw = True
                    break
                
                move_start = time.time()
                move_count = len(game.moves)
                
                if move_count < 16:
                    current_temp = 1.0
                else:
                    current_temp = 0.0
                
                best_move, mcts_policy = worker.search(game, temperature=current_temp)

                # Try to apply the move
                move_applied = game.push(best_move)
                if not move_applied:
                    print(f" [Worker {worker_id}] ❌ Illegal move from MCTS: {best_move}")
                    # Optional: print some context
                    print(f" [Worker {worker_id}]   Legal moves: {game.legal_moves()}")
                    break  # Abort this game; don't pretend it finished normally

                # Only record data if the move was applied
                game_data.append({
                    "state": game.to_tensor(),
                    "policy": mcts_policy,
                    "turn": game.turn_player,
                })

                
                dur = time.time() - move_start
                nps = SIMULATIONS / dur if dur > 0 else 0
                print(f" [Worker {worker_id}] Move {move_count+1}: {best_move} ({dur:.2f}s | {nps:.0f} sim/s)")
        except Exception as e:
            print(f" [Worker {worker_id}] ❌ ERROR during game {i+1}: {e}")
            import traceback
            traceback.print_exc()
            continue

        if forced_draw:
            print(f" [Worker {worker_id}] Game {i+1} ended in FORCED DRAW (Max moves {MAX_MOVES_PER_GAME})")
            result = "1/2-1/2"
        else:
            result = game.result
        
        final_winner = 0.0
        if result == "1-0":
            final_winner = 1.0
        elif result == "0-1":
            final_winner = 0.0
        elif result == "1/2-1/2":
            final_winner = 0.5
        
        values = []
        for g in game_data:
            if final_winner == 0.5:
                penalty = FORCED_DRAW_PENALTY if forced_draw else DRAW_PENALTY
                values.append(penalty)
            elif g['turn'] == final_winner:
                values.append(1.0)
            else:
                values.append(-1.0)
        
        timestamp = int(time.time())
        filename = f"{iter_dir}/w{worker_id}_g{i}_{timestamp}.npz"
        
        np.savez_compressed(filename,
                          states=np.array([g["state"] for g in game_data]),
                          policies=np.array([g["policy"] for g in game_data]),
                          values=np.array(values, dtype=np.float32))

        print(f" [Worker {worker_id}] Finished Game {i+1} in {time.time()-game_start:.1f}s | Total Moves {len(game.moves)} | Result {result}")
        
        gc.collect()

def run_server_wrapper(server):
    setup_child_logging()
    if hasattr(os, 'sched_setaffinity'):
        try:
            os.sched_setaffinity(0, {44, 45, 46, 47})
        except Exception:
            pass  # CPU affinity is optional; ignore if not supported
        
    monitor = threading.Thread(target=queue_monitor_thread, args=(server.input_queue,))
    monitor.daemon = True 
    monitor.start()
    server.loop()

# ==========================================
#        BALANCED GCP CONFIG (T4 OPTIMIZED)
# ==========================================

# --- PATHS ---
STOCKFISH_PATH = "/usr/games/stockfish" 
LOG_FILE = "training_log.txt"
MODEL_DIR = "game_engine/model"
DATA_DIR = "data/self_play"
BEST_MODEL = f"{MODEL_DIR}/best_model.pth"
CANDIDATE_MODEL = f"{MODEL_DIR}/candidate.pth"

# --- CUDA ---
CUDA_TIMEOUT_INFERENCE = 1.0
CUDA_STREAMS = 8 
CUDA_BATCH_SIZE = 4096

# --- EXECUTION ---
RESUME_ITERATION = None
ITERATIONS = 1000
NUM_WORKERS = 150            
WORKER_BATCH_SIZE = 320       
GAMES_PER_WORKER = 2        

# --- QUALITY ---
SIMULATIONS = 1600           
EVAL_SIMULATIONS = 1600      

# --- EVALUATION CONFIG ---
EVAL_WORKERS = 10           
GAMES_PER_EVAL_WORKER = 4   
STOCKFISH_GAMES = 20
STOCKFISH_ELO = 1320        

# --- RULES ---
MAX_MOVES_PER_GAME = 800   
EVAL_MAX_MOVES_PER_GAME = 800 

current_iter = get_start_iteration(DATA_DIR) - 1
if current_iter < 10:
    DRAW_PENALTY = -0.1
elif current_iter < 20:
    DRAW_PENALTY = -0.2
else:
    DRAW_PENALTY = -0.25        

FORCED_DRAW_PENALTY = 0.0

# Training
TRAIN_EPOCHS = 4 
TRAIN_WINDOW = 15           
TRAIN_BATCH_SIZE = 1024
TRAIN_LR = 0.001       

# --- DRY WORKER WRAPPERS ---

def run_arena_batch_worker(worker_id, queue, num_games, cand_model, champ_model, sims, max_moves):
    setup_child_logging()
    np.random.seed(worker_id + int(time.time()) % 10000)
    torch.manual_seed(worker_id + int(time.time()) % 10000)
    try:
        arena = Arena(cand_model, champ_model, sims)
        w, d, l, fd = 0, 0, 0, 0
        for game_num in range(num_games):
            result = arena.play_game(game_num, max_moves=max_moves)
            if result == "CAND_WIN":
                w += 1
            elif result == "CHAMP_WIN":
                l += 1
            elif result == "DRAW":
                d += 1
            elif result == "DRAW_FORCED":
                fd += 1
        
        result_str = f"Worker {worker_id}: {w}W - {d}D - {l}L - {fd}FD"
        print(f" [Arena] {result_str}")
        queue.put({"wins": w, "draws": d, "losses": l, "forced_draws": fd})
    
    except Exception as e:  # ← ADD THIS
        print(f"[Arena Worker {worker_id}] ❌ CRASHED: {e}")
        import traceback
        traceback.print_exc()
        queue.put({"wins": 0, "draws": 0, "losses": 0, "forced_draws": 0})  # ← ALWAYS send result

# --- PHASES ---

def run_self_play_phase(iteration):
    print(f"\n=== ITERATION {iteration}: SELF-PLAY PHASE (Batched MCTS) ===")
    cleanup_memory() # Clear RAM before starting
    
    server = InferenceServer(BEST_MODEL, batch_size=CUDA_BATCH_SIZE, timeout=CUDA_TIMEOUT_INFERENCE, streams=CUDA_STREAMS)
    worker_queues = [server.register_worker(i) for i in range(NUM_WORKERS)]
    
    server_process = mp.Process(target=run_server_wrapper, args=(server,))
    server_process.start()
    time.sleep(5) 
    
    # Start queue monitor to detect stalls
    def monitor_queues(server_ref, input_q, interval=30):
        """Monitor queue sizes and detect stuck situations"""
        last_size = 0
        stall_count = 0
        
        while server_process.is_alive():
            current_size = input_q.qsize()
            
            # If queue size hasn't changed in 3 checks (90 seconds), likely stuck
            if current_size == last_size and current_size > 0:
                stall_count += 1
                if stall_count >= 3:
                    print(f"⚠️ QUEUE STALL DETECTED: {current_size} requests stuck for ~90s")
            else:
                stall_count = 0
            
            last_size = current_size
            time.sleep(interval)
    
    monitor_thread = threading.Thread(
        target=monitor_queues,
        args=(server, server.input_queue),
        daemon=True
    )
    monitor_thread.start()

    workers = []
    for i in range(NUM_WORKERS):
        p = mp.Process(target=run_worker_batch, 
                       args=(i, server.input_queue, worker_queues[i], GAMES_PER_WORKER, iteration))
        p.start()
        workers.append(p)
        
    # ACTIVE MONITORING LOOP
    try:
        while True:
            # 1. Check if Server is alive
            if not server_process.is_alive():
                print("🚨 CRITICAL: Inference Server died unexpectedly! Terminating workers...")
                raise RuntimeError("Inference Server died during self-play.")
            
            # 2. Check if Workers are done
            alive_workers = [p for p in workers if p.is_alive()]
            if not alive_workers:
                print("✅ All workers finished successfully.")
                break
            
            # 3. Sleep to prevent CPU burn
            time.sleep(2)
            
    except Exception as e:
        print(f"❌ Exception in Phase 1 Loop: {e}")
        for p in workers:
            if p.is_alive():
                p.terminate()
        if server_process.is_alive():
            server_process.terminate()
        raise e
        
    finally:
        print("🧹 Cleaning up Phase 1 processes...")
        
        # Kill any straggler workers
        for p in workers:
            if p.is_alive():
                p.terminate()
                p.join(timeout=1)
        
        # Stop Server
        if server_process.is_alive():
            server.input_queue.put("STOP")
            server_process.join(timeout=10)
            if server_process.is_alive():
                print("⚠️ Server did not stop gracefully. Force killing...")
                server_process.terminate()
                server_process.join(timeout=5)
                if server_process.is_alive():
                    server_process.kill()
                    server_process.join()

def run_training_phase(iteration):
    print(f"\n=== ITERATION {iteration}: TRAINING PHASE ===")
    cleanup_memory() # Clear VRAM before training
    
    p_loss, v_loss = train_model(data_path=DATA_DIR, 
                input_model_path=BEST_MODEL, 
                output_model_path=CANDIDATE_MODEL,
                epochs=TRAIN_EPOCHS,
                batch_size=TRAIN_BATCH_SIZE,
                lr=TRAIN_LR,
                window_size=TRAIN_WINDOW)
    
    return p_loss, v_loss

def run_evaluation_phase(iteration, logger, p_loss, v_loss):
    print(f"\n=== ITERATION {iteration}: EVALUATION PHASE ===")
    cleanup_memory() # Clear VRAM before launching multiple evaluation workers

    # 1. ARENA EVALUATION
    print(f" [Arena] Playing {EVAL_WORKERS * GAMES_PER_EVAL_WORKER} games...")
    ctx = mp.get_context('spawn')
    arena_queue = ctx.Queue()
    arena_workers = []

    for i in range(EVAL_WORKERS):
        p = ctx.Process(
            target=run_arena_batch_worker,
            args=(i, arena_queue, GAMES_PER_EVAL_WORKER, CANDIDATE_MODEL, BEST_MODEL, EVAL_SIMULATIONS, EVAL_MAX_MOVES_PER_GAME)
        )
        p.start()
        arena_workers.append(p)

    # --- ACTIVE MONITORING FOR ARENA ---
    try:
        while True:
            alive = [p for p in arena_workers if p.is_alive()]
            if not alive:
                print("✅ All arena workers finished.")
                break
            time.sleep(1)
    except Exception as e:
        print(f"❌ Arena evaluation failed: {e}")
        for p in arena_workers:
            if p.is_alive():
                p.terminate()
                p.join(timeout=1)
        raise

    
    # Collect Arena Results
    total_wins, total_draws, total_losses, total_forced_draws = 0, 0, 0, 0
    while not arena_queue.empty():
        res = arena_queue.get()
        total_wins += res['wins']
        total_draws += res['draws']
        total_losses += res['losses']
        total_forced_draws += res['forced_draws']
    
    total_score = total_wins + 0.75 * total_forced_draws + 0.5 * total_draws
    total_game_count = total_wins + total_draws + total_forced_draws + total_losses
    win_rate = total_score / total_game_count if total_game_count > 0 else 0
    
    print(f" [Arena] Final Result: {win_rate*100:.1f}% Win Rate ({total_wins}W - {total_draws}D - {total_forced_draws}FD - {total_losses}L)")
    
    est_elo = None
    arena_promoted = False

    # 2. PROMOTION LOGIC
    if win_rate >= 0.55:
        print(f" [Arena] ⭐ Candidate PROMOTED! (WR > 55%) ⭐")
        shutil.copyfile(CANDIDATE_MODEL, BEST_MODEL)
        arena_promoted = True  # ← ADD THIS LINE
    else:  # ← MOVE THIS FROM WRONG PLACE
        print(f" [Arena] Candidate rejected (WR <= 55%). Running Stockfish evaluation anyway...")
        
    # 3. FETCH LAST CHAMPION ELO FROM METRICS
    last_champion_elo = None
    try:
        if os.path.exists("game_engine/model/metrics.json"):
            with open("game_engine/model/metrics.json", "r") as f:
                metrics_history = json.load(f)
                if metrics_history and len(metrics_history) > 0:
                    # Get last entry with valid model elo (not stockfish_elo)
                    for entry in reversed(metrics_history):
                        if entry.get("elo") is not None:  # ← FIXED: 'elo' not 'stockfish_elo'
                            last_champion_elo = entry.get("elo")
                            print(f" [Metrics] Last Champion Model Elo: {last_champion_elo:.0f}")
                            break
    except Exception as e:
        print(f" [Metrics] Warning: Could not read metrics.json: {e}")

    
    # 4. STOCKFISH EVALUATION
    print(f" [Stockfish/BayesElo] Playing {STOCKFISH_GAMES} games vs Elo {STOCKFISH_ELO}...")
    cleanup_memory()
    
    try:
        sf_eval = StockfishEvaluator(STOCKFISH_PATH, EVAL_SIMULATIONS)
        pgn_path = f"game_engine/evaluation/pgn/iter_{iteration}_{int(time.time())}.pgn"
        
        bayeselo_results = sf_eval.evaluate_with_bayeselo(
            model_path=CANDIDATE_MODEL,
            pgn_output_path=pgn_path,
            num_games=STOCKFISH_GAMES,
            stockfish_elo=STOCKFISH_ELO,
            max_moves=EVAL_MAX_MOVES_PER_GAME
        )
        
        if bayeselo_results:
            est_elo = bayeselo_results['model_elo']
            print(f" [BayesElo] ✅ Model Elo: {est_elo:.0f}")
            print(f" [BayesElo] Record: {bayeselo_results['win_count']}-{bayeselo_results['draw_count']}-{bayeselo_results['loss_count']}")
            
            if not arena_promoted:
                # 5. PROMOTION LOGIC: STOCKFISH-BASED
                if last_champion_elo is not None and est_elo > last_champion_elo:
                    print(f" [Stockfish] 🚀 CANDIDATE PROMOTED! ({est_elo:.0f} > {last_champion_elo:.0f})")
                    shutil.copyfile(CANDIDATE_MODEL, BEST_MODEL)
                elif last_champion_elo is None and est_elo > 1000:
                    # Fallback: if no last Elo, promote if >1000 (basic competence)
                    print(f" [Stockfish] 🚀 CANDIDATE PROMOTED! (First Elo: {est_elo:.0f})")
                    shutil.copyfile(CANDIDATE_MODEL, BEST_MODEL)
                else:
                    print(f" [Stockfish] ❌ Candidate not promoted. ({est_elo:.0f} <= {last_champion_elo if last_champion_elo else None})")
        else:
            est_elo = None
            print(f" [BayesElo] ❌ Failed to compute")
            
    except Exception as e:
        print(f" [BayesElo] ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        est_elo = None
            
    logger.log(iteration, p_loss, v_loss, win_rate, est_elo, stockfish_elo=STOCKFISH_ELO)

if __name__ == "__main__":
    setup_child_logging()
    mp.set_start_method('spawn', force=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    if not os.path.exists(BEST_MODEL):
        print("Initializing random model...")
        torch.save(ChessCNN(upgraded=True).state_dict(), BEST_MODEL)

    timeout_handler.start()
    print("⏱️ Deadlock timeout: 30 hour per iteration")

    # RESUMPTION LOGIC
    start_iter = get_start_iteration(DATA_DIR)
    
    print("=" * 60)
    print(f"STARTING RUN")
    print(f"Resuming from Iteration: {start_iter}")
    print(f"Workers: {NUM_WORKERS} | Sims: {SIMULATIONS} | Batch: {WORKER_BATCH_SIZE}")
    print("=" * 60)

    killer = GracefulKiller()
    
    try:
        for it in range(start_iter, ITERATIONS + 1):
            
            # CHECK BEFORE ITERATION STARTS
            if killer.kill_now:
                print("\n[Main] ⚠️  Kill signal received BEFORE iteration start")
                print(f"[Main] Gracefully exiting. Next run will resume from Iteration {it}")
                break
            
            iter_start = time.time()
            
            # === PHASE 1: SELF-PLAY ===
            print(f"\n{'='*60}")
            print(f"ITERATION {it} - PHASE 1: SELF-PLAY")
            print(f"{'='*60}")
            
            if RESUME_ITERATION is not None and it == RESUME_ITERATION:
                print(f"⏭️  ITERATION {it} - SKIPPING SELF-PLAY (using existing data)")
                print(f"✅ ITERATION {it} - PHASE 1 COMPLETE (skipped)")
            else:
                try:
                    run_self_play_phase(it)
                    print(f"✅ ITERATION {it} - PHASE 1 COMPLETE")
                except Exception as e:
                    print(f"❌ ITERATION {it} - PHASE 1 FAILED: {e}")
                    if killer.kill_now:
                        print("[Main] Kill signal during Phase 1. Exiting...")
                        break
                    raise
            
            # CHECK AFTER PHASE 1
            if killer.kill_now:
                print("\n[Main] ⚠️  Kill signal received AFTER Phase 1")
                print("[Main] Saving state and exiting. Training/Eval will run on next startup.")
                break
            
            # === PHASE 2: TRAINING ===
            print(f"\n{'='*60}")
            print(f"ITERATION {it} - PHASE 2: TRAINING")
            print(f"{'='*60}")
            
            try:
                p_loss, v_loss = run_training_phase(it)
                print(f"\n✅ ITERATION {it} - PHASE 2 COMPLETE (Policy Loss: {p_loss:.4f}, Value Loss: {v_loss:.4f})")
            except Exception as e:
                print(f"\n❌ ITERATION {it} - PHASE 2 FAILED: {e}")
                if killer.kill_now:
                    print("[Main] Kill signal during Phase 2. Exiting...")
                    break
                raise
            
            # CHECK AFTER PHASE 2
            if killer.kill_now:
                print("\n[Main] ⚠️  Kill signal received AFTER Phase 2")
                print("[Main] Saving state and exiting. Eval will run on next startup.")
                break
            
            # === PHASE 3: EVALUATION ===
            print(f"\n{'='*60}")
            print(f"ITERATION {it} - PHASE 3: EVALUATION")
            print(f"{'='*60}")
            
            try:
                run_evaluation_phase(it, MetricsLogger(), p_loss, v_loss)
                print(f"\n✅ ITERATION {it} - PHASE 3 COMPLETE")
            except Exception as e:
                print(f"\n❌ ITERATION {it} - PHASE 3 FAILED: {e}")
                if killer.kill_now:
                    print("[Main] Kill signal during Phase 3. Exiting...")
                    break
                raise
            
            # === ITERATION COMPLETE ===
            iter_end = time.time()
            elapsed = iter_end - iter_start
            hours, remainder = divmod(elapsed, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            print(f"\n{'='*60}")
            if hours > 0:
                print(f"✅ ITERATION {it} COMPLETE: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
            else:
                print(f"✅ ITERATION {it} COMPLETE: {int(minutes)}m {seconds:.2f}s")
            print(f"{'='*60}\n")
            
            timeout_handler.reset()
            print(f"✅ Iteration {it} completed - timeout reset")
            
            # CHECK BEFORE NEXT ITERATION
            if killer.kill_now:
                print("[Main] ⚠️  Kill signal received. Exiting gracefully...")
                break
    
    except KeyboardInterrupt:
        print("\n\n[Main] ❌ USER INTERRUPTED - EXITING")
    
    except Exception as e:
        print(f"\n\n[Main] ❌ FATAL ERROR: {e}")
        raise
    
    finally:
        print("\n[Main] Cleanup: Closing threads and processes...")
        cleanup_memory()
        print("[Main] ✅ Shutdown complete")
