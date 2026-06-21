import multiprocessing as mp
import threading
import os
import sys
import time
import shutil
import gc
import json
import collections
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
import signal
import sys

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

# Per-iteration deadlock guard (SIGALRM), reset after each iteration completes. With the small
# WORKER_BATCH_SIZE the search now does ~100 inference round-trips/move, so draw-heavy iterations
# (games up to MAX_MOVES_PER_GAME) run far longer than under the old 1-round-trip config. Default
# 48h gives margin over a worst-case long iteration while still catching a genuine hang. Env-tunable.
timeout_handler = TimeoutHandler(timeout_seconds=int(os.environ.get("ITERATION_TIMEOUT", 172800)))

# Ensure project root is in path
sys.path.append(os.getcwd())

from game_engine.neural_net import InferenceServer
from game_engine.mcts_worker_cpp import MCTSWorker
from game_engine.chess_env import ChessGame
from game_engine.trainer import train_model
from game_engine.evaluation import MetricsLogger
from game_engine.cnn import ChessCNN
from game_engine.aux_labels import derive_aux_labels

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
    # Pin this process to 1 CPU thread. We parallelise across worker PROCESSES; letting each also
    # spin up a torch/MKL intra-op thread pool (≈cores per process) thrashes the box (load ~570 on
    # 32 vCPUs). Belt-and-suspenders to the OMP_NUM_THREADS=1 env in hyperparams.env.sh.
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

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

# Phase-level checkpoint. get_start_iteration() only sees self-play data dirs, so it treats
# "self-play done" as "iteration done" and would skip an iteration's training+eval if the run
# stopped between phases. run_state.json records the last completed phase so a restart resumes
# at the NEXT phase of the same iteration.
_PHASES = ("self_play", "training", "eval")

def save_phase(iteration, phase):
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp = RUN_STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"iteration": iteration, "phase": phase}, f)
    os.replace(tmp, RUN_STATE_FILE)  # atomic write

def load_resume_point():
    """Return (start_iteration, start_phase) where start_phase is 1=self-play, 2=training,
    3=eval — the phase the FIRST iteration of this run should begin at. Falls back to the
    data-dir scan (fresh self-play of the next iteration) when no phase checkpoint exists."""
    data_iter = get_start_iteration(DATA_DIR)
    if not os.path.exists(RUN_STATE_FILE):
        return data_iter, 1
    try:
        with open(RUN_STATE_FILE) as f:
            st = json.load(f)
        it = int(st["iteration"])
        done = _PHASES.index(st["phase"]) + 1          # phases completed so far
        if done >= len(_PHASES):
            return it + 1, 1                           # whole iteration done → next, fresh
        return it, done + 1                            # resume at the phase after the last done
    except Exception:
        return data_iter, 1

def cleanup_memory():
    """Forces garbage collection and clears CUDA cache to prevent OOM"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _promote_best():
    """Copy candidate → best, unless NO_PROMOTE (dry-run eval that must not mutate the champion)."""
    if NO_PROMOTE:
        print("  (NO_PROMOTE: keeping best_model.pth unchanged)")
    else:
        shutil.copyfile(CANDIDATE_MODEL, BEST_MODEL)

def _start_inference_server(model_path, n_workers, worker_batch, cuda_batch, streams, timeout):
    """Start an InferenceServer (own process) hosting `model_path`, with SHM buffers + per-worker
    queues — same batched-GPU path self-play uses. Returns (server, proc, worker_queues, shm tuple)."""
    if SHM_TRANSPORT:
        shm_inp = torch.empty((n_workers, worker_batch, 120, 8, 8), dtype=torch.float32).share_memory_()
        shm_pol = torch.empty((n_workers, worker_batch, 4672), dtype=torch.float32).share_memory_()
        shm_val = torch.empty((n_workers, worker_batch, 3), dtype=torch.float32).share_memory_()
    else:
        shm_inp = shm_pol = shm_val = None
    server = InferenceServer(model_path, batch_size=cuda_batch, timeout=timeout, streams=streams,
                             shm_inp=shm_inp, shm_pol=shm_pol, shm_val=shm_val)
    wq = [server.register_worker(i) for i in range(n_workers)]
    sp = mp.Process(target=run_server_wrapper, args=(server,))
    sp.start()
    time.sleep(5)  # server CUDA init + model load
    return server, sp, wq, (shm_inp, shm_pol, shm_val)

def _stop_inference_server(server, sp):
    """STOP the server and free its VRAM before the next phase/server."""
    try:
        server.input_queue.put("STOP")
        sp.join(timeout=10)
    except Exception:
        pass
    if sp.is_alive():
        sp.terminate(); sp.join(timeout=5)
        if sp.is_alive():
            sp.kill(); sp.join()
    cleanup_memory()

def run_worker_batch(worker_id, input_queue, output_queue, game_limit, iteration, games_counter=None,
                     shm_inp=None, shm_pol=None, shm_val=None):
    # Generate unique seed for this worker (same for python and C++ RNG).
    # SEED_BASE (env) makes runs reproducible for A/B correctness tests; default = wall clock.
    mcts_seed = int(os.environ.get("SEED_BASE", int(time.time()))) + worker_id
    np.random.seed(mcts_seed)  # Use same seed for numpy RNG
    
    if hasattr(os, 'sched_setaffinity'):
        try:
            total = len(os.sched_getaffinity(0)) or os.cpu_count() or 32
            # Reserve the top RESERVED_CORES for the OS + the GPU-feeding inference server, so
            # workers never contend with the server (a starved server can fall behind, trip its
            # deadlock timeout, self-kill, and then every worker hangs). Workers fill cores
            # [0 .. total-reserve-1]; e.g. 32 vCPU, reserve=2 → workers on 0-29, server/OS on 30-31.
            reserve = int(os.environ.get("RESERVED_CORES", 0))
            worker_cores = max(1, total - reserve)
            os.sched_setaffinity(0, {worker_id % worker_cores})
        except Exception:
            pass  # CPU affinity is optional; ignore if not supported

    setup_child_logging()
    
    iter_dir = os.path.join(DATA_DIR, f"iter_{iteration}")
    os.makedirs(iter_dir, exist_ok=True)
    
    # PASS SEED TO MCTSWorker for C++ RNG seeding
    worker = MCTSWorker(worker_id, input_queue, output_queue,
                        simulations=SIMULATIONS,
                        batch_size=WORKER_BATCH_SIZE,
                        seed=mcts_seed,
                        shm_inp=shm_inp, shm_pol=shm_pol, shm_val=shm_val)
    
    for i in range(game_limit):
        # Pacing: don't start game i until no worker is more than MAX_WORKER_LEAD behind.
        if games_counter is not None and i > 0:
            wait_start = time.time()
            pacing_logged = False
            while True:
                min_done = min(games_counter)
                if games_counter[worker_id] <= min_done + MAX_WORKER_LEAD:
                    break
                if not pacing_logged:
                    print(f" [Worker {worker_id}] ⏸ Pacing: waiting for slowest worker ({min_done} games done)")
                    pacing_logged = True
                if time.time() - wait_start > 7200:
                    print(f" [Worker {worker_id}] ⚠️ Pacing timeout (2h) — proceeding")
                    break
                time.sleep(0.5)

        print(f" [Worker {worker_id}] Starting Game {i+1}...")

        game_start = time.time()
        game = ChessGame()
        game_data = []
        forced_draw = False
        # Per-game position history: chess.Board copies, most recent first.
        # Populated before each search so the NN sees the last 7 board states.
        position_history = collections.deque(maxlen=7)

        # Reset MCTS tree cache at game start so we never carry state across games.
        worker.reset_cache()
        # Restore full simulations (a previous game may have reduced them once decided).
        worker.mcts_engine.simulations = SIMULATIONS
        decided_streak = 0
        noprogress_streak = 0
        reduced_mode = False
        game_aborted = False

        try:
            while not game.is_over:
                if len(game.moves) >= MAX_MOVES_PER_GAME:
                    forced_draw = True
                    break

                move_start = time.time()
                move_count = len(game.moves)

                if move_count < TEMP_MOVES:
                    current_temp = 1.0
                else:
                    current_temp = 0.0

                # Playout-cap randomization: FULL search (recorded) with prob FULL_SEARCH_PROB,
                # else FAST search (played but not recorded, Dirichlet off). Off → record every move.
                if FULL_SEARCH_PROB > 0.0:
                    is_full = (np.random.random() < FULL_SEARCH_PROB)
                    base_sims = REDUCED_SIMULATIONS if reduced_mode else SIMULATIONS
                    # Once the decided/no-progress cut fires, base_sims drops to REDUCED_SIMULATIONS.
                    # Clamp the fast count to it too — otherwise the 75% fast moves keep running at
                    # FAST_SIMULATIONS (200 > 100), defeating the cut (and costing more than the
                    # recorded full moves). min() leaves normal mode unchanged (fast stays 200).
                    fast_sims = min(FAST_SIMULATIONS, base_sims)
                    worker.mcts_engine.simulations = base_sims if is_full else fast_sims
                else:
                    is_full = True

                # Snapshot history for this search (excludes current board)
                history_snapshot = list(position_history)
                best_move, mcts_policy = worker.search(
                    game, temperature=current_temp, history=history_snapshot,
                    use_dirichlet=is_full
                )

                # KataGo-style decided-game detection: if the position has been lopsided for
                # DECIDED_PATIENCE consecutive moves, cap sims for the rest of the game (still
                # played to completion — no resignation, so labels stay honest).
                if not reduced_mode:
                    if abs(worker.last_root_value) >= DECIDED_VALUE_THRESHOLD:
                        decided_streak += 1
                    else:
                        decided_streak = 0
                    if decided_streak >= DECIDED_PATIENCE:
                        worker.mcts_engine.simulations = REDUCED_SIMULATIONS
                        reduced_mode = True
                        print(f" [Worker {worker_id}] Game {i+1} move {move_count}: decided "
                              f"(|v|={abs(worker.last_root_value):.2f}) — sims "
                              f"{SIMULATIONS}→{REDUCED_SIMULATIONS} to completion")

                    # No-progress (dead-draw) cut: the decided cut above only catches WON games
                    # (|v|>=threshold). A long game whose value sits NEAR ZERO is a shuffling draw
                    # heading to MAX_MOVES that no value threshold can catch — once it's gone
                    # NOPROGRESS_MIN_MOVES and stayed within ±NOPROGRESS_VALUE_BAND for
                    # NOPROGRESS_PATIENCE consecutive moves, cap sims for the rest. Still played to
                    # completion (honest labels), MAX_MOVES unchanged. (Mutually exclusive with the
                    # decided cut: |v| can't be both >=0.9 and <0.2 in the same move.)
                    if abs(worker.last_root_value) < NOPROGRESS_VALUE_BAND:
                        noprogress_streak += 1
                    else:
                        noprogress_streak = 0
                    if (not reduced_mode and move_count >= NOPROGRESS_MIN_MOVES
                            and noprogress_streak >= NOPROGRESS_PATIENCE):
                        worker.mcts_engine.simulations = REDUCED_SIMULATIONS
                        reduced_mode = True
                        print(f" [Worker {worker_id}] Game {i+1} move {move_count}: no-progress draw "
                              f"(|v|<{NOPROGRESS_VALUE_BAND} for {NOPROGRESS_PATIENCE} moves) — sims "
                              f"{SIMULATIONS}→{REDUCED_SIMULATIONS} to completion")

                # Capture state and turn BEFORE push — MCTS searched this position.
                # Storing post-push state would mismatch the policy targets.
                current_turn = game.turn_player
                state_tensor = game.to_tensor(history_snapshot)

                # Advance tree cache to the played move before the next search.
                # This retains the explored subtree, saving redundant simulations.
                worker.advance_root(best_move)

                # Try to apply the move
                move_applied = game.push(best_move)
                if not move_applied:
                    print(f" [Worker {worker_id}] ❌ Illegal move from MCTS: {best_move}")
                    print(f" [Worker {worker_id}]   Legal moves: {game.legal_moves()}")
                    worker.reset_cache()
                    game_aborted = True
                    break

                # Update history with the committed board for the next move's search
                position_history.appendleft(game.board.copy())

                if is_full:   # playout-cap: only full-search positions enter the training data
                    game_data.append({
                        "state": state_tensor,
                        "policy": mcts_policy,
                        "turn": current_turn,
                    })

                
                dur = time.time() - move_start
                # Use the engine's CURRENT sim count, not the module constant — KataGo
                # decided-game reduction may have lowered it, so SIMULATIONS would over-report.
                cur_sims = worker.mcts_engine.simulations
                nps = cur_sims / dur if dur > 0 else 0
                print(f" [Worker {worker_id}] Move {move_count+1}: {best_move} "
                      f"({dur:.2f}s | {nps:.0f} sim/s | {cur_sims} sims | v={worker.last_root_value:+.2f})")
        except Exception as e:
            print(f" [Worker {worker_id}] ❌ ERROR during game {i+1}: {e}")
            import traceback
            traceback.print_exc()
            if games_counter is not None:
                games_counter[worker_id] += 1
            continue

        if game_aborted or not game_data:
            print(f" [Worker {worker_id}] Game {i+1} aborted or empty — skipping save")
            gc.collect()
            if games_counter is not None:
                games_counter[worker_id] += 1
            continue

        if forced_draw:
            print(f" [Worker {worker_id}] Game {i+1} ended in FORCED DRAW (Max moves {MAX_MOVES_PER_GAME})")
            result = "1/2-1/2"
        else:
            result = game.result

        # WDL class indices: 0=win, 1=draw, 2=loss (from perspective of player to move)
        values = []
        if result == "1/2-1/2":
            values = [1] * len(game_data)
        else:
            winner_turn = 1.0 if result == "1-0" else 0.0
            for g in game_data:
                values.append(0 if g['turn'] == winner_turn else 2)

        timestamp = int(time.time())
        filename = f"{iter_dir}/w{worker_id}_g{i}_{timestamp}.npz"
        
        states_arr   = np.array([g["state"] for g in game_data])
        policies_arr = np.array([g["policy"] for g in game_data])
        values_arr   = np.array(values, dtype=np.int8)
        # Bake auxiliary trunk-regularizer labels so the trainer reads them directly (matches
        # migrate_aux.py for old games). See local/plans/auxiliary-targets.md.
        material, plies_left, reply = derive_aux_labels(states_arr, policies_arr, values_arr)
        np.savez_compressed(filename,
                          states=states_arr,
                          policies=policies_arr,
                          values=values_arr,
                          material=material,
                          plies_left=plies_left,
                          reply=reply)

        print(f" [Worker {worker_id}] Finished Game {i+1} in {time.time()-game_start:.1f}s | Total Moves {len(game.moves)} | Result {result}")

        gc.collect()
        if games_counter is not None:
            games_counter[worker_id] += 1

def run_server_wrapper(server):
    setup_child_logging()
    if hasattr(os, 'sched_setaffinity'):
        try:
            available = sorted(os.sched_getaffinity(0))
            # Pin the server to the RESERVED top cores — dedicated, since workers are confined to
            # [0 .. total-reserve-1] (see run_worker_batch). This keeps the GPU-feeding server off
            # the contended worker cores. Fall back to the last 4 when no cores are reserved.
            reserve = int(os.environ.get("RESERVED_CORES", 0))
            n = reserve if reserve > 0 else min(4, len(available))
            server_cpus = set(available[-n:]) if available else set()
            if server_cpus:
                os.sched_setaffinity(0, server_cpus)
        except Exception:
            pass  # CPU affinity is optional; ignore if not supported
        
    monitor = threading.Thread(target=queue_monitor_thread, args=(server.input_queue,))
    monitor.daemon = True 
    monitor.start()
    server.loop()

# ==========================================
#        RUNTIME CONFIGURATION
# All values can be overridden via environment variables for SLURM/CARC runs.
# ==========================================

# --- PATHS ---
STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH", "/usr/games/stockfish")
# Absolute, anchored to chess_ai/ (parent of game_engine/) so the Logger always writes
# chess_ai/training_log.txt regardless of a process's cwd — prevents a duplicate
# training_log.txt being created under game_engine/.
LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training_log.txt")
MODEL_DIR = "game_engine/model"
DATA_DIR = "data/self_play"
RUN_STATE_FILE = os.path.join(DATA_DIR, "run_state.json")
BEST_MODEL = f"{MODEL_DIR}/best_model.pth"
CANDIDATE_MODEL = f"{MODEL_DIR}/candidate.pth"

# --- CUDA ---
CUDA_TIMEOUT_INFERENCE = float(os.environ.get("CUDA_TIMEOUT_INFERENCE", 0.02))
CUDA_STREAMS = int(os.environ.get("CUDA_STREAMS", 8))
CUDA_BATCH_SIZE = int(os.environ.get("CUDA_BATCH_SIZE", 4096))
# Shared-memory inference transport: pass leaf/result tensors via shared buffers and send only
# tiny (worker_id, N) signals through the queues (vs pickling tensors). "0" = legacy path.
SHM_TRANSPORT = os.environ.get("SHM_TRANSPORT", "1") == "1"

# --- EXECUTION ---
RESUME_ITERATION = None
ITERATIONS = int(os.environ.get("ITERATIONS", 1000))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 150))
WORKER_BATCH_SIZE = int(os.environ.get("WORKER_BATCH_SIZE", 320))
GAMES_PER_WORKER = int(os.environ.get("GAMES_PER_WORKER", 2))

# --- QUALITY ---
SIMULATIONS = int(os.environ.get("SIMULATIONS", 1600))
EVAL_SIMULATIONS = int(os.environ.get("EVAL_SIMULATIONS", 1600))

# --- KataGo-style decided-game playout (self-play only) ---
# When the root value has been lopsided (|P(win)-P(loss)| >= DECIDED_VALUE_THRESHOLD) for
# DECIDED_PATIENCE consecutive moves, cap simulations at REDUCED_SIMULATIONS for the rest of
# the game. The game is PLAYED TO COMPLETION (no resignation) so value labels stay honest —
# KataGo found resignation records positions incorrectly; reducing visits just finishes decided
# games cheaply. Set DECIDED_PATIENCE huge to disable.
DECIDED_VALUE_THRESHOLD = float(os.environ.get("DECIDED_VALUE_THRESHOLD", 0.9))
DECIDED_PATIENCE = int(os.environ.get("DECIDED_PATIENCE", 5))
REDUCED_SIMULATIONS = int(os.environ.get("REDUCED_SIMULATIONS", 100))

# --- Playout-cap randomization (KataGo §3.1; self-play only) ---
# When FULL_SEARCH_PROB > 0: each move does a FULL search (normal sims, Dirichlet on) with this
# probability and is RECORDED; otherwise a FAST search (FAST_SIMULATIONS, Dirichlet off) that is
# PLAYED but NOT recorded. Decouples value-data volume from policy-target quality. 0 = off (record
# every move = current behavior). See local/plans/upgrades-1-6.md.
FULL_SEARCH_PROB = float(os.environ.get("FULL_SEARCH_PROB", 0.0))
FAST_SIMULATIONS = int(os.environ.get("FAST_SIMULATIONS", 200))
# Opening exploration: sample the played move (temperature=1) for the first TEMP_MOVES plies, then
# greedy. Lower = stay closer to the prior's lines (less off-distribution opening drift). AZ used ~30.
TEMP_MOVES = int(os.environ.get("TEMP_MOVES", 16))

# --- No-progress (dead-draw) cut (self-play only) ---
# The decided-game cut above only fires on WON positions (|value| high). Long DRAWN games have
# value ≈ 0 — no threshold catches them, so they grind full sims to MAX_MOVES. This cuts them:
# once a game passes NOPROGRESS_MIN_MOVES AND its root value has stayed within ±NOPROGRESS_VALUE_BAND
# for NOPROGRESS_PATIENCE consecutive moves (a shuffling stalemate), cap sims at REDUCED_SIMULATIONS
# for the rest. Game is still played to completion; MAX_MOVES_PER_GAME is unchanged.
NOPROGRESS_MIN_MOVES  = int(os.environ.get("NOPROGRESS_MIN_MOVES", 250))
NOPROGRESS_VALUE_BAND = float(os.environ.get("NOPROGRESS_VALUE_BAND", 0.2))
NOPROGRESS_PATIENCE   = int(os.environ.get("NOPROGRESS_PATIENCE", 30))

# --- EVALUATION CONFIG ---
EVAL_WORKERS = int(os.environ.get("EVAL_WORKERS", 10))
GAMES_PER_EVAL_WORKER = int(os.environ.get("GAMES_PER_EVAL_WORKER", 4))
STOCKFISH_GAMES = int(os.environ.get("STOCKFISH_GAMES", 20))
# Stockfish workers run a full Stockfish engine each (CPU-heavy), so decouple from arena's
# EVAL_WORKERS. Falls back to EVAL_WORKERS if unset.
STOCKFISH_WORKERS = int(os.environ.get("STOCKFISH_WORKERS", EVAL_WORKERS))
STOCKFISH_ELO = int(os.environ.get("STOCKFISH_ELO", 1320))
# Stockfish move budget: fixed NODES (reproducible across machines/versions) when >0, else the
# legacy wall-time (0.1s, CPU-dependent). Set to the A_low anchor (anchors.json: 100000) so the
# loop's per-iter Elo lands on the SAME scale as the round-robin ladder. UCI_Elo limiting still
# applies on top (skill cap), so this just fixes the search budget reproducibly.
STOCKFISH_NODES = int(os.environ.get("STOCKFISH_NODES", "0"))
# Measure the CANDIDATE's Elo vs Stockfish EVERY iteration (not just on promotion) — an absolute-
# strength trend independent of the promotion gate, so we can see if the loop is climbing even when
# nothing promotes. Default on.
STOCKFISH_EVERY_ITER = os.environ.get("STOCKFISH_EVERY_ITER", "1") == "1"
# Skip Phase 3 (arena + Stockfish eval) entirely — for local self-play/train validation.
SKIP_EVAL = os.environ.get("SKIP_EVAL", "0") == "1"
# Run eval but NEVER overwrite best_model.pth (dry-run promotion) — for testing the eval phase
# without mutating the champion.
NO_PROMOTE = os.environ.get("NO_PROMOTE", "0") == "1"
# Arena promotion gate + early-stop. Once the full-arena outcome is mathematically decided —
# candidate can't reach the gate even winning every remaining game (reject), or has already
# clinched it even losing every remaining game (promote) — signal workers to stop so the saved L4
# time goes to self-play. Zero risk: both are hard bounds on the final win-rate. See
# local/plans/arena-early-stop.md.
PROMOTION_WIN_RATE = float(os.environ.get("PROMOTION_WIN_RATE", 0.55))
ARENA_EARLY_STOP = os.environ.get("ARENA_EARLY_STOP", "1") == "1"
# Run the offline probes (probe_all + search_probe) automatically each iteration before the arena,
# logging the headline markers to metrics.json for a per-iteration trend. See
# local/plans/baked-in-probes.md.
PROBE_ON_ITER = os.environ.get("PROBE_ON_ITER", "1") == "1"

# --- RULES ---
MAX_MOVES_PER_GAME = int(os.environ.get("MAX_MOVES_PER_GAME", 800))
EVAL_MAX_MOVES_PER_GAME = int(os.environ.get("EVAL_MAX_MOVES_PER_GAME", 800))

# --- WORKER PACING ---
# Slow workers hold back fast ones: a worker won't start game N+1 until the
# slowest worker has completed at least game (N+1 - MAX_WORKER_LEAD).
MAX_WORKER_LEAD = int(os.environ.get("MAX_WORKER_LEAD", 5))

current_iter = get_start_iteration(DATA_DIR) - 1

# Training
TRAIN_EPOCHS = int(os.environ.get("TRAIN_EPOCHS", 4))
TRAIN_WINDOW = int(os.environ.get("TRAIN_WINDOW", 20))   # train on the last 20 iterations' self-play data
# Train-from-lineage (AlphaZero continual): each iter builds on the latest candidate, not the frozen
# champion, so gains accumulate in the weights. Anchor stays pinned to a frozen pretrained snapshot.
TRAIN_FROM_LINEAGE = os.environ.get("TRAIN_FROM_LINEAGE", "0") == "1"
# TRAIN_BATCH_SIZE is sized for the 22 GB L4 (uses ~20 GB). Override low (e.g. 128) for
# small local GPUs — backprop activation memory across 20 blocks scales with batch size.
TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", 2048))
TRAIN_LR = float(os.environ.get("TRAIN_LR", 3e-4))   # gentler AdamW fine-tune from pretrained; fewer fp16 overflows
# Warmup: generate self-play only until this many iterations of data exist, then start training.
# Avoids overwriting the pretrained net on a tiny first-iteration window. 1 = train every iter.
MIN_TRAIN_ITERS = int(os.environ.get("MIN_TRAIN_ITERS", 1))

# --- DRY WORKER WRAPPERS ---

def run_stockfish_server_worker(worker_id, in_q, out_q, n_games, stockfish_path, sims, worker_batch,
                                sf_elo, max_moves, result_q, shm_inp, shm_pol, shm_val,
                                tally=None, total_games=0):
    """Play n_games of Model-vs-Stockfish. The model's MCTS inference is routed to the shared GPU
    InferenceServer (batched, like self-play); Stockfish runs on CPU as the opponent. Pushes one
    result dict (or None on failure) per game to result_q."""
    setup_child_logging()
    import chess, chess.engine, chess.pgn, io
    from datetime import datetime
    worker = MCTSWorker(worker_id, in_q, out_q, simulations=sims, batch_size=worker_batch,
                        seed=worker_id + int(time.time()) % 10000,
                        shm_inp=shm_inp, shm_pol=shm_pol, shm_val=shm_val)
    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        try:
            if "UCI_Elo" not in engine.options:
                raise RuntimeError("engine exposes no UCI_Elo option")
            engine.configure({"UCI_LimitStrength": True, "UCI_Elo": sf_elo})
        except Exception as e:
            # Could NOT apply the strength limit (e.g. SF 14.x floor is 1350 ≠ SF 16's 1320, or no
            # UCI_Elo). Running unlimited gives a garbage 0/N anchor — SKIP Stockfish this iteration
            # instead (surface the error, return None for every game so the eval reports "skipped").
            if worker_id == 0:
                print(f"[SF] ❌ UCI_Elo={sf_elo} could not be applied ({e}) — SKIPPING Stockfish this "
                      f"iteration (refusing to measure against an UNLIMITED engine).")
            try:
                engine.quit()
            except Exception:
                pass
            for _ in range(n_games):
                result_q.put(None)
            return
    except Exception as e:
        print(f"[SF Worker {worker_id}] engine launch failed: {e}")
        for _ in range(n_games):
            result_q.put(None)
        return

    for gi in range(n_games):
        game_id = worker_id * n_games + gi
        game = ChessGame()
        agent_is_white = (game_id % 2 == 0)
        history = collections.deque(maxlen=7)
        worker.reset_cache()
        try:
            while not game.is_over and len(game.moves) < max_moves:
                is_agent = ((game.board.turn == chess.WHITE) == agent_is_white)
                if is_agent:
                    mv, _ = worker.search(game, temperature=0.0, history=list(history), use_dirichlet=False)
                else:
                    _sf_limit = chess.engine.Limit(nodes=STOCKFISH_NODES) if STOCKFISH_NODES > 0 \
                        else chess.engine.Limit(time=0.1)
                    mv = engine.play(game.board, _sf_limit).move.uci()
                if mv is None:
                    break
                worker.advance_root(mv)
                history.appendleft(game.board.copy())
                game.push(mv)
        except Exception as e:
            print(f"[SF Worker {worker_id}] game {game_id} error: {e}")
            result_q.put(None)
            continue

        result = game.result
        white = "Model" if agent_is_white else "Stockfish"
        black = "Stockfish" if agent_is_white else "Model"
        pgn = chess.pgn.Game()
        pgn.headers["Event"] = "Model vs Stockfish"
        pgn.headers["Date"]  = datetime.now().strftime("%Y.%m.%d")
        pgn.headers["Round"] = str(game_id + 1)
        pgn.headers["White"] = white
        pgn.headers["Black"] = black
        pgn.headers["Result"] = result
        node = pgn
        for m in game.moves:
            try:
                node = node.add_variation(chess.Move.from_uci(m))
            except Exception:
                pass
        buf = io.StringIO(); print(pgn, file=buf, end="\n\n")
        result_q.put({"result": result, "pgn_str": buf.getvalue(),
                      "agent_is_white": agent_is_white, "game_id": game_id, "num_moves": len(game.moves)})
        # Running GLOBAL tally across all SF workers (shared mp.Array) + model WR — same format and
        # formula as the arena: W is from the MODEL's POV, WR = (W + 0.5D + 0.5FD) / total.
        if   result == "1-0": cat = 0 if agent_is_white else 2   # model win / loss
        elif result == "0-1": cat = 2 if agent_is_white else 0   # model loss / win
        elif result == "*":   cat = 3                            # reached move cap = forced draw
        else:                 cat = 1                            # draw
        line = f"[SF Worker {worker_id}] Game {game_id+1}: {result} ({white} vs {black}) | Moves: {len(game.moves)}"
        if tally is not None:
            with tally.get_lock():
                tally[cat] += 1
                tw, td, tl, tfd = tally[0], tally[1], tally[2], tally[3]
            tot = tw + td + tl + tfd
            wr = (tw + 0.5 * td + 0.5 * tfd) / tot if tot else 0.0
            line += f"  ||  W-D-L-FD {tw}-{td}-{tl}-{tfd}  WR {100*wr:4.1f}%  ({tot}/{total_games})"
        print(line)
    try:
        engine.quit()
    except Exception:
        pass


def run_stockfish_eval_gpu(model_path, num_games, stockfish_path, sims, sf_elo, max_moves, pgn_path):
    """Server-routed Stockfish eval: the evaluated model runs on ONE GPU InferenceServer (batched),
    Stockfish on CPU. Returns the Elo result dict (or None if no games completed)."""
    from game_engine.bayeselo_runner import BayesEloRunner
    if not os.path.exists(stockfish_path):
        print(f"❌ Stockfish not found at {stockfish_path}")
        return None

    n_workers = max(1, min(num_games, STOCKFISH_WORKERS))
    per = [num_games // n_workers + (1 if i < num_games % n_workers else 0) for i in range(n_workers)]

    print(f"  Stockfish eval: model on GPU server + {n_workers} CPU-Stockfish workers ({num_games} games)")
    # Size the server batch to THIS phase's worker count — self-play's CUDA_BATCH_SIZE oversizes it
    # (the server then never fills a batch and only timeout-flushes). Matches self-play's invariant.
    eval_cuda_batch = n_workers * WORKER_BATCH_SIZE
    server, sp, wq, (shm_inp, shm_pol, shm_val) = _start_inference_server(
        model_path, n_workers, WORKER_BATCH_SIZE, eval_cuda_batch, CUDA_STREAMS, CUDA_TIMEOUT_INFERENCE)

    result_q = mp.Queue()
    sf_tally = mp.Array('i', 4)   # shared running [model_wins, draws, losses, forced_draws] for the live line
    workers = []
    for i in range(n_workers):
        p = mp.Process(target=run_stockfish_server_worker,
                       args=(i, server.input_queue, wq[i], per[i], stockfish_path, sims,
                             WORKER_BATCH_SIZE, sf_elo, max_moves, result_q, shm_inp, shm_pol, shm_val,
                             sf_tally, num_games))
        p.start(); workers.append(p)

    results = []
    deadline = time.time() + 1800   # global cap so one slow game can't abort the whole collection
    for _ in range(num_games):
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        try:
            r = result_q.get(timeout=min(900, remaining))
            if r is not None:
                results.append(r)
        except Exception:
            continue   # a single slow/missing game shouldn't drop all subsequent finished games
    for p in workers:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate(); p.join(timeout=5)
    _stop_inference_server(server, sp)

    if not results:
        print("❌ All Stockfish eval games failed")
        return None

    w = l = d = fd = 0
    parts = []
    for r in sorted(results, key=lambda x: x["game_id"]):
        parts.append(r["pgn_str"]); res = r["result"]; aw = r["agent_is_white"]
        if res == "1-0":
            if aw: w += 1
            else:  l += 1
        elif res == "0-1":
            if aw: l += 1
            else:  w += 1
        elif res == "*":
            fd += 1            # reached the move cap w/o a decisive result = forced draw (matches arena)
        else:
            d += 1

    os.makedirs(os.path.dirname(pgn_path), exist_ok=True)
    with open(pgn_path, "w") as f:
        f.write("".join(parts))
    print(f"  Saved {len(results)} games to {pgn_path}  (W{w}/D{d}/FD{fd}/L{l})")

    n = len(results)
    # Sweep guard (NOT a formula — a degenerate-case detector): BayesElo returns a prior-capped number
    # on a 0%/100% result, which is pure noise. Null it and flag the anchor instead of recording a fake.
    if n and (w == n or l == n):
        kind = "raise" if w == n else "lower"
        note = f"{'100%' if w == n else '0%'} sweep over {n} games — Elo unbounded, {kind} the anchor"
        print(f"  [Elo] not measurable — {note}")
        elo = elo_lo = elo_hi = None
    else:
        # Third-party BayesElo is the sole Elo source. Needs a NATIVE binary (run_gcp.sh builds it).
        r = BayesEloRunner(stockfish_elo=sf_elo).run(pgn_path)
        if r and r.get('model_elo') is not None:
            elo = r['model_elo']; elo_lo = r.get('model_ci_lower'); elo_hi = r.get('model_ci_upper')
            note = f"BayesElo {elo:.0f} vs SF {sf_elo} over {n} games"
            print(f"  [Elo] {elo:.0f}  vs SF {sf_elo}  ({note})")
        else:
            elo = elo_lo = elo_hi = None
            note = "BayesElo failed (is the native binary built on this box? run_gcp.sh step 3b)"
            print(f"  [Elo] BayesElo failed — {note}")
    return {
        'model_elo': elo, 'elo_lower': elo_lo, 'elo_upper': elo_hi, 'elo_note': note,
        'win_count': w, 'loss_count': l, 'draw_count': d, 'forced_draw_count': fd,
        'total_games': n,
        'win_rate': (w + 0.5 * d + 0.5 * fd) / n if n else 0.0,
    }


def run_arena_server_worker(worker_id, n_games, cand_in_q, cand_out_q, champ_in_q, champ_out_q,
                            sims, worker_batch, max_moves, result_q, cand_shm, champ_shm,
                            tally=None, total_games=0, stop_flag=None):
    """Play n_games of Candidate-vs-Champion. Each model's MCTS inference is routed to its OWN
    shared GPU InferenceServer (candidate server + champion server), so the GPU holds exactly 2
    model copies regardless of worker count — vs the old inline path that loaded 2 models ×
    EVAL_WORKERS processes (OOM risk). Only one model searches at a time (turn-based), so the two
    per-worker MCTSWorkers never have overlapping in-flight requests. Pushes one aggregate result
    dict to result_q."""
    setup_child_logging()
    import chess, random
    seed = worker_id + int(time.time()) % 10000
    np.random.seed(seed); random.seed(seed)
    cand = MCTSWorker(worker_id, cand_in_q, cand_out_q, simulations=sims, batch_size=worker_batch,
                      seed=seed, shm_inp=cand_shm[0], shm_pol=cand_shm[1], shm_val=cand_shm[2])
    champ = MCTSWorker(worker_id, champ_in_q, champ_out_q, simulations=sims, batch_size=worker_batch,
                       seed=seed, shm_inp=champ_shm[0], shm_pol=champ_shm[1], shm_val=champ_shm[2])
    w = d = l = fd = 0
    try:
        for gi in range(n_games):
            if stop_flag is not None and stop_flag.value != 0:
                break  # arena early-stop: the promotion gate is already mathematically decided
            game_id = worker_id * n_games + gi
            game = ChessGame()
            cand_is_white = (game_id % 2 == 0)
            cand_label = "Cand" if cand_is_white else "Champ"
            champ_label = "Champ" if cand_is_white else "Cand"
            history = collections.deque(maxlen=7)
            cand.reset_cache(); champ.reset_cache()
            # Opening variety: one random legal first move (both trees track it).
            legal = list(game.board.legal_moves)
            if legal:
                opening = random.choice(legal).uci()
                history.appendleft(game.board.copy())
                game.push(opening)
                cand.advance_root(opening); champ.advance_root(opening)
            try:
                while not game.is_over and len(game.moves) < max_moves:
                    is_cand_turn = ((game.board.turn == chess.WHITE) == cand_is_white)
                    mover = cand if is_cand_turn else champ
                    mv, _ = mover.search(game, temperature=0.0, history=list(history), use_dirichlet=False)
                    if mv is None:
                        break
                    # Both trees track the same game line.
                    cand.advance_root(mv); champ.advance_root(mv)
                    history.appendleft(game.board.copy())
                    game.push(mv)
            except Exception as e:
                print(f"[Arena Worker {worker_id}] game {game_id} error: {e}")
                continue
            # Candidate-perspective outcome: 0=win 1=draw 2=loss 3=forced-draw
            if not game.is_over and len(game.moves) >= max_moves:
                fd += 1; cat = 3; disp = "*"; tag = " (FORCED DRAW)"
            else:
                disp = game.result; tag = ""
                if disp == "1-0":
                    cat = 0 if cand_is_white else 2
                elif disp == "0-1":
                    cat = 2 if cand_is_white else 0
                else:
                    cat = 1
                if   cat == 0: w += 1
                elif cat == 2: l += 1
                else:          d += 1
            # Running GLOBAL tally across all arena workers (shared mp.Array) + WR using the same
            # formula as the promotion gate: (W + 0.5D + 0.5FD) / total, ≥0.55 promotes.
            line = f"Arena Game {game_id}: {disp} ({cand_label} vs {champ_label}) | Moves: {len(game.moves)}{tag}"
            if tally is not None:
                with tally.get_lock():
                    tally[cat] += 1
                    tw, td, tl, tfd = tally[0], tally[1], tally[2], tally[3]
                tot = tw + td + tl + tfd
                wr = (tw + 0.5 * td + 0.5 * tfd) / tot if tot else 0.0
                line += (f"  ||  W-D-L-FD {tw}-{td}-{tl}-{tfd}  WR {100*wr:4.1f}%"
                         f"  ({tot}/{total_games})")
                # Early-stop once the FULL-total_games gate is mathematically decided. Hard bounds
                # (zero risk): score = W + 0.5(D+FD); a loss is the worst possible remaining game.
                if ARENA_EARLY_STOP and stop_flag is not None and total_games > 0:
                    score = tw + 0.5 * td + 0.5 * tfd
                    need = PROMOTION_WIN_RATE * total_games
                    if score + (total_games - tot) < need:   # can't reach the gate even winning out
                        stop_flag.value = 1                  # → reject
                    elif score >= need:                      # already clinched even losing out
                        stop_flag.value = 2                  # → promote
            print(line)
    except Exception as e:
        print(f"[Arena Worker {worker_id}] ❌ CRASHED: {e}")
        import traceback; traceback.print_exc()
    result_q.put({"wins": w, "draws": d, "losses": l, "forced_draws": fd})


def run_arena_eval_gpu(cand_model, champ_model, num_games, sims, max_moves):
    """Server-routed arena: candidate on one GPU InferenceServer, champion on another (2 model
    copies total), CPU arena workers route each move to the right server. Returns aggregate
    {wins, draws, losses, forced_draws} from the candidate's perspective."""
    n_workers = max(1, min(num_games, EVAL_WORKERS))
    per = [num_games // n_workers + (1 if i < num_games % n_workers else 0) for i in range(n_workers)]

    print(f"  Arena: candidate + champion on 2 GPU servers, {n_workers} CPU workers ({num_games} games)")
    # Size each server to THIS phase's worker count (self-play's CUDA_BATCH_SIZE oversizes it).
    arena_cuda_batch = n_workers * WORKER_BATCH_SIZE
    cand_server, cand_sp, cand_wq, cand_shm = _start_inference_server(
        cand_model, n_workers, WORKER_BATCH_SIZE, arena_cuda_batch, CUDA_STREAMS, CUDA_TIMEOUT_INFERENCE)
    champ_server, champ_sp, champ_wq, champ_shm = _start_inference_server(
        champ_model, n_workers, WORKER_BATCH_SIZE, arena_cuda_batch, CUDA_STREAMS, CUDA_TIMEOUT_INFERENCE)

    result_q = mp.Queue()
    tally = mp.Array('i', 4)   # shared running [wins, draws, losses, forced_draws] across workers
    stop_flag = mp.Value('i', 0)   # arena early-stop: 0=run, 1=eliminated(reject), 2=clinched(promote)
    workers = []
    for i in range(n_workers):
        p = mp.Process(target=run_arena_server_worker,
                       args=(i, per[i], cand_server.input_queue, cand_wq[i],
                             champ_server.input_queue, champ_wq[i], sims, WORKER_BATCH_SIZE,
                             max_moves, result_q, cand_shm, champ_shm, tally, num_games, stop_flag))
        p.start(); workers.append(p)

    totals = {"wins": 0, "draws": 0, "losses": 0, "forced_draws": 0}
    for _ in range(n_workers):
        try:
            r = result_q.get(timeout=1800)
            for k in totals:
                totals[k] += r[k]
        except Exception:
            pass
    for p in workers:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate(); p.join(timeout=5)
    _stop_inference_server(cand_server, cand_sp)
    _stop_inference_server(champ_server, champ_sp)

    # The result_q aggregate silently drops a whole worker's games if that worker times out or is
    # terminated above — biasing the promotion gate. The shared mp.Array tally is incremented
    # per-game and survives worker death, so it is the authoritative count for the gate.
    tw, td, tl, tfd = int(tally[0]), int(tally[1]), int(tally[2]), int(tally[3])
    tally_games = tw + td + tl + tfd
    es = int(stop_flag.value)
    if es == 1:
        print(f"  ⏹ Arena early-stop: candidate ELIMINATED after {tally_games}/{num_games} games "
              f"(can't reach {100*PROMOTION_WIN_RATE:.0f}% even winning out) → reject "
              f"(~{num_games - tally_games} games of self-play time reclaimed).")
    elif es == 2:
        print(f"  ⏹ Arena early-stop: candidate CLINCHED ≥{100*PROMOTION_WIN_RATE:.0f}% after "
              f"{tally_games}/{num_games} games (champion can't catch up) → promote "
              f"(~{num_games - tally_games} games of self-play time reclaimed).")
    elif tally_games != sum(totals.values()) or tally_games < num_games:
        print(f"  ⚠️ Arena count: result_q aggregate={sum(totals.values())} vs shared tally="
              f"{tally_games}/{num_games} games — using the shared tally for the gate.")
    return {"wins": tw, "draws": td, "losses": tl, "forced_draws": tfd}

# --- PHASES ---

def run_self_play_phase(iteration):
    print(f"\n=== ITERATION {iteration}: SELF-PLAY PHASE (Batched MCTS) ===")
    cleanup_memory() # Clear RAM before starting
    
    # Shared-memory transport buffers (allocated ONCE in the main process before spawn; torch's
    # reducer pickles only the shm handle when passed via Process args). Per-worker slots sized
    # to WORKER_BATCH_SIZE leaves. ~26 MB total at 64×8. None → legacy pickle-through-queue path.
    if SHM_TRANSPORT:
        shm_inp = torch.empty((NUM_WORKERS, WORKER_BATCH_SIZE, 120, 8, 8), dtype=torch.float32).share_memory_()
        shm_pol = torch.empty((NUM_WORKERS, WORKER_BATCH_SIZE, 4672), dtype=torch.float32).share_memory_()
        shm_val = torch.empty((NUM_WORKERS, WORKER_BATCH_SIZE, 3), dtype=torch.float32).share_memory_()
        print(f"SHM transport ON: buffers inp/pol/val for {NUM_WORKERS}×{WORKER_BATCH_SIZE} slots")
    else:
        shm_inp = shm_pol = shm_val = None

    server = InferenceServer(BEST_MODEL, batch_size=CUDA_BATCH_SIZE, timeout=CUDA_TIMEOUT_INFERENCE, streams=CUDA_STREAMS,
                             shm_inp=shm_inp, shm_pol=shm_pol, shm_val=shm_val)
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

    games_counter = mp.Array('i', [0] * NUM_WORKERS)

    workers = []
    for i in range(NUM_WORKERS):
        p = mp.Process(target=run_worker_batch,
                       args=(i, server.input_queue, worker_queues[i], GAMES_PER_WORKER, iteration, games_counter,
                             shm_inp, shm_pol, shm_val))
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

def _update_swa():
    """SWA (KataGo): maintain swa_model.pth = decay·swa + (1−decay)·candidate across iterations.
    Offline only (Option A) — NOT used by the pipeline; probe it vs candidate. Off by default
    (SWA_ENABLE=1 to turn on). Note: benefit is uncertain in our discrete-iteration loop (each
    candidate is an independent fine-tune of best_model, not a continuous trajectory)."""
    if os.environ.get("SWA_ENABLE", "0") != "1" or not os.path.exists(CANDIDATE_MODEL):
        return
    decay = float(os.environ.get("SWA_DECAY", 0.75))
    swa_path = f"{MODEL_DIR}/swa_model.pth"
    cand = torch.load(CANDIDATE_MODEL, map_location="cpu", weights_only=False)
    cand_sd = cand["model_state_dict"] if isinstance(cand, dict) and "model_state_dict" in cand else cand
    if not os.path.exists(swa_path):
        torch.save({"model_state_dict": cand_sd}, swa_path)
        print("  [SWA] initialized swa_model.pth from candidate"); return
    swa = torch.load(swa_path, map_location="cpu", weights_only=False)
    swa_sd = swa["model_state_dict"] if isinstance(swa, dict) and "model_state_dict" in swa else swa
    for k in swa_sd:
        if k in cand_sd and torch.is_floating_point(swa_sd[k]):
            swa_sd[k].mul_(decay).add_(cand_sd[k].float(), alpha=1.0 - decay)
    torch.save({"model_state_dict": swa_sd}, swa_path)
    print(f"  [SWA] updated swa_model.pth (decay={decay})")


def run_training_phase(iteration):
    print(f"\n=== ITERATION {iteration}: TRAINING PHASE ===")
    cleanup_memory()  # Clear VRAM before training

    # Wait for the inference server subprocess's CUDA memory to be reclaimed by the
    # GPU driver before allocating training tensors. The server process has exited by
    # this point but the driver can take a few seconds to return pages to the free pool.
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        free_gb = free / (1024 ** 3)
        total_gb = total / (1024 ** 3)
        # Target enough free VRAM for training. 8 GB suits the 22 GB L4; on a small GPU
        # (e.g. 6 GB laptop) 8 GB is unreachable, so scale to 70% of total. Env-overridable.
        target_gb = float(os.environ.get("VRAM_DRAIN_TARGET_GB", min(8.0, total_gb * 0.7)))
        if free_gb < target_gb:
            print(f"  GPU: {free_gb:.1f}/{total_gb:.1f} GB free — waiting for VRAM to drain (target {target_gb:.1f})...")
            deadline = time.time() + 120
            while time.time() < deadline:
                time.sleep(5)
                torch.cuda.empty_cache()
                free, total = torch.cuda.mem_get_info(0)
                free_gb = free / (1024 ** 3)
                print(f"  GPU: {free_gb:.1f} GB free")
                if free_gb >= target_gb:
                    break
            if free_gb < target_gb:
                print(f"  ⚠️ VRAM did not drain after 120s — proceeding ({free_gb:.1f} GB free)")
    
    # Frozen KL-anchor reference = a one-time pretrained snapshot. best_model is pretrained until the
    # first promotion, so capture it now (before any 0.50-gate promotion can change it).
    pretrained_anchor = f"{MODEL_DIR}/pretrained_anchor.pth"
    if not os.path.exists(pretrained_anchor):
        shutil.copy(BEST_MODEL, pretrained_anchor)
        print(f"  Captured frozen KL-anchor reference → {pretrained_anchor}")

    # Train-from-lineage: build on the latest candidate (continual), not the frozen champion, so
    # gains compound. Anchor stays pinned to pretrained for stability.
    train_base = BEST_MODEL
    if TRAIN_FROM_LINEAGE and os.path.exists(CANDIDATE_MODEL):
        train_base = CANDIDATE_MODEL
        print(f"  Train-from-lineage: base = candidate.pth (anchor pinned to pretrained)")

    p_loss, v_loss, train_metrics = train_model(data_path=DATA_DIR,
                input_model_path=train_base,
                output_model_path=CANDIDATE_MODEL,
                epochs=TRAIN_EPOCHS,
                batch_size=TRAIN_BATCH_SIZE,
                lr=TRAIN_LR,
                window_size=TRAIN_WINDOW,
                total_iterations=ITERATIONS,
                anchor_model_path=pretrained_anchor)

    _update_swa()
    return p_loss, v_loss, train_metrics

def run_probes(iteration):
    """Offline diagnostics (read-only) run before the arena: value-head calibration vs the champion
    anchor (probe_all) and policy-target sharpness (search_probe). Subprocess the standalone scripts
    so their full output lands in training_log.txt, then parse the headline markers for metrics.json
    (a per-iteration trend). NEVER raises — diagnostics must not break the iteration. CPU-only, so it
    won't contend with the arena's GPU servers. See local/plans/baked-in-probes.md."""
    import subprocess, re
    if not PROBE_ON_ITER:
        return None
    here = os.path.dirname(os.path.abspath(__file__))   # .../game_engine
    cwd = os.path.dirname(here)                          # .../chess_ai (scripts use game_engine/ paths)
    markers = {}
    print(f"\n=== ITERATION {iteration}: PROBES (offline diagnostics) ===")

    # probe_all: champion (best_model) anchor vs candidate (+ swa) on the window. value_acc_gap =
    # cand − champ is the drift-controlled value-head marker.
    try:
        cmd = [sys.executable, "game_engine/probe_all.py",
               f"champ={BEST_MODEL}", f"cand={CANDIDATE_MODEL}", f"swa={MODEL_DIR}/swa_model.pth",
               "--data", DATA_DIR, "--games-per-iter", "60"]
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=900).stdout
        print(out)
        accs = dict(re.findall(r"^\s*(\w+)\s+mean\|v\|=.*?acc=\s*(\d+)%", out, re.M))
        if "cand" in accs and "champ" in accs:
            markers["value_acc_cand"] = int(accs["cand"])
            markers["value_acc_champ"] = int(accs["champ"])
            markers["value_acc_gap"] = int(accs["cand"]) - int(accs["champ"])
        # Value-head sharpness (mean|v|) and decisiveness (|v|>=0.9 share) per model — the headline
        # value-head trend for the cand-vs-champ comparison chart.
        meanv = dict(re.findall(r"^\s*(\w+)\s+mean\|v\|=([\d.]+)", out, re.M))
        dec   = dict(re.findall(r"^\s*(\w+)\s+mean\|v\|=.*?\|v\|>=0\.9:\s*(\d+)%", out, re.M))
        for who in ("cand", "champ"):
            if who in meanv: markers[f"value_meanv_{who}"] = float(meanv[who])
            if who in dec:   markers[f"value_decisive_{who}"] = int(dec[who])
    except Exception as e:
        print(f"  [probe] probe_all failed: {e}")

    # search_probe: candidate vs the stored MCTS targets on THIS iter's data (meaningful from iter-10).
    try:
        cmd = [sys.executable, "game_engine/search_probe.py",
               "--model", CANDIDATE_MODEL, "--data", DATA_DIR, "--iter", str(iteration)]
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=900).stdout
        print(out)
        mkl = re.search(r"KL\(MCTS.*?:\s*([\d.]+)\s*nats", out)
        mov = re.search(r"override.*?:\s*([\d.]+)%", out)
        mt1 = re.search(r"top1-agree.*?:\s*([\d.]+)%", out)
        mde = re.search(r"Δ=([+\-]?[\d.]+)", out)
        if mkl: markers["search_kl"] = float(mkl.group(1))
        if mov: markers["search_override"] = float(mov.group(1))
        if mt1: markers["search_top1"] = float(mt1.group(1))
        if mde: markers["search_dentropy"] = float(mde.group(1))
    except Exception as e:
        print(f"  [probe] search_probe failed: {e}")

    print(f"  [probe] markers iter {iteration}: {markers}")
    return markers or None


def run_evaluation_phase(iteration, logger, p_loss, v_loss, train_metrics=None):
    print(f"\n=== ITERATION {iteration}: EVALUATION PHASE ===")
    probe_markers = run_probes(iteration)
    cleanup_memory() # Clear VRAM before launching multiple evaluation workers

    # 1. ARENA EVALUATION — candidate vs champion, each on its own GPU InferenceServer.
    total_games = EVAL_WORKERS * GAMES_PER_EVAL_WORKER
    print(f" [Arena] Playing {total_games} games (candidate vs champion, 2 GPU servers)...")
    arena = run_arena_eval_gpu(CANDIDATE_MODEL, BEST_MODEL, total_games,
                               EVAL_SIMULATIONS, EVAL_MAX_MOVES_PER_GAME)
    total_wins, total_draws = arena['wins'], arena['draws']
    total_losses, total_forced_draws = arena['losses'], arena['forced_draws']

    total_score = total_wins + 0.5 * total_forced_draws + 0.5 * total_draws
    total_game_count = total_wins + total_draws + total_forced_draws + total_losses
    win_rate = total_score / total_game_count if total_game_count > 0 else 0

    print(f" [Arena] Final Result: {win_rate*100:.1f}% Win Rate ({total_wins}W - {total_draws}D - {total_forced_draws}FD - {total_losses}L)")

    est_elo = None
    arena_promoted = False

    # 2. PROMOTION — the arena win-rate is the SOLE promotion gate.
    if win_rate >= PROMOTION_WIN_RATE:
        print(f" [Arena] ⭐ Candidate PROMOTED! (WR >= {100*PROMOTION_WIN_RATE:.0f}%) ⭐")
        _promote_best()
        arena_promoted = True
    else:
        print(f" [Arena] Candidate rejected (WR < {100*PROMOTION_WIN_RATE:.0f}%).")

    # 3. STOCKFISH EVALUATION — Elo vs Stockfish. With STOCKFISH_EVERY_ITER (default on) we measure
    #    the CANDIDATE every iteration to get an absolute-strength trend independent of the promotion
    #    gate (after a promotion the candidate IS the new champion, so it doubles as the champion's
    #    Elo). Metrics-only: it never promotes. NO_PROMOTE is the dry-run/test mode.
    if STOCKFISH_EVERY_ITER or arena_promoted or NO_PROMOTE:
        tag = "new champion" if arena_promoted else "reigning champion"
        print(f" [Stockfish/BayesElo] Playing {STOCKFISH_GAMES} games vs Elo {STOCKFISH_ELO} ({tag})...")
        cleanup_memory()
        try:
            pgn_path = f"game_engine/evaluation/pgn/iter_{iteration}_{int(time.time())}.pgn"
            # Measure best_model (the champion / self-play generator): _promote_best() already ran, so
            # it's the new champion if promoted, else the reigning one — a rejected candidate is
            # discarded and its Elo is meaningless. Ties the Elo trend to actual strength. EXCEPTION:
            # NO_PROMOTE never updates best_model, so the dry-run measures the candidate under test.
            sf_target = CANDIDATE_MODEL if NO_PROMOTE else BEST_MODEL
            bayeselo_results = run_stockfish_eval_gpu(
                model_path=sf_target,
                num_games=STOCKFISH_GAMES,
                stockfish_path=STOCKFISH_PATH,
                sims=EVAL_SIMULATIONS,
                sf_elo=STOCKFISH_ELO,
                max_moves=EVAL_MAX_MOVES_PER_GAME,
                pgn_path=pgn_path,
            )
            if bayeselo_results:
                est_elo = bayeselo_results['model_elo']
                sf_w = bayeselo_results['win_count']; sf_d = bayeselo_results['draw_count']
                sf_fd = bayeselo_results.get('forced_draw_count', 0); sf_l = bayeselo_results['loss_count']
                sf_total = sf_w + sf_d + sf_fd + sf_l
                sf_wr = (sf_w + 0.5 * sf_fd + 0.5 * sf_d) / sf_total if sf_total > 0 else 0
                note = bayeselo_results.get('elo_note', '')
                print(f" [Stockfish] Final Result: {sf_wr*100:.1f}% Win Rate ({sf_w}W - {sf_d}D - {sf_fd}FD - {sf_l}L)")
                if est_elo is not None:
                    print(f" [Elo] ✅ {tag.capitalize()} Elo: {est_elo:.0f}  ({note})")
                else:                       # 0%/100% sweep — unbounded, logged as unmeasured
                    print(f" [Elo] ⚠️  Not measurable — {note}")
            else:
                print(f" [Elo] ❌ No games completed")
        except Exception as e:
            print(f" [Elo] ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            est_elo = None
    else:
        print(f" [Stockfish] Skipped (STOCKFISH_EVERY_ITER=0, no promotion).")

    logger.log(iteration, p_loss, v_loss, win_rate, est_elo, stockfish_elo=STOCKFISH_ELO,
               probe=probe_markers, train=train_metrics)

if __name__ == "__main__":
    setup_child_logging()
    mp.set_start_method('spawn', force=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    if not os.path.exists(BEST_MODEL):
        print("Initializing random model...")
        torch.save(ChessCNN().state_dict(), BEST_MODEL)

    timeout_handler.start()
    print(f"⏱️ Deadlock timeout: {timeout_handler.timeout_seconds/3600:.0f}h per iteration")

    # RESUMPTION LOGIC
    start_iter, start_phase = load_resume_point()

    print("=" * 60)
    print(f"STARTING RUN")
    print(f"Resuming from Iteration: {start_iter} (phase {start_phase}: {_PHASES[start_phase-1]})")
    print(f"Workers: {NUM_WORKERS} | Sims: {SIMULATIONS} | Batch: {WORKER_BATCH_SIZE}")
    print("=" * 60)

    killer = GracefulKiller()

    def _phase_dur(secs):
        h, rem = divmod(secs, 3600); m, s = divmod(rem, 60)
        return f"{int(h)}h {int(m)}m {s:.0f}s" if h >= 1 else f"{int(m)}m {s:.0f}s"
    
    try:
        for it in range(start_iter, ITERATIONS + 1):
            
            # CHECK BEFORE ITERATION STARTS
            if killer.kill_now:
                print("\n[Main] ⚠️  Kill signal received BEFORE iteration start")
                print(f"[Main] Gracefully exiting. Next run will resume from Iteration {it}")
                break
            
            iter_start = time.time()
            # Only the first (resumed) iteration may start mid-way; every later iteration
            # runs all phases from the top. p_loss/v_loss default for the resume-into-eval case.
            resume_phase = start_phase if it == start_iter else 1
            p_loss, v_loss = 0.0, 0.0
            train_metrics = {}

            # === PHASE 1: SELF-PLAY ===
            if resume_phase <= 1:
                print(f"\n{'='*60}")
                print(f"ITERATION {it} - PHASE 1: SELF-PLAY")
                print(f"{'='*60}")
                _t_phase = time.time()

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
                print(f"⏱️  ITER {it} self-play took {_phase_dur(time.time() - _t_phase)}")
                save_phase(it, "self_play")

            # CHECK AFTER PHASE 1
            if killer.kill_now:
                print("\n[Main] ⚠️  Kill signal received AFTER Phase 1")
                print("[Main] Saving state and exiting. Training/Eval will resume on next startup.")
                break

            # Lever 3: warmup — skip training/eval until enough iterations of data accumulate
            # (champion unchanged; just keep generating self-play to fill the replay window).
            do_train = it >= MIN_TRAIN_ITERS
            if resume_phase <= 2 and not do_train:
                print(f"\n⏭  ITERATION {it} - WARMUP: {it}/{MIN_TRAIN_ITERS} data iters — self-play only, skipping training+eval")
                save_phase(it, "eval")

            # === PHASE 2: TRAINING ===
            if resume_phase <= 2 and do_train:
                print(f"\n{'='*60}")
                print(f"ITERATION {it} - PHASE 2: TRAINING")
                print(f"{'='*60}")
                _t_phase = time.time()

                try:
                    p_loss, v_loss, train_metrics = run_training_phase(it)
                    print(f"\n✅ ITERATION {it} - PHASE 2 COMPLETE (Policy Loss: {p_loss:.4f}, Value Loss: {v_loss:.4f})")
                except Exception as e:
                    print(f"\n❌ ITERATION {it} - PHASE 2 FAILED: {e}")
                    if killer.kill_now:
                        print("[Main] Kill signal during Phase 2. Exiting...")
                        break
                    raise
                print(f"⏱️  ITER {it} training took {_phase_dur(time.time() - _t_phase)}")
                save_phase(it, "training")

            # CHECK AFTER PHASE 2
            if killer.kill_now:
                print("\n[Main] ⚠️  Kill signal received AFTER Phase 2")
                print("[Main] Saving state and exiting. Eval will resume on next startup.")
                break

            # === PHASE 3: EVALUATION ===
            if resume_phase <= 3 and do_train:
                _t_phase = time.time()
                if SKIP_EVAL:
                    print(f"\n{'='*60}")
                    print(f"ITERATION {it} - PHASE 3: EVALUATION SKIPPED (SKIP_EVAL=1)")
                    print(f"{'='*60}")
                else:
                    print(f"\n{'='*60}")
                    print(f"ITERATION {it} - PHASE 3: EVALUATION")
                    print(f"{'='*60}")

                    try:
                        run_evaluation_phase(it, MetricsLogger(), p_loss, v_loss, train_metrics)
                        print(f"\n✅ ITERATION {it} - PHASE 3 COMPLETE")
                    except Exception as e:
                        print(f"\n❌ ITERATION {it} - PHASE 3 FAILED: {e}")
                        if killer.kill_now:
                            print("[Main] Kill signal during Phase 3. Exiting...")
                            break
                        raise
                print(f"⏱️  ITER {it} evaluation took {_phase_dur(time.time() - _t_phase)}")
                save_phase(it, "eval")
            
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
