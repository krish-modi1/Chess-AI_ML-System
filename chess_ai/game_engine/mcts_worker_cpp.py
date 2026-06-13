"""
MCTSWorker wrapper for C++ MCTS backend with BATCHED INFERENCE
Matches the interface of mcts.py MCTSWorker class exactly

KEY FIX: This version uses a callback-based architecture where:
1. C++ handles fast tree traversal (selection, PUCT, backprop)
2. Python handles batched neural network inference via GPU server
3. Each MCTS iteration batch gets REAL neural network evaluations

Performance comparison:
- Old C++ version: Only root inference, uniform policy for leaves
- New C++ version: Full batched inference for ALL leaves
- Result: Proper MCTS with neural network guidance at every node
"""

import numpy as np
import torch
import sys
import os
import time

# Max wait for ONE inference round-trip (not per-move). Individual calls stay fast even with
# heavy worker oversubscription — the server clears all pending workers in ~1 batch — so 300s
# only ever fires on a genuinely dead/hung server. Env-tunable for safety under unusual load.
_INFERENCE_TIMEOUT_MS = int(os.environ.get("INFERENCE_TIMEOUT_MS", 300000))

sys.path.append(os.getcwd())

import mcts_engine_cpp


def _chess_game_to_cpp_board(chess_game):
    """Build ChessBoard by replaying all moves from the starting position.

    Creates from FEN alone would reset the chess-library's repetition history.
    Replaying preserves it, which matters for isRepetition() in to_tensor().
    """
    cpp_board = mcts_engine_cpp.ChessBoard()
    for move_uci in chess_game.moves:
        cpp_board.push(move_uci)
    return cpp_board


def _cpp_board_from_history(b):
    """Build a ChessBoard for a history position by REPLAYING its move stack, so repetition
    tracking (isRepetition() in to_tensor) is preserved. Constructing from FEN alone resets the
    chess-library's repetition history → the history frames' repetition planes would be all-zero.
    """
    cb = mcts_engine_cpp.ChessBoard()
    for mv in b.move_stack:
        cb.push(mv.uci())
    return cb


class MCTSWorker:
    """
    MCTS Worker using C++ backend with callback-based batched inference.

    Architecture:
    ─────────────────────────────────────────────────────────────────────

    OLD (Broken):
        Python → C++ search(root_state, root_policy, root_value)
                      │
                      └─► C++ runs entire MCTS with uniform_policy
                          (NO neural network for leaves!)

    NEW (Fixed):
        Python → C++ search(root_state, root_policy, root_value, callback)
                      │
                      ├─► C++ Selection Phase (fast tree traversal)
                      │
                      ├─► C++ calls callback(leaf_states)
                      │         │
                      │         └─► Python batches to GPU server
                      │                    │
                      │         ◄──────────┘ returns (policies, values)
                      │
                      └─► C++ Expansion & Backprop with REAL NN values

    ─────────────────────────────────────────────────────────────────────
    """

    def __init__(self, worker_id, input_queue, output_queue, simulations=800, batch_size=8, seed=0,
                 shm_inp=None, shm_pol=None, shm_val=None):
        """
        Initialize MCTSWorker with C++ backend.

        Args:
            worker_id: Unique ID for this worker (for queue routing)
            input_queue: Queue to send inference requests to GPU server
            output_queue: Queue to receive inference results from GPU server
            simulations: Total number of MCTS simulations
            batch_size: Number of leaves to evaluate per iteration
            seed: Random seed for exploration diversity
            shm_inp/shm_pol/shm_val: shared-memory transport buffers
                (NUM_WORKERS, WORKER_BATCH_SIZE, …). When present, leaf/result tensors are
                exchanged via this worker's per-id slot and only (worker_id, N) signals go
                through the queues. None → legacy pickle-through-queue path.
        """
        self.worker_id = worker_id
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.simulations = simulations
        self.batch_size = batch_size
        self.cpu = 1.0
        self.seed = seed

        # Per-worker shared-memory views (this worker only ever touches row `worker_id`).
        self.shm = shm_inp is not None
        if self.shm:
            assert shm_inp.is_shared(), "shm_inp not shared — check Process args under spawn"
            self._inp_view = shm_inp[worker_id]   # (WORKER_BATCH_SIZE, 120, 8, 8)
            self._pol_view = shm_pol[worker_id]   # (WORKER_BATCH_SIZE, 4672)
            self._val_view = shm_val[worker_id]   # (WORKER_BATCH_SIZE, 3)

        # Root NN value (P(win)-P(loss), side-to-move POV, in [-1,1]) from the most recent
        # search(). Sim-independent (computed from the single root inference), so it stays a
        # reliable "is this position decided?" signal even when simulations are reduced.
        self.last_root_value = 0.0

        # History for current game — list of ChessBoard copies, most recent first.
        # Set by search() on each call; consumed by _batch_inference_callback.
        self._cpp_history = []

        # Create C++ MCTS engine instance
        self.mcts_engine = mcts_engine_cpp.MCTSEngine(simulations, batch_size)

    # ═══════════════════════════════════════════════════════════════════════════
    # NON-BLOCKING QUEUE POLLING
    # ═══════════════════════════════════════════════════════════════════════════

    def _get_inference_result(self, timeout_ms=_INFERENCE_TIMEOUT_MS):
        """
        Poll for inference results with non-blocking timeout.

        This prevents workers from blocking indefinitely and allows
        the queue to fill properly with batched requests.
        """
        start_time = time.time()

        while True:
            try:
                # Poll with 1ms timeout
                result = self.output_queue.get(timeout=0.001)
                return result
            except:
                elapsed_ms = (time.time() - start_time) * 1000
                if elapsed_ms > timeout_ms:
                    raise TimeoutError(
                        f"[Worker {self.worker_id}] No inference response in {timeout_ms}ms"
                    )

    def _flush_output_queue(self):
        """Discard any stale result still in the queue (e.g. a late arrival from a previously
        timed-out request). Called before issuing a new request so the single-outstanding-request
        invariant can't desync the SHM slot ↔ response pairing after a timeout."""
        try:
            while True:
                self.output_queue.get_nowait()
        except Exception:
            pass

    # ═══════════════════════════════════════════════════════════════════════════
    # INFERENCE CALLBACK - Called by C++ during MCTS search
    # ═══════════════════════════════════════════════════════════════════════════

    def _batch_inference_callback(self, leaf_states):
        """
        Called by C++ with a batch of leaf positions.
        leaf_states is List[mcts_engine_cpp.ChessBoard] (C++ objects exposed to Python).
        Uses self._cpp_history set at search() start (frozen at root time) — leaf nodes at
        depth N use root-era history, which is an approximation that worsens with tree depth
        but matches the behavior of the previous Python implementation.
        """
        batch_size = len(leaf_states)

        if batch_size == 0:
            return (
                np.zeros((0, 4672), dtype=np.float32),
                np.zeros((0,), dtype=np.float32)
            )

        tensors = [state.to_tensor(self._cpp_history) for state in leaf_states]
        arr = np.array(tensors)   # (N, 120, 8, 8) float32

        # ── Shared-memory path ───────────────────────────────────────────────────
        if self.shm:
            n = batch_size
            self._inp_view[:n] = torch.from_numpy(arr)          # write our slot
            self._flush_output_queue()                          # drop any stale late result
            self.input_queue.put((self.worker_id, n))           # tiny signal, no tensor
            try:
                self._get_inference_result()                    # block for (n,) sentinel
            except TimeoutError:
                print(f"[Worker {self.worker_id}] ❌ Inference timeout in callback")
                return (
                    np.zeros((batch_size, 4672), dtype=np.float32),
                    np.zeros((batch_size,), dtype=np.float32)
                )
            # .clone() is mandatory: the next request will overwrite this slot.
            policies_np = self._pol_view[:n].clone().numpy()
            v = self._val_view[:n].clone()                      # (n, 3) raw logits
            probs = torch.softmax(v, dim=1)
            values_scalar = (probs[:, 0] - probs[:, 2]).numpy().astype(np.float32)
            return (policies_np, values_scalar)

        # ── Legacy path ──────────────────────────────────────────────────────────
        batch_tensor = torch.from_numpy(arr)

        self._flush_output_queue()
        self.input_queue.put((self.worker_id, batch_tensor))

        try:
            policies, values = self._get_inference_result()
        except TimeoutError:
            print(f"[Worker {self.worker_id}] ❌ Inference timeout in callback")
            return (
                np.zeros((batch_size, 4672), dtype=np.float32),
                np.zeros((batch_size,), dtype=np.float32)
            )

        if isinstance(policies, torch.Tensor):
            policies_np = policies.detach().cpu().numpy()
        else:
            policies_np = np.array(policies, dtype=np.float32)

        if policies_np.ndim == 1:
            policies_np = policies_np.reshape(1, -1)

        if isinstance(values, torch.Tensor):
            v = values.detach().cpu()
        else:
            v = torch.from_numpy(np.array(values, dtype=np.float32))
        if v.ndim == 1:
            v = v.unsqueeze(0)
        probs = torch.softmax(v, dim=1)
        values_scalar = (probs[:, 0] - probs[:, 2]).numpy().astype(np.float32)

        return (policies_np, values_scalar)

    # ═══════════════════════════════════════════════════════════════════════════
    # MAIN SEARCH METHOD
    # ═══════════════════════════════════════════════════════════════════════════

    def search(self, root_state, temperature=1.0, history=None, use_dirichlet=True):
        """
        Perform MCTS search (server-routed).

        Args:
            root_state: ChessGame object for root position
            temperature: Temperature for move selection
            history: list[chess.Board], most recent first, up to 7 entries.
            use_dirichlet: root exploration noise — True for self-play, False for eval (greedy).
        Returns:
            Tuple of (best_move: str, policy_vector: np.ndarray)
        """
        # Convert ChessGame → ChessBoard by replaying moves (preserves repetition tracking)
        cpp_root = _chess_game_to_cpp_board(root_state)

        # Convert python-chess Board history → ChessBoard history by REPLAYING moves so the
        # repetition planes in history frames are correct (FEN-only construction zeroed them).
        self._cpp_history = [_cpp_board_from_history(b) for b in (history or [])]

        # Get root position evaluation from inference server (N=1, same slot mechanism).
        root_np = cpp_root.to_tensor(self._cpp_history)   # (120, 8, 8) float32

        if self.shm:
            self._inp_view[:1] = torch.from_numpy(root_np).unsqueeze(0)
            self._flush_output_queue()
            self.input_queue.put((self.worker_id, 1))
            try:
                self._get_inference_result()
            except TimeoutError:
                print(f"[Worker {self.worker_id}] ❌ Server timeout - no root inference")
                raise RuntimeError("Server communication timeout")
            policy_np = self._pol_view[:1].clone().numpy()[0]   # (4672,)
            v = self._val_view[:1].clone()                      # (1, 3) raw logits
            probs = torch.softmax(v, dim=1)
            value_f = float(probs[0, 0] - probs[0, 2])
            self.last_root_value = value_f
        else:
            root_tensor = torch.from_numpy(root_np).unsqueeze(0)
            self._flush_output_queue()
            self.input_queue.put((self.worker_id, root_tensor))

            try:
                policy, value = self._get_inference_result()
            except TimeoutError:
                print(f"[Worker {self.worker_id}] ❌ Server timeout - no root inference")
                raise RuntimeError("Server communication timeout")

            if isinstance(policy, torch.Tensor):
                policy_np = policy.detach().cpu().numpy()
            else:
                policy_np = np.array(policy, dtype=np.float32)

            if policy_np.ndim == 2:
                policy_np = policy_np[0]

            if isinstance(value, torch.Tensor):
                v = value.detach().cpu()
            else:
                v = torch.from_numpy(np.array(value, dtype=np.float32))
            if v.ndim == 1:
                v = v.unsqueeze(0)
            probs = torch.softmax(v, dim=1)
            value_f = float(probs[0, 0] - probs[0, 2])
            self.last_root_value = value_f   # exposed for KataGo-style decided-game detection

        best_move, policy_vector = self.mcts_engine.search(
            cpp_root,
            policy_np,
            value_f,
            temperature,
            self.seed,
            self._batch_inference_callback,
            use_dirichlet,
        )

        return best_move, policy_vector

    # ═══════════════════════════════════════════════════════════════════════════
    # DIRECT SEARCH (for evaluation without queue server)
    # ═══════════════════════════════════════════════════════════════════════════

    def search_direct(self, root_state, model, temperature=1.0, use_dirichlet=True, history=None):
        """
        Direct MCTS search with direct model access (no queue server).
        Used by the standalone dev/eval scripts (eval_elo.py, play_*.py, verify_heads.py).
        use_dirichlet: accepted for API compatibility; C++ engine handles Dirichlet internally.
        """
        cpp_root = _chess_game_to_cpp_board(root_state)
        cpp_history = [_cpp_board_from_history(b) for b in (history or [])]

        device = next(model.parameters()).device

        def direct_inference_callback(leaf_states):
            if len(leaf_states) == 0:
                return (
                    np.zeros((0, 4672), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32)
                )
            tensors = [s.to_tensor(cpp_history) for s in leaf_states]
            batch = torch.from_numpy(np.array(tensors)).to(device)
            with torch.no_grad():
                policies, values = model(batch)
            values_probs = torch.softmax(values, dim=1)
            values_scalar = (values_probs[:, 0] - values_probs[:, 2]).cpu().numpy()
            return (
                policies.cpu().numpy(),
                values_scalar
            )

        root_tensor = torch.from_numpy(cpp_root.to_tensor(cpp_history)).unsqueeze(0).to(device)
        with torch.no_grad():
            root_policy, root_value = model(root_tensor)

        policy_np = root_policy[0].cpu().numpy()
        root_value_probs = torch.softmax(root_value, dim=1)
        value_f = float(root_value_probs[0, 0] - root_value_probs[0, 2])
        self.last_root_value = value_f   # exposed for KataGo-style decided-game detection

        best_move, policy_vector = self.mcts_engine.search(
            cpp_root,
            policy_np,
            value_f,
            temperature,
            self.seed,
            direct_inference_callback,
            use_dirichlet,
        )

        return best_move, policy_vector

    # ═══════════════════════════════════════════════════════════════════════════
    # TREE REUSE
    # ═══════════════════════════════════════════════════════════════════════════

    def advance_root(self, played_move: str) -> bool:
        """
        Advance the MCTS tree cache to the subtree rooted at played_move.

        Call this immediately after search() returns, before the next search.
        Returns True if the subtree was reused, False if the cache was cleared.
        """
        return self.mcts_engine.advance_root(played_move)

    def reset_cache(self):
        """Discard the cached tree. Call at game start or after a game ends."""
        self.mcts_engine.reset_cache()


# ═══════════════════════════════════════════════════════════════════════════════
# USAGE NOTES
# ═══════════════════════════════════════════════════════════════════════════════
#
# In main.py, use exactly as before:
#   from game_engine.mcts_worker_cpp import MCTSWorker
#
# The interface is IDENTICAL to the old version, but now:
#   ✅ Every leaf batch gets REAL neural network evaluation
#   ✅ GPU server receives properly batched requests
#   ✅ MCTS quality matches AlphaZero paper
#
# Performance expectations:
#   - Each MCTS iteration: C++ selection → Python callback → C++ expansion
#   - Callback overhead: ~1ms for Python/numpy conversion
#   - GPU inference: ~5-15ms per batch (amortized across batch_size leaves)
#   - Total per move: Similar to Python MCTS but with proper NN guidance
#
# ═══════════════════════════════════════════════════════════════════════════════
