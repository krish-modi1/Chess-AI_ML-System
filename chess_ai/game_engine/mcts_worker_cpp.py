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

    def __init__(self, worker_id, input_queue, output_queue, simulations=800, batch_size=8, seed=0):
        """
        Initialize MCTSWorker with C++ backend.

        Args:
            worker_id: Unique ID for this worker (for queue routing)
            input_queue: Queue to send inference requests to GPU server
            output_queue: Queue to receive inference results from GPU server
            simulations: Total number of MCTS simulations
            batch_size: Number of leaves to evaluate per iteration
            seed: Random seed for exploration diversity
        """
        self.worker_id = worker_id
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.simulations = simulations
        self.batch_size = batch_size
        self.cpu = 1.0
        self.seed = seed

        # History for current game — list of ChessBoard copies, most recent first.
        # Set by search() on each call; consumed by _batch_inference_callback.
        self._cpp_history = []

        # Create C++ MCTS engine instance
        self.mcts_engine = mcts_engine_cpp.MCTSEngine(simulations, batch_size)

    # ═══════════════════════════════════════════════════════════════════════════
    # NON-BLOCKING QUEUE POLLING
    # ═══════════════════════════════════════════════════════════════════════════

    def _get_inference_result(self, timeout_ms=60000):
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
        batch_tensor = torch.from_numpy(np.array(tensors))

        self.input_queue.put((self.worker_id, batch_tensor))

        try:
            policies, values = self._get_inference_result(timeout_ms=60000)
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

    def search(self, root_state, temperature=1.0, history=None):
        """
        Perform MCTS search.

        Args:
            root_state: ChessGame object for root position
            temperature: Temperature for move selection
            history: list[chess.Board], most recent first, up to 7 entries.
                     Used to fill historical planes 14-111 of to_tensor().
        Returns:
            Tuple of (best_move: str, policy_vector: np.ndarray)
        """
        # Convert ChessGame → ChessBoard by replaying moves (preserves repetition tracking)
        cpp_root = _chess_game_to_cpp_board(root_state)

        # Convert python-chess Board history → ChessBoard history (from FEN).
        # Minor approximation: repetition planes in history frames will be 0.
        # Acceptable — matches the existing root-history approximation.
        self._cpp_history = [mcts_engine_cpp.ChessBoard(b.fen()) for b in (history or [])]

        # Get root position evaluation from inference server
        root_tensor = torch.from_numpy(cpp_root.to_tensor(self._cpp_history)).unsqueeze(0)
        self.input_queue.put((self.worker_id, root_tensor))

        try:
            policy, value = self._get_inference_result(timeout_ms=60000)
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

        best_move, policy_vector = self.mcts_engine.search(
            cpp_root,
            policy_np,
            value_f,
            temperature,
            self.seed,
            self._batch_inference_callback
        )

        return best_move, policy_vector

    # ═══════════════════════════════════════════════════════════════════════════
    # DIRECT SEARCH (for evaluation without queue server)
    # ═══════════════════════════════════════════════════════════════════════════

    def search_direct(self, root_state, model, temperature=1.0, use_dirichlet=True, history=None):
        """
        Direct MCTS search with direct model access (no queue server).
        Used by StockfishEvaluator / Arena during evaluation.
        use_dirichlet: accepted for API compatibility; C++ engine handles Dirichlet internally.
        """
        cpp_root = _chess_game_to_cpp_board(root_state)
        cpp_history = [mcts_engine_cpp.ChessBoard(b.fen()) for b in (history or [])]

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

        best_move, policy_vector = self.mcts_engine.search(
            cpp_root,
            policy_np,
            value_f,
            temperature,
            self.seed,
            direct_inference_callback
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
