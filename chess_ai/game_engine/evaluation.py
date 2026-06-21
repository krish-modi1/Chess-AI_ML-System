"""Training-metrics logging.

NOTE: the old `EvalMCTS`, `Arena`, and `StockfishEvaluator` classes (plus the `__main__` CLI)
were removed — they were dead relative to the live pipeline, which uses the server-routed
`run_arena_eval_gpu` / `run_stockfish_eval_gpu` in main.py. Only `MetricsLogger` is used (imported
by main.py).
"""
import os
import json
from datetime import datetime


class MetricsLogger:
    """Log training metrics to JSON file."""

    @staticmethod
    def log(iteration, p_loss, v_loss, arena_win_rate, elo, stockfish_elo=None, probe=None, train=None):
        """
        Log iteration metrics.

        Args:
            iteration: Iteration number
            p_loss: TRAIN policy loss (stored as policy_loss for back-compat)
            v_loss: TRAIN value loss (stored as value_loss for back-compat)
            arena_win_rate: Win rate from arena evaluation (0-1)
            elo: Model Elo rating from BayesElo (or None)
            stockfish_elo: Stockfish Elo used for evaluation
            probe: Optional dict of offline-probe markers (value_acc_gap, search_kl, ...) merged in
            train: Optional dict of extra training metrics (val_policy_loss, val_value_loss,
                   train_acc, val_acc, kl_anchor, grad_norm) merged in
        """
        _metrics_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
        os.makedirs(_metrics_dir, exist_ok=True)
        metrics_file = os.path.join(_metrics_dir, "metrics.json")

        metrics = []
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    if not isinstance(metrics, list):
                        metrics = []
            except Exception:
                metrics = []

        # No promotion this iter (elo is None) → champion (best_model) is unchanged, so carry forward
        # its last measured Elo instead of logging null — keeps a continuous trend line. elo_measured
        # flags fresh measurements vs held carry-forwards. (Stale right after a MANUAL champion swap
        # until the next real measurement — seed it with a one-off eval after a rollback.)
        elo_measured = elo is not None
        if elo is None:
            for m in reversed(metrics):
                if m.get("model_elo") is not None:
                    elo = m["model_elo"]
                    break

        data = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "policy_loss": float(p_loss),
            "value_loss": float(v_loss),
            "arena_win_rate": float(arena_win_rate),
            "model_elo": float(elo) if elo is not None else None,
            "elo_measured": elo_measured,
            "stockfish_elo": int(stockfish_elo) if stockfish_elo is not None else None,
        }
        if train:
            data.update(train)
        if probe:
            data.update(probe)

        metrics.append(data)

        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
