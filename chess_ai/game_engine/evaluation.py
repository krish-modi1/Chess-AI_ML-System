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
    def log(iteration, p_loss, v_loss, arena_win_rate, elo, stockfish_elo=None, probe=None):
        """
        Log iteration metrics.

        Args:
            iteration: Iteration number
            p_loss: Policy loss from training
            v_loss: Value loss from training
            arena_win_rate: Win rate from arena evaluation (0-1)
            elo: Model Elo rating from BayesElo (or None)
            stockfish_elo: Stockfish Elo used for evaluation
            probe: Optional dict of offline-probe markers (value_acc_gap, search_kl, ...) merged in
        """
        data = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "policy_loss": float(p_loss),
            "value_loss": float(v_loss),
            "arena_win_rate": float(arena_win_rate),
            "model_elo": float(elo) if elo is not None else None,
            "stockfish_elo": int(stockfish_elo) if stockfish_elo is not None else None,
        }
        if probe:
            data.update(probe)

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

        metrics.append(data)

        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
