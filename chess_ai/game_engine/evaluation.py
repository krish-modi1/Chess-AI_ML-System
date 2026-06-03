import torch
import os
import sys
import chess
import chess.engine
import chess.pgn
import json
import random
import collections
import multiprocessing as mp
from datetime import datetime

# Ensure project root is in path
sys.path.append(os.getcwd())

from game_engine.mcts_worker_cpp import MCTSWorker
from game_engine.cnn import ChessCNN
from game_engine.chess_env import ChessGame
from game_engine.bayeselo_runner import BayesEloRunner


class EvalMCTS:
    """Evaluation wrapper using MCTSWorker.search_direct()"""
    
    def __init__(self, model_path, simulations=1200, batch_size=8, device=None):
        self.simulations = simulations
        self.batch_size = batch_size
        
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ChessCNN().to(self.device)
        
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                print(f"[Eval] Loaded model from {model_path}")
            except Exception as e:
                print(f"[Eval] ❌ Failed to load checkpoint: {e}")
                raise
        else:
            print(f"[Eval] ⚠️ Model not found at {model_path}, using random weights")
        
        self.model.eval()
        
        # Create MCTSWorker for search (no queues needed for eval)
        self.mcts = MCTSWorker(
            worker_id=0,
            input_queue=None,
            output_queue=None,
            simulations=simulations,
            batch_size=batch_size
        )
    
    def get_move(self, game, temperature=0.0, use_dirichlet=False, history=None):
        """
        Get best move using MCTS search with the model.

        Args:
            game: ChessGame instance
            temperature: 0.0 for greedy, >0 for sampling
            use_dirichlet: Whether to add exploration noise
            history: list of chess.Board copies, most recent first (up to 7)

        Returns:
            Best action (UCI string) or None if search fails
        """
        try:
            action, _ = self.mcts.search_direct(
                game,
                model=self.model,
                temperature=temperature,
                use_dirichlet=use_dirichlet,
                history=history,
            )
            return action
        except Exception as e:
            print(f"[EvalMCTS] ❌ Search error: {e}")
            return None

    def advance_root(self, played_move: str):
        """Advance the MCTS tree cache to the subtree for played_move."""
        self.mcts.advance_root(played_move)

    def reset_cache(self):
        """Discard the cached MCTS tree. Call at the start of each game."""
        self.mcts.reset_cache()


class Arena:
    """Arena for comparing two models."""
    
    def __init__(self, candidate_path, champion_path, simulations=1200, max_moves=400):
        self.candidate = EvalMCTS(candidate_path, simulations=simulations)
        self.champion = EvalMCTS(champion_path, simulations=simulations)
        self.max_moves = max_moves
    
    def play_game(self, game_id, max_moves=None):
        """
        Play one game between candidate and champion.
        Alternates colors for fairness.
        
        Returns:
            "CAND_WIN", "CHAMP_WIN", "DRAW", or "DRAW_FORCED"
        """
        if max_moves is None:
            max_moves = self.max_moves
        
        game = ChessGame()
        cand_is_white = (game_id % 2 == 0)
        cand_label = "Cand" if cand_is_white else "Champ"
        champ_label = "Champ" if cand_is_white else "Cand"
        position_history = collections.deque(maxlen=7)

        # Reset tree caches so we never carry state across games
        self.candidate.reset_cache()
        self.champion.reset_cache()

        # Opening variety
        legal_moves = list(game.board.legal_moves)
        if legal_moves:
            opening_move = random.choice(legal_moves).uci()
            position_history.appendleft(game.board.copy())
            game.push(opening_move)
            # Both trees must track the opening move
            self.candidate.advance_root(opening_move)
            self.champion.advance_root(opening_move)

        while not game.is_over and len(game.moves) < max_moves:
            history_snapshot = list(position_history)
            if game.board.turn == chess.WHITE:
                move = (self.candidate.get_move(game, temperature=0.0, history=history_snapshot)
                        if cand_is_white else
                        self.champion.get_move(game, temperature=0.0, history=history_snapshot))
            else:
                move = (self.champion.get_move(game, temperature=0.0, history=history_snapshot)
                        if cand_is_white else
                        self.candidate.get_move(game, temperature=0.0, history=history_snapshot))

            if move is None:
                break

            # Advance both trees — they both track the same game position
            self.candidate.advance_root(move)
            self.champion.advance_root(move)

            position_history.appendleft(game.board.copy())
            game.push(move)
        
        # Determine result
        if not game.is_over and len(game.moves) >= max_moves:
            print(f" [Arena] Game {game_id} ended in FORCED DRAW (Max moves {max_moves})")
            print(f"Arena Game {game_id}: * ({cand_label} vs {champ_label}) | Total Moves: {len(game.moves)}")
            return "DRAW_FORCED"
        
        result = game.result
        if result == "1-0":
            outcome = "CAND_WIN" if cand_is_white else "CHAMP_WIN"
        elif result == "0-1":
            outcome = "CAND_WIN" if cand_is_white else "CHAMP_WIN"
        else:
            outcome = "DRAW"
        
        # Log like your rich output:
        #   Arena Game N: 1-0 (Cand vs Champ) | Total Moves: X
        print(f"Arena Game {game_id}: {result} ({cand_label} vs {champ_label}) | Total Moves: {len(game.moves)}")
        return outcome


def _play_stockfish_game(args):
    """
    Play one game between EvalMCTS and Stockfish. Top-level for mp.Pool pickling.

    Args:
        args: tuple of (game_id, model_path, stockfish_path, simulations, stockfish_elo, max_moves)

    Returns:
        dict with keys: result, pgn_str, agent_is_white, game_id, num_moves
        Returns None if the game fails (engine error, model error, etc.)
    """
    import io as _io
    game_id, model_path, stockfish_path, simulations, stockfish_elo, max_moves = args

    # Force CPU to avoid multi-process GPU OOM when running 20 workers in parallel
    try:
        agent = EvalMCTS(model_path, simulations=simulations, device=torch.device('cpu'))
    except Exception as e:
        print(f"[Stockfish Worker {game_id}] Failed to load model: {e}")
        return None

    if not os.path.exists(stockfish_path):
        print(f"[Stockfish Worker {game_id}] Stockfish not found at {stockfish_path}")
        return None

    game = ChessGame()
    agent_is_white = (game_id % 2 == 0)
    position_history = collections.deque(maxlen=7)
    agent.reset_cache()

    try:
        with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
            try:
                engine.configure({"UCI_LimitStrength": True, "UCI_Elo": stockfish_elo})
            except Exception:
                pass  # Older Stockfish versions may not support UCI_Elo

            while not game.is_over and len(game.moves) < max_moves:
                is_agent_turn = (
                    (game.board.turn == chess.WHITE and agent_is_white) or
                    (game.board.turn == chess.BLACK and not agent_is_white)
                )

                move = None
                if is_agent_turn:
                    history_snapshot = list(position_history)
                    move = agent.get_move(
                        game, temperature=0.0, use_dirichlet=False, history=history_snapshot
                    )
                    if move is None:
                        print(f"[Stockfish Worker {game_id}] Agent returned None move, stopping game")
                else:
                    try:
                        res = engine.play(game.board, chess.engine.Limit(time=0.1))
                        move = res.move.uci()
                    except Exception as e:
                        print(f"[Stockfish Worker {game_id}] Stockfish engine error: {e}")

                if move is None:
                    break

                agent.advance_root(move)
                position_history.appendleft(game.board.copy())
                game.push(move)

    except Exception as e:
        print(f"[Stockfish Worker {game_id}] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return None

    result = game.result

    # Build PGN
    white_name = "Model" if agent_is_white else f"Stockfish {stockfish_elo}"
    black_name = f"Stockfish {stockfish_elo}" if agent_is_white else "Model"

    pgn_game = chess.pgn.Game()
    pgn_game.headers["Event"] = "Model vs Stockfish"
    pgn_game.headers["Site"] = "Localhost"
    pgn_game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    pgn_game.headers["Round"] = str(game_id + 1)
    pgn_game.headers["White"] = white_name
    pgn_game.headers["Black"] = black_name
    pgn_game.headers["Result"] = result

    node = pgn_game
    for move_uci in game.moves:
        try:
            node = node.add_variation(chess.Move.from_uci(move_uci))
        except Exception:
            pass

    buf = _io.StringIO()
    print(pgn_game, file=buf, end="\n\n")
    pgn_str = buf.getvalue()

    if not game.is_over and len(game.moves) >= max_moves:
        print(f"[Stockfish Worker {game_id}] Forced draw (max moves {max_moves})")

    print(
        f"[Stockfish Worker {game_id}] Game {game_id + 1}: {result} "
        f"({white_name} vs {black_name}) | Moves: {len(game.moves)}"
    )

    return {
        "result": result,
        "pgn_str": pgn_str,
        "agent_is_white": agent_is_white,
        "game_id": game_id,
        "num_moves": len(game.moves),
    }


class StockfishEvaluator:
    """Evaluate model against Stockfish with BayesElo rating."""
    
    def __init__(self, stockfish_path, simulations=1200):
        self.stockfish_path = stockfish_path
        self.simulations = simulations
    
    def evaluate_with_bayeselo(self, model_path, pgn_output_path, num_games=20,
                               stockfish_elo=1320, max_moves=400):
        print(f"\n{'='*70}\n📊 STOCKFISH EVALUATION - {num_games} GAMES (parallel)\n{'='*70}\n")

        if not os.path.exists(self.stockfish_path):
            print(f"❌ Stockfish not found at {self.stockfish_path}")
            return None

        # Build args list — one tuple per game
        args_list = [
            (i, model_path, self.stockfish_path, self.simulations, stockfish_elo, max_moves)
            for i in range(num_games)
        ]

        # Spawn one worker per game. Use 'spawn' to avoid CUDA fork issues.
        ctx = mp.get_context('spawn')
        print(f"  Launching {num_games} parallel workers...")
        try:
            num_workers = min(num_games, os.cpu_count() or 4)
            with ctx.Pool(processes=num_workers) as pool:
                game_results = pool.map(_play_stockfish_game, args_list)
        except Exception as e:
            print(f"❌ Pool execution error: {e}")
            import traceback
            traceback.print_exc()
            return None

        # Filter out failed games
        game_results = [r for r in game_results if r is not None]
        if not game_results:
            print("❌ All worker games failed")
            return None

        # Aggregate results
        win_count = 0
        loss_count = 0
        draw_count = 0
        pgn_parts = []

        for r in sorted(game_results, key=lambda x: x["game_id"]):
            result = r["result"]
            agent_is_white = r["agent_is_white"]
            pgn_parts.append(r["pgn_str"])

            if result == "1-0":
                if agent_is_white:
                    win_count += 1
                else:
                    loss_count += 1
            elif result == "0-1":
                if agent_is_white:
                    loss_count += 1
                else:
                    win_count += 1
            else:
                draw_count += 1

        # Save PGN file
        os.makedirs(os.path.dirname(pgn_output_path), exist_ok=True)
        try:
            with open(pgn_output_path, 'w') as f:
                f.write("".join(pgn_parts))
            print(f"\n✅ Saved {len(game_results)} games to {pgn_output_path}")
        except Exception as e:
            print(f"❌ Failed to save PGN: {e}")
            return None

        # Run BayesElo
        try:
            runner = BayesEloRunner(stockfish_elo=stockfish_elo)
            bayeselo_results = runner.run(pgn_output_path)

            if bayeselo_results:
                print(f"\n{'='*70}\n🏆 BAYESELO RESULTS\n{'='*70}")
                print(
                    f"Model Strength:     {bayeselo_results['model_elo']:.0f} "
                    f"± {(bayeselo_results['model_ci_upper']-bayeselo_results['model_ci_lower'])/2:.0f} Elo"
                )
                print(f"Vs Stockfish:       {stockfish_elo} Elo")
                print(
                    f"Difference:         "
                    f"{bayeselo_results['diff_elo']:+.0f} Elo "
                    f"[{bayeselo_results['diff_ci_lower']:.0f}, {bayeselo_results['diff_ci_upper']:.0f}]"
                )
                print(f"{'='*70}\n")

                bayeselo_results['win_count'] = win_count
                bayeselo_results['loss_count'] = loss_count
                bayeselo_results['draw_count'] = draw_count
                bayeselo_results['total_games'] = len(game_results)
                bayeselo_results['win_rate'] = win_count / len(game_results) if game_results else 0.0

                return bayeselo_results
            else:
                print("❌ BayesElo computation failed")
                return None

        except Exception as e:
            print(f"❌ BayesElo Error: {e}")
            import traceback
            traceback.print_exc()
            return None


class MetricsLogger:
    """Log training metrics to JSON file."""
    
    @staticmethod
    def log(iteration, p_loss, v_loss, arena_win_rate, elo, stockfish_elo=None):
        """
        Log iteration metrics.
        
        Args:
            iteration: Iteration number
            p_loss: Policy loss from training
            v_loss: Value loss from training
            arena_win_rate: Win rate from arena evaluation (0-1)
            elo: Model Elo rating from BayesElo (or None)
            stockfish_elo: Stockfish Elo used for evaluation
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
        
        os.makedirs("game_engine/model", exist_ok=True)
        metrics_file = "game_engine/model/metrics.json"
        
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


if __name__ == "__main__":
    """
    Quick test evaluation.
    
    Usage:
        python game_engine/evaluation.py --model game_engine/model/best_model.pth --games 5 --sims 200
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate model vs Stockfish with BayesElo")
    parser.add_argument("--model", type=str, default="game_engine/model/best_model.pth",
                        help="Model checkpoint path")
    parser.add_argument("--stockfish", type=str, default="/usr/games/stockfish",
                        help="Stockfish executable path")
    parser.add_argument("--games", type=int, default=10, help="Number of games to play")
    parser.add_argument("--elo", type=int, default=1320, help="Stockfish Elo rating")
    parser.add_argument("--sims", type=int, default=800, help="MCTS simulations per move")
    parser.add_argument("--max-moves", type=int, default=400,
                        help="Max moves per game before forced draw")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🏆 CHESS MODEL EVALUATION")
    print("=" * 70)
    print(f"Model:        {args.model}")
    print(f"Stockfish:    {args.stockfish} ({args.elo} Elo)")
    print(f"Games:        {args.games}")
    print(f"Simulations:  {args.sims}")
    print(f"Max moves:    {args.max_moves}")
    print("=" * 70 + "\n")
    
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        sys.exit(1)
    
    if not os.path.exists(args.stockfish):
        print(f"❌ Stockfish not found: {args.stockfish}")
        sys.exit(1)
    
    evaluator = StockfishEvaluator(
        stockfish_path=args.stockfish,
        simulations=args.sims
    )
    
    pgn_path = f"game_engine/evaluation/pgn/eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pgn"
    
    results = evaluator.evaluate_with_bayeselo(
        model_path=args.model,
        pgn_output_path=pgn_path,
        num_games=args.games,
        stockfish_elo=args.elo,
        max_moves=args.max_moves
    )
    
    if results:
        print("\n" + "=" * 70)
        print("✅ EVALUATION COMPLETE")
        print("=" * 70)
        print(f"Model Elo:      {results['model_elo']:.0f}")
        print(f"Stockfish Elo:  {args.elo}")
        print(f"Difference:     {results['diff_elo']:+.0f} Elo")
        print(f"Win Rate:       {results['win_rate']:.1%}")
        print(f"Record:         {results['win_count']}-{results['draw_count']}-{results['loss_count']}")
        print(f"PGN Saved:      {pgn_path}")
        print("=" * 70)
        sys.exit(0)
    else:
        print("\n❌ EVALUATION FAILED")
        sys.exit(1)
