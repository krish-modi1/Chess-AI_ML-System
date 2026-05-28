import torch
import os
import sys
import chess
import chess.engine
import chess.pgn
import json
import random
import time
import shutil
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

        # Use upgraded=True to match training model (256 filters)
        self.model = ChessCNN(upgraded=True).to(self.device)
        
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
    
    def get_move(self, game, temperature=0.0, use_dirichlet=False):
        """
        Get best move using MCTS search with the model.
        
        Args:
            game: ChessGame instance
            temperature: 0.0 for greedy, >0 for sampling
            use_dirichlet: Whether to add exploration noise
        
        Returns:
            Best action (UCI string) or None if search fails
        """
        try:
            action, _ = self.mcts.search_direct(
                game,
                model=self.model,
                temperature=temperature,
                use_dirichlet=use_dirichlet
            )
            return action
        except Exception as e:
            print(f"[EvalMCTS] ❌ Search error: {e}")
            return None


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
        
        # Optional: opening variety like your old version
        legal_moves = list(game.board.legal_moves)
        if legal_moves:
            game.push(random.choice(legal_moves).uci())
        
        while not game.is_over and len(game.moves) < max_moves:
            if game.board.turn == chess.WHITE:
                move = self.candidate.get_move(game, temperature=0.0) if cand_is_white else self.champion.get_move(game, temperature=0.0)
            else:
                move = self.champion.get_move(game, temperature=0.0) if cand_is_white else self.candidate.get_move(game, temperature=0.0)
            
            if move is None:
                break
            
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


class StockfishEvaluator:
    """Evaluate model against Stockfish with BayesElo rating."""
    
    def __init__(self, stockfish_path, simulations=1200):
        self.stockfish_path = stockfish_path
        self.simulations = simulations
    
    def evaluate_with_bayeselo(self, model_path, pgn_output_path, num_games=20,
                               stockfish_elo=1320, max_moves=400):
        """
        Play games against Stockfish, save PGN, and run BayesElo for rating.
        
        Args:
            model_path: Path to trained model
            pgn_output_path: Where to save PGN file
            num_games: Number of games to play
            stockfish_elo: Stockfish skill level (Elo)
            max_moves: Maximum moves per game before forced draw
        
        Returns:
            Dictionary with BayesElo results or None on failure
        """
        print(f"\n{'='*70}\n📊 STOCKFISH EVALUATION - {num_games} GAMES\n{'='*70}\n")
        
        agent = EvalMCTS(model_path, simulations=self.simulations)
        
        if not os.path.exists(self.stockfish_path):
            print(f"❌ Stockfish not found at {self.stockfish_path}")
            return None
        
        pgn_games = []
        win_count = 0
        loss_count = 0
        draw_count = 0
        
        try:
            with chess.engine.SimpleEngine.popen_uci(self.stockfish_path) as engine:
                # Try to configure Elo-limited Stockfish
                try:
                    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": stockfish_elo})
                except Exception:
                    print("[Warning] Stockfish version may not support UCI_Elo configuration")
                
                for i in range(num_games):
                    game = ChessGame()
                    agent_is_white = (i % 2 == 0)
                    
                    while not game.is_over and len(game.moves) < max_moves:
                        is_agent = (
                            (game.board.turn == chess.WHITE and agent_is_white) or
                            (game.board.turn == chess.BLACK and not agent_is_white)
                        )
                        
                        if is_agent:
                            move = agent.get_move(game, temperature=0.0, use_dirichlet=False)
                            if move is None:
                                print(f" [Stockfish] Game {i+1}: Agent move failed, stopping game")
                                break
                        else:
                            try:
                                res = engine.play(game.board, chess.engine.Limit(time=0.1))
                                move = res.move.uci()
                            except Exception as e:
                                print(f" [Stockfish] Game {i+1}: Stockfish error: {e}")
                                break
                        
                        if move is None:
                            break
                        game.push(move)
                    
                    # Forced draw logging
                    if not game.is_over and len(game.moves) >= max_moves:
                        print(f" [Stockfish] Game {i+1} ended in FORCED DRAW (Max moves {max_moves})")
                    
                    result = game.result
                    
                    # Count results from model perspective and log nicely
                    if result == "1-0":
                        if agent_is_white:
                            win_count += 1
                            result_log = "1-0 (Model vs Stockfish)"
                        else:
                            loss_count += 1
                            result_log = "1-0 (Stockfish vs Model)"
                    elif result == "0-1":
                        if agent_is_white:
                            loss_count += 1
                            result_log = "0-1 (Model vs Stockfish)"
                        else:
                            win_count += 1
                            result_log = "0-1 (Stockfish vs Model)"
                    else:
                        draw_count += 1
                        result_log = "1/2-1/2 (Model vs Stockfish)"
                    
                    white_name = "Model" if agent_is_white else f"Stockfish {stockfish_elo}"
                    black_name = f"Stockfish {stockfish_elo}" if agent_is_white else "Model"
                    print(f"Game {i+1}/{num_games}: {result} ({white_name} vs {black_name}) | Total Moves: {len(game.moves)}")
                    
                    # Build PGN game
                    pgn_game = chess.pgn.Game()
                    pgn_game.headers["Event"] = "Model vs Stockfish"
                    pgn_game.headers["Site"] = "Localhost"
                    pgn_game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
                    pgn_game.headers["Round"] = str(i + 1)
                    pgn_game.headers["White"] = white_name
                    pgn_game.headers["Black"] = black_name
                    pgn_game.headers["Result"] = result
                    
                    node = pgn_game
                    for move_uci in game.moves:
                        try:
                            move_obj = chess.Move.from_uci(move_uci)
                            node = node.add_variation(move_obj)
                        except Exception:
                            pass
                    
                    pgn_games.append(pgn_game)
        
        except Exception as e:
            print(f"❌ Evaluation Error: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        # Save PGN file
        os.makedirs(os.path.dirname(pgn_output_path), exist_ok=True)
        try:
            with open(pgn_output_path, 'w') as f:
                for pgn_game in pgn_games:
                    f.write(str(pgn_game))
                    f.write("\n\n")
            print(f"\n✅ Saved {len(pgn_games)} games to {pgn_output_path}")
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
                bayeselo_results['total_games'] = num_games
                bayeselo_results['win_rate'] = win_count / num_games if num_games > 0 else 0.0
                
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
