import subprocess
import os
import re
import sys


class BayesEloRunner:
    """Run BayesElo on PGN files and parse output"""
    
    def __init__(self, project_root=None, stockfish_elo=1350):
        """
        Initialize BayesElo runner
        
        Args:
            project_root: Path to project root (contains BayesElo folder)
                         Defaults to parent of game_engine/
            stockfish_elo: The Elo rating of Stockfish (used as baseline)
        """
        if project_root is None:
            # Auto-detect: game_engine is 1 level up from this script
            game_engine_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(game_engine_dir)
        
        self.project_root = project_root
        self.bayeselo_path = os.path.join(project_root, "BayesElo", "bayeselo")
        self.stockfish_elo = stockfish_elo  # Baseline for anchoring
        
        if not os.path.exists(self.bayeselo_path):
            raise FileNotFoundError(
                f"BayesElo not found at {self.bayeselo_path}\n"
                f"Expected structure: {project_root}/BayesElo/bayeselo\n"
                f"Download: https://www.remi-coulom.fr/Bayesian-Elo/"
            )
        
        self.output_dir = os.path.join(project_root, "game_engine", "evaluation", "metrics")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def run(self, pgn_filepath, iteration=0):
        """
        Run BayesElo on PGN file and return ratings
        
        Args:
            pgn_filepath: Path to PGN file
            iteration: Iteration number (for logging)
        
        Returns:
            {
                'model_elo': float (absolute rating),
                'model_ci_lower': float,
                'model_ci_upper': float,
                'sf_elo': float (baseline, 1350),
                'sf_ci_lower': float,
                'sf_ci_upper': float,
                'diff_elo': float (model - sf, should be ~127),
                'diff_ci_lower': float,
                'diff_ci_upper': float,
                'raw_output': str
            }
            or None if failed
        """
        
        if not os.path.exists(pgn_filepath):
            print(f"❌ PGN file not found: {pgn_filepath}")
            return None
        
        # BayesElo commands
        commands = f"""readpgn {pgn_filepath}
elo
mm
exactdist
ratings
x
x
"""
        
        try:
            print(f"🔄 Running BayesElo on {os.path.basename(pgn_filepath)}...")
            
            process = subprocess.Popen(
                [self.bayeselo_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.project_root
            )
            
            stdout, stderr = process.communicate(input=commands, timeout=300)
            
            # Parse output
            results = self._parse_output(stdout)
            
            if results:
                results['raw_output'] = stdout
                print(f"✅ BayesElo Complete")
                
                # Extract absolute model strength
                model_abs = results['model_elo']
                ci_size = (results['model_ci_upper'] - results['model_ci_lower']) / 2
                print(f"   Model: {model_abs:.0f} ± {ci_size:.0f} Elo")
                print(f"   vs Stockfish ({self.stockfish_elo})")
                
                diff = model_abs - self.stockfish_elo
                print(f"   Difference: +{diff:.0f} Elo")
                
                return results
            else:
                print(f"❌ Failed to parse BayesElo output")
                if stdout:
                    print(f"Output (first 500 chars):\n{stdout[:500]}")
                return None
        
        except subprocess.TimeoutExpired:
            print(f"❌ BayesElo timeout (exceeded 5 minutes)")
            return None
        except Exception as e:
            print(f"❌ BayesElo error: {e}")
            return None
    
    def _parse_output(self, output):
        """Parse BayesElo output to extract ratings"""
        lines = output.split('\n')
        results = {}
        
        model_relative = None
        model_ci_plus = None
        model_ci_minus = None
        sf_relative = None
        sf_ci_plus = None
        sf_ci_minus = None
        
        # Look for the ratings table output
        # Format: rank name elo +ci -ci games score oppo draws
        for line in lines:
            tokens = line.split()
            
            # Must have at least: rank name elo +ci -ci games score oppo draws
            if len(tokens) >= 5:
                try:
                    # Check if first token is a digit (rank)
                    rank = int(tokens[0])
                    name = tokens[1]
                    elo = float(tokens[2])
                    plus_ci = float(tokens[3])
                    minus_ci = float(tokens[4])
                    
                    # Store relative Elo values
                    if name == "Model":
                        model_relative = elo
                        model_ci_plus = plus_ci
                        model_ci_minus = minus_ci
                    elif name == "Stockfish":
                        sf_relative = elo
                        sf_ci_plus = plus_ci
                        sf_ci_minus = minus_ci
                except (ValueError, IndexError):
                    pass
        
        # Compute results if we have both
        if model_relative is not None and sf_relative is not None:
            # Convert from relative to absolute ratings
            # BayesElo reports relative: Model + Stockfish = 0
            # We need absolute: use Stockfish baseline (1350 Elo)
            
            model_abs = self.stockfish_elo + model_relative
            sf_abs = self.stockfish_elo  # Stockfish is at baseline
            
            results['model_elo'] = model_abs
            results['model_ci_lower'] = model_abs - model_ci_minus
            results['model_ci_upper'] = model_abs + model_ci_plus
            
            results['sf_elo'] = sf_abs
            results['sf_ci_lower'] = sf_abs - sf_ci_minus
            results['sf_ci_upper'] = sf_abs + sf_ci_plus
            
            # Compute difference
            diff = model_abs - sf_abs
            results['diff_elo'] = diff
            results['diff_ci_lower'] = diff - (sf_ci_plus + model_ci_minus)
            results['diff_ci_upper'] = diff + (model_ci_plus + sf_ci_minus)
            
            return results
        
        return None
