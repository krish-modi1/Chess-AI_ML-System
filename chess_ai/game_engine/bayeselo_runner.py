import subprocess
import os


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
            print(
                f"[Warning] BayesElo not found at {self.bayeselo_path}. "
                f"Expected structure: {project_root}/BayesElo/bayeselo. "
                f"BayesElo rating will be skipped. "
                f"Download: https://www.remi-coulom.fr/Bayesian-Elo/"
            )
            self.bayeselo_path = None

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
        
        if self.bayeselo_path is None:
            print("[Warning] BayesElo binary unavailable, skipping rating computation.")
            return None

        if not os.path.exists(pgn_filepath):
            print(f"❌ PGN file not found: {pgn_filepath}")
            return None
        
        abs_pgn = os.path.abspath(pgn_filepath)
        commands = f"readpgn {abs_pgn}\nelo\nmm\ncovariance\nratings\nx\nx\n"
        
        try:
            os.chmod(self.bayeselo_path, 0o755)
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
        # Format: rank name [name2] elo +ci -ci games score oppo draws
        # Player names may have spaces (e.g. "Stockfish 1320"), so detect
        # the ELO column by checking whether tokens[2] is the Stockfish ELO
        # suffix rather than a rating value.
        for line in lines:
            tokens = line.split()

            if len(tokens) < 5:
                continue
            try:
                int(tokens[0])  # rank — must be numeric
            except ValueError:
                continue

            try:
                name = tokens[1]

                if name == "Model":
                    model_relative = float(tokens[2])
                    model_ci_plus  = float(tokens[3])
                    model_ci_minus = float(tokens[4])

                elif name.startswith("Stockfish"):
                    # Detect two-word name "Stockfish 1320": tokens[2] is the
                    # ELO suffix, not the relative rating — shift columns by 1.
                    shift = 1 if tokens[2] == str(self.stockfish_elo) else 0
                    sf_relative = float(tokens[2 + shift])
                    sf_ci_plus  = float(tokens[3 + shift])
                    sf_ci_minus = float(tokens[4 + shift])

            except (ValueError, IndexError):
                pass

        # Compute results if we have both
        if model_relative is not None and sf_relative is not None:
            # BayesElo reports ratings relative to the group mean (≈0).
            # Anchor to Stockfish's known absolute ELO:
            #   model_abs = stockfish_actual + (model_relative - sf_relative)
            model_abs = self.stockfish_elo + model_relative - sf_relative
            sf_abs    = self.stockfish_elo
            
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
