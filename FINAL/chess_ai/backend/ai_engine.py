import os
import sys
import torch
import threading
from pathlib import Path
from typing import Optional

# Add game_engine to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from game_engine.chess_env import ChessGame


class AIEngine:
    """Wrapper for model-based move selection using MCTS (non-blocking)."""
    
    def __init__(self, model_path="game_engine/model/best_model.pth", 
                 simulations=800, batch_size=8):
        """
        Initialize AI engine with trained model (lazy-loads in background).
        
        Args:
            model_path: Path to best_model.pth checkpoint
            simulations: Number of MCTS simulations per move
            batch_size: Batch size for neural net inference
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.simulations = simulations
        self.batch_size = batch_size
        self.model_path = model_path
        
        self.model = None
        self.mcts_worker = None
        self.is_ready = False
        self._load_lock = threading.Lock()
        self._loading = False
        
        # Start background loading
        self._start_background_load()
        
        print(f"[AI Engine] ✅ Initialized (lazy-loading in background) | Device: {self.device}")
    
    def _start_background_load(self):
        """Load model in background thread to avoid blocking."""
        if self._loading:
            return
        
        self._loading = True
        thread = threading.Thread(target=self._load_model_background, daemon=True)
        thread.start()
    
    def _load_model_background(self):
        """Background thread: load model without blocking Flask."""
        print(f"[AI Engine] Loading model in background...")
        
        with self._load_lock:
            try:
                # Import here to avoid blocking
                from game_engine.cnn import ChessCNN
                from game_engine.mcts import MCTSWorker
                
                # Create model with v2 (upgraded) architecture
                self.model = ChessCNN(upgraded=True).to(self.device)
                
                # Load weights if file exists
                if os.path.exists(self.model_path):
                    try:
                        checkpoint = torch.load(self.model_path, map_location=self.device)
                        
                        # Handle different checkpoint formats
                        if isinstance(checkpoint, dict):
                            if 'model_state_dict' in checkpoint:
                                self.model.load_state_dict(checkpoint['model_state_dict'])
                            elif 'state_dict' in checkpoint:
                                self.model.load_state_dict(checkpoint['state_dict'])
                            else:
                                self.model.load_state_dict(checkpoint)
                        else:
                            self.model.load_state_dict(checkpoint)
                        
                        print(f"[AI Engine] ✅ Model weights loaded from {self.model_path}")
                    except Exception as e:
                        print(f"[AI Engine] ⚠️ Error loading weights: {e}")
                        print(f"[AI Engine] Using randomly initialized weights")
                else:
                    print(f"[AI Engine] ⚠️ Model file not found at {self.model_path}")
                    print(f"[AI Engine] Using randomly initialized weights")
                
                self.model.eval()
                
                # Initialize MCTS worker
                self.mcts_worker = MCTSWorker(
                    worker_id=0,
                    input_queue=None,
                    output_queue=None,
                    simulations=self.simulations,
                    batch_size=self.batch_size
                )
                
                self.is_ready = True
                print(f"[AI Engine] ✅ Model ready for inference")
                
            except Exception as e:
                print(f"[AI Engine] ❌ Error during model loading: {e}")
                self.is_ready = False
    
    def _ensure_loaded(self, timeout=5.0):
        """
        Wait for model to finish loading (with timeout).
        
        Args:
            timeout: Max seconds to wait
            
        Returns:
            True if model is ready, False if timeout
        """
        import time
        
        start = time.time()
        while time.time() - start < timeout:
            if self.is_ready:
                return True
            time.sleep(0.1)
        
        return False
    
    def get_best_move(self, game: ChessGame, temperature=0.0, verbose=False) -> Optional[str]:
        """
        Get best move using MCTS + trained neural network.
        
        Args:
            game: ChessGame instance at current position
            temperature: 0.0 for greedy, >0.0 for sampling
            verbose: Print debug info
            
        Returns:
            UCI move string (e.g., "e2e4") or None
        """
        if game.is_over:
            if verbose:
                print(f"[AI Engine] Game is over")
            return None
        
        legal_moves = list(game.board.legal_moves)
        if not legal_moves:
            if verbose:
                print(f"[AI Engine] No legal moves")
            return None
        
        # Wait for model to load (with timeout)
        if not self.is_ready:
            print(f"[AI Engine] ⏳ Waiting for model to load...")
            if not self._ensure_loaded(timeout=10.0):
                print(f"[AI Engine] ⚠️ Model still loading, using random fallback")
                import random
                return random.choice(legal_moves).uci()
        
        try:
            if verbose:
                print(f"[AI Engine] Searching with {self.simulations} simulations...")
            
            # Use MCTSWorker.search_direct() for single-threaded search
            move, visit_count = self.mcts_worker.search_direct(
                game,
                model=self.model,
                temperature=temperature,
                use_dirichlet=False
            )
            
            if verbose:
                print(f"[AI Engine] Move: {move} | Visits: {visit_count}")
            
            return move
            
        except Exception as e:
            print(f"[AI Engine] ❌ MCTS search failed: {e}")
            # Fallback to random legal move
            import random
            fallback_move = random.choice(legal_moves).uci()
            print(f"[AI Engine] Falling back to random move: {fallback_move}")
            return fallback_move
    
    def get_move_with_stats(self, game: ChessGame, temperature=0.0):
        """
        Get move and return associated stats (for UI display).
        
        Returns:
            {
                'move': 'e2e4',
                'simulations': 800,
                'success': True,
                'model_ready': bool
            }
        """
        move = self.get_best_move(game, temperature=temperature)
        return {
            'move': move,
            'simulations': self.simulations,
            'success': move is not None,
            'model_ready': self.is_ready
        }


# ============================================================================
# Global AI Engine Instance (Lazy Initialization - NON-BLOCKING)
# ============================================================================

_ai_engine: Optional[AIEngine] = None
_ai_engine_lock = threading.Lock()


def get_ai_engine(model_path="game_engine/model/best_model.pth", 
                  simulations=800, batch_size=8) -> AIEngine:
    """
    Get or create global AI engine instance (singleton pattern - NON-BLOCKING).
    
    Returns immediately without waiting for model to load.
    Model loads in background thread.
    
    Args:
        model_path: Path to trained model checkpoint
        simulations: MCTS simulations per move
        batch_size: Neural net batch size
        
    Returns:
        AIEngine instance (reused across multiple games)
    """
    global _ai_engine
    
    with _ai_engine_lock:
        if _ai_engine is None:
            print(f"[AI Engine Manager] Creating AI engine (non-blocking)...")
            _ai_engine = AIEngine(
                model_path=model_path,
                simulations=simulations,
                batch_size=batch_size
            )
    
    return _ai_engine


def reset_ai_engine():
    """Force reload AI engine (e.g., when model is updated)."""
    global _ai_engine
    with _ai_engine_lock:
        _ai_engine = None
    print(f"[AI Engine Manager] AI engine reset - will reinitialize on next use")


def is_ai_ready() -> bool:
    """Check if AI engine is ready for inference."""
    global _ai_engine
    if _ai_engine is None:
        return False
    return _ai_engine.is_ready
