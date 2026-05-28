"""
Chess AI Configuration Settings

Centralized configuration for game environment, training, and evaluation.
"""

import os
from pathlib import Path

class Config:
    """Base configuration"""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODEL_DIR = PROJECT_ROOT / "game_engine"
    EVALUATION_DIR = PROJECT_ROOT / "evaluation"
    BACKEND_DIR = PROJECT_ROOT / "backend"
    FRONTEND_DIR = PROJECT_ROOT / "frontend"
    
    # Ensure directories exist
    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / "training_data").mkdir(exist_ok=True)
    (DATA_DIR / "lichess_raw").mkdir(exist_ok=True)
    (DATA_DIR / "played_games").mkdir(exist_ok=True)
    (MODEL_DIR / "model").mkdir(exist_ok=True)
    EVALUATION_DIR.mkdir(exist_ok=True)
    
    # Chess Environment Settings
    CHESS_CONFIG = {
        "time_controls": {
            "classical_min": 900,    # 15 minutes
            "classical_max": 1800,   # 30 minutes
            "description": "Standard classical chess"
        },
        "variant": "standard",
        "min_elo": 1750,
        "initial_time": 600  # 10 minutes for gameplay
    }
    
    # Training Data
    TRAINING_DATA = {
        "min_positions": 1_000_000,
        "train_split": 0.8,
        "val_split": 0.1,
        "test_split": 0.1,
        "batch_size": 128,
        "dtype": "float16"  # Memory efficient
    }
    
    # Model Architecture (CNN)
    MODEL = {
        "input_channels": 12,      # 6 piece types x 2 colors
        "board_size": 8,
        "output_moves": 8192,      # All possible moves + promotions
        "conv_layers": [64, 128, 256],
        "policy_head_hidden": 128,
        "value_head_hidden": 128
    }
    
    # Training Hyperparameters
    TRAINING = {
        "epochs": 100,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "loss_policy": "cross_entropy",
        "loss_value": "mse",
        "regularization": 0.0001,
        "early_stopping_patience": 10
    }
    
    # Evaluation Settings
    EVALUATION = {
        "elo_quick_games": 20,
        "elo_full_games": 100,
        "stockfish_depth": 15,
        "move_analysis_time": 0.1  # seconds per move
    }
    
    # Logging and Checkpoints
    LOGGING = {
        "log_dir": PROJECT_ROOT / "logs",
        "checkpoint_dir": MODEL_DIR / "checkpoints",
        "save_frequency": 5,  # Save every N epochs
        "plot_frequency": 1   # Plot every N epochs
    }
    LOGGING["log_dir"].mkdir(exist_ok=True)
    LOGGING["checkpoint_dir"].mkdir(exist_ok=True)
    
    # Backend Flask Configuration
    FLASK = {
        "HOST": "127.0.0.1",
        "PORT": 5000,
        "DEBUG": False,
        "CORS_ORIGINS": ["http://localhost:*", "http://127.0.0.1:*"]
    }
    
    # API Settings
    API = {
        "lichess_token": os.getenv("LICHESS_TOKEN", None),
        "stockfish_path": "/usr/bin/stockfish"
    }

    # AI Engine Configuration
    AI = {
        "model_path": "game_engine/model/best_model.pth",
        "simulations": 200,  # Sims per move (200-400 for fast frontend play)
        "batch_size": 8,
        "device": "cuda",  # or "cpu"
    }


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False


class TestingConfig(Config):
    """Testing configuration - FIXED to properly override"""
    DEBUG = False
    TESTING = True
    
    # FIXED: Properly override TRAINING_DATA dict
    TRAINING_DATA = {
        "min_positions": 10000,  # Smaller for testing
        "train_split": 0.8,
        "val_split": 0.1,
        "test_split": 0.1,
        "batch_size": 32,  # Smaller batch for testing
        "dtype": "float16"
    }


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False


def get_config(env=None):
    """Get configuration based on environment"""
    if env is None:
        env = os.getenv("FLASK_ENV", "development")
    
    configs = {
        "development": DevelopmentConfig,
        "testing": TestingConfig,
        "production": ProductionConfig
    }
    
    return configs.get(env, DevelopmentConfig)


# For direct import (often cleaner in code)
def get_settings():
    """Get settings singleton"""
    return get_config()
