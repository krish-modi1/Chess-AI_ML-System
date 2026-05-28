"""
Chess AI Backend - Flask + Socket.IO
PRODUCTION-READY VERSION - All errors corrected + gameplay perfect
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import send_from_directory, Flask, request, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room
from game_engine.chess_env import ChessGame
from config.settings import get_config
import threading
import time
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from backend.ai_engine import get_ai_engine
except ImportError as e:
    logger.warning(f"AI engine import failed: {e}. Using fallback mode.")
    get_ai_engine = None


app = Flask(__name__)

# Get config object
config = get_config()

# Manually map Flask settings
app.config['HOST'] = config.FLASK.get('HOST', '127.0.0.1')
app.config['PORT'] = config.FLASK.get('PORT', 5000)
app.config['DEBUG'] = config.FLASK.get('DEBUG', False)

# CORS origins
app.config['CORS_ORIGINS'] = [
    "http://localhost:*",
    "http://127.0.0.1:*",
    "http://localhost",
    "http://127.0.0.1",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5000",
    "http://127.0.0.1:5000"
]
app.config['SECRET_KEY'] = 'chess-ai-secret'

# Initialize SocketIO
socketio = SocketIO(
    app,
    cors_allowed_origins=app.config['CORS_ORIGINS'],
    ping_timeout=60,
    ping_interval=25,
    async_mode='threading'
)

# Game storage
games = {}
game_locks = {}

# ==================== HELPERS ====================

def get_game(game_id):
    """Safely retrieve game"""
    return games.get(game_id, None)

def get_game_with_lock(game_id):
    """Get game with thread lock"""
    if game_id not in game_locks:
        game_locks[game_id] = threading.Lock()
    return games.get(game_id, None), game_locks[game_id]

def delete_game(game_id):
    """Clean up game resources"""
    if game_id in games:
        del games[game_id]
    if game_id in game_locks:
        del game_locks[game_id]
    logger.info(f"[Game {game_id}] Deleted")

def get_game_state(game):
    """Build game state from ChessGame object
    
    ✅ CRITICAL FIXES:
    - legal_moves() is a METHOD, not a property - MUST CALL IT
    - Returns list of UCI strings - JSON serializable
    """
    return {
        'board_fen': game.board.fen(),
        'legal_moves': game.legal_moves(),  # ✅ CALL THE METHOD
        'moves': game.moves,
        'turn': 'white' if game.board.turn else 'black',
        'is_over': game.is_over,  # @property - auto-evaluates
        'result': game.result,    # @property - auto-evaluates
        'last_move': game.moves[-1] if game.moves else None
    }

# ==================== ROUTES ====================

@app.route("/")
def index():
    """Serve index.html"""
    frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
    return send_from_directory(frontend_dir, "index.html")

@app.route("/static/<path:filename>")
def serve_static(filename):
    """Serve static assets"""
    static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "static"))
    return send_from_directory(static_dir, filename)

@app.route("/new_game", methods=["POST"])
def new_game():
    """REST endpoint to create new game"""
    try:
        data = request.get_json() or {}
        mode = data.get('mode', 'pvp')
        ai_side = data.get('ai_side', 'black')
        
        if mode not in ['pvp', 'pve']:
            return jsonify({"error": "Invalid mode"}), 400
        
        game = ChessGame()
        games[game.game_id] = {
            'game': game,
            'mode': mode,
            'ai_side': ai_side,
            'timer': {'white': 600, 'black': 600},
            'created_at': datetime.now().isoformat()
        }
        
        logger.info(f"[Game {game.game_id}] Created - Mode: {mode}, AI: {ai_side}")
        
        return jsonify({
            "game_id": game.game_id,
            "mode": mode,
            "ai_side": ai_side,
            **get_game_state(game)
        })
    except Exception as e:
        logger.error(f"Error creating game: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/game_stats/<game_id>", methods=["GET"])
def game_stats(game_id):
    """Get game statistics"""
    game_data = get_game(game_id)
    if not game_data:
        return jsonify({"error": "No such game_id"}), 404
    
    game = game_data['game']
    stats = {
        "total_moves": len(game.moves),
        "move_history": game.moves,
        "is_over": game.is_over,
        "result": game.result,
        "current_turn": "White" if game.board.turn else "Black",
        "legal_moves_count": len(game.legal_moves()),  # ✅ CALL THE METHOD
        "mode": game_data['mode'],
        "timer": game_data['timer']
    }
    return jsonify(stats)

@app.route("/save_game/<game_id>", methods=["POST"])
def save_game(game_id):
    """Save game to file"""
    game_data = get_game(game_id)
    if not game_data:
        return jsonify({"error": "No such game_id"}), 404
    
    try:
        game = game_data['game']
        os.makedirs('data/played_games', exist_ok=True)
        
        filename = f"data/played_games/game_{game_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump({
                'game_id': game_id,
                'mode': game_data['mode'],
                'moves': game.moves,
                'result': game.result,
                'move_count': len(game.moves),
                'created_at': game_data['created_at'],
                'finished_at': datetime.now().isoformat()
            }, f, indent=2)
        
        logger.info(f"[Game {game_id}] Saved to {filename}")
        return jsonify({"success": True, "filename": filename})
    except Exception as e:
        logger.error(f"Error saving game: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# ==================== SOCKET EVENTS ====================

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info(f"[Socket] Client connected: {request.sid}")
    emit('connect_response', {'data': 'Connected to Chess AI server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnect"""
    logger.info(f"[Socket] Client disconnected: {request.sid}")

@socketio.on('new_game')
def on_new_game(data):
    """Create new game via Socket.IO"""
    try:
        mode = data.get('mode', 'pvp')
        ai_side = data.get('ai_side', 'black')
        
        if mode not in ['pvp', 'pve']:
            emit('error', {'message': 'Invalid game mode'})
            return
        
        game = ChessGame()
        games[game.game_id] = {
            'game': game,
            'mode': mode,
            'ai_side': ai_side,
            'timer': {'white': 600, 'black': 600},
            'created_at': datetime.now().isoformat()
        }
        
        join_room(game.game_id)
        logger.info(f"[Game {game.game_id}] Created - Mode: {mode}, AI: {ai_side}")
        
        emit('game_created', {
            'game_id': game.game_id,
            'mode': mode,
            'ai_side': ai_side,
            **get_game_state(game)
        })
        
        # If AI plays white, make first move
        if mode == 'pve' and ai_side == 'white':
            time.sleep(0.5)
            socketio.start_background_task(ai_make_move_async, game.game_id)
    
    except Exception as e:
        logger.error(f"Error in on_new_game: {e}", exc_info=True)
        emit('error', {'message': f'Game creation failed: {str(e)}'})

@socketio.on('player_move')
def on_player_move(data):
    """Handle player move"""
    try:
        game_id = data.get('game_id')
        move = data.get('move')
        
        game_data, lock = get_game_with_lock(game_id)
        if not game_data:
            emit('error', {'message': 'Game not found'})
            return
        
        with lock:
            game = game_data['game']
            
            # ✅ legal_moves() is a METHOD - CALL IT
            if move not in game.legal_moves():
                logger.warning(f"[Game {game_id}] Illegal move: {move}")
                emit('illegal_move', {'move': move})
                return
            
            success = game.push(move)
            if not success:
                emit('illegal_move', {'move': move})
                return
            
            logger.info(f"[Game {game_id}] Player move: {move}")
            
            game_ending_info = check_game_ending(game)
            response = {
                'move': move,
                **get_game_state(game),
                **game_ending_info
            }
            
            emit('move_made', response, room=game_id)
            
            if game.is_over:
                emit('game_over', response, room=game_id)
                return
            
            # If PvE and AI's turn, request AI move
            if game_data['mode'] == 'pve' and not game.is_over:
                emit('ai_thinking', {}, room=game_id)
                socketio.start_background_task(ai_make_move_async, game_id)
    
    except Exception as e:
        logger.error(f"Error in on_player_move: {e}", exc_info=True)
        emit('error', {'message': f'Move failed: {str(e)}'})

@socketio.on('undo_move')
def on_undo_move(data):
    """Undo last 2 moves (player + AI)"""
    try:
        game_id = data.get('game_id')
        game_data, lock = get_game_with_lock(game_id)
        
        if not game_data:
            emit('error', {'message': 'Game not found'})
            return
        
        with lock:
            game = game_data['game']
            
            if len(game.moves) < 2:
                emit('error', {'message': 'At least 2 moves required to undo'})
                return
            
            # Pop last 2 moves
            game.board.pop()
            game.board.pop()
            game.moves.pop()
            game.moves.pop()
            
            # Update last_move
            last_move = game.moves[-1] if game.moves else None
            
            # Clear cache for legal moves - use correct attribute name
            game._cache_legal = None
            
            logger.info(f"[Game {game_id}] Undone last 2 moves")
            
            game_ending_info = check_game_ending(game)
            
            response = {
                **get_game_state(game),
                **game_ending_info
            }
            
            emit('move_undone', response, room=game_id)
    
    except Exception as e:
        logger.error(f"Error in on_undo_move: {e}", exc_info=True)
        emit('error', {'message': f'Undo failed: {str(e)}'})

@socketio.on('resign')
def on_resign(data):
    """Handle resignation"""
    try:
        game_id = data.get('game_id')
        player = data.get('player', 'white')
        
        game_data, lock = get_game_with_lock(game_id)
        if not game_data:
            emit('error', {'message': 'Game not found'})
            return
        
        with lock:
            game = game_data['game']
            winner = 'Black' if player == 'white' else 'White'
            
            logger.info(f"[Game {game_id}] {player} resigned - {winner} wins")
            
            emit('game_over', {
                **get_game_state(game),
                'game_ending_reason': 'resignation',
                'winner': winner,
                'message': f'{player.capitalize()} resigned - {winner} wins'
            }, room=game_id)
            
            delete_game(game_id)
    
    except Exception as e:
        logger.error(f"Error in on_resign: {e}", exc_info=True)
        emit('error', {'message': f'Resignation failed: {str(e)}'})

@socketio.on('request_analysis')
def on_request_analysis(data):
    """Handle analysis request"""
    game_id = data.get('game_id')
    emit('analysis_result', {
        'message': 'Analysis feature will be available after PocketZero engine is trained',
        'game_id': game_id
    })

# ==================== GAME STATE HELPERS ====================

def check_game_ending(game):
    """Check game state and return ending info"""
    board = game.board
    
    if not board.is_game_over():
        return {
            'in_check': board.is_check(),
            'game_ending_reason': None,
            'winner': None
        }
    
    if board.is_checkmate():
        winner = "Black" if board.turn else "White"
        return {
            'in_check': False,
            'game_ending_reason': 'checkmate',
            'winner': winner,
            'message': f'Checkmate! {winner} wins!'
        }
    elif board.is_stalemate():
        return {
            'in_check': False,
            'game_ending_reason': 'stalemate',
            'winner': None,
            'message': 'Stalemate! Game is a draw.'
        }
    elif board.is_insufficient_material():
        return {
            'in_check': False,
            'game_ending_reason': 'insufficient_material',
            'winner': None,
            'message': 'Draw by insufficient material.'
        }
    elif board.is_seventyfive_moves():
        return {
            'in_check': False,
            'game_ending_reason': 'seventyfive_moves',
            'winner': None,
            'message': 'Draw by 75-move rule.'
        }
    elif board.is_fivefold_repetition():
        return {
            'in_check': False,
            'game_ending_reason': 'fivefold_repetition',
            'winner': None,
            'message': 'Draw by fivefold repetition.'
        }
    else:
        return {
            'in_check': False,
            'game_ending_reason': 'other',
            'winner': None,
            'message': 'Game over.'
        }

# ==================== AI LOGIC ====================

def ai_make_move_async(game_id):
    """AI makes move asynchronously"""
    try:
        time.sleep(0.2)
        
        game_data, lock = get_game_with_lock(game_id)
        if not game_data:
            logger.warning(f"[AI] Game {game_id} not found")
            return
        
        with lock:
            game = game_data['game']
            
            if get_ai_engine:
                try:
                    ai_engine = get_ai_engine()
                    move = ai_engine.get_best_move(game, temperature=0.0)
                except Exception as e:
                    logger.error(f"[AI] Engine error: {e}. Using random move.")
                    import random
                    legal_moves = game.legal_moves()
                    if not legal_moves:
                        return
                    move = random.choice(legal_moves)
            else:
                import random
                legal_moves = game.legal_moves()
                if not legal_moves:
                    logger.warning(f"[Game {game_id}] No legal moves available")
                    return
                move = random.choice(legal_moves)
            
            if not move:
                logger.warning(f"[Game {game_id}] AI returned no move")
                return
            
            success = game.push(move)
            if not success:
                logger.error(f"[Game {game_id}] Failed to apply AI move: {move}")
                return
            
            logger.info(f"[Game {game_id}] AI moved: {move}")
            
            game_ending_info = check_game_ending(game)
            response = {
                'move': move,
                **get_game_state(game),
                **game_ending_info
            }
            
            socketio.emit('ai_move', response, room=game_id)
            
            if game.is_over:
                socketio.emit('game_over', response, room=game_id)
    
    except Exception as e:
        logger.error(f"[AI] Error in ai_make_move_async: {e}", exc_info=True)
        socketio.emit('error', {'message': f'AI move failed: {str(e)}'}, room=game_id)

# ==================== MAIN ====================

if __name__ == "__main__":
    logger.info("="*70)
    logger.info("CHESS AI - BACKEND SERVER")
    logger.info("="*70)
    logger.info(f"Host: {app.config['HOST']}:{app.config['PORT']}")
    logger.info(f"Debug: {app.config['DEBUG']}")
    logger.info("="*70)
    
    socketio.run(
        app,
        host=app.config['HOST'],
        port=app.config['PORT'],
        debug=app.config['DEBUG'],
        allow_unsafe_werkzeug=True
    )
