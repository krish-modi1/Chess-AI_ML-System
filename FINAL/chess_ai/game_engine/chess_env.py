import chess
import uuid
import numpy as np
import copy

class ChessGame:
    def __init__(self, fen=None, _board=None):
        if _board:
            self.board = _board
        else:
            self.board = chess.Board(fen) if fen else chess.Board()
            
        self.game_id = str(uuid.uuid4())
        self.moves = []       
        self._cache_legal = None  

    @property
    def turn_player(self):
        return 1.0 if self.board.turn == chess.WHITE else 0.0

    @property
    def is_over(self):
        return self.board.is_game_over()

    @property
    def result(self):
        return self.board.result()

    def legal_moves(self):
        if self._cache_legal is None:
            self._cache_legal = [move.uci() for move in self.board.legal_moves]
        return self._cache_legal

    def push(self, move_uci):
        try:
            # Optimization: Check cache first if available to avoid parsing
            if self._cache_legal and move_uci not in self._cache_legal:
                return False
                
            move = chess.Move.from_uci(move_uci)
            if move in self.board.legal_moves:
                self.board.push(move)
                self.moves.append(move_uci)
                self._cache_legal = None 
                return True
            return False
        except ValueError:
            return False

    def copy(self):
        new_board = self.board.copy()
        new_game = ChessGame(_board=new_board)
        new_game.moves = self.moves.copy()
        return new_game

    def __deepcopy__(self, memo):
        return self.copy()

    def to_tensor(self):
        """
        Tensor representation: 16 x 8 x 8
        Planes 0-5: White Pieces [P, N, B, R, Q, K]
        Planes 6-11: Black Pieces [P, N, B, R, Q, K]
        Plane 12: Turn (1.0 if Black's turn, else 0.0)
        Plane 13: Repetition (1.0 if position has occurred >= 1 time before)
        Plane 14: Total Move Count (Normalized: count / 400.0)  <-- UPDATED
        Plane 15: No Progress Count (Normalized: halfmoves / 100.0)
        """
        tensor = np.zeros((16, 8, 8), dtype=np.float32)
        
        # Optimization: Map types directly to indices
        piece_map = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }
        
        # Optimization: use piece_map() to get only existing pieces
        for square, piece in self.board.piece_map().items():
            offset = 0 if piece.color == chess.WHITE else 6
            idx = offset + piece_map[piece.piece_type]
            
            # Bottom-Up orientation (Matches training)
            row = square // 8
            col = square % 8
            
            tensor[idx][row][col] = 1.0
                
        if self.board.turn == chess.BLACK:
             tensor[12, :, :] = 1.0
             
        # Plane 13: Repetition Warning
        if self.board.is_repetition(2):
            tensor[13, :, :] = 1.0

        # Plane 14: Total Move Count (Normalized 0-1)
        # UPDATED: Cap at 400 moves (800 plies) to allow deep endgames
        move_count_norm = min(self.board.fullmove_number, 400) / 400.0
        tensor[14, :, :] = move_count_norm

        # Plane 15: No Progress Count (Normalized 0-1)
        # 50-move rule = 100 halfmoves.
        no_progress_norm = min(self.board.halfmove_clock, 100) / 100.0
        tensor[15, :, :] = no_progress_norm
             
        return tensor

    def get_reward_for_turn(self, turn_val):
        res = self.board.result()
        if res == "1-0": return 1.0 if turn_val == 1.0 else -1.0
        if res == "0-1": return 1.0 if turn_val == 0.0 else -1.0
        return 0.0