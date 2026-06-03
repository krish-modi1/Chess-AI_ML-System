import chess
import uuid
import numpy as np

_PIECE_MAP = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
}

def _fill_board_planes(tensor, board, offset):
    """
    Write 14 planes into tensor starting at offset.
      offset+0..5  : White piece bitboards [P, N, B, R, Q, K]
      offset+6..11 : Black piece bitboards [P, N, B, R, Q, K]
      offset+12    : 1.0 if position has occurred >= 2 times in game history
      offset+13    : 1.0 if position has occurred >= 3 times in game history
    """
    for square, piece in board.piece_map().items():
        idx = offset + (0 if piece.color == chess.WHITE else 6) + _PIECE_MAP[piece.piece_type]
        tensor[idx, square // 8, square % 8] = 1.0

    if board.is_repetition(2):
        tensor[offset + 12, :, :] = 1.0
    if board.is_repetition(3):
        tensor[offset + 13, :, :] = 1.0


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

    def to_tensor(self, history=None):
        """
        Returns a (120, 8, 8) float32 tensor.

        Layout:
          Planes   0-13  : Frame 0 — current position (12 piece + 2 repetition planes)
          Planes  14-27  : Frame 1 — 1 move ago
          Planes  28-41  : Frame 2 — 2 moves ago
          ...
          Planes 98-111  : Frame 7 — 7 moves ago
          Plane  112     : White kingside castling right
          Plane  113     : White queenside castling right
          Plane  114     : Black kingside castling right
          Plane  115     : Black queenside castling right
          Plane  116     : En passant target square (single cell = 1.0)
          Plane  117     : Side to move (1.0 = Black, 0.0 = White)
          Plane  118     : 50-move clock  (normalized: halfmoves / 100)
          Plane  119     : Total move count (normalized: fullmove / 400)

        history: list of chess.Board objects, most recent first (up to 7 entries).
                 Boards must be full copies so is_repetition() works correctly.
                 If None or shorter than 7, missing frames are left as zero planes.
        """
        tensor = np.zeros((120, 8, 8), dtype=np.float32)

        # Frame 0: current position
        _fill_board_planes(tensor, self.board, offset=0)

        # Frames 1-7: history (zero-padded if history is short)
        if history:
            for i, hist_board in enumerate(history[:7]):
                _fill_board_planes(tensor, hist_board, offset=14 * (i + 1))

        # Auxiliary planes (current position only)
        tensor[112, :, :] = float(self.board.has_kingside_castling_rights(chess.WHITE))
        tensor[113, :, :] = float(self.board.has_queenside_castling_rights(chess.WHITE))
        tensor[114, :, :] = float(self.board.has_kingside_castling_rights(chess.BLACK))
        tensor[115, :, :] = float(self.board.has_queenside_castling_rights(chess.BLACK))

        if self.board.ep_square is not None:
            tensor[116, self.board.ep_square // 8, self.board.ep_square % 8] = 1.0

        if self.board.turn == chess.BLACK:
            tensor[117, :, :] = 1.0

        tensor[118, :, :] = min(self.board.halfmove_clock, 100) / 100.0
        tensor[119, :, :] = min(self.board.fullmove_number, 400) / 400.0

        return tensor

    def get_reward_for_turn(self, turn_val):
        res = self.board.result()
        if res == "1-0": return 1.0 if turn_val == 1.0 else -1.0
        if res == "0-1": return 1.0 if turn_val == 0.0 else -1.0
        return 0.0
