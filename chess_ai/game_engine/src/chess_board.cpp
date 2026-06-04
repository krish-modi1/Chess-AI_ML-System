#include "chess_board.h"

ChessBoard::ChessBoard() : board_(), ep_square_(-1) {}

ChessBoard::ChessBoard(const std::string &fen) : board_(fen), ep_square_(-1)
{
    // Initialise ep_square_ from the FEN if the library parsed one
    auto ep = board_.enpassantSq();
    if (ep != chess::Square::NO_SQ)
        ep_square_ = ep.index();
}

ChessBoard::ChessBoard(const chess::Board& b, int ep_sq)
    : board_(b), ep_square_(ep_sq) {}

ChessBoard ChessBoard::copy() const
{
    return ChessBoard(board_, ep_square_);
}

bool ChessBoard::push(const std::string &uci)
{
    // Precondition: uci must be from legal_moves() — MCTS only calls push with
    // moves it generated, so uciToMove will not throw in normal operation.
    chess::Move move = chess::uci::uciToMove(board_, uci);
    if (move == chess::Move::NO_MOVE)
        return false;
    if (!chess::movegen::isLegal(board_, move))
        return false;

    // Track EP square using python-chess semantics: any pawn double-push sets
    // ep_square_ to the target square regardless of whether capture is possible.
    // chess.hpp only sets ep_sq_ when a capture is actually available, which
    // diverges from python-chess and would corrupt plane 116 of to_tensor().
    ep_square_ = -1;
    auto from_sq = move.from();
    auto to_sq   = move.to();
    auto piece   = board_.at(from_sq);
    if (piece.type() == chess::PieceType::PAWN) {
        int dist = to_sq.index() - from_sq.index();
        if (dist == 16 || dist == -16) {
            // Double push: EP target is the square between from and to
            ep_square_ = (from_sq.index() + to_sq.index()) / 2;
        }
    }

    board_.makeMove(move);
    legal_moves_cache_.clear();
    return true;
}

const std::vector<std::string>& ChessBoard::legal_moves() const
{
    if (legal_moves_cache_.empty()) {
        chess::Movelist moves;
        chess::movegen::legalmoves(moves, board_);
        legal_moves_cache_.reserve(moves.size());
        for (const auto &m : moves)
        {
            legal_moves_cache_.push_back(chess::uci::moveToUci(m));
        }
    }
    return legal_moves_cache_;
}

bool ChessBoard::is_over() const
{
    auto [reason, result] = board_.isGameOver();
    return reason != chess::GameResultReason::NONE;
}

float ChessBoard::turn_player() const
{
    return (board_.sideToMove() == chess::Color::WHITE) ? 1.0f : 0.0f;
}

float ChessBoard::get_reward_for_turn(float turn_player_val) const
{
    auto [reason, result] = board_.isGameOver();
    if (reason == chess::GameResultReason::NONE)
        return 0.0f;
    if (reason != chess::GameResultReason::CHECKMATE)
        return 0.0f; // all other endings are draws
    // On checkmate: the player currently to move is the one who has been checkmated (lost)
    bool white_is_checkmated = (board_.sideToMove() == chess::Color::WHITE);
    if (turn_player_val == 1.0f)
    { // asking from White's perspective
        return white_is_checkmated ? -1.0f : 1.0f;
    }
    else
    { // asking from Black's perspective
        return white_is_checkmated ? 1.0f : -1.0f;
    }
}

std::string ChessBoard::fen() const
{
    return board_.getFen();
}

void ChessBoard::fill_planes(float* t, const chess::Board& board, int offset)
{
    for (int sq = 0; sq < 64; sq++) {
        auto piece = board.at(chess::Square(sq));
        if (piece != chess::Piece::NONE) {
            int pt = static_cast<int>(piece.type());  // 0-5
            int color_off = (piece.color() == chess::Color::BLACK) ? 6 : 0;
            int plane = offset + pt + color_off;
            t[plane * 64 + (sq / 8) * 8 + (sq % 8)] = 1.0f;
        }
    }
    if (board.isRepetition(1))   // 1 prior occurrence = 2 total = python-chess is_repetition(2)
        for (int i = 0; i < 64; i++) t[(offset + 12) * 64 + i] = 1.0f;
    if (board.isRepetition(2))   // 2 prior occurrences = 3 total = python-chess is_repetition(3)
        for (int i = 0; i < 64; i++) t[(offset + 13) * 64 + i] = 1.0f;
}

py::array_t<float> ChessBoard::to_tensor(py::object history_py) const
{
    auto result = py::array_t<float>({120, 8, 8});
    float* raw = result.mutable_data();
    std::fill(raw, raw + 120 * 64, 0.0f);

    fill_planes(raw, board_, 0);

    if (!history_py.is_none()) {
        auto history = history_py.cast<std::vector<ChessBoard>>();
        int hist_count = std::min((int)history.size(), 7);
        for (int i = 0; i < hist_count; i++) {
            fill_planes(raw, history[i].board_, 14 * (i + 1));
        }
    }

    // Castling rights (planes 112-115)
    using Side = chess::Board::CastlingRights::Side;
    auto cr = board_.castlingRights();
    const struct { chess::Color c; Side s; } rights[4] = {
        {chess::Color::WHITE, Side::KING_SIDE},
        {chess::Color::WHITE, Side::QUEEN_SIDE},
        {chess::Color::BLACK, Side::KING_SIDE},
        {chess::Color::BLACK, Side::QUEEN_SIDE},
    };
    for (int p = 0; p < 4; p++) {
        if (cr.has(rights[p].c, rights[p].s))
            for (int i = 0; i < 64; i++) raw[(112 + p) * 64 + i] = 1.0f;
    }

    // En passant (plane 116)
    // Use our manually-tracked ep_square_ which mirrors python-chess behaviour:
    // set on any double pawn push, even when no enemy pawn can capture.
    if (ep_square_ >= 0 && ep_square_ < 64)
        raw[116 * 64 + (ep_square_ / 8) * 8 + (ep_square_ % 8)] = 1.0f;

    // Side to move (plane 117)
    if (board_.sideToMove() == chess::Color::BLACK)
        for (int i = 0; i < 64; i++) raw[117 * 64 + i] = 1.0f;

    // 50-move clock (plane 118)
    float hmc = std::min((uint32_t)board_.halfMoveClock(), (uint32_t)100) / 100.0f;
    for (int i = 0; i < 64; i++) raw[118 * 64 + i] = hmc;

    // Fullmove number (plane 119)
    float fmn = std::min((uint32_t)board_.fullMoveNumber(), (uint32_t)400) / 400.0f;
    for (int i = 0; i < 64; i++) raw[119 * 64 + i] = fmn;

    return result;
}
