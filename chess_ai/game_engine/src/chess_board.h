#pragma once
#include "chess.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>

namespace py = pybind11;

class ChessBoard
{
public:
    chess::Board board_;

    ChessBoard();                                // starting position
    explicit ChessBoard(const std::string &fen); // from FEN string

    ChessBoard copy() const;
    bool push(const std::string &uci);
    const std::vector<std::string>& legal_moves() const;

    bool is_over() const;
    float turn_player() const;
    float get_reward_for_turn(float turn_player_val) const;

    py::array_t<float> to_tensor(py::object history_py = py::none()) const;

    std::string fen() const;

private:
    explicit ChessBoard(const chess::Board& b, int ep_sq = -1);  // used by copy()
    mutable std::vector<std::string> legal_moves_cache_;

    // EP square in python-chess convention (0-63, -1 = none).
    // Tracked manually because chess.hpp only sets ep_sq_ when a capture is
    // actually possible, but python-chess always records it on any double-push.
    int ep_square_ = -1;

    static void fill_planes(float* tensor, const chess::Board& board, int offset);
};
