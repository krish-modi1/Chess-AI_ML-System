#include "mcts_engine.h"

PYBIND11_MODULE(mcts_engine_cpp, m)
{
    m.doc() = "MCTS C++ Engine with native chess board — no Python GIL crossings in tree operations";

    // ChessBoard exposed to Python so inference callbacks can call .to_tensor()
    py::class_<ChessBoard>(m, "ChessBoard")
        .def(py::init<>(),
             "Create board at starting position")
        .def(py::init<const std::string&>(),
             py::arg("fen"),
             "Create board from FEN string")
        .def("copy", &ChessBoard::copy,
             "Return a deep copy of this board")
        .def("push", &ChessBoard::push,
             py::arg("uci"),
             "Apply a move in UCI notation. Returns True if the move was legal.")
        .def("legal_moves", &ChessBoard::legal_moves,
             "Return list of legal move UCI strings")
        .def("is_over", &ChessBoard::is_over,
             "Return True if the game is over")
        .def("turn_player", &ChessBoard::turn_player,
             "Return 1.0 for White to move, 0.0 for Black to move")
        .def("get_reward_for_turn", &ChessBoard::get_reward_for_turn,
             py::arg("turn_player_val"),
             "Return +1/-1/0 from the perspective of turn_player_val (1.0=White)")
        .def("to_tensor", &ChessBoard::to_tensor,
             py::arg("history") = py::none(),
             "Return (120,8,8) float32 array. history: optional list of ChessBoard (most recent first)")
        .def("fen", &ChessBoard::fen,
             "Return FEN string for current position");

    py::class_<MCTSEngine>(m, "MCTSEngine")
        .def(py::init<int, int>(),
             py::arg("simulations") = 800,
             py::arg("batch_size") = 8)

        .def("search", &MCTSEngine::search,
             py::arg("root_state"),
             py::arg("initial_policy"),
             py::arg("initial_value"),
             py::arg("temperature") = 1.0f,
             py::arg("seed") = 0u,
             py::arg("inference_callback"))

        .def("advance_root", &MCTSEngine::advance_root,
             py::arg("played_move"))

        .def("reset_cache", &MCTSEngine::reset_cache)

        .def_readwrite("simulations", &MCTSEngine::simulations)
        .def_readwrite("batch_size", &MCTSEngine::batch_size);
}
