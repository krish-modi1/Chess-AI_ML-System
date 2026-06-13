#pragma once

#include <vector>
#include <unordered_map>
#include <memory>
#include <string>
#include <cmath>
#include <algorithm>
#include <random>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include "chess_board.h"

namespace py = pybind11;

constexpr float VIRTUAL_LOSS = 3.0f;
constexpr float CPUCT_INIT = 1.25f;   // AlphaZero value — more exploration → broader position coverage
constexpr int CPUCT_BASE = 19652;
constexpr float DIRICHLET_ALPHA = 0.3f;
constexpr float DIRICHLET_FRAC = 0.25f;

// AlphaZero 4672 policy encoding: 64 source squares × 73 planes.
// Planes 0-55: queen-like moves (8 directions × 7 distances).
//   dir 0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW; plane = dir*7 + (dist-1)
// Planes 56-63: knight moves (8 L-shape directions).
// Planes 64-72: underpromotions (3 file-delta dirs × 3 pieces: N=0,B=1,R=2).
//   dir 0=left(df<0), 1=straight(df=0), 2=right(df>0)
inline int move_to_index(const std::string& uci) {
    int src_file = uci[0] - 'a';
    int src_rank = uci[1] - '1';
    int dst_file = uci[2] - 'a';
    int dst_rank = uci[3] - '1';
    int src = src_file + src_rank * 8;
    int df = dst_file - src_file;
    int dr = dst_rank - src_rank;

    // Underpromotion (non-queen promotion)
    if (uci.size() == 5 && uci[4] != 'q') {
        char promo = uci[4];
        int piece = (promo == 'n') ? 0 : (promo == 'b') ? 1 : 2;
        int dir   = (df < 0) ? 0 : (df == 0) ? 1 : 2;
        return src * 73 + 64 + dir * 3 + piece;
    }

    int abs_df = std::abs(df);
    int abs_dr = std::abs(dr);

    // Knight move
    if ((abs_df == 1 && abs_dr == 2) || (abs_df == 2 && abs_dr == 1)) {
        int kdir;
        if      (df ==  1 && dr ==  2) kdir = 0;
        else if (df ==  2 && dr ==  1) kdir = 1;
        else if (df ==  2 && dr == -1) kdir = 2;
        else if (df ==  1 && dr == -2) kdir = 3;
        else if (df == -1 && dr == -2) kdir = 4;
        else if (df == -2 && dr == -1) kdir = 5;
        else if (df == -2 && dr ==  1) kdir = 6;
        else                           kdir = 7;  // df==-1, dr==2
        return src * 73 + 56 + kdir;
    }

    // Queen-like move (includes queen promotion)
    int dir, dist;
    if (df == 0) {
        dir  = (dr > 0) ? 0 : 4;
        dist = abs_dr - 1;
    } else if (dr == 0) {
        dir  = (df > 0) ? 2 : 6;
        dist = abs_df - 1;
    } else {
        dist = abs_df - 1;  // == abs_dr - 1 for diagonals
        if      (df > 0 && dr > 0) dir = 1;   // NE
        else if (df < 0 && dr > 0) dir = 7;   // NW
        else if (df > 0 && dr < 0) dir = 3;   // SE
        else                       dir = 5;   // SW
    }
    return src * 73 + dir * 7 + dist;
}

class MCTSNode : public std::enable_shared_from_this<MCTSNode> {
public:
    ChessBoard board;
    std::unordered_map<std::string, std::shared_ptr<MCTSNode>> children;
    std::weak_ptr<MCTSNode> parent;

    int visit_count = 0;
    float value_sum = 0.0f;
    float prior = 0.0f;
    float virtual_loss = 0.0f;

    MCTSNode() = default;
    MCTSNode(ChessBoard b, std::shared_ptr<MCTSNode> p = nullptr, float pr = 0.0f)
        : board(std::move(b)), parent(p), prior(pr) {}

    MCTSNode(const MCTSNode&) = delete;
    MCTSNode& operator=(const MCTSNode&) = delete;
    MCTSNode(MCTSNode&&) = default;
    MCTSNode& operator=(MCTSNode&&) = default;

    ~MCTSNode() {
        children.clear();
    }

    bool is_expanded() const { return !children.empty(); }

    float value() const {
        float visits = visit_count + virtual_loss;
        return (visits > 0) ? value_sum / visits : 0.0f;
    }

    std::pair<std::string, std::shared_ptr<MCTSNode>> select_child(bool self_play = true);
    void expand(const std::vector<std::string>& valid_moves,
                const std::vector<float>& policy_logits);
    std::string best_action() const;
};

class MCTSEngine {
public:
    int simulations;
    int batch_size;
    std::mt19937 rng;
    std::shared_ptr<MCTSNode> cached_root;

    MCTSEngine(int sims = 800, int bs = 8) : simulations(sims), batch_size(bs) {}

    std::pair<std::string, py::array_t<float>> search(
        ChessBoard root_state,
        const py::array_t<float>& initial_policy,
        float initial_value,
        float temperature,
        uint32_t seed,
        py::function inference_callback,
        bool use_dirichlet = true
    );

    bool advance_root(const std::string& played_move);
    void reset_cache();

private:
    void backpropagate(const std::vector<std::shared_ptr<MCTSNode>>& path,
                      float value, float leaf_turn_player);

    py::array_t<float> get_policy_vector(
        const std::shared_ptr<MCTSNode>& root,
        float temperature = 1.0f);

    void add_dirichlet_noise(std::shared_ptr<MCTSNode>& root);
    void clear_tree(std::shared_ptr<MCTSNode>& root);
};
