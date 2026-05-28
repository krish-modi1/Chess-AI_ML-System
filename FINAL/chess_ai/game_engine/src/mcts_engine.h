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

namespace py = pybind11;

// === CONSTANTS (from mcts.py) ===
constexpr float VIRTUAL_LOSS = 3.0f;
constexpr float CPUCT_INIT = 1.0f;
constexpr int CPUCT_BASE = 19652;
constexpr float DIRICHLET_ALPHA = 0.3f;
constexpr float DIRICHLET_FRAC = 0.25f;

// === Move Encoding (matches mcts.py move_to_index exactly) ===
inline int move_to_index(const std::string& move_str) {
    int src = (move_str[0] - 'a') + (move_str[1] - '1') * 8;
    int dst = (move_str[2] - 'a') + (move_str[3] - '1') * 8;
    int idx = src * 64 + dst;
    
    if (move_str.length() == 5 && move_str[4] != 'q') {
        int promo_type = 0;
        if (move_str[4] == 'n') promo_type = 0;
        else if (move_str[4] == 'b') promo_type = 1;
        else if (move_str[4] == 'r') promo_type = 2;
        
        int src_col = src % 8;
        int dst_col = dst % 8;
        int direction = (dst_col - src_col) + 1;
        idx = 4096 + (src_col * 9) + (direction * 3) + promo_type;
    }
    return (idx < 8192) ? idx : 0;
}

// === MCTSNode Class (matches mcts.py Node class) ===
class MCTSNode : public std::enable_shared_from_this<MCTSNode> {
public:
    py::object py_state;
    std::unordered_map<std::string, std::shared_ptr<MCTSNode>> children;
    
    // ═══════════════════════════════════════════════════════════════════
    // FIX 1: Change parent to weak_ptr to break circular reference
    // ═══════════════════════════════════════════════════════════════════
    std::weak_ptr<MCTSNode> parent;  // Was: std::shared_ptr<MCTSNode>
    
    int visit_count = 0;
    float value_sum = 0.0f;
    float prior = 0.0f;
    float virtual_loss = 0.0f;
    
    MCTSNode() = default;
    MCTSNode(py::object state, std::shared_ptr<MCTSNode> p = nullptr, float pr = 0.0f)
        : py_state(state), parent(p), prior(pr) {}  // weak_ptr auto-converts from shared_ptr
    
    // Delete copy
    MCTSNode(const MCTSNode&) = delete;
    MCTSNode& operator=(const MCTSNode&) = delete;
    MCTSNode(MCTSNode&&) = default;
    MCTSNode& operator=(MCTSNode&&) = default;
    
    // ═══════════════════════════════════════════════════════════════════
    // FIX 2: Destructor to explicitly release Python object
    // ═══════════════════════════════════════════════════════════════════
    ~MCTSNode() {
        // Clear children first to help with destruction order
        children.clear();
        // Release Python object reference
        py_state = py::none();
    }
    
    bool is_expanded() const { 
        return !children.empty(); 
    }
    
    float value() const {
        float visits = visit_count + virtual_loss;
        return (visits > 0) ? value_sum / visits : 0.0f;
    }
    
    std::pair<std::string, std::shared_ptr<MCTSNode>> select_child();
    
    void expand(const std::vector<std::string>& valid_moves, 
                const std::vector<float>& policy_logits);
    
    std::string best_action() const;
};

// === MCTSEngine Class ===
class MCTSEngine {
public:
    int simulations;
    int batch_size;
    std::mt19937 rng;
    
    MCTSEngine(int sims = 800, int bs = 8) : simulations(sims), batch_size(bs) {}
    
    std::pair<std::string, py::array_t<float>> search(
        py::object root_state,
        const py::array_t<float>& initial_policy,
        float initial_value,
        float temperature,
        uint32_t seed,
        py::function inference_callback
    );
    
private:
    void backpropagate(const std::vector<std::shared_ptr<MCTSNode>>& path,
                      float value, float leaf_turn_player);
    
    py::array_t<float> get_policy_vector(
        const std::shared_ptr<MCTSNode>& root,
        float temperature = 1.0f);
    
    void add_dirichlet_noise(std::shared_ptr<MCTSNode>& root);
    
    // ═══════════════════════════════════════════════════════════════════
    // FIX 3: Explicit tree cleanup helper
    // ═══════════════════════════════════════════════════════════════════
    void clear_tree(std::shared_ptr<MCTSNode>& root);
};
