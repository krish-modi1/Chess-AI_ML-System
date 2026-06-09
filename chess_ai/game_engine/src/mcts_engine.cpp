#include "mcts_engine.h"
#include <cmath>
#include <algorithm>
#include <numeric>

// === select_child() — unchanged ===
std::pair<std::string, std::shared_ptr<MCTSNode>> MCTSNode::select_child() {
    float best_score = -1e9f;
    std::string best_action;
    std::shared_ptr<MCTSNode> best_child;

    float parent_visits = visit_count + virtual_loss;
    float cpuct = CPUCT_INIT + std::log((parent_visits + CPUCT_BASE) / CPUCT_BASE);
    float sqrt_parent_visits = std::sqrt(std::max(1.0f, parent_visits));

    for (auto& [action, child] : children) {
        float child_visits = child->visit_count + child->virtual_loss;
        float q_value = -child->value();
        float u_value = cpuct * child->prior * sqrt_parent_visits / (1.0f + child_visits);
        float score = q_value + u_value;

        if (score > best_score) {
            best_score = score;
            best_action = action;
            best_child = child;
        }
    }

    return {best_action, best_child};
}

// === expand() — pure C++, no Python copies ===
void MCTSNode::expand(const std::vector<std::string>& valid_moves,
                     const std::vector<float>& policy_logits) {
    if (is_expanded()) return;   // guard: second call in same batch is a no-op
    std::unordered_map<std::string, float> move_probs;
    float policy_sum = 0.0f;

    for (const auto& move_str : valid_moves) {
        int idx = move_to_index(move_str);
        float logit = (idx < (int)policy_logits.size()) ? policy_logits[idx] : -10.0f;
        float prob = std::exp(logit);
        move_probs[move_str] = prob;
        policy_sum += prob;
    }

    for (const auto& move : valid_moves) {
        float normalized_prior = (policy_sum > 0) ?
            move_probs[move] / policy_sum :
            1.0f / valid_moves.size();

        ChessBoard child_board = board.copy();   // pure C++ — no GIL crossing
        child_board.push(move);                  // pure C++

        auto child = std::make_shared<MCTSNode>(
            std::move(child_board), shared_from_this(), normalized_prior);
        children[move] = child;
    }
}

// === best_action() — unchanged ===
std::string MCTSNode::best_action() const {
    int most_visits = -1;
    std::string best = "";

    for (const auto& [action, child] : children) {
        if (child->visit_count > most_visits) {
            most_visits = child->visit_count;
            best = action;
        }
    }

    return best;
}

// === backpropagate() — C++ turn_player, no GIL crossing ===
void MCTSEngine::backpropagate(const std::vector<std::shared_ptr<MCTSNode>>& path,
                               float value, float leaf_turn_player) {
    for (auto& node : path) {
        // Undo the virtual loss added to EVERY node on this path during selection
        // (see the selection loop). Removing it leaf-only — as the previous code did —
        // left intermediate nodes permanently penalized within a batch, so all
        // batch_size (e.g. 320) selections funneled down nearly the same path instead
        // of diversifying. Whole-path add/remove is the standard leaf-parallel scheme.
        node->virtual_loss -= VIRTUAL_LOSS;
        node->value_sum    += VIRTUAL_LOSS;

        node->visit_count += 1;
        float turn_val = node->board.turn_player();   // pure C++ — no GIL crossing
        if (turn_val == leaf_turn_player) {
            node->value_sum += value;
        } else {
            node->value_sum -= value;
        }
    }
}

// === add_dirichlet_noise() — unchanged ===
void MCTSEngine::add_dirichlet_noise(std::shared_ptr<MCTSNode>& root) {
    if (root->children.empty()) return;

    size_t num_actions = root->children.size();
    std::gamma_distribution<float> gamma(DIRICHLET_ALPHA, 1.0f);

    std::vector<float> noise(num_actions);
    float noise_sum = 0.0f;
    for (size_t i = 0; i < num_actions; i++) {
        noise[i] = gamma(rng);
        noise_sum += noise[i];
    }

    if (noise_sum > 0) {
        for (auto& n : noise) n /= noise_sum;
    }

    size_t i = 0;
    for (auto& [action, child] : root->children) {
        child->prior = (1.0f - DIRICHLET_FRAC) * child->prior + DIRICHLET_FRAC * noise[i];
        i++;
    }
}

// === get_policy_vector() ===
py::array_t<float> MCTSEngine::get_policy_vector(const std::shared_ptr<MCTSNode>& root, float temperature) {
    std::vector<float> policy(4672, 0.0f);
    std::unordered_map<int, float> visits;

    for (const auto& [action_uci, child] : root->children) {
        int idx = move_to_index(action_uci);
        if (idx < 4672) visits[idx] = (float)child->visit_count;
    }

    if (visits.empty()) return py::array_t<float>(4672, policy.data());

    std::vector<float> counts;
    std::vector<int> indices;
    for (const auto& [idx, v] : visits) {
        indices.push_back(idx);
        counts.push_back(v);
    }

    // Greedy (temperature → 0): one-hot on the most-visited move(s), split among ties.
    // NOTE: the previous code used pow(count, 1e6) here, which overflows to inf for any
    // visit count >= 2, making total=inf and count/total = inf/inf = NaN. That corrupted
    // every policy target after the temperature schedule drops to 0.
    if (temperature < 1e-6f) {
        float max_c = *std::max_element(counts.begin(), counts.end());
        int n_max = 0;
        for (float c : counts) if (c == max_c) n_max++;
        for (size_t i = 0; i < indices.size(); i++) {
            policy[indices[i]] = (counts[i] == max_c) ? 1.0f / n_max : 0.0f;
        }
        return py::array_t<float>(4672, policy.data());
    }

    // Temperature scaling: count^(1/T), normalized.
    float exponent = 1.0f / temperature;
    float total = 0.0f;
    for (auto& c : counts) {
        c = std::pow(c, exponent);
        total += c;
    }

    if (total > 0) {
        for (size_t i = 0; i < indices.size(); i++) {
            policy[indices[i]] = counts[i] / total;
        }
    }

    return py::array_t<float>(4672, policy.data());
}

// === clear_tree() — ChessBoard destructs automatically, no py_state release ===
void MCTSEngine::clear_tree(std::shared_ptr<MCTSNode>& root) {
    if (!root) return;

    std::vector<std::shared_ptr<MCTSNode>> nodes_to_clear;
    nodes_to_clear.push_back(root);

    while (!nodes_to_clear.empty()) {
        auto node = nodes_to_clear.back();
        nodes_to_clear.pop_back();

        if (!node) continue;

        for (auto& [action, child] : node->children) {
            if (child) nodes_to_clear.push_back(child);
        }

        node->children.clear();
        node->parent.reset();
        // ChessBoard member 'board' destructs automatically with the node
    }

    root.reset();
}

// === advance_root() — unchanged ===
bool MCTSEngine::advance_root(const std::string& played_move) {
    if (!cached_root) return false;

    auto it = cached_root->children.find(played_move);
    if (it == cached_root->children.end()) {
        clear_tree(cached_root);
        cached_root.reset();
        return false;
    }

    auto new_root = it->second;
    cached_root->children.erase(it);
    clear_tree(cached_root);

    cached_root = new_root;
    cached_root->parent.reset();
    return true;
}

// === reset_cache() — unchanged ===
void MCTSEngine::reset_cache() {
    if (cached_root) {
        clear_tree(cached_root);
        cached_root.reset();
    }
}

// === search() — root_state is ChessBoard; all tree ops are pure C++ ===
std::pair<std::string, py::array_t<float>> MCTSEngine::search(
    ChessBoard root_state,
    const py::array_t<float>& initial_policy,
    float initial_value,
    float temperature,
    uint32_t seed,
    py::function inference_callback,
    bool use_dirichlet) {

    rng.seed(seed);

    std::shared_ptr<MCTSNode> root;
    if (cached_root && cached_root->is_expanded()) {
        root = cached_root;
    } else {
        root = std::make_shared<MCTSNode>(std::move(root_state));

        auto policy_buf = initial_policy.request();
        std::vector<float> policy_vec((float*)policy_buf.ptr,
                                      (float*)policy_buf.ptr + policy_buf.size);

        const auto& legal_moves = root->board.legal_moves();   // pure C++
        root->expand(legal_moves, policy_vec);
    }

    if (use_dirichlet) add_dirichlet_noise(root);

    int num_iterations = std::max(1, simulations / batch_size);

    for (int iter = 0; iter < num_iterations; iter++) {

        std::vector<std::shared_ptr<MCTSNode>> leaves;
        std::vector<std::vector<std::shared_ptr<MCTSNode>>> paths;
        std::vector<py::object> leaf_states;

        leaves.reserve(batch_size);
        paths.reserve(batch_size);
        leaf_states.reserve(batch_size);

        // ── SELECTION ────────────────────────────────────────────────────────
        for (int i = 0; i < batch_size; i++) {
            auto node = root;
            std::vector<std::shared_ptr<MCTSNode>> path;
            path.reserve(64);
            path.push_back(node);

            while (node->is_expanded()) {
                std::uniform_real_distribution<float> epsilon_dist(0.0f, 1.0f);
                constexpr float epsilon = 0.05f;

                // Epsilon-greedy is an exploration aid for self-play only. Gate it on
                // use_dirichlet so evaluation/arena (use_dirichlet=false) plays deterministic
                // PUCT — otherwise 5% random selections add noise to strength/ELO measurements.
                if (use_dirichlet && epsilon_dist(rng) < epsilon && node->children.size() > 1) {
                    std::uniform_int_distribution<int> child_dist(0, node->children.size() - 1);
                    int random_idx = child_dist(rng);
                    auto it = node->children.begin();
                    std::advance(it, random_idx);
                    path.push_back(it->second);
                    node = it->second;
                } else {
                    auto [action, next_node] = node->select_child();
                    path.push_back(next_node);
                    node = next_node;
                }
            }

            // Apply virtual loss to EVERY node on the path (not just the leaf) so the
            // next selection in this batch is steered away from the whole explored path,
            // not only its tip. Removed again in backpropagate().
            for (auto& n : path) {
                n->virtual_loss += VIRTUAL_LOSS;
                n->value_sum    -= VIRTUAL_LOSS;
            }

            if (node->board.is_over()) {                       // pure C++ — no GIL crossing
                float tp = node->board.turn_player();
                float reward = node->board.get_reward_for_turn(tp);
                backpropagate(path, reward, tp);
            } else {
                leaves.push_back(node);
                paths.push_back(std::move(path));
                // py::cast creates a Python-wrapped copy of ChessBoard for the callback.
                // Copy is intentional — callback must not mutate MCTS tree nodes.
                leaf_states.push_back(py::cast(node->board));
            }
        }

        if (leaves.empty()) continue;

        // ── INFERENCE (Python callback — GPU inference must cross into Python) ──
        py::list py_leaf_states;
        for (auto& state : leaf_states) {
            py_leaf_states.append(state);
        }

        py::object result = inference_callback(py_leaf_states);

        py::array_t<float> policies_array = result.attr("__getitem__")(0).cast<py::array_t<float>>();
        py::array_t<float> values_array   = result.attr("__getitem__")(1).cast<py::array_t<float>>();

        auto policies_buf = policies_array.request();
        auto values_buf   = values_array.request();

        float* policies_ptr = (float*)policies_buf.ptr;
        float* values_ptr   = (float*)values_buf.ptr;

        // ── EXPANSION & BACKPROPAGATION ──────────────────────────────────────
        for (size_t i = 0; i < leaves.size(); i++) {
            auto node = leaves[i];
            auto& path = paths[i];

            const auto& next_legal = node->board.legal_moves();   // pure C++

            std::vector<float> leaf_policy(4672);
            for (int j = 0; j < 4672; j++) {
                leaf_policy[j] = policies_ptr[i * 4672 + j];
            }

            node->expand(next_legal, leaf_policy);

            float leaf_value = values_ptr[i];
            float tp = node->board.turn_player();                  // pure C++
            backpropagate(path, leaf_value, tp);
        }

        leaves.clear();
        paths.clear();
        leaf_states.clear();
    }

    std::string best_move = root->best_action();
    auto policy = get_policy_vector(root, temperature);
    cached_root = root;

    return {best_move, policy};
}
