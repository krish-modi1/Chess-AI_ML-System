#include "mcts_engine.h"
#include <cmath>
#include <algorithm>
#include <numeric>

// === select_child() ===
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

// === expand() ===
void MCTSNode::expand(const std::vector<std::string>& valid_moves,
                     const std::vector<float>& policy_logits) {
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
        float normalized_prior;
        if (policy_sum > 0) {
            normalized_prior = move_probs[move] / policy_sum;
        } else {
            normalized_prior = 1.0f / valid_moves.size();
        }
        
        py::object next_state = py_state.attr("copy")();
        next_state.attr("push")(move);
        auto child = std::make_shared<MCTSNode>(next_state, shared_from_this(), normalized_prior);
        children[move] = child;
    }
}

// === best_action() ===
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

// === backpropagate() ===
void MCTSEngine::backpropagate(const std::vector<std::shared_ptr<MCTSNode>>& path,
                               float value, float leaf_turn_player) {
    auto leaf = path.back();
    leaf->virtual_loss -= VIRTUAL_LOSS;
    leaf->value_sum += VIRTUAL_LOSS;
    
    for (auto& node : path) {
        node->visit_count += 1;
        float turn_val = py::cast<float>(node->py_state.attr("turn_player"));
        if (turn_val == leaf_turn_player) {
            node->value_sum += value;
        } else {
            node->value_sum -= value;
        }
    }
}

// === add_dirichlet_noise() ===
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
        for (auto& n : noise) {
            n /= noise_sum;
        }
    }
    
    size_t i = 0;
    for (auto& [action, child] : root->children) {
        child->prior = (1.0f - DIRICHLET_FRAC) * child->prior + DIRICHLET_FRAC * noise[i];
        i++;
    }
}

// === get_policy_vector() ===
py::array_t<float> MCTSEngine::get_policy_vector(const std::shared_ptr<MCTSNode>& root, float temperature) {
    std::vector<float> policy(8192, 0.0f);
    std::unordered_map<int, float> visits;
    
    for (const auto& [action_uci, child] : root->children) {
        int idx = move_to_index(action_uci);
        if (idx < 8192) {
            visits[idx] = (float)child->visit_count;
        }
    }
    
    if (visits.empty()) {
        return py::array_t<float>(8192, policy.data());
    }
    
    std::vector<float> counts;
    std::vector<int> indices;
    for (const auto& [idx, v] : visits) {
        indices.push_back(idx);
        counts.push_back(v);
    }
    
    float total = 0.0f;
    for (auto& c : counts) {
        float exponent = 1.3f;
        c = std::pow(c, exponent);
        total += c;
    }
    
    if (total > 0) {
        for (size_t i = 0; i < indices.size(); i++) {
            policy[indices[i]] = counts[i] / total;
        }
    }
    
    return py::array_t<float>(8192, policy.data());
}

// ═══════════════════════════════════════════════════════════════════════════════
// FIX 3: Explicit tree cleanup - breaks parent references and clears children
// This ensures proper destruction even if there are lingering shared_ptrs
// ═══════════════════════════════════════════════════════════════════════════════
void MCTSEngine::clear_tree(std::shared_ptr<MCTSNode>& root) {
    if (!root) return;
    
    // Use iterative approach to avoid stack overflow on deep trees
    std::vector<std::shared_ptr<MCTSNode>> nodes_to_clear;
    nodes_to_clear.push_back(root);
    
    while (!nodes_to_clear.empty()) {
        auto node = nodes_to_clear.back();
        nodes_to_clear.pop_back();
        
        if (!node) continue;
        
        // Queue all children for clearing
        for (auto& [action, child] : node->children) {
            if (child) {
                nodes_to_clear.push_back(child);
            }
        }
        
        // Clear this node's children map (releases shared_ptrs)
        node->children.clear();
        
        // Reset parent weak_ptr
        node->parent.reset();
        
        // Release Python object
        node->py_state = py::none();
    }
    
    // Finally reset the root itself
    root.reset();
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN SEARCH
// ═══════════════════════════════════════════════════════════════════════════════
std::pair<std::string, py::array_t<float>> MCTSEngine::search(
    py::object root_state,
    const py::array_t<float>& initial_policy,
    float initial_value,
    float temperature,
    uint32_t seed,
    py::function inference_callback) {
    
    rng.seed(seed);
    
    auto root = std::make_shared<MCTSNode>(root_state);
    
    auto policy_buf = initial_policy.request();
    std::vector<float> policy_vec((float*)policy_buf.ptr,
                                   (float*)policy_buf.ptr + policy_buf.size);
    
    auto legal_moves = py::cast<std::vector<std::string>>(
        root_state.attr("legal_moves")());
    root->expand(legal_moves, policy_vec);
    
    add_dirichlet_noise(root);
    
    int num_iterations = std::max(1, simulations / batch_size);
    
    // ════════════════════════════════════════════════════════════════════════
    // MAIN MCTS LOOP
    // ════════════════════════════════════════════════════════════════════════
    for (int iter = 0; iter < num_iterations; iter++) {
        
        std::vector<std::shared_ptr<MCTSNode>> leaves;
        std::vector<std::vector<std::shared_ptr<MCTSNode>>> paths;
        std::vector<py::object> leaf_states;
        
        // Reserve space to reduce reallocations
        leaves.reserve(batch_size);
        paths.reserve(batch_size);
        leaf_states.reserve(batch_size);
        
        // ════════════════════════════════════════════════════════════════════
        // SELECTION PHASE
        // ════════════════════════════════════════════════════════════════════
        for (int i = 0; i < batch_size; i++) {
            auto node = root;
            std::vector<std::shared_ptr<MCTSNode>> path;
            path.reserve(64);  // Typical max depth
            path.push_back(node);
            
            while (node->is_expanded()) {
                std::uniform_real_distribution<float> epsilon_dist(0.0f, 1.0f);
                constexpr float epsilon = 0.05f;
                
                if (epsilon_dist(rng) < epsilon && node->children.size() > 1) {
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
            
            node->virtual_loss += VIRTUAL_LOSS;
            node->value_sum -= VIRTUAL_LOSS;
            
            bool is_over = py::cast<bool>(node->py_state.attr("is_over"));
            if (is_over) {
                float reward = py::cast<float>(
                    node->py_state.attr("get_reward_for_turn")(
                        node->py_state.attr("turn_player")));
                backpropagate(path, reward, py::cast<float>(
                    node->py_state.attr("turn_player")));
            } else {
                leaves.push_back(node);
                paths.push_back(std::move(path));
                leaf_states.push_back(node->py_state);
            }
        }
        
        if (leaves.empty()) continue;
        
        // ════════════════════════════════════════════════════════════════════
        // INFERENCE PHASE
        // ════════════════════════════════════════════════════════════════════
        py::list py_leaf_states;
        for (auto& state : leaf_states) {
            py_leaf_states.append(state);
        }
        
        py::object result = inference_callback(py_leaf_states);
        
        py::array_t<float> policies_array = result.attr("__getitem__")(0).cast<py::array_t<float>>();
        py::array_t<float> values_array = result.attr("__getitem__")(1).cast<py::array_t<float>>();
        
        auto policies_buf = policies_array.request();
        auto values_buf = values_array.request();
        
        float* policies_ptr = (float*)policies_buf.ptr;
        float* values_ptr = (float*)values_buf.ptr;
        
        // ════════════════════════════════════════════════════════════════════
        // EXPANSION & BACKPROPAGATION PHASE
        // ════════════════════════════════════════════════════════════════════
        for (size_t i = 0; i < leaves.size(); i++) {
            auto node = leaves[i];
            auto& path = paths[i];
            
            auto next_legal = py::cast<std::vector<std::string>>(
                node->py_state.attr("legal_moves")());
            
            std::vector<float> leaf_policy(8192);
            for (int j = 0; j < 8192; j++) {
                leaf_policy[j] = policies_ptr[i * 8192 + j];
            }
            
            node->expand(next_legal, leaf_policy);
            
            float leaf_value = values_ptr[i];
            
            backpropagate(path, leaf_value, py::cast<float>(
                node->py_state.attr("turn_player")));
        }
        
        // Clear temporary vectors explicitly
        leaves.clear();
        paths.clear();
        leaf_states.clear();
    }
    
    // ════════════════════════════════════════════════════════════════════════
    // GET RESULT BEFORE CLEANUP
    // ════════════════════════════════════════════════════════════════════════
    std::string best_move = root->best_action();
    auto policy = get_policy_vector(root, temperature);
    
    // ════════════════════════════════════════════════════════════════════════
    // FIX: Explicitly clear tree to free memory
    // ════════════════════════════════════════════════════════════════════════
    clear_tree(root);
    
    return {best_move, policy};
}
