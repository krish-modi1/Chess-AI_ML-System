#include "mcts_engine.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstdlib>

// Search-upgrade knobs, read ONCE from the environment. Defaults reproduce the current/AlphaZero
// behavior, so a rebuilt .so is a no-op until these are set in hyperparams.env.sh.
// See local/plans/upgrades-1-6.md.
static float env_float(const char* name, float def) {
    const char* v = std::getenv(name);
    return v ? (float)std::atof(v) : def;
}
static int env_int(const char* name, int def) {
    const char* v = std::getenv(name);
    return v ? std::atoi(v) : def;
}
static const float FPU_REDUCTION    = env_float("FPU_REDUCTION", 0.0f);     // 0 = off (unvisited Q=0)
static const float CPUCT_FACTOR     = env_float("CPUCT_FACTOR", 1.0f);      // 1 = current log term
static const float FORCED_PLAYOUT_K = env_float("FORCED_PLAYOUT_K", 0.0f);  // 0 = off (KataGo k=2)

// Gumbel AlphaZero (self-play only, gated on use_dirichlet). Default OFF → rebuilt .so is a no-op.
static const int   GUMBEL_ENABLE   = env_int("GUMBEL_ENABLE", 0);      // 1 = use Gumbel self-play search
static const int   GUMBEL_M        = env_int("GUMBEL_M", 16);          // sampled root actions (Gumbel-top-k)
static const float GUMBEL_C_VISIT  = env_float("GUMBEL_C_VISIT", 50.0f); // σ(q)=(c_visit+maxN)·c_scale·q
static const float GUMBEL_C_SCALE  = env_float("GUMBEL_C_SCALE", 1.0f);
static const int   GUMBEL_INTERIOR = env_int("GUMBEL_INTERIOR", 0);    // 1 = deterministic non-root rule (full)

// === select_child() — FPU + cpuct factor + forced playouts (all env-gated) ===
std::pair<std::string, std::shared_ptr<MCTSNode>> MCTSNode::select_child(bool self_play) {
    float best_score = -1e9f;
    std::string best_action;
    std::shared_ptr<MCTSNode> best_child;

    float parent_visits = visit_count + virtual_loss;
    float cpuct = CPUCT_INIT + CPUCT_FACTOR * std::log((parent_visits + CPUCT_BASE) / CPUCT_BASE);
    float sqrt_parent_visits = std::sqrt(std::max(1.0f, parent_visits));

    // FPU: score truly-unvisited children at (parent value − reduction·√explored-policy-mass)
    // instead of a flat 0. Disabled at the root so Dirichlet-noised root moves all get explored.
    const bool is_root = parent.expired();
    const bool fpu_on  = (FPU_REDUCTION > 0.0f) && !is_root;
    float parent_value = 0.0f, explored_prior_sum = 0.0f;
    if (fpu_on) {
        parent_value = this->value();
        for (auto& [a, c] : children)
            if (c->visit_count > 0) explored_prior_sum += c->prior;
    }
    // Forced playouts (self-play + root only): force each visited child to ≥ n_forced visits.
    // Gated on self_play so eval/arena plays pure PUCT (strongest move, no forced exploration).
    const bool forced_on = (FORCED_PLAYOUT_K > 0.0f) && is_root && self_play;
    const float total_root_visits = (float)visit_count;

    for (auto& [action, child] : children) {
        float child_visits = child->visit_count + child->virtual_loss;
        float q_value;
        if (fpu_on && child->visit_count == 0 && child->virtual_loss == 0.0f) {
            q_value = parent_value - FPU_REDUCTION * std::sqrt(explored_prior_sum);
        } else {
            q_value = -child->value();
        }
        float u_value = cpuct * child->prior * sqrt_parent_visits / (1.0f + child_visits);
        float score = q_value + u_value;

        if (forced_on && child->visit_count > 0) {
            float n_forced = std::sqrt(FORCED_PLAYOUT_K * child->prior * total_root_visits);
            if ((float)child->visit_count < n_forced) score = 1e9f;   // force-select until met
        }

        // `!best_child` forces the first child to always be selected, so this never returns a
        // null child even when every score is NaN. A non-finite NN prior/value makes `score`
        // NaN, and `NaN > x` is always false — previously that left best_child null, and the
        // caller then dereferenced a null node (reading children at offset 0x140 → segfault).
        if (!best_child || score > best_score) {
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
        // A non-finite NN logit (fp16 overflow → inf/nan) would make exp() inf/nan and poison
        // every PUCT score with NaN. Clamp to a finite range so priors stay well-defined.
        if (!std::isfinite(logit)) logit = -10.0f;
        if (logit > 30.0f) logit = 30.0f;          // exp(30)≈1e13; exp(88)=inf → avoid overflow
        float prob = std::exp(logit);
        move_probs[move_str] = prob;
        policy_sum += prob;
    }

    for (const auto& move : valid_moves) {
        float normalized_prior = (policy_sum > 0 && std::isfinite(policy_sum)) ?
            move_probs[move] / policy_sum :
            1.0f / valid_moves.size();
        if (!std::isfinite(normalized_prior)) normalized_prior = 1.0f / valid_moves.size();

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

    std::vector<int> indices;
    std::vector<float> counts;
    std::vector<float> priors;
    float total_visits = 0.0f;
    for (const auto& [action_uci, child] : root->children) {
        int idx = move_to_index(action_uci);
        if (idx < 4672) {
            indices.push_back(idx);
            counts.push_back((float)child->visit_count);
            priors.push_back(child->prior);
            total_visits += (float)child->visit_count;
        }
    }

    if (indices.empty()) return py::array_t<float>(4672, policy.data());

    // ── Policy-target pruning (KataGo §3.2): strip forced-playout / noise visits from the TARGET.
    //    Tree visit counts are untouched (best_action / played move unaffected) — only the stored
    //    policy target is sharpened. Only runs when forced playouts are on. ──
    if (FORCED_PLAYOUT_K > 0.0f && total_visits > 1.0f) {
        size_t star = 0;
        for (size_t i = 1; i < counts.size(); i++) if (counts[i] > counts[star]) star = i;
        float cpuct = CPUCT_INIT + CPUCT_FACTOR * std::log((total_visits + CPUCT_BASE) / CPUCT_BASE);
        float sqrtN = std::sqrt(std::max(1.0f, total_visits));
        auto upuct = [&](float prior, float v) { return cpuct * prior * sqrtN / (1.0f + v); };
        float puct_star = upuct(priors[star], counts[star]);
        for (size_t i = 0; i < counts.size(); i++) {
            if (i == star) continue;
            float n_forced = std::sqrt(FORCED_PLAYOUT_K * priors[i] * total_visits);
            float removed = 0.0f;
            while (counts[i] > 1.0f && removed < n_forced &&
                   upuct(priors[i], counts[i] - 1.0f) < puct_star) {
                counts[i] -= 1.0f; removed += 1.0f;
            }
            if (removed > 0.0f && counts[i] <= 1.0f) counts[i] = 0.0f;   // prune outright
        }
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

// === gumbel_interior_select() — full-Gumbel non-root rule (GUMBEL_INTERIOR=1) ===
// Pick the child maximizing [ π'(a) − N(a)/(1+ΣN) ], where π' is the node-local improved policy
// softmax(logit + σ(completedQ)). Drives interior visits toward the improved policy (better Q
// estimates at low sims than PUCT's noisy exploration term). node value stands in for v_π in v_mix.
std::shared_ptr<MCTSNode> MCTSEngine::gumbel_interior_select(const std::shared_ptr<MCTSNode>& node) {
    const size_t A = node->children.size();
    std::vector<std::shared_ptr<MCTSNode>> kids; kids.reserve(A);
    std::vector<float> logit(A), q(A);
    std::vector<int> N(A);
    float sumN = 0.0f, wq = 0.0f, wp = 0.0f, maxN = 0.0f;
    size_t i = 0;
    for (auto& [a, c] : node->children) {
        kids.push_back(c);
        logit[i] = std::log(std::max(c->prior, 1e-12f));
        q[i]     = -c->value();                 // node-STM Q of the child
        N[i]     = c->visit_count;
        sumN += (float)N[i];
        maxN = std::max(maxN, (float)N[i]);
        if (N[i] > 0) { wp += c->prior; wq += c->prior * q[i]; }
        i++;
    }
    float v_node = node->value();
    float v_mix = (sumN > 0.0f && wp > 1e-12f) ? (v_node + sumN * (wq / wp)) / (1.0f + sumN) : v_node;
    float scale = (GUMBEL_C_VISIT + maxN) * GUMBEL_C_SCALE;

    // improved policy π' = softmax(logit + σ(completedQ))
    std::vector<float> pi(A);
    float mx = -1e30f;
    for (size_t j = 0; j < A; j++) {
        float cq = (N[j] > 0) ? q[j] : v_mix;
        pi[j] = logit[j] + scale * cq;
        mx = std::max(mx, pi[j]);
    }
    float Z = 0.0f;
    for (size_t j = 0; j < A; j++) { pi[j] = std::exp(pi[j] - mx); Z += pi[j]; }
    if (Z > 0.0f) for (auto& p : pi) p /= Z;

    // argmax_a [ π'(a) − N(a)/(1+ΣN) ]
    float denom = 1.0f + sumN;
    size_t best = 0; float best_score = -1e30f;
    for (size_t j = 0; j < A; j++) {
        float score = pi[j] - (float)N[j] / denom;
        if (j == 0 || score > best_score) { best_score = score; best = j; }
    }
    return kids[best];
}

// === run_gumbel() — Gumbel-top-k + sequential halving at the root; improved-policy target ===
std::pair<std::string, py::array_t<float>> MCTSEngine::run_gumbel(
    std::shared_ptr<MCTSNode>& root, float root_value, float temperature,
    py::function& inference_callback) {

    // Index the root children.
    std::vector<std::string> acts;
    std::vector<std::shared_ptr<MCTSNode>> kids;
    acts.reserve(root->children.size());
    kids.reserve(root->children.size());
    for (auto& [a, c] : root->children) { acts.push_back(a); kids.push_back(c); }
    const int A = (int)acts.size();

    if (A == 0) { cached_root = root; return {"", get_policy_vector(root, temperature)}; }
    if (A == 1) {
        std::vector<float> pol(4672, 0.0f);
        int idx = move_to_index(acts[0]); if (idx < 4672) pol[idx] = 1.0f;
        cached_root = root;
        return {acts[0], py::array_t<float>(4672, pol.data())};
    }

    // logits = log(prior) (recovers NN logits up to a constant, which cancels in top-k & softmax).
    // Gumbel noise g(a); zeroed when temperature≈0 so late-game self-play exploits deterministically
    // (mirrors the AlphaZero temperature schedule — early explore, late greedy).
    const bool explore = (temperature >= 1e-6f);
    std::vector<float> logit(A), g(A);
    std::uniform_real_distribution<float> U(1e-12f, 1.0f);
    for (int i = 0; i < A; i++) {
        logit[i] = std::log(std::max(kids[i]->prior, 1e-12f));
        g[i] = explore ? -std::log(-std::log(U(rng))) : 0.0f;
    }

    // Gumbel-top-k: the m best actions by g+logit become the sequential-halving candidates.
    const int m = std::min(GUMBEL_M, A);
    std::vector<int> order(A);
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + m, order.end(),
        [&](int x, int y) { return g[x] + logit[x] > g[y] + logit[y]; });
    std::vector<int> live(order.begin(), order.begin() + m);

    const int n = std::max(1, simulations);
    const int num_phases = std::max(1, (int)std::ceil(std::log2((double)m)));

    auto q_hat     = [&](int i) -> float { return -kids[i]->value(); };   // root-STM Q of action i
    auto max_visit = [&]() -> float { float mx = 0.0f; for (auto& c : kids) mx = std::max(mx, (float)c->visit_count); return mx; };

    // One descent from a given root child → leaf (PUCT interior, or Gumbel-interior if enabled).
    auto descend = [&](int child_idx) -> std::vector<std::shared_ptr<MCTSNode>> {
        std::vector<std::shared_ptr<MCTSNode>> path;
        path.reserve(64);
        path.push_back(root);
        auto node = kids[child_idx];
        path.push_back(node);
        while (node->is_expanded()) {
            std::shared_ptr<MCTSNode> nxt = GUMBEL_INTERIOR
                ? gumbel_interior_select(node)
                : node->select_child(false).second;   // PUCT, no forced playouts
            path.push_back(nxt);
            node = nxt;
        }
        return path;
    };

    // Sequential halving: each phase gives every live candidate `per` visits, then keep the top half
    // ranked by g+logit+σ(q̂). Descents are batched at most ONE per candidate per batch (so a candidate
    // is never scheduled twice against its own still-unexpanded child → no duplicate leaf).
    while ((int)live.size() > 1) {
        const int k = (int)live.size();
        const int per = std::max(1, n / (num_phases * k));
        std::vector<int> need(k, per);
        int total_need = per * k;

        while (total_need > 0) {
            std::vector<std::shared_ptr<MCTSNode>> leaves;
            std::vector<std::vector<std::shared_ptr<MCTSNode>>> paths;
            std::vector<py::object> leaf_states;
            int scheduled = 0;

            for (int t = 0; t < k && scheduled < batch_size; t++) {
                if (need[t] <= 0) continue;
                need[t]--; total_need--; scheduled++;

                auto path = descend(live[t]);
                for (auto& nd : path) { nd->virtual_loss += VIRTUAL_LOSS; nd->value_sum -= VIRTUAL_LOSS; }
                auto leaf = path.back();
                if (leaf->board.is_over()) {
                    float tp = leaf->board.turn_player();
                    backpropagate(path, leaf->board.get_reward_for_turn(tp), tp);
                } else {
                    leaves.push_back(leaf);
                    paths.push_back(std::move(path));
                    leaf_states.push_back(py::cast(leaf->board));
                }
            }

            if (leaves.empty()) continue;

            py::list py_leaf_states;
            for (auto& s : leaf_states) py_leaf_states.append(s);
            py::object result = inference_callback(py_leaf_states);
            py::array_t<float> policies_array = result.attr("__getitem__")(0).cast<py::array_t<float>>();
            py::array_t<float> values_array   = result.attr("__getitem__")(1).cast<py::array_t<float>>();
            float* policies_ptr = (float*)policies_array.request().ptr;
            float* values_ptr   = (float*)values_array.request().ptr;

            for (size_t i = 0; i < leaves.size(); i++) {
                auto node = leaves[i];
                const auto& next_legal = node->board.legal_moves();
                std::vector<float> leaf_policy(4672);
                for (int j = 0; j < 4672; j++) leaf_policy[j] = policies_ptr[i * 4672 + j];
                node->expand(next_legal, leaf_policy);
                float tp = node->board.turn_player();
                backpropagate(paths[i], values_ptr[i], tp);
            }
        }

        // Rank live candidates and keep the top half.
        const float mv = max_visit();
        std::sort(live.begin(), live.end(), [&](int x, int y) {
            float sx = g[x] + logit[x] + (GUMBEL_C_VISIT + mv) * GUMBEL_C_SCALE * q_hat(x);
            float sy = g[y] + logit[y] + (GUMBEL_C_VISIT + mv) * GUMBEL_C_SCALE * q_hat(y);
            return sx > sy;
        });
        live.resize(std::max(1, k / 2));
    }

    const std::string best_move = acts[live[0]];

    // Improved-policy TARGET over ALL root actions: π′ = softmax(logit + σ(completedQ)),
    // completedQ = q̂ for visited actions, v_mix for unvisited. A valid policy improvement at any n.
    float sumN = 0.0f, wq = 0.0f, wp = 0.0f;
    for (int i = 0; i < A; i++) {
        int N = kids[i]->visit_count;
        sumN += (float)N;
        if (N > 0) { wp += kids[i]->prior; wq += kids[i]->prior * q_hat(i); }
    }
    const float v_mix = (sumN > 0.0f && wp > 1e-12f) ? (root_value + sumN * (wq / wp)) / (1.0f + sumN) : root_value;
    const float mv = max_visit();
    const float scale = (GUMBEL_C_VISIT + mv) * GUMBEL_C_SCALE;

    std::vector<float> improved(A);
    float mx = -1e30f;
    for (int i = 0; i < A; i++) {
        float cq = (kids[i]->visit_count > 0) ? q_hat(i) : v_mix;
        improved[i] = logit[i] + scale * cq;
        mx = std::max(mx, improved[i]);
    }
    float Z = 0.0f;
    for (int i = 0; i < A; i++) { improved[i] = std::exp(improved[i] - mx); Z += improved[i]; }

    std::vector<float> policy(4672, 0.0f);
    if (Z > 0.0f) {
        for (int i = 0; i < A; i++) {
            int idx = move_to_index(acts[i]);
            if (idx < 4672) policy[idx] = improved[i] / Z;
        }
    }

    cached_root = root;
    return {best_move, py::array_t<float>(4672, policy.data())};
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

    // Reseed per call but ADVANCE by a per-engine counter so each move/game gets fresh randomness
    // (still reproducible given a fixed SEED_BASE). The old `rng.seed(seed)` reset to the SAME value
    // every move → identical Dirichlet noise each move and the same opening sampled every game.
    rng.seed(seed + search_calls++);

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

    // Gumbel self-play path (env-gated, self-play only). Gumbel noise replaces Dirichlet + epsilon,
    // so we branch BEFORE add_dirichlet_noise. Eval/arena (use_dirichlet=false) never enters here.
    if (GUMBEL_ENABLE && use_dirichlet) {
        return run_gumbel(root, initial_value, temperature, inference_callback);
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
                    auto [action, next_node] = node->select_child(use_dirichlet);
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

    // Move selection. temperature≈0 → greedy (most-visited), matching eval/arena (use_dirichlet=false,
    // temperature=0). temperature>0 (self-play's first TEMP_MOVES plies) → SAMPLE the played move from
    // the visit distribution ∝ visit_count^(1/T), the AlphaZero opening-diversity mechanism. Previously
    // the played move was ALWAYS argmax, so the temperature schedule only shaped the stored target and
    // never diversified actual play (a primary cause of the opening collapse to a single move).
    std::string best_move;
    if (temperature < 1e-6f) {
        best_move = root->best_action();
    } else {
        std::vector<std::string> actions;
        std::vector<double> weights;
        const double inv_t = 1.0 / temperature;
        for (const auto& [action, child] : root->children) {
            if (child->visit_count > 0) {
                actions.push_back(action);
                weights.push_back(std::pow((double)child->visit_count, inv_t));
            }
        }
        if (actions.empty()) {
            best_move = root->best_action();   // no visited child (shouldn't happen) → greedy fallback
        } else {
            std::discrete_distribution<size_t> dist(weights.begin(), weights.end());
            best_move = actions[dist(rng)];
        }
    }
    auto policy = get_policy_vector(root, temperature);
    cached_root = root;

    return {best_move, policy};
}
