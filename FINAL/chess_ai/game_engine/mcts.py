import torch
import math
import copy
import numpy as np

# --- Constants ---
VIRTUAL_LOSS = 3.0  

# PUCT Constants
CPUCT_BASE = 19652
CPUCT_INIT = 1.0

def move_to_index(move_str):
    """
    Robust conversion of UCI move string to policy index.
    0-4095: Standard Moves (From-To). Includes Queen Promotions.
    4096-4191: Underpromotions (Knight, Bishop, Rook).
    """
    src = (ord(move_str[0]) - 97) + (ord(move_str[1]) - 49) * 8
    dst = (ord(move_str[2]) - 97) + (ord(move_str[3]) - 49) * 8
    
    idx = src * 64 + dst
    
    if len(move_str) == 5:
        promotion = move_str[4]
        if promotion != 'q':
            type_map = {'n': 0, 'b': 1, 'r': 2}
            promo_type = type_map.get(promotion, 0)
            src_col = src % 8
            dst_col = dst % 8
            direction = (dst_col - src_col) + 1 
            idx = 4096 + (src_col * 9) + (direction * 3) + promo_type

    return idx if idx < 8192 else 0

class Node:
    def __init__(self, state, parent=None, prior=0):
        self.state = state
        self.children = {}  
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior  
        self.virtual_loss = 0 
        
    def is_expanded(self):
        return len(self.children) > 0

    def value(self):
        # Virtual loss increases the denominator (visits) and decreases numerator (value_sum)
        # This effectively reduces the win rate, discouraging other threads.
        visits = self.visit_count + self.virtual_loss
        if visits <= 0: return 0
        return self.value_sum / visits

    def select_child(self):
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        parent_visits = self.visit_count + self.virtual_loss
        
        # --- DYNAMIC PUCT FORMULA (Section 3.1.1) ---
        cpuct = CPUCT_INIT + math.log((parent_visits + CPUCT_BASE) / CPUCT_BASE)
        sqrt_parent_visits = math.sqrt(max(1, parent_visits))
        
        for action, child in self.children.items():
            child_visits = child.visit_count + child.virtual_loss
            
            q_value = -child.value() # Negate because child value is from opponent's perspective
            
            u_value = cpuct * child.prior * sqrt_parent_visits / (1 + child_visits)
            score = q_value + u_value
            
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
                
        return best_action, best_child

    def expand(self, valid_moves, policy_logits):
        move_probs = {}
        policy_sum = 0
        
        for move_str in valid_moves:
            idx = move_to_index(move_str)
            logit = policy_logits[idx] if idx < len(policy_logits) else -10.0
            prob = math.exp(logit)
            move_probs[move_str] = prob
            policy_sum += prob
            
        for move in valid_moves:
            if policy_sum > 0:
                normalized_prior = move_probs[move] / policy_sum
            else:
                normalized_prior = 1.0 / len(valid_moves)
            
            next_state = self.state.copy()
            next_state.push(move)
            self.children[move] = Node(next_state, parent=self, prior=normalized_prior)

    def best_action(self):
        most_visits = -1
        best_action = None
        for action, child in self.children.items():
            if child.visit_count > most_visits:
                most_visits = child.visit_count
                best_action = action
        return best_action

class MCTSWorker:
    def __init__(self, worker_id, input_queue, output_queue, simulations=800, batch_size=8):
        self.worker_id = worker_id
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.simulations = simulations
        self.batch_size = batch_size 
        self.cpu = 1.0 

    def get_policy_vector(self, root, alpha: float = 1.3):
        policy_vector = np.zeros(8192, dtype=np.float32)
        
        visits = {}
        for action_uci, child in root.children.items():
            idx = move_to_index(action_uci)
            if idx < 8192:
                visits[idx] = child.visit_count
        
        if not visits:
            return policy_vector
        
        counts = np.array(list(visits.values()), dtype=np.float32)
        indices = np.array(list(visits.keys()), dtype=np.int32)
        
        sharpened = counts ** alpha
        total = sharpened.sum()
        
        if total <= 0:
            return policy_vector
        
        probs = sharpened / total
        policy_vector[indices] = probs
        
        return policy_vector

    def search(self, root_state, temperature=1.0):
        root = Node(root_state)
        
        # 1. Expand Root
        tensor = torch.from_numpy(root.state.to_tensor())
        self.input_queue.put((self.worker_id, tensor)) 
        
        try:
            policy, value = self.output_queue.get(timeout=60)
        except Exception:
            print(f"[Worker {self.worker_id}] ❌ Server timeout - no response in 60")
            raise RuntimeError("Server communication timeout")

        root.expand(root.state.legal_moves(), policy)
        
        self._add_noise_recursive(root, depth=0)
        
        # 2. Simulation Loop
        num_iterations = max(1, self.simulations // self.batch_size)
        
        for _ in range(num_iterations):
            leaves = []
            paths = []
            tensors = []
            
            # Selection Phase
            for _ in range(self.batch_size):
                node = root
                path = [node]

                while node.is_expanded():
                    action, node = node.select_child()
                    path.append(node)
                
                # CRITICAL FIX 1: Apply Virtual Loss ONLY to the leaf node (outside the loop)
                # CRITICAL FIX 2: Correct signs (+= to VL count, -= to Value sum)
                node.virtual_loss += VIRTUAL_LOSS
                node.value_sum -= VIRTUAL_LOSS  # Penalize to repel other threads
                
                # Terminal state handling
                if node.state.is_over: # Removed can_claim_draw() to fix peaceful loop
                    reward = node.state.get_reward_for_turn(node.state.turn_player)
                    self.backpropagate(path, reward, node.state.turn_player, is_terminal=True)
                else:
                    leaves.append(node)
                    paths.append(path)
                    tensors.append(node.state.to_tensor())

            if not leaves: continue

            # Inference Phase
            batch_tensor = torch.from_numpy(np.array(tensors))
            self.input_queue.put((self.worker_id, batch_tensor))
            
            try:
                policies, values = self.output_queue.get(timeout=60)
            except Exception:
                print(f"[Worker {self.worker_id}] ❌ Server timeout - no response in 60")
                raise RuntimeError("Server communication timeout")

            # Expansion & Backprop Phase
            for i, node in enumerate(leaves):
                path = paths[i]
                node.expand(node.state.legal_moves(), policies[i])
                self.backpropagate(path, values[i], node.state.turn_player, is_terminal=False)

        return self.get_result(root, temperature)

    def backpropagate(self, path, value, leaf_turn_player, is_terminal):
        # CRITICAL FIX: Remove Virtual Loss from the LEAF only
        leaf = path[-1]
        leaf.virtual_loss -= VIRTUAL_LOSS
        leaf.value_sum += VIRTUAL_LOSS  # Add back to restore the real value (undo the penalty)

        # Standard Backprop
        for node in reversed(path):
            node.visit_count += 1
            if node.state.turn_player == leaf_turn_player:
                node.value_sum += value
            else:
                node.value_sum -= value

    def _add_noise_recursive(self, node, depth=0):
        '''Add Dirichlet noise ONLY at the root node'''
        if depth == 0:
            self.add_exploration_noise(node, depth=0)

    def add_exploration_noise(self, node, depth=0):
        '''Add Dirichlet noise with depth-aware scaling'''
        actions = list(node.children.keys())
        if not actions: return
        
        alpha = 0.3 if depth == 0 else 0.1
        frac = 0.25 if depth == 0 else 0.05
        
        noise = np.random.dirichlet([alpha] * len(actions))
        for i, action in enumerate(actions):
            node.children[action].prior = (
                node.children[action].prior * (1 - frac) + 
                noise[i] * frac
            )

    def get_result(self, root, temperature):
        if temperature == 0: 
            return root.best_action(), self.get_policy_vector(root)
            
        actions = list(root.children.keys())
        if not actions: return None, self.get_policy_vector(root)

        visits = np.array([root.children[a].visit_count for a in actions], dtype=np.float32)
        
        if temperature != 1.0:
            visits = np.power(visits, 1.0 / temperature)
        
        probs = visits / np.sum(visits) if np.sum(visits) > 0 else np.ones(len(visits))/len(visits)
        chosen_action = np.random.choice(actions, p=probs)
        return chosen_action, self.get_policy_vector(root)
    
    def search_direct(self, root_state, model, temperature=1.0, use_dirichlet=True):
        """
        Direct MCTS search without queue communication.
        Used for evaluation where we have direct model access.
        """
        root = Node(root_state)
        device = next(model.parameters()).device
        
        # 1. Expand Root
        tensor = torch.from_numpy(root.state.to_tensor()).unsqueeze(0).to(device)
        with torch.no_grad():
            policy, value = model(tensor)
        
        root.expand(root.state.legal_moves(), policy.cpu().numpy())
        
        if use_dirichlet:
            self._add_noise_recursive(root, depth=0)
        
        # 3. Batch simulation loop
        num_iterations = max(1, self.simulations // self.batch_size)
        
        for _ in range(num_iterations):
            leaves = []
            paths = []
            tensors = []
            
            # Selection Phase
            for _ in range(self.batch_size):
                node = root
                path = [node]
                
                while node.is_expanded():
                    action, node = node.select_child()
                    path.append(node)
                
                # CRITICAL FIX: Match logic with search()
                # Apply VL only to leaf, correct signs
                node.virtual_loss += VIRTUAL_LOSS
                node.value_sum -= VIRTUAL_LOSS
                
                if node.state.is_over: # Removed can_claim_draw()
                    reward = node.state.get_reward_for_turn(node.state.turn_player)
                    self.backpropagate(path, reward, node.state.turn_player, is_terminal=True)
                else:
                    leaves.append(node)
                    paths.append(path)
                    tensors.append(node.state.to_tensor())
            
            if not leaves:
                continue
            
            # Inference Phase (Direct)
            batch_tensor = torch.from_numpy(np.array(tensors)).to(device)
            with torch.no_grad():
                policies, values = model(batch_tensor)
            
            # Expansion & Backprop Phase
            for i, node in enumerate(leaves):
                path = paths[i]
                node.expand(node.state.legal_moves(), policies[i].cpu().numpy())
                self.backpropagate(path, values[i].item(), node.state.turn_player, is_terminal=False)
        
        return self.get_result(root, temperature)