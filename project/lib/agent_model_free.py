"""
Model-Free RL Agent for Ergodic Market Making.

Uses a softmax tabular policy over discretised action space.
State: inventory q (discrete, q_lower to q_upper)
Action: (delta_sell, delta_buy) from a discrete grid

For the ergodic (continuing) setting, we use differential (average-reward) 
REINFORCE. The key idea:
  - Maintain a running average reward R_bar
  - Use differential return: r_t - R_bar as the advantage
  - Update policy proportional to (r_t - R_bar) * grad log pi

References:
  - Sutton & Barto Ch.13 (Policy Gradient), Ch.10 (Average Reward)
  - mbt_gym PolicyGradientAgent (finite-horizon version)
"""

import numpy as np


class ModelFreeAgent:
    def __init__(self, q_lower, q_upper, 
                 delta_min=0.01, delta_max=1.0, n_delta=10,
                 lr=0.01, avg_reward_lr=0.01, temperature=1.0,
                 seed=42):
        """
        Args:
            q_lower, q_upper: inventory bounds
            delta_min, delta_max: range of depth values
            n_delta: number of discrete depth levels
            lr: learning rate for policy parameters
            avg_reward_lr: learning rate for average reward estimate
            temperature: softmax temperature (higher = more exploration)
        """
        self.q_lower = q_lower
        self.q_upper = q_upper
        self.n_states = q_upper - q_lower + 1
        
        # Discretise action space
        self.delta_grid = np.linspace(delta_min, delta_max, n_delta)
        self.n_delta = n_delta
        # Joint action = (delta_sell_idx, delta_buy_idx)
        # Total actions = n_delta * n_delta
        self.n_actions = n_delta * n_delta
        
        # Policy parameters: theta[q_idx, action_idx]
        # Initialised to 0 -> uniform policy
        self.rng = np.random.default_rng(seed)
        self.theta = np.zeros((self.n_states, self.n_actions))
        
        # Learning rates
        self.lr = lr
        self.avg_reward_lr = avg_reward_lr
        self.temperature = temperature
        
        # Running average reward (for differential return)
        self.R_bar = 0.0
        
        # History for tracking
        self.R_bar_history = [0.0]

    def _q_to_idx(self, q):
        """Convert inventory q to state index."""
        return int(np.clip(q - self.q_lower, 0, self.n_states - 1))

    def _action_to_deltas(self, action_idx):
        """Convert flat action index to (delta_sell, delta_buy)."""
        sell_idx = action_idx // self.n_delta
        buy_idx = action_idx % self.n_delta
        return self.delta_grid[sell_idx], self.delta_grid[buy_idx]

    def _softmax(self, q_idx):
        """Compute softmax policy for state q_idx."""
        logits = self.theta[q_idx] / self.temperature
        logits -= np.max(logits)  # numerical stability
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits)

    def get_action(self, q):
        """
        Sample action from current policy.
        
        Args:
            q: current inventory
            
        Returns:
            delta_sell, delta_buy, action_idx
        """
        q_idx = self._q_to_idx(q)
        probs = self._softmax(q_idx)
        action_idx = self.rng.choice(self.n_actions, p=probs)
        delta_sell, delta_buy = self._action_to_deltas(action_idx)
        
        # Enforce inventory constraints
        if q <= self.q_lower:
            delta_sell = 1e20  # don't sell
        if q >= self.q_upper:
            delta_buy = 1e20  # don't buy
            
        return delta_sell, delta_buy, action_idx

    def update(self, q, action_idx, reward):
        """
        Online policy gradient update (differential REINFORCE).
        
        Args:
            q: state when action was taken
            action_idx: action that was taken
            reward: instantaneous reward r_t
        """
        q_idx = self._q_to_idx(q)
        
        # Differential reward
        delta_r = reward - self.R_bar
        
        # Update average reward estimate
        self.R_bar += self.avg_reward_lr * (reward - self.R_bar)
        self.R_bar_history.append(self.R_bar)
        
        # Policy gradient: grad log pi(a|s) * (r - R_bar)
        # For softmax: grad log pi(a|s) = e_a - pi(s)
        probs = self._softmax(q_idx)
        grad = -probs.copy()          # -pi(a'|s) for all a'
        grad[action_idx] += 1.0       # +1 for the taken action
        
        # Update theta
        self.theta[q_idx] += self.lr * delta_r * grad

    def get_learned_policy(self):
        """
        Extract the learned deterministic policy (greedy).
        
        Returns:
            dict mapping q -> (delta_sell, delta_buy)
        """
        policy = {}
        for q in range(self.q_lower, self.q_upper + 1):
            q_idx = self._q_to_idx(q)
            probs = self._softmax(q_idx)
            best_action = np.argmax(probs)
            ds, db = self._action_to_deltas(best_action)
            policy[q] = (ds, db)
        return policy
