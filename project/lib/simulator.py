# Shared market environment for comparing model-based and model-free agents
# in the ERGODIC setting (infinite horizon, no terminal time).
#
# Environment dynamics (fixed for both agents):
# 1. Mid-price: Brownian motion  dS = sigma * dW
# 2. Market order arrival: Poisson process with intensity lambda
# 3. Fill probability: exponential  P(fill) = exp(-kappa_true * delta)
# 4. Inventory penalty: phi * q^2 * dt

import numpy as np


class MarketSimulator:
    """
    Fixed market environment for comparing model-based and model-free agents.
    Both agents use the same simulator instance to ensure fair comparison.
    
    Ergodic setting: no terminal time, regret = gamma(kappa*) * T - objective(T).
    """

    def __init__(self, sigma, kappa_true, lambda_buy=1.0, lambda_sell=1.0,
                 dt=0.001, S0=10.0, q_upper=10, q_lower=-10, phi=0.01, seed=42):
        # environment parameters (fixed for both agents)
        self.sigma       = sigma
        self.kappa_true  = kappa_true
        self.lambda_buy  = lambda_buy
        self.lambda_sell = lambda_sell
        self.dt          = dt
        self.q_upper     = q_upper
        self.q_lower     = q_lower
        self.phi         = phi

        # initial state
        self.S0 = S0
        self.S  = S0
        self.Q  = 0       # inventory
        self.X  = 0.0     # cash
        self.t  = 0.0     # elapsed time
        self.cumulative_penalty = 0.0  # cumulative phi * q^2 * dt
        self.seed_val = seed
        self.rng = np.random.default_rng(seed)

    def reset(self, seed=None):
        """Reset state to initial values."""
        if seed is not None:
            self.seed_val = seed
            self.rng = np.random.default_rng(seed)
        self.S = self.S0
        self.Q = 0
        self.X = 0.0
        self.t = 0.0
        self.cumulative_penalty = 0.0
        return self._get_state()

    def _get_state(self):
        """Return current state as dict."""
        return {
            'S': self.S,
            'Q': self.Q,
            'X': self.X,
            't': self.t,
        }

    def step(self, delta_sell, delta_buy):
        """
        Execute one time step.
        
        Args:
            delta_sell: depth for sell limit order (ask side)
            delta_buy:  depth for buy limit order (bid side)
            
        Returns:
            state_dict, reward, info
            
        reward = dX + S*dQ - phi*q^2*dt  (instantaneous contribution to objective)
        """
        S_old = self.S
        Q_old = self.Q
        X_old = self.X

        # 1. mid-price update (Brownian motion)
        S_new = S_old + self.sigma * self.rng.standard_normal() * np.sqrt(self.dt)

        # 2. market order arrival (Poisson)
        buyMOs  = self.rng.poisson(self.lambda_buy  * self.dt)
        sellMOs = self.rng.poisson(self.lambda_sell * self.dt)

        # 3. fill probability
        prob_sell = np.exp(-self.kappa_true * delta_sell)
        prob_buy  = np.exp(-self.kappa_true * delta_buy)

        # 4. actual fills (Binomial) + inventory boundary protection
        dN_sell = self.rng.binomial(buyMOs,  prob_sell) if Q_old > self.q_lower else 0
        dN_buy  = self.rng.binomial(sellMOs, prob_buy)  if Q_old < self.q_upper else 0

        # 5. state update
        Q_new = Q_old + dN_buy - dN_sell
        X_new = X_old + (S_old + delta_sell) * dN_sell \
                      - (S_old - delta_buy)  * dN_buy

        # inventory penalty this step
        penalty = self.phi * Q_old**2 * self.dt
        self.cumulative_penalty += penalty

        # update state
        self.S = S_new
        self.Q = Q_new
        self.X = X_new
        self.t += self.dt

        # instantaneous reward (contribution to ergodic objective)
        # objective = (X_T + S_T*Q_T) - (X_0 + S_0*Q_0) - phi * integral(q^2 dt)
        # reward per step ≈ d(X + S*Q) - phi*q^2*dt
        reward = (X_new + S_new * Q_new) - (X_old + S_old * Q_old) - penalty

        info = {
            'dN_sell': dN_sell,
            'dN_buy': dN_buy,
            'buyMOs': buyMOs,
            'sellMOs': sellMOs,
            'penalty': penalty,
        }

        return self._get_state(), reward, info

    def objective(self):
        """
        Current value of the ergodic objective function:
        (X_T + S_T*Q_T) - (X_0 + S_0*Q_0) - phi * integral(q^2 dt)
        """
        return (self.X + self.S * self.Q) - (0 + self.S0 * 0) - self.cumulative_penalty
