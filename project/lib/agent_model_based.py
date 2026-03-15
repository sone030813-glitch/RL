"""
Model-Based Agent for Ergodic Market Making.

Algorithm (from slides, Algorithm 1):
1. Start with initial guess kappa_0
2. At each MO arrival, update kappa via regularised MLE
3. Solve ergodic HJB with current kappa estimate -> get optimal delta(q)
4. Post limit orders at those depths

This is an online learning agent: it learns kappa while trading.
"""

import numpy as np
from scipy.optimize import fsolve
from HJB import ErgodicCP


class ModelBasedAgent:
    def __init__(self, lambda_buy, lambda_sell, q_upper, q_lower, phi,
                 kappa_init, K_lower=1.0, K_upper=50.0, delta0=1e-5, update_freq=1):
        self.lambda_buy = lambda_buy
        self.lambda_sell = lambda_sell
        self.q_upper = q_upper
        self.q_lower = q_lower
        self.phi = phi
        self.delta0 = delta0
        self.K_lower = K_lower
        self.K_upper = K_upper
        self.update_freq = update_freq
        self._obs_count = 0

        # current kappa estimate
        self.kappa = kappa_init
        self.kappa_history = [kappa_init]

        # solve ergodic HJB with initial kappa
        self._update_control()

        # history for MLE
        self.history_depths = []      # posted depths
        self.history_MOs = []         # number of MOs that arrived
        self.history_fills = []       # number of fills

    def _update_control(self):
        """Re-solve ergodic HJB with current kappa estimate."""
        self.ecp = ErgodicCP(
            lambda_buy=self.lambda_buy,
            lambda_sell=self.lambda_sell,
            q_upper=self.q_upper,
            q_lower=self.q_lower,
            phi=self.phi,
            kappa=self.kappa,
        )
        self._delta_sell, self._delta_buy = self.ecp.EControl

    def get_action(self, q):
        """Get (delta_sell, delta_buy) for current inventory q."""
        return self.ecp.get_control_for_q(q)

    def update(self, delta_sell, delta_buy, info):
        """
        Update kappa estimate after observing market data.
        """
        # Record sell-side observation
        if info['buyMOs'] > 0 and delta_sell < 1e15:
            self.history_depths.append(delta_sell)
            self.history_MOs.append(info['buyMOs'])
            self.history_fills.append(info['dN_sell'])

        # Record buy-side observation
        if info['sellMOs'] > 0 and delta_buy < 1e15:
            self.history_depths.append(delta_buy)
            self.history_MOs.append(info['sellMOs'])
            self.history_fills.append(info['dN_buy'])

        # Update kappa every update_freq observations to save time
        self._obs_count += 1
        if self._obs_count % self.update_freq == 0 and len(self.history_depths) > 0:
            self._update_kappa()

    def _update_kappa(self):
        """Regularised MLE for kappa (vectorised)."""
        depths = np.array(self.history_depths)
        MOs = np.array(self.history_MOs)
        fills = np.array(self.history_fills)
        delta0 = self.delta0

        # Filter out huge depths
        mask = depths < 1e15
        depths = depths[mask]
        MOs = MOs[mask]
        fills = fills[mask]

        if len(depths) == 0:
            self.kappa_history.append(self.kappa)
            return

        def f(x):
            x = float(x)
            if x <= 0:
                return 1e10
            reg = delta0 * (np.exp(-x * delta0) / (1 - np.exp(-x * delta0)) - 1)
            exps = np.exp(-x * depths)
            denoms = 1 - exps
            denoms = np.where(np.abs(denoms) < 1e-15, 1e-15, denoms)
            terms = -depths * MOs + depths * (MOs - fills) / denoms
            return reg + np.sum(terms)

        try:
            kappa_new = fsolve(f, self.kappa, full_output=False)[0]
        except:
            kappa_new = self.kappa

        kappa_new = np.clip(kappa_new, self.K_lower, self.K_upper)
        self.kappa = float(kappa_new)
        self.kappa_history.append(self.kappa)
        self._update_control()
