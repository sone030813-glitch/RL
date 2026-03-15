import numpy as np
import scipy.linalg


class ErgodicCP:
    """
    Solve the Ergodic Control Problem for the classical market making model.
    
    Given parameters (lambda_buy, lambda_sell, kappa, phi, q_bounds),
    compute:
      - ergodic constant gamma
      - value function v(q)
      - optimal controls delta_sell(q), delta_buy(q)
    """
    def __init__(self, lambda_buy, lambda_sell, q_upper, q_lower, phi, kappa):
        self.lambda_buy = lambda_buy
        self.lambda_sell = lambda_sell
        self.q_upper = q_upper
        self.q_lower = q_lower
        self.phi = phi
        self.kappa = kappa
        self._dim = q_upper - q_lower + 1

    @property
    def A(self):
        """Generator matrix A for computing ergodic constant."""
        n = self._dim
        A = np.zeros([n, n])
        for i in range(n):
            for j in range(n):
                if j == i:
                    A[j, i] = -self.phi * self.kappa * (self.q_upper - j)**2
                elif j == i - 1:
                    A[j, i] = self.lambda_buy * np.e**(-1)
                elif j == i + 1:
                    A[j, i] = self.lambda_sell * np.e**(-1)
        return A

    @property
    def EConst(self):
        """Ergodic constant gamma = max_eigenvalue(A) / kappa."""
        evalues, _ = scipy.linalg.eig(self.A)
        gamma = np.real(max(evalues)) / self.kappa
        return gamma

    @property
    def CoefMatrix(self):
        """Coefficient matrix C for solving the ergodic HJB."""
        n = self._dim
        C = np.zeros([n, n])
        for i in range(n):
            for j in range(n):
                if j == i:
                    C[j, i] = -self.kappa * (self.phi * (self.q_upper - j)**2 + self.EConst)
                elif j == i - 1:
                    C[j, i] = self.lambda_buy * np.e**(-1)
                elif j == i + 1:
                    C[j, i] = self.lambda_sell * np.e**(-1)
        return C

    def _solu_HomoEQ(self, C):
        """Find eigenvector associated with minimum eigenvalue of C^T C."""
        e_vals, e_vecs = np.linalg.eig(np.dot(C.T, C))
        return e_vecs[:, np.argmin(e_vals)]

    @property
    def ValueFunction(self):
        """Value function v(q) for each inventory level."""
        C = self.CoefMatrix
        solu = self._solu_HomoEQ(C)
        if solu[0] < 0:
            solu = -1 * solu
        return np.log(solu) / self.kappa

    @property
    def EControl(self):
        """
        Optimal ergodic controls.
        
        Returns:
            delta_sell: array of size (q_upper - q_lower,), indexed q_lower+1 to q_upper
            delta_buy:  array of size (q_upper - q_lower,), indexed q_lower to q_upper-1
        """
        C = self.CoefMatrix
        solu = np.abs(self._solu_HomoEQ(C))
        solu = np.log(solu) / self.kappa

        n = self.q_upper - self.q_lower
        delta_sell = np.zeros(n)
        delta_buy = np.zeros(n)

        for i in range(n):
            delta_sell[i] = 1/self.kappa + solu[i] - solu[i+1]
        for i in range(n):
            delta_buy[i] = 1/self.kappa + solu[i+1] - solu[i]

        # Reverse so that index 0 = q_lower, index -1 = q_upper
        return delta_sell[::-1], delta_buy[::-1]

    def get_control_for_q(self, q):
        """
        Get optimal (delta_sell, delta_buy) for a given inventory q.
        
        Returns:
            delta_sell, delta_buy
        """
        delta_sell_arr, delta_buy_arr = self.EControl
        # delta_sell is defined for q in [q_lower+1, q_upper]
        # delta_buy is defined for q in [q_lower, q_upper-1]
        
        if q <= self.q_lower:
            ds = 1e20  # don't sell when at min inventory
        else:
            idx_sell = int(q - self.q_lower - 1)
            idx_sell = np.clip(idx_sell, 0, len(delta_sell_arr) - 1)
            ds = delta_sell_arr[idx_sell]
        
        if q >= self.q_upper:
            db = 1e20  # don't buy when at max inventory
        else:
            idx_buy = int(q - self.q_lower)
            idx_buy = np.clip(idx_buy, 0, len(delta_buy_arr) - 1)
            db = delta_buy_arr[idx_buy]
        
        return ds, db
