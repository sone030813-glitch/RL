import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


# The class is to solve an Ergodic Control Problem for classical market makiing model
class ErgodicCP:
    def __init__(
        self,
        lambda_buy, 
        lambda_sell,
        q_upper,
        q_lower,
        phi,
        kappa,
    ):
        self.lambda_buy = lambda_buy
        self.lambda_sell = lambda_sell
        self.q_upper = q_upper
        self.q_lower = q_lower
        self.phi = phi 
        self.kappa = kappa

    @property
    def A(self):
        A = np.zeros([self.q_upper-self.q_lower+1, self.q_upper-self.q_lower+1])   

        for i in range(self.q_upper-self.q_lower+1):
            # j denotes the row number
            for j in range(self.q_upper-self.q_lower+1):
                if j == i:
                    # q = q_upper - j
                    A[j,i] = - self.phi*self.kappa*(self.q_upper-j)**2
                elif j == i-1:
                    A[j,i] = self.lambda_buy*np.e**(-1)
                elif j == i+1:
                    A[j,i] = self.lambda_sell*np.e**(-1)

        return A

    @property
    def EConst(self):
        import scipy.linalg
        A = self.A
        evalues, _ = scipy.linalg.eig(A)
        gamma = np.real(max(evalues)) / self.kappa
        return gamma
    
    @property
    def CoefMatrix(self):
        # Let C denote a square matrix
        C = np.zeros([self.q_upper-self.q_lower+1, self.q_upper-self.q_lower+1])

        # Notice we assume kappa_buy = kappa_sell in the model
        # i denotes the column number
        for i in range(self.q_upper-self.q_lower+1):
            # j denotes the row number
            for j in range(self.q_upper-self.q_lower+1):
                if j == i:
                    # q = q_upper - j
                    C[j,i] = - self.kappa*(self.phi*(self.q_upper-j)**2 + self.EConst)
                elif j == i-1:
                    C[j,i] = self.lambda_buy*np.e**(-1)
                elif j == i+1:
                    C[j,i] = self.lambda_sell*np.e**(-1)

        return C

    def solu_HomoEQ(self, C):
        # find the eigenvalues and eigenvector of C(transpose).C
        e_vals, e_vecs = np.linalg.eig(np.dot(C.T, C))  
        # extract the eigenvector (column) associated with the minimum eigenvalue
        return e_vecs[:, np.argmin(e_vals)]

    @property
    def ValueFunction(self):
        C = self.CoefMatrix
        solu = self.solu_HomoEQ(C)
        if solu[0] < 0:
            solu = -1 * solu
        
        solu = np.log(solu) / self.kappa
        return solu


    @property
    def EControl(self):
        C = self.CoefMatrix
        solu = np.abs(self.solu_HomoEQ(C))
        # if solu[0] < 0:
        #     solu = -1 * solu
        
        solu = np.log(solu) / self.kappa

        delta_buy = np.zeros(self.q_upper - self.q_lower)
        delta_sell = np.zeros(self.q_upper - self.q_lower)

        for i in range(self.q_upper-self.q_lower):
            # q = q_upper - i
            delta_sell[i] = 1/self.kappa + solu[i] - solu[i+1]
            # delta_sell[i-1] = 1/self.kappa - solu[i-1] + solu[i] (original)
        for i in range(self.q_upper-self.q_lower):
            # q = q_upper - 1 - j
            delta_buy[i] = 1/self.kappa + solu[i+1] - solu[i]
            # delta_buy[i] = 1/self.kappa + solu[i] - solu[i+1] (original)
        return delta_sell[::-1], delta_buy[::-1]
    
    @property
    def plot_EControl(self):
        import matplotlib.pyplot as plt

        inventory = np.arange(self.q_lower, self.q_upper+1, 1)
        delta_sell, delta_buy = self.EControl

        plt.plot(inventory[1:], delta_sell, 'o',label='sell depth $\delta^{+}$')
        plt.plot(inventory[:-1], delta_buy, 'o',label='buy depth $\delta^{-}$')
        plt.xlabel("q")
        plt.ylabel('$\delta^{\pm}(\$)$')
        plt.title("Optimal Strategy $\delta^{\pm, *}(q)$ for Ergodic Control Problem")
        plt.legend()
        plt.show()  


# The class is to simulate the agent's parameter learning
class Agent:
    def __init__(
        self,
        lambda_buy, 
        lambda_sell,
        q_upper,
        q_lower,
        phi,
        kappa_est, 
        T, 
        Q0 = 0, 
        S0 = 10, 
        X0 = 0,
        dt = 0.01,
        K_upper = 50, 
        K_lower = 1, 
        delta0 = 1e-5,
    ):
        self.lambda_buy = lambda_buy
        self.lambda_sell = lambda_sell
        self.q_upper = q_upper
        self.q_lower = q_lower
        self.phi = phi 
        self.delta0 = delta0

        # update kappa/control via learning
        self.kappa_est = kappa_est
        self.kappa = kappa_est
        self.control = ErgodicCP(
            lambda_buy=lambda_buy,
            lambda_sell=lambda_sell,
            q_upper=q_upper,
            q_lower=q_lower,
            phi=phi,
            kappa=kappa_est,
        ).EControl

        self.T = T
        self.dt = dt
        self.ts = np.linspace(0, self.T, int(self.T/self.dt)+1)
        self.length = len(self.ts)
        self.step = 0

        # initialise the 
        self.stateX = np.zeros(self.length)
        self.stateS = np.zeros(self.length)
        self.stateQ = np.zeros(self.length)
        self.stateX[0] = X0
        self.stateS[0] = S0
        self.stateQ[0] = Q0

        self.K_upper = K_upper
        self.K_lower = K_lower

        self.kappa_learnlist = [self.kappa]

        self.objective = np.zeros(self.length)

    def learning(self, sigma, kappa_true):
        """
        Simulate the learning process
        """
        # create nparray to log the number of coming MOs 
        self.coming_sellMOs = np.zeros(self.length - 1)
        self.coming_buyMOs = np.zeros(self.length - 1)
        # create nparray to log the hit number of LOs posted by agent
        self.hitsellLOs = np.zeros(self.length - 1)
        self.hitbuyLOs = np.zeros(self.length - 1)
        # create nparray to log the depth posted by agent in each interval (batch_size, N)
        self.postselldepth = np.zeros(self.length - 1)
        self.postbuydepth = np.zeros(self.length - 1)

        for idx, t in enumerate(self.ts[1:]):
            # update time step 
            self.step += 1

            Q_old = self.stateQ[idx]
            S_old = self.stateS[idx]
            X_old = self.stateX[idx]




            brownian_increments = np.random.randn() * np.sqrt(self.dt)
            # update midprice
            S_new = (
                S_old + sigma * brownian_increments
            )



            sell_depth = self.control[0][int(self.q_upper + Q_old - 1)] if Q_old > self.q_lower else 1e20
            buy_depth = self.control[1][int(self.q_upper + Q_old)] if Q_old < self.q_upper else 1e20            




            self.postselldepth[idx] = sell_depth
            self.postbuydepth[idx] = buy_depth

            prob_sellside = np.exp(-sell_depth * kappa_true)
            prob_buyside = np.exp(-buy_depth * kappa_true)
            prob_sellside = 1 if prob_sellside > 1 else prob_sellside
            prob_buyside = 1 if prob_buyside > 1 else prob_buyside

            # update coming market orders as Poisson process
            sellMOs = np.random.poisson(self.lambda_sell*self.dt)
            buyMOs = np.random.poisson(self.lambda_buy*self.dt)
            self.coming_sellMOs[idx] = sellMOs
            self.coming_buyMOs[idx] = buyMOs

            # update the hit LOs posted by Agent
            dN_sell = np.random.binomial(buyMOs, prob_sellside) 
            dN_buy = np.random.binomial(sellMOs, prob_buyside) 
            self.hitsellLOs[idx] = dN_sell
            self.hitbuyLOs[idx] = dN_buy

            # update inventory 
            Q_new = (
                Q_old + dN_buy - dN_sell
            )

            # update cash
            X_new = (
                X_old + (S_old + sell_depth) * dN_sell - (S_old - buy_depth) * dN_buy
            ) 

            # update state
            self.stateQ[idx+1] = Q_new
            self.stateS[idx+1] = S_new
            self.stateX[idx+1] = X_new

            # calculate objective function
            objective_value = self.Objective_Function()    
            self.objective[idx+1] = objective_value

            # update kappa/control
            self.estimator(delta0=self.delta0)

        return None
    
    def learning_myopic(self, sigma, kappa_true):
        """
        Simulate the learning process
        """

        # create nparray to log the number of coming MOs 
        self.coming_sellMOs = np.zeros(self.length - 1)
        self.coming_buyMOs = np.zeros(self.length - 1)
        # create nparray to log the hit number of LOs posted by agent
        self.hitsellLOs = np.zeros(self.length - 1)
        self.hitbuyLOs = np.zeros(self.length - 1)
        # create nparray to log the depth posted by agent in each interval (batch_size, N)
        self.postselldepth = np.zeros(self.length - 1)
        self.postbuydepth = np.zeros(self.length - 1)

        for idx, t in enumerate(self.ts[1:]):
            # update time step 
            self.step += 1

            Q_old = self.stateQ[idx]
            S_old = self.stateS[idx]
            X_old = self.stateX[idx]

            brownian_increments = np.random.randn() * np.sqrt(self.dt)
            # update midprice
            S_new = (
                S_old + sigma * brownian_increments
            )
            
            sell_depth = 1 / self.kappa
            buy_depth =  1 / self.kappa
            
            self.postselldepth[idx] = sell_depth
            self.postbuydepth[idx] = buy_depth

            prob_sellside = np.exp(-sell_depth * kappa_true)
            prob_buyside = np.exp(-buy_depth * kappa_true)
            prob_sellside = 1 if prob_sellside > 1 else prob_sellside
            prob_buyside = 1 if prob_buyside > 1 else prob_buyside

            # update coming market orders as Poisson process
            sellMOs = np.random.poisson(self.lambda_sell*self.dt)
            buyMOs = np.random.poisson(self.lambda_buy*self.dt)
            self.coming_sellMOs[idx] = sellMOs
            self.coming_buyMOs[idx] = buyMOs

            # update the hit LOs posted by Agent
            dN_sell = np.random.binomial(buyMOs, prob_sellside) 
            dN_buy = np.random.binomial(sellMOs, prob_buyside) 
            self.hitsellLOs[idx] = dN_sell
            self.hitbuyLOs[idx] = dN_buy

            # update inventory 
            Q_new = (
                Q_old + dN_buy - dN_sell
            )

            # update cash
            X_new = (
                X_old + (S_old + sell_depth) * dN_sell - (S_old - buy_depth) * dN_buy
            ) 

            # update state
            self.stateQ[idx+1] = Q_new
            self.stateS[idx+1] = S_new
            self.stateX[idx+1] = X_new

            # calculate objective function
            objective_value = self.Objective_Function()    
            self.objective[idx+1] = objective_value

            # update kappa/control
            self.estimator(delta0=self.delta0)

        return None
        
    def regret(self, kappa_true):
        """
        Simualte the achieved regret
        """
        gamma = ErgodicCP(
            lambda_buy=self.lambda_buy,
            lambda_sell=self.lambda_sell,
            q_upper=self.q_upper,
            q_lower=self.q_lower,
            phi=self.phi,
            kappa=kappa_true,
        ).EConst

        reg = gamma*self.ts - self.objective
        return reg

    def Objective_Function(self):
        """
        Calculate the objective function
        """
        # current time step, step starts from 1
        current_step = self.step
        X_T = self.stateX[current_step]
        S_T = self.stateS[current_step]
        Q_T = self.stateQ[current_step]

        X_0 = self.stateX[0]
        S_0 = self.stateS[0]
        Q_0 = self.stateQ[0] 

        Q = self.stateQ[:current_step]   

        return (X_T + S_T*Q_T) - (X_0 + S_0*Q_0) - self.phi*np.sum(Q**2*self.dt)

    def estimator(self, delta0 = 1e-5):
        """
        Implement a regularised-maximum-likelihood-estimator
        """
        from scipy.optimize import fsolve
        # current time step
        current_step = self.step

        # Collect historical informaton and update kappa
        flag = self.coming_buyMOs[:current_step] != 0
        coming_buyMOs = self.coming_buyMOs[:current_step][flag]
        hitsellLOs = self.hitsellLOs[:current_step][flag]
        postselldepth = self.postselldepth[:current_step][flag]

        # also use sell MOs to laern kappa
        flag = self.coming_sellMOs[:current_step] != 0
        coming_sellMOs = self.coming_sellMOs[:current_step][flag]
        hitbuyLOs = self.hitbuyLOs[:current_step][flag]
        postbuydepth = self.postbuydepth[:current_step][flag]

        coming_MOs = np.concatenate([coming_buyMOs, coming_sellMOs])
        hitLOs = np.concatenate([hitsellLOs, hitbuyLOs])
        postdepth = np.concatenate([postselldepth, postbuydepth])

        def f(x):
            mle = delta0 * (np.exp(-x*delta0)/(1 - np.exp(-x*delta0)) - 1)
            for idx, trials in enumerate(coming_MOs):
                if postdepth[idx] > 1e15: 
                    continue
                temp = ( 
                    - postdepth[idx] * trials + 
                    postdepth[idx] * (trials - hitLOs[idx]) / (1 - np.exp(-x*postdepth[idx]))
                )
                mle += temp
            return mle
        
        def f_upper(x):
            mle = delta0 * (np.exp(-self.K_upper*delta0)/(1 - np.exp(-self.K_upper*delta0)) - 1)
            for idx, trials in enumerate(coming_MOs):
                if postdepth[idx] > 1e15: 
                    continue
                temp = ( 
                    - postdepth[idx] * trials + 
                    postdepth[idx] * (trials - hitLOs[idx]) / (1 - np.exp(-self.K_upper*postdepth[idx]))
                )

                temp2 = (
                    - delta0**2*(np.exp(-self.K_upper*delta0))/(1-np.exp(-self.K_upper*delta0))**2 
                    - (trials - hitLOs[idx])*postdepth[idx]**2*(np.exp(-self.K_upper*postdepth[idx]))/(1 - np.exp(-self.K_upper*postdepth[idx]))**2
                ) * (x - self.K_upper)
                mle = mle + temp + temp2
            return mle

        # kappa_new = fsolve(f, np.random.uniform(self.K_lower, self.K_upper))[0]
        if len(coming_MOs) != 0:
            # kappa_new = fsolve(f, np.random.uniform(self.K_lower, self.K_upper))[0]
            kappa_new = fsolve(f, self.kappa)[0]
        
            if kappa_new > self.K_upper: 
                # kappa_new = fsolve(f_upper, np.random.uniform(self.K_lower, self.K_upper))[0]
                kappa_new = fsolve(f_upper, self.kappa)[0]
            # if the updated kappa is not in the boundary
            # if kappa_new > upper_b or kappa_new < lower_b:
            #     kappa_new = self.kappa
            # if the updated kappa is not in the boundary
            if kappa_new > self.K_upper:
                kappa_new = self.K_upper
            
            elif kappa_new < self.K_lower:
                kappa_new = self.K_lower
            
        else:
            kappa_new = self.kappa
        # kappa_new = fsolve(f, self.kappa)[0]
        # kappa_new = kappa_new if kappa_new <= 65 and kappa_new >= 1 else np.random.uniform(35, 45)
        
        # update kappa 
        self.kappa_learnlist.append(kappa_new)
        self.kappa = kappa_new
        # update control 
        self.control = ErgodicCP(
            lambda_buy=self.lambda_buy,
            lambda_sell=self.lambda_sell,
            q_upper=self.q_upper,
            q_lower=self.q_lower,
            phi=self.phi,
            kappa=self.kappa,
        ).EControl

        return None    

    def standard_estimator(self):
        """
        Implement the standard maximum-likelihood-estimator
        """
        from scipy.optimize import fsolve
        # current time step
        current_step = self.step

        # Collect historical informaton and update kappa
        flag = self.coming_buyMOs[:current_step] != 0
        coming_buyMOs = self.coming_buyMOs[:current_step][flag]
        hitsellLOs = self.hitsellLOs[:current_step][flag]
        postselldepth = self.postselldepth[:current_step][flag]

        # also use sell MOs to laern kappa
        flag = self.coming_sellMOs[:current_step] != 0
        coming_sellMOs = self.coming_sellMOs[:current_step][flag]
        hitbuyLOs = self.hitbuyLOs[:current_step][flag]
        postbuydepth = self.postbuydepth[:current_step][flag]

        coming_MOs = np.concatenate([coming_buyMOs, coming_sellMOs])
        hitLOs = np.concatenate([hitsellLOs, hitbuyLOs])
        postdepth = np.concatenate([postselldepth, postbuydepth])

        def f(x):
            mle = 0
            for idx, trials in enumerate(coming_MOs):
                if postdepth[idx] > 1e15: 
                    continue
                temp = ( 
                    - postdepth[idx] * trials + 
                    postdepth[idx] * (trials - hitLOs[idx]) / (1 - np.exp(-x*postdepth[idx]))
                )
                mle += temp
            return mle


        # kappa_new = fsolve(f, np.random.uniform(self.K_lower, self.K_upper))[0]
        if len(coming_MOs) != 0:
            # kappa_new = fsolve(f, np.random.uniform(self.K_lower, self.K_upper))[0]
            kappa_new = fsolve(f, self.kappa)[0]
    
            # if the updated kappa is not in the boundary
            if kappa_new > self.K_upper or kappa_new < self.K_lower:
                kappa_new = self.kappa
            
        else:
            kappa_new = self.kappa
        # kappa_new = fsolve(f, self.kappa)[0]
        # kappa_new = kappa_new if kappa_new <= 65 and kappa_new >= 1 else np.random.uniform(35, 45)
        
        # update kappa 
        self.kappa_learnlist.append(kappa_new)
        self.kappa = kappa_new
        # update control 
        self.control = ErgodicCP(
            lambda_buy=self.lambda_buy,
            lambda_sell=self.lambda_sell,
            q_upper=self.q_upper,
            q_lower=self.q_lower,
            phi=self.phi,
            kappa=self.kappa,
        ).EControl

        return None    
    

class Agent_nonstationary:
    def __init__(
        self,
        lambda_buy, 
        lambda_sell,
        q_upper,
        q_lower,
        phi,
        kappa_est, 
        T, 
        non_stationary_interval,
        method, 
        Q0 = 0, 
        S0 = 10, 
        X0 = 0,
        dt = 0.01,
        K_upper = 60, 
        K_lower = 1, 
        delta0 = 1e-5,
        alpha = 0.8,
        window_size = 30,
    ):
        self.lambda_buy = lambda_buy
        self.lambda_sell = lambda_sell
        self.q_upper = q_upper
        self.q_lower = q_lower
        self.phi = phi 
        self.delta0 = delta0
        self.non_stationary_interval = non_stationary_interval
        self.method = method # either 'EWMA' or 'SW'
        self.alpha = alpha # only used in EWMA method
        self.window_size = window_size # only used in SW method

        # update kappa/control via learning
        self.kappa_est = kappa_est
        self.kappa = kappa_est
        self.control = ErgodicCP(
            lambda_buy=lambda_buy,
            lambda_sell=lambda_sell,
            q_upper=q_upper,
            q_lower=q_lower,
            phi=phi,
            kappa=kappa_est,
        ).EControl

        self.T = T
        self.dt = dt
        self.ts = np.linspace(0, self.T, int(self.T/self.dt)+1)
        self.length = len(self.ts)
        self.step = 0

        # initialise the 
        self.stateX = np.zeros(self.length)
        self.stateS = np.zeros(self.length)
        self.stateQ = np.zeros(self.length)
        self.stateX[0] = X0
        self.stateS[0] = S0
        self.stateQ[0] = Q0

        self.K_upper = K_upper
        self.K_lower = K_lower

        self.kappa_learnlist = [self.kappa]

        self.objective = np.zeros(self.length)

    def learning(self, sigma, kappa_true_list):
        """
        Simulate the learning process in a non-stationary market, where 
            we use kappa_true_list to simulate the non-stationary parameter
        """
        # create nparray to log the number of coming MOs 
        self.coming_sellMOs = np.zeros(self.length - 1)
        self.coming_buyMOs = np.zeros(self.length - 1)
        # create nparray to log the hit number of LOs posted by agent
        self.hitsellLOs = np.zeros(self.length - 1)
        self.hitbuyLOs = np.zeros(self.length - 1)
        # create nparray to log the depth posted by agent in each interval (batch_size, N)
        self.postselldepth = np.zeros(self.length - 1)
        self.postbuydepth = np.zeros(self.length - 1)

        for idx, t in enumerate(self.ts[1:]):
            # update time step 
            self.step += 1

            Q_old = self.stateQ[idx]
            S_old = self.stateS[idx]
            X_old = self.stateX[idx]

            brownian_increments = np.random.randn() * np.sqrt(self.dt)
            # update midprice
            S_new = (
                S_old + sigma * brownian_increments
            )
            
            sell_depth = self.control[0][int(self.q_upper + Q_old - 1)] if Q_old > self.q_lower else 1e20
            buy_depth = self.control[1][int(self.q_upper + Q_old)] if Q_old < self.q_upper else 1e20            
            
            self.postselldepth[idx] = sell_depth
            self.postbuydepth[idx] = buy_depth

            kappa_idx = idx // int(self.non_stationary_interval /self.dt)

            kappa_true = kappa_true_list[kappa_idx]
            prob_sellside = np.exp(-sell_depth * kappa_true)
            prob_buyside = np.exp(-buy_depth * kappa_true)
            prob_sellside = 1 if prob_sellside > 1 else prob_sellside
            prob_buyside = 1 if prob_buyside > 1 else prob_buyside

            # update coming market orders as Poisson process
            sellMOs = np.random.poisson(self.lambda_sell*self.dt)
            buyMOs = np.random.poisson(self.lambda_buy*self.dt)
            self.coming_sellMOs[idx] = sellMOs
            self.coming_buyMOs[idx] = buyMOs

            # update the hit LOs posted by Agent
            dN_sell = np.random.binomial(buyMOs, prob_sellside) 
            dN_buy = np.random.binomial(sellMOs, prob_buyside) 
            self.hitsellLOs[idx] = dN_sell
            self.hitbuyLOs[idx] = dN_buy

            # update inventory 
            Q_new = (
                Q_old + dN_buy - dN_sell
            )

            # update cash
            X_new = (
                X_old + (S_old + sell_depth) * dN_sell - (S_old - buy_depth) * dN_buy
            ) 

            # update state
            self.stateQ[idx+1] = Q_new
            self.stateS[idx+1] = S_new
            self.stateX[idx+1] = X_new

            # calculate objective function
            objective_value = self.Objective_Function()    
            self.objective[idx+1] = objective_value

            # update kappa/control
            if self.method == 'EWMA':
                self.estimator_ewma(delta0=self.delta0, alpha=self.alpha)
            elif self.method == 'SW':
                self.estimator_sw(delta0=self.delta0, window_size=self.window_size)
            else:
                self.standard_estimator()

        return None
        
    def regret(self, kappa_true_list):
        """
        Simualte the achieved regret
        """
        gamma_list = []
        for kappa_true in kappa_true_list:
            gamma = ErgodicCP(
                lambda_buy=self.lambda_buy,
                lambda_sell=self.lambda_sell,
                q_upper=self.q_upper,
                q_lower=self.q_lower,
                phi=self.phi,
                kappa=kappa_true,
            ).EConst
            gamma_list.append(gamma)

        optimal_reward = np.zeros(self.length)
        for idx, gamma in enumerate(gamma_list):
            start_idx = int(idx * (self.non_stationary_interval/self.dt))
            end_idx = int((idx + 1) * int(self.non_stationary_interval/self.dt))

            if idx == 0:
                optimal_reward[start_idx:end_idx] = gamma*np.linspace(0, self.non_stationary_interval, int(self.non_stationary_interval/self.dt)+1)[:-1]
            optimal_reward[start_idx: end_idx] = optimal_reward[start_idx-1] + gamma*np.linspace(0, self.non_stationary_interval, int(self.non_stationary_interval/self.dt)+1)[1:]
        
        reg = optimal_reward - self.objective
        return reg

    def Objective_Function(self):
        """
        Calculate the objective function
        """
        # current time step, step starts from 1
        current_step = self.step
        X_T = self.stateX[current_step]
        S_T = self.stateS[current_step]
        Q_T = self.stateQ[current_step]

        X_0 = self.stateX[0]
        S_0 = self.stateS[0]
        Q_0 = self.stateQ[0] 

        Q = self.stateQ[:current_step]   

        return (X_T + S_T*Q_T) - (X_0 + S_0*Q_0) - self.phi*np.sum(Q**2*self.dt)

    def estimator_sw(self, delta0 = 1e-5, window_size = 30):
        """
        Perform a sliding window method to
            the regularised-maximum-likelihood-estimator 
        """
        from scipy.optimize import fsolve
        # current time step
        current_step = self.step

        # Use only the most recent `window_size` data points
        start_idx = max(0, current_step - window_size)        

        # Collect historical informaton and update kappa
        flag = self.coming_buyMOs[start_idx:current_step] != 0
        coming_buyMOs = self.coming_buyMOs[start_idx:current_step][flag]
        hitsellLOs = self.hitsellLOs[start_idx:current_step][flag]
        postselldepth = self.postselldepth[start_idx:current_step][flag]

        # also use sell MOs to laern kappa
        flag = self.coming_sellMOs[start_idx:current_step] != 0
        coming_sellMOs = self.coming_sellMOs[start_idx:current_step][flag]
        hitbuyLOs = self.hitbuyLOs[start_idx:current_step][flag]
        postbuydepth = self.postbuydepth[start_idx:current_step][flag]

        coming_MOs = np.concatenate([coming_buyMOs, coming_sellMOs])
        hitLOs = np.concatenate([hitsellLOs, hitbuyLOs])
        postdepth = np.concatenate([postselldepth, postbuydepth])

        def f(x):
            mle = delta0 * (np.exp(-x*delta0)/(1 - np.exp(-x*delta0)) - 1)
            for idx, trials in enumerate(coming_MOs):
                if postdepth[idx] > 1e15: 
                    continue
                temp = ( 
                    - postdepth[idx] * trials + 
                    postdepth[idx] * (trials - hitLOs[idx]) / (1 - np.exp(-x*postdepth[idx]))
                )
                mle += temp
            return mle
        
        def f_upper(x):
            mle = delta0 * (np.exp(-self.K_upper*delta0)/(1 - np.exp(-self.K_upper*delta0)) - 1)
            for idx, trials in enumerate(coming_MOs):
                if postdepth[idx] > 1e15: 
                    continue
                temp = ( 
                    - postdepth[idx] * trials + 
                    postdepth[idx] * (trials - hitLOs[idx]) / (1 - np.exp(-self.K_upper*postdepth[idx]))
                )

                temp2 = (
                    - delta0**2*(np.exp(-self.K_upper*delta0))/(1-np.exp(-self.K_upper*delta0))**2 
                    - (trials - hitLOs[idx])*postdepth[idx]**2*(np.exp(-self.K_upper*postdepth[idx]))/(1 - np.exp(-self.K_upper*postdepth[idx]))**2
                ) * (x - self.K_upper)
                mle = mle + temp + temp2
            return mle

        # kappa_new = fsolve(f, np.random.uniform(self.K_lower, self.K_upper))[0]
        if len(coming_MOs) != 0:
            # kappa_new = fsolve(f, np.random.uniform(self.K_lower, self.K_upper))[0]
            kappa_new = fsolve(f, self.kappa)[0]
        
            if kappa_new > self.K_upper: 
                # kappa_new = fsolve(f_upper, np.random.uniform(self.K_lower, self.K_upper))[0]
                kappa_new = fsolve(f_upper, self.kappa)[0]
            # if the updated kappa is not in the boundary
            if kappa_new > self.K_upper:
                kappa_new = self.K_upper
            
            elif kappa_new < self.K_lower:
                kappa_new = self.K_lower
            
        else:
            kappa_new = self.kappa
        # kappa_new = fsolve(f, self.kappa)[0]
        # kappa_new = kappa_new if kappa_new <= 65 and kappa_new >= 1 else np.random.uniform(35, 45)
        
        # # exponetially weighted moving average
        # kappa_new = alpha * kappa_new + (1 - alpha) * self.kappa

        # update kappa 
        self.kappa_learnlist.append(kappa_new)
        self.kappa = kappa_new
        # update control 
        self.control = ErgodicCP(
            lambda_buy=self.lambda_buy,
            lambda_sell=self.lambda_sell,
            q_upper=self.q_upper,
            q_lower=self.q_lower,
            phi=self.phi,
            kappa=self.kappa,
        ).EControl

        return None    
    
    def estimator_ewma(self, alpha=0.9, delta0 = 1e-5):
        """
        Implement the exponentially moving average maximum-likelihood-estimator
        """
        from scipy.optimize import fsolve
        # current time step
        current_step = self.step
        time_differences = self.ts[current_step-1] - self.ts[:current_step]

        # Collect historical informaton and update kappa
        flag = self.coming_buyMOs[:current_step] != 0
        coming_buyMOs = self.coming_buyMOs[:current_step][flag]
        hitsellLOs = self.hitsellLOs[:current_step][flag]
        postselldepth = self.postselldepth[:current_step][flag]
        time_differences_buy = time_differences[flag]

        # also use sell MOs to laern kappa
        flag = self.coming_sellMOs[:current_step] != 0
        coming_sellMOs = self.coming_sellMOs[:current_step][flag]
        hitbuyLOs = self.hitbuyLOs[:current_step][flag]
        postbuydepth = self.postbuydepth[:current_step][flag]
        time_differences_sell = time_differences[flag]

        coming_MOs = np.concatenate([coming_buyMOs, coming_sellMOs])
        hitLOs = np.concatenate([hitsellLOs, hitbuyLOs])
        postdepth = np.concatenate([postselldepth, postbuydepth])
        wights = np.concatenate([np.exp(- alpha * time_differences_buy), np.exp(- alpha * time_differences_sell)])

        def f(x):
            mle = delta0 * (np.exp(-x*delta0)/(1 - np.exp(-x*delta0)) - 1)
            for idx, trials in enumerate(coming_MOs):
                if postdepth[idx] > 1e15: 
                    continue
                temp = ( 
                    - postdepth[idx] * trials + 
                    postdepth[idx] * (trials - hitLOs[idx]) / (1 - np.exp(-x*postdepth[idx]))
                )
                mle += temp*wights[idx]
            return mle
        
        def f_upper(x):
            mle = delta0 * (np.exp(-self.K_upper*delta0)/(1 - np.exp(-self.K_upper*delta0)) - 1)
            for idx, trials in enumerate(coming_MOs):
                if postdepth[idx] > 1e15: 
                    continue
                temp = ( 
                    - postdepth[idx] * trials + 
                    postdepth[idx] * (trials - hitLOs[idx]) / (1 - np.exp(-self.K_upper*postdepth[idx]))
                )

                temp2 = (
                    - delta0**2*(np.exp(-self.K_upper*delta0))/(1-np.exp(-self.K_upper*delta0))**2 
                    - (trials - hitLOs[idx])*postdepth[idx]**2*(np.exp(-self.K_upper*postdepth[idx]))/(1 - np.exp(-self.K_upper*postdepth[idx]))**2
                ) * (x - self.K_upper)
                mle = mle + temp*wights[idx] + temp2*wights[idx]
            return mle

        # kappa_new = fsolve(f, np.random.uniform(self.K_lower, self.K_upper))[0]
        if len(coming_MOs) != 0:
            # kappa_new = fsolve(f, np.random.uniform(self.K_lower, self.K_upper))[0]
            kappa_new = fsolve(f, self.kappa)[0]
        
            if kappa_new > self.K_upper: 
                # kappa_new = fsolve(f_upper, np.random.uniform(self.K_lower, self.K_upper))[0]
                kappa_new = fsolve(f_upper, self.kappa)[0]

            # if the updated kappa is not in the boundary
            if kappa_new > self.K_upper:
                kappa_new = self.K_upper
            
            elif kappa_new < self.K_lower:
                kappa_new = self.K_lower
            
        else:
            kappa_new = self.kappa
        # kappa_new = fsolve(f, self.kappa)[0]
        # kappa_new = kappa_new if kappa_new <= 65 and kappa_new >= 1 else np.random.uniform(35, 45)
        
        # # exponetially weighted moving average
        # kappa_new = alpha * kappa_new + (1 - alpha) * self.kappa

        # update kappa 
        self.kappa_learnlist.append(kappa_new)
        self.kappa = kappa_new
        # update control 
        self.control = ErgodicCP(
            lambda_buy=self.lambda_buy,
            lambda_sell=self.lambda_sell,
            q_upper=self.q_upper,
            q_lower=self.q_lower,
            phi=self.phi,
            kappa=self.kappa,
        ).EControl

        return None

    def standard_estimator(self):
        """
        Implement the standard maximum-likelihood-estimator
        """
        from scipy.optimize import fsolve
        # current time step
        current_step = self.step

        # Collect historical informaton and update kappa
        flag = self.coming_buyMOs[:current_step] != 0
        coming_buyMOs = self.coming_buyMOs[:current_step][flag]
        hitsellLOs = self.hitsellLOs[:current_step][flag]
        postselldepth = self.postselldepth[:current_step][flag]

        # also use sell MOs to laern kappa
        flag = self.coming_sellMOs[:current_step] != 0
        coming_sellMOs = self.coming_sellMOs[:current_step][flag]
        hitbuyLOs = self.hitbuyLOs[:current_step][flag]
        postbuydepth = self.postbuydepth[:current_step][flag]

        coming_MOs = np.concatenate([coming_buyMOs, coming_sellMOs])
        hitLOs = np.concatenate([hitsellLOs, hitbuyLOs])
        postdepth = np.concatenate([postselldepth, postbuydepth])

        def f(x):
            mle = 0
            for idx, trials in enumerate(coming_MOs):
                if postdepth[idx] > 1e15: 
                    continue
                temp = ( 
                    - postdepth[idx] * trials + 
                    postdepth[idx] * (trials - hitLOs[idx]) / (1 - np.exp(-x*postdepth[idx]))
                )
                mle += temp
            return mle


        # kappa_new = fsolve(f, np.random.uniform(self.K_lower, self.K_upper))[0]
        if len(coming_MOs) != 0:
            # kappa_new = fsolve(f, np.random.uniform(self.K_lower, self.K_upper))[0]
            kappa_new = fsolve(f, self.kappa)[0]
    
            # if the updated kappa is not in the boundary
            if kappa_new > self.K_upper or kappa_new < self.K_lower:
                kappa_new = self.kappa
            
        else:
            kappa_new = self.kappa
        # kappa_new = fsolve(f, self.kappa)[0]
        # kappa_new = kappa_new if kappa_new <= 65 and kappa_new >= 1 else np.random.uniform(35, 45)
        
        # update kappa 
        self.kappa_learnlist.append(kappa_new)
        self.kappa = kappa_new
        # update control 
        self.control = ErgodicCP(
            lambda_buy=self.lambda_buy,
            lambda_sell=self.lambda_sell,
            q_upper=self.q_upper,
            q_lower=self.q_lower,
            phi=self.phi,
            kappa=self.kappa,
        ).EControl

        return None   