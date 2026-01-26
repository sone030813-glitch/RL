# Reinforcement Learning — MDP / DPP / KL / Q-learning 思维图（可直接放入 .md）

> 结构：知识点（并列/递进） + 公式（如有） + 公式如何用 + 有什么用 + 金融价值

---

## 1. RL 的目标（Optimization Target）
- **知识点**：在未知环境中学策略，使长期累计 cost 最小（或 reward 最大）
- **公式**：
  $$V^\pi(s)=\mathbb{E}_s^\pi\left[\sum_{t=0}^\infty \gamma^t\,c(s_t,a_t)\right]$$
- **如何用**：定义“好策略”的评判标准；所有算法都在逼近/优化它
- **有什么用**：把决策问题转为可优化目标
- **金融价值**：把交易策略定义为“长期风险/成本最小”的可优化对象（而非单次预测）

---

## 2. MDP（Markov Decision Process）
- **知识点**：用五元组描述序列决策：$(S,A,P,c,\gamma)$
- **转移核（Transition Kernel）**
  - **公式**：$$s_{t+1}\sim P(\cdot\mid s_t,a_t)$$
  - **如何用**：理论推导用它写期望；RL 中不需要知道 $P$，只需采样 $(s,a,s')$
  - **金融价值**：类似“给定当前状态与操作，下期市场状态的概率分布”

---

## 3. Policy（策略）
- **知识点**：策略是“状态到动作（或动作分布）的映射”
- **公式**：
  - 确定性：$$a=\pi(s)$$
  - 随机性：$$\pi(\mathrm{d}a\mid s)\in\mathcal{P}(A)$$
- **如何用**：随机策略可自然表达探索与稳定性约束
- **金融价值**：避免极端确定性换仓；可以表达“分散下单/仓位随机化”的稳健性

---

## 4. Value / Q function（价值函数）
- **知识点**：$V$ 评估状态好坏，$Q$ 评估状态-动作好坏
- **公式**：
  $$Q^\pi(s,a)=c(s,a)+\gamma\int_S V^\pi(s')\,P(\mathrm{d}s'\mid s,a)$$
- **如何用**：先学 $Q$，再由 $Q$ 推出“该选什么动作”
- **金融价值**：$Q$ 可直接对应“在当前市场下采取某个仓位动作的长期代价”

---

## 5. DPP / Bellman（动态规划原理）
### 5.1 经典 Bellman（$\tau=0$）
- **公式**：
  $$V^*(s)=\min_{a\in A}\left[c(s,a)+\gamma\int_S V^*(s')\,P(\mathrm{d}s'\mid s,a)\right]$$
- **如何用**：若模型已知，可 value iteration / policy iteration 直接算
- **有什么用**：给出最优性不动点方程
- **金融价值**：用“当前成本 + 未来最优”定义最优交易/对冲路径

---

## 6. 维度灾难（Curse of Dimensionality）
- **知识点**：状态/动作高维或连续时，表格法不可行
- **如何用**：引入函数逼近（线性/NN）近似 $V,Q,\pi$
- **金融价值**：市场状态通常高维（价格、波动、因子、仓位、流动性等）

---

## 7. Function Approximation（函数逼近）
### 7.1 线性逼近（数学上稳定）
- **公式**：
  $$f(x)\approx \sum_{k=1}^M \theta_k\,\phi_k(x)$$
- **如何用**：选 basis/features，再拟合参数 $\theta$
- **金融价值**：可解释、易控（适合风控与审计）

### 7.2 神经网络逼近（表达能力强）
- **公式**：$$f(x)\approx \text{NN}_\theta(x)$$
- **如何用**：用梯度下降拟合 TD target / Bellman residual
- **金融价值**：能表示非线性与 regime、尾部风险结构，但稳定性更难保证

---

## 8. Relative Entropy / KL（相对熵）
- **知识点**：衡量策略/分布偏离参考分布 $\mu$ 的代价
- **公式**：
  $$\mathrm{KL}(\nu\mid\mu)=\int_A \ln\left(\frac{\mathrm{d}\nu}{\mathrm{d}\mu}\right)\,\mathrm{d}\nu \quad (\nu\ll\mu)$$
- **如何用**：作为正则项约束策略变化（稳定、可控）
- **有什么用**：保证 $\mathrm{KL}\ge 0$，且等号当且仅当 $\nu=\mu$
- **金融价值**：解释为交易成本/换仓惩罚/风险偏好约束，抑制策略跳变

---

## 9. Relaxed formulation（松弛策略：选“分布”而非单点动作）
- **知识点**：把动作选择从 $a$ 扩展为动作分布 $m\in\mathcal{P}(A)$
- **如何用**：使问题更“平滑”，便于分析与得到 soft 策略闭式解
- **金融价值**：对应“连续仓位调整”和“稳健执行”，避免激进离散跳仓

---

## 10. Entropy-regularized DPP（$\tau>0$ 的 soft Bellman）
- **知识点**：在 Bellman 里加入 KL 正则（对策略偏离惩罚）
- **价值方程（核心形态）**：
  $$V^*_\tau(s)=\inf_{m\in\mathcal{P}(A)}\int_A \left[c(s,a)+\tau\ln\left(\frac{\mathrm{d}m}{\mathrm{d}\mu}\right)(a)+\gamma\int_S V^*_\tau(s')P(\mathrm{d}s'\mid s,a)\right]\,m(\mathrm{d}a)$$
- **如何用**：通过变分公式可推出 softmin / softmax 结构
- **最优策略（softmax 形态）**：
  $$\pi^*(a\mid s)\propto \exp\left(-\frac{Q^*_\tau(s,a)}{\tau}\right)\,\mu(a)$$
- **金融价值**：
  - 连续仓位调整
  - 风险平滑而非跳跃
  - 可解释为“偏离基准策略会被收费”

---

## 11. Regret（后悔度，Bandit）
- **知识点**：衡量学习过程中相对“事后最优臂”的累计损失
- **公式（常见定义）**：
  $$R(N)=\sum_{t=1}^N\left(r^*-r_{a_t}\right)$$
  其中 $r^*$ 是最优臂期望回报，$r_{a_t}$ 是你选的臂的回报
- **如何用**：分析探索-利用策略的性能上界（如 $O(\log N)$）
- **金融价值**：衡量“试错成本”；决定探索力度（例如新因子上线的资金占比）

---

## 12. Classical Bandit Methods（探索-利用）
### 12.1 Explore-then-exploit
- **用法**：先每个臂试 $M$ 次，再选均值最优的臂
- **金融价值**：先小资金跑多策略试验，再集中资金到表现最佳者

### 12.2 $\varepsilon$-greedy
- **用法**：以 $1-\varepsilon$ 选当前最优臂，以 $\varepsilon$ 随机探索
- **金融价值**：大部分时间执行主策略，小部分时间测试替代策略

### 12.3 UCB（Upper Confidence Bound）
- **思想**：均值估计 + 不确定性奖励（越不确定越鼓励探索）
- **金融价值**：优先给“高 alpha 且样本少”的新策略更多试运行机会

---

## 13. “Solving RL” 的课程定义（这页 slide 的结论）
- **知识点**：我们说“解决了 RL”当且仅当能在以下条件下学到 near-optimal policy：
  - 不知道 $c$ 与 $P$
  - 自选 $\gamma>0,\ \tau\ge 0$
  - 只能通过 simulator 反复采样交互
  - 初始状态来自分布 $\rho$，到终止或 reset
- **金融价值**：现实市场 $P,c$ 不可得，只能靠数据/仿真/回测逼近最优策略

---

## 14. Q-learning（$\varepsilon$-greedy 元算法，$\tau=0$ 版本）
- **Bellman target**：
  $$y=c+\gamma\min_{a'}Q_\theta(s',a')$$
- **损失函数（拟合 Bellman）**：
  $$L(\theta)=\mathbb{E}\left[(y-Q_\theta(s,a))^2\right]$$
- **如何用**：
  - 用 buffer 存 $(s,a,c,s')$
  - 采样 batch，算 target $y$
  - 梯度下降更新 $\theta$
- **金融价值**：无需模型 $P$；只要历史/模拟即可进行策略学习

---

## 15. Soft / Entropy-regularized Q-learning（$\tau>0$）
- **知识点**：把 Bellman “min” 替换成 softmin，并加入 log-ratio 项
- **soft 策略（由 $Q$ 给出）**：
  $$\pi_k(a\mid s)\propto \exp\left(-\frac{Q_k(s,a)}{\tau}\right)\mu(a)$$
- **TD 形式更新的核心结构**：
  $$Q_{k+1}(s,a)=Q_k(s,a)+\alpha\left[c+\gamma\left(Q_k(s',a')+\tau\ln\frac{\mathrm{d}\pi_k}{\mathrm{d}\mu}(a'\mid s')\right)-Q_k(s,a)\right]$$
- **如何用**：
  - 用采样得到 $(s,a,c,s')$
  - 从 $\pi_k(\cdot\mid s')$ 采样 $a'$
  - 计算带 KL 的 target，做 TD 更新
- **金融价值**：
  - 自带稳定性与“换仓收费”
  - 对连续动作（仓位）更自然
  - 比 $\varepsilon$-greedy 更平滑、可控

---

## 16. 统一金融案例（贯穿全套知识点）
- **状态 $s$**：价格/波动/因子/持仓/流动性/风险指标
- **动作 $a$**：调仓幅度、对冲比例、杠杆水平
- **转移 $P$**：市场演化（未知，只能采样）
- **cost $c$**：负收益、回撤、VaR/ES 惩罚、交易成本
- **KL 正则**：限制与基准仓位/上一期仓位的偏离，控制换仓与冲击
- **目标**：学到长期回撤可控且收益稳健的策略（near-optimal）

---
