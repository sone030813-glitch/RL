# Reinforcement Learning — MDP / DPP / KL / Q-learning 思维图

---

## 1. Reinforcement Learning (RL) 的本质
- **定义**
  - RL = 在未知环境中，通过交互学习最优决策规则（policy）
- **核心目标**
  - 最小化长期期望 cost（或最大化 reward）

### 金融价值
- 策略交易 = 连续决策问题
- 你不知道真实市场分布，只能通过历史和模拟学习

---

## 2. Markov Decision Process (MDP)

### 2.1 MDP 的组成
- **状态空间**：`S`
- **动作空间**：`A`
- **转移核**：`P(ds' | s, a)`
- **单步 cost**：`c(s, a)`
- **折扣因子**：`γ ∈ (0,1)`

### 2.2 转移核 P 的含义
- **定义**
  - 给定当前状态和动作，下一个状态的概率分布
- **公式**
  \[
  s_{t+1} \sim P(\cdot | s_t, a_t)
  \]

### 如何用
- 不需要知道解析形式
- 通过样本 `(s_t, a_t, s_{t+1})` 近似

### 金融价值
- 类似：  
  > “当前仓位 + 操作 → 下一期市场状态的分布”
- 例如：  
  - 当前风险暴露 + 加仓 → 下期回撤概率分布

---

## 3. Policy（策略）

### 3.1 确定性策略
- \[
  a = π(s)
  \]

### 3.2 随机策略（更一般）
- \[
  π(da|s) ∈ \mathcal P(A)
  \]

### 为什么要随机策略？
- 探索（exploration）
- 稳定性（避免过拟合）
- 理论上更容易证明最优性

### 金融价值
- 不每次都 All-in 同一操作
- 控制换仓频率、交易冲击

---

## 4. Value Function（价值函数）

### 4.1 状态价值函数 V
\[
V^π(s)
=
\mathbb E_s^π \Big[\sum_{t=0}^\infty γ^t c(s_t,a_t)\Big]
\]

### 4.2 状态-动作价值函数 Q
\[
Q^π(s,a)
=
c(s,a) + γ \mathbb E[V^π(s')]
\]

### 如何用
- V：评估“在这个状态好不好”
- Q：评估“在这个状态做这个动作好不好”

### 金融价值
- V：当前市场环境下长期风险水平
- Q：当前是否该加仓 / 减仓 / 对冲

---

## 5. Dynamic Programming Principle (DPP)

### 5.1 Bellman 方程（τ = 0）
\[
V^*(s)
=
\min_{a∈A}
\Big[
c(s,a) + γ \int V^*(s')P(ds'|s,a)
\Big]
\]

### 含义
- “现在最优 + 未来最优 = 整体最优”

### 金融价值
- 动态资产配置
- 不是“这一步赚不赚”，而是“这一步是否改善长期路径”

---

## 6. Curse of Dimensionality（维度灾难）

### 问题
- S / A 连续或高维 → 无法 tabular 计算
- Bellman 方程无法直接算

### 解决方案
- **函数逼近（Function Approximation）**

---

## 7. Function Approximation（函数逼近）

### 7.1 线性逼近（数学家最爱）
\[
V(s) ≈ \sum_{k=1}^M θ_k φ_k(s)
\]

- 优点：可解释、稳定
- 缺点：表达能力有限

### 7.2 神经网络逼近（工程主流）
\[
V(s) ≈ \text{NN}_θ(s)
\]

### 金融价值
- 非线性风险结构
- regime switching
- 尾部风险

---

## 8. Entropy / Relative Entropy（KL 散度）

### 8.1 KL 定义
\[
KL(ν||μ)
=
\int \ln\frac{dν}{dμ} dν
\]

### 含义
- 两个策略/分布的“距离”
- 偏离参考策略的代价

### 图像含义
- `s ln s ≥ s − 1`
- KL ≥ 0，且 =0 当且仅当两个分布相同

### 金融价值
- 约束策略变化幅度
- 防止过度换仓

---

## 9. Relaxed MDP（松弛形式）

### 核心思想
- 不选“一个动作”
- 而是选“一个动作分布”

### 带 KL 正则的目标
\[
\min_π
\mathbb E\Big[
\sum γ^t
(c(s_t,a_t) + τ KL(π_t||μ))
\Big]
\]

### 金融价值
- 稳定交易策略
- 控制交易冲击与模型不确定性

---

## 10. DPP with τ > 0（熵正则 Bellman）

### Bellman（软）
\[
V^*_τ(s)
=
\inf_{m∈\mathcal P(A)}
\int
\Big(
c(s,a) + τ \ln\frac{dm}{dμ}(a)
+ γ \mathbb E[V^*_τ(s')]
\Big)
m(da)
\]

### 结果
- 最优策略是 **softmax**
\[
π^*(a|s)
∝
\exp(-Q^*_τ(s,a)/τ)\,μ(a)
\]

### 金融价值
- 连续仓位调整
- 风险平滑，而非跳跃

---

## 11. Regret（后悔度，Bandit）

### 定义
\[
R(N)
=
\sum_{t=1}^N
(r^* - r_{a_t})
\]

### 含义
- 和“事后最优”相比损失了多少

### 好的学习
- Regret = O(log N)

### 金融价值
- 策略学习初期的“试错成本”
- 控制探索代价

---

## 12. Classical Bandit Methods

### ε-greedy
- 概率 1−ε 选最优
- 概率 ε 探索

### UCB
\[
\text{score}
=
\hat μ + \text{uncertainty bonus}
\]

### 金融价值
- 在多个 alpha / 策略之间动态分配资金

---

## 13. Q-learning（核心算法）

### Bellman for Q
\[
Q^*(s,a)
=
c(s,a) + γ \min_{a'} Q^*(s',a')
\]

### Sample-based 更新
\[
Q_{k+1}
=
Q_k
+
α
\Big[
c + γ \min Q_k(s') - Q_k
\Big]
\]

### 金融价值
- 不需要知道转移分布
- 只用历史数据 / 回测 / 模拟

---

## 14. Entropy-Regularized Q-learning（Soft Q）

### 更新公式
\[
Q_{k+1}
=
Q_k
+
α
\Big[
c + γ(Q_k(s',a') + τ \ln\tfrac{dπ}{dμ}) - Q_k
\Big]
\]

### 策略
\[
π(a|s) ∝ \exp(-Q(s,a)/τ)
\]

### 金融价值
- 稳定、可控、可解释的 RL
- SAC / modern RL 的数学基础

---

## 15. “Solving RL” 的真正含义

### 不意味着
- 精确解 Bellman 方程

### 而是
- 在未知 P、c 下
- 通过模拟器
- 学到 **近似最优且稳定的策略**

### 金融现实
- 市场不是已知模型
- RL = 风险受控的自适应决策系统

---

## 总结一句话（你可以放在笔记最顶端）

> RL =  
> 在未知动态系统中，  
> 用样本 + Bellman 结构 + 函数逼近 + 正则化，  
> 学一个 **长期稳健、风险可控的决策规则**。

