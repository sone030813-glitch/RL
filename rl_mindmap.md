# Reinforcement Learning — MDP / DPP / KL / Q-learning 思维图

> **目标读者**：零基础入门，最终理解如何用 RL 做金融交易策略
>
> **结构**：知识点 → 直觉解释 → 公式 → 金融价值

---

# Part 0: 什么是强化学习？（给完全不懂的人）

## 0.1 一句话定义
**强化学习 = 通过"试错"学习最优决策的方法**

想象你在玩一个游戏：
- 你不知道游戏规则（环境未知）
- 每做一个动作，游戏会给你反馈（奖励或惩罚）
- 你的目标：通过不断尝试，找到让**长期总收益最大**的策略

## 0.2 和监督学习的区别

| 监督学习 | 强化学习 |
|----------|----------|
| 有标准答案（标签） | 没有标准答案，只有奖惩信号 |
| 一次性预测 | 序列决策（一连串动作） |
| 数据独立 | 当前动作影响未来状态 |
| 例：预测明天涨跌 | 例：学习一整套交易策略 |

## 0.3 金融中的直觉

把交易想象成"游戏"：
- **状态**：当前市场情况（价格、波动率、持仓等）
- **动作**：你的操作（买入、卖出、持有、调仓比例）
- **奖励**：收益（正）或亏损（负）
- **目标**：学一个策略，让**长期累计收益最大**（不是单次预测准）

---

# Part 1: RL 的核心概念

## 1.1 RL 的优化目标

### 知识点
在未知环境中学策略，使**长期累计收益最大**（或成本最小）

### 直觉
不是追求"每一步都最优"，而是"长期总和最优"。
就像下棋：有时候牺牲一个子（短期亏损）是为了赢棋（长期收益）。

### 公式
价值函数（从状态 s 出发，按策略 π 行动的期望总成本）：

```
V^π(s) = E[ c₀ + γc₁ + γ²c₂ + γ³c₃ + ... ]
       = E[ Σ(t=0 到 ∞) γᵗ · c(sₜ, aₜ) ]
```

其中：
- `c(s,a)` = 在状态 s 做动作 a 的即时成本（或负收益）
- `γ` (gamma) = 折扣因子，0 < γ < 1，表示"未来的钱不如现在值钱"
- `E` = 期望（因为环境有随机性）

### γ 的直觉
| γ 值 | 含义 |
|------|------|
| γ = 0.99 | 很看重未来（长期投资者）|
| γ = 0.5 | 未来很快贬值（短线交易者）|
| γ = 0 | 只看当前（贪婪，不考虑后果）|

### 金融价值
把"好策略"量化为一个可优化的目标，而不是主观判断。

---

## 1.2 MDP（马尔可夫决策过程）

### 知识点
MDP 是描述"序列决策问题"的数学框架：

```
MDP = (S, A, P, c, γ)
```

| 符号 | 含义 | 金融例子 |
|------|------|----------|
| S | 状态空间 | 价格、波动率、持仓、因子值 |
| A | 动作空间 | 买/卖/持有，或仓位比例 [0,1] |
| P | 转移概率 | 给定当前状态和动作，下一状态的概率分布 |
| c | 成本函数 | 交易成本 + 负收益 + 风险惩罚 |
| γ | 折扣因子 | 对未来收益的重视程度 |

### 转移概率 P 的含义

```
s_{t+1} ~ P(·| sₜ, aₜ)
```

读作："下一个状态服从概率分布 P，这个分布取决于当前状态和动作"

### 马尔可夫性质
"未来只取决于现在，与过去无关"

### 金融价值
- 实际中我们**不知道 P**（不知道市场怎么变化）
- RL 的强大之处：**不需要知道 P，只需要能采样**（历史数据/回测/模拟器）

---

## 1.3 策略（Policy）

### 两种类型

**确定性策略**：给定状态，输出唯一动作
```
a = π(s)
例：如果价格 > 均线，就买入
```

**随机策略**：给定状态，输出动作的概率分布
```
a ~ π(·|s)
例：70% 概率买入，30% 概率持有
```

### 为什么要随机策略？
1. **探索**：避免陷入局部最优
2. **稳健**：避免极端操作，分散风险
3. **数学更好处理**：优化问题更平滑

### 金融价值
避免"全仓梭哈"，可以表达"分批建仓"等稳健策略

---

## 1.4 价值函数 V 和 Q

### 定义
- **V(s)**：状态价值 —— "这个状态有多好"
- **Q(s,a)**：动作价值 —— "在这个状态做这个动作有多好"

### 公式

```
Q^π(s,a) = c(s,a) + γ · E[ V^π(s') ]
         = 即时成本 + 折扣 × 下一状态的价值
```

### 直觉
- 有了 Q，就知道该怎么选动作：**选 Q 最小（成本最低）的动作**

### 金融价值
Q(s,a) = "在当前市场状态下，采取某个仓位的长期代价"

---

# Part 2: Bellman 方程与动态规划

## 2.1 Bellman 方程（核心！）

### 核心思想
> **今天的最优 = 今天的成本 + 未来的最优**

### 公式

```
V*(s) = min_a [ c(s,a) + γ · E[ V*(s') ] ]
```

读作："最优价值 = 选择让（即时成本 + 未来价值）最小的动作"

### 金融价值
用"当前交易成本 + 未来最优路径"定义最优策略

---

## 2.2 维度灾难

### 问题
金融状态是**高维连续**的（价格、波动率、因子、持仓...），表格法不可行

### 解决方案：函数逼近

```
Q(s,a) ≈ Q_θ(s,a)
```

### 两种方式

| 线性逼近 | 神经网络 |
|----------|----------|
| `Q ≈ Σ θₖ·φₖ(s,a)` | `Q ≈ NN_θ(s,a)` |
| 可解释、稳定 | 表达能力强 |
| 需要手工特征 | 自动学特征 |
| 监管友好 | 黑箱 |

---

# Part 3: 探索与利用

## 3.1 核心矛盾

- **利用**：选当前认为最好的
- **探索**：尝试不确定的，可能发现更好的

## 3.2 Regret（后悔度）

```
Regret(N) = N × 最优收益 - 实际总收益
```

衡量"因为探索损失了多少"

## 3.3 经典策略

### ε-greedy
```
以概率 (1-ε)：选最优（利用）
以概率 ε：随机选（探索）
```
**金融**：90% 资金主策略，10% 测试新策略

### UCB
```
选择：估计均值 + 不确定性奖励
```
**金融**：优先给"高 alpha 但样本少"的新因子更多机会

---

# Part 4: KL 散度与熵正则化

## 4.1 KL 散度

### 知识点
衡量两个概率分布的"差异" / "偏离成本"

### 公式

```
KL(ν|μ) = E_ν[ ln(ν/μ) ] ≥ 0
```

- KL = 0 当且仅当 ν = μ

### 金融价值
- 解释为**换仓成本**：偏离基准越多，成本越高
- 抑制策略剧烈跳变

---

## 4.2 熵正则化（Soft Bellman）

### 公式

```
V*_τ(s) = min_m [ ∫ (c(s,a) + τ·ln(m/μ) + γ·E[V*_τ(s')]) dm(a) ]
```

### τ 的作用

| τ 值 | 效果 |
|------|------|
| τ = 0 | 标准 Bellman（硬决策）|
| τ 小 | 稍微平滑 |
| τ 大 | 更随机，更接近参考分布 |

### 最优策略（Softmax 形式）

```
π*(a|s) ∝ exp(-Q*(s,a)/τ) · μ(a)
```

### 金融价值
- 连续仓位调整（不是 0/1 跳变）
- "偏离基准要收费" → 天然换仓成本

---

# Part 5: Q-learning

## 5.1 标准 Q-learning（τ = 0）

### 核心
不需要知道 P，只需要**采样经验**来学习 Q

### 算法

```
1. 在状态 s，用 ε-greedy 选动作 a
2. 执行 a，观察成本 c 和下一状态 s'
3. 计算 target：y = c + γ · min_{a'} Q(s', a')
4. 更新 Q 最小化：(y - Q(s,a))²
```

### 损失函数

```
L(θ) = E[ (c + γ·min_{a'}Q(s',a') - Q(s,a))² ]
```

---

## 5.2 Soft Q-learning（τ > 0）

### Soft 策略

```
π(a|s) ∝ exp(-Q(s,a)/τ) · μ(a)
```

### 对比

| 标准 Q-learning | Soft Q-learning |
|-----------------|-----------------|
| min 选动作 | softmin（概率采样）|
| 策略可能跳变 | 策略平滑变化 |
| 探索靠 ε-greedy | 自带探索（熵奖励）|
| 离散动作友好 | 连续动作更自然 |

### 金融价值
- 自带稳定性与换仓惩罚
- 适合连续仓位调整

---

# Part 6: 金融应用完整案例

## 6.1 问题设定

### 状态 s
```
价格、收益率、波动率、因子值、当前持仓、现金、风险敞口
```

### 动作 a
```
离散：买/卖/持有
连续：目标仓位比例 [0, 1]
```

### 成本 c
```
c = -收益 + 交易成本 + 风险惩罚(VaR) + 回撤惩罚
```

### KL 正则
```
换仓越大，惩罚越大
```

---

## 6.2 代码框架

```python
import numpy as np

class TradingEnv:
    """交易环境（模拟器）"""

    def __init__(self, prices, transaction_cost=0.001):
        self.prices = prices
        self.tc = transaction_cost
        self.t = 0
        self.position = 0.0

    def reset(self):
        self.t = 0
        self.position = 0.0
        return self._get_state()

    def step(self, action):
        """执行动作，返回 (下一状态, 成本, 是否结束)"""
        # action = 目标仓位
        trade = action - self.position

        # 计算成本
        returns = (self.prices[self.t+1] - self.prices[self.t]) / self.prices[self.t]
        pnl = self.position * returns
        tc = self.tc * abs(trade)
        cost = -pnl + tc  # 负收益 + 交易成本

        # 更新状态
        self.position = action
        self.t += 1
        done = (self.t >= len(self.prices) - 1)

        return self._get_state(), cost, done

    def _get_state(self):
        # 简化：用最近 N 天收益率作为状态
        return self.prices[max(0,self.t-10):self.t+1]


class QLearningAgent:
    """Q-learning 智能体"""

    def __init__(self, state_dim, n_actions, gamma=0.99, lr=0.001, epsilon=0.1):
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.n_actions = n_actions

        # 简化：用线性函数逼近 Q
        self.theta = np.zeros((n_actions, state_dim))

    def get_q(self, state, action):
        return np.dot(self.theta[action], state)

    def choose_action(self, state):
        """ε-greedy 选动作"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            q_values = [self.get_q(state, a) for a in range(self.n_actions)]
            return np.argmin(q_values)  # 选成本最小的

    def update(self, state, action, cost, next_state, done):
        """Q-learning 更新"""
        if done:
            target = cost
        else:
            next_q_values = [self.get_q(next_state, a) for a in range(self.n_actions)]
            target = cost + self.gamma * min(next_q_values)

        current_q = self.get_q(state, action)
        td_error = target - current_q
        self.theta[action] += self.lr * td_error * state


def train(env, agent, n_episodes=1000):
    """训练循环"""
    for episode in range(n_episodes):
        state = env.reset()
        total_cost = 0

        while True:
            action = agent.choose_action(state)
            next_state, cost, done = env.step(action)
            agent.update(state, action, cost, next_state, done)

            total_cost += cost
            state = next_state

            if done:
                break

        if episode % 100 == 0:
            print(f"Episode {episode}, Total Cost: {total_cost:.4f}")
```

---

## 6.3 实际部署

### 流程
1. 历史数据 / 模拟器中训练
2. 小资金实盘验证
3. 逐步增加资金

### 风险控制
- 硬约束：最大仓位、最大亏损
- 软约束：KL 正则、风险惩罚

---

# 速查表

## 符号对照

| 符号 | 含义 | 金融解释 |
|------|------|----------|
| s | 状态 | 市场信息 + 持仓 |
| a | 动作 | 交易决策 |
| c(s,a) | 即时成本 | 负收益 + 交易成本 |
| γ | 折扣因子 | 对未来的重视程度 |
| π | 策略 | 交易规则 |
| V(s) | 状态价值 | 期望总成本 |
| Q(s,a) | 动作价值 | 做这个交易的期望总成本 |
| τ | 温度参数 | 换仓惩罚强度 |
| KL | 相对熵 | 策略变化的成本 |

## 核心公式

```
Bellman:     V*(s) = min_a [ c(s,a) + γ·E[V*(s')] ]

Q-learning:  Q(s,a) ← Q(s,a) + α·[ c + γ·min_{a'}Q(s',a') - Q(s,a) ]

Soft Policy: π(a|s) ∝ exp(-Q(s,a)/τ)

KL:          KL(ν|μ) = E_ν[ ln(ν/μ) ] ≥ 0
```

## 算法选择

| 场景 | 推荐 |
|------|------|
| 离散动作（买/卖/持有） | Q-learning + ε-greedy |
| 连续动作（仓位比例） | Soft Q-learning / SAC |
| 需要可解释 | 线性函数逼近 |
| 复杂模式 | 神经网络（DQN）|
