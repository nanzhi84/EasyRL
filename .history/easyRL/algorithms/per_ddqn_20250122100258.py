import numpy as np
import torch
import random
from .ddqn import DDQN

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(left + 1, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

class PER_DDQN(DDQN):
    def __init__(self, 
                 feature_dim,
                 action_dim,
                 learning_rate=0.01,
                 gamma=0.9,
                 epsilon_start=0.9,
                 epsilon_end=0.1,
                 epsilon_decay=0.995,
                 memory_size=500,
                 batch_size=32,
                 replace_target_iter=300,
                 alpha=0.6,
                 beta=0.4,
                 beta_increment=0.001):
        
        super().__init__(
            feature_dim, action_dim, learning_rate, gamma, 
            epsilon_start, epsilon_end, epsilon_decay, 
            memory_size, batch_size, replace_target_iter
        )
        
        # PER特有参数
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.tree = SumTree(memory_size)
        self.max_priority = 1.0  # 初始优先级
        self.epsilon = 1e-5  # 防止除零

    def store_transition(self, s, a, r, s_):
        """存储transition并添加初始优先级"""
        transition = np.hstack((s, [a, r], s_))
        self.tree.add(self.max_priority, transition)

    def learn(self, done):
        if self.tree.n_entries < self.batch_size:
            return

        # 采样并计算重要性采样权重
        segment = self.tree.total() / self.batch_size
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        batch_idx, batch_data, weights = [], [], []
        for i in range(self.batch_size):
            s = random.uniform(segment*i, segment*(i+1))
            idx, p, data = self.tree.get(s)
            batch_idx.append(idx)
            batch_data.append(data)
            
            # 计算重要性采样权重
            prob = p / self.tree.total()
            weights.append((prob * self.tree.n_entries) ** -self.beta)
            
        weights = torch.FloatTensor(weights / np.max(weights)).unsqueeze(1)
        batch = np.array(batch_data)

        # 复用父类的数据预处理
        states = torch.FloatTensor(np.vstack(batch[:, :self.feature_dim]))
        actions = torch.LongTensor(batch[:, self.feature_dim].astype(int)).unsqueeze(1)
        rewards = torch.FloatTensor(batch[:, self.feature_dim+1])
        next_states = torch.FloatTensor(np.vstack(batch[:, -self.feature_dim:]))

        # 复用父类的Q值计算逻辑
        q_eval, q_target = super()._calculate_q_values(states, actions, next_states, done)

        # 计算TD误差并更新优先级
        td_errors = (q_target - q_eval).abs().detach().numpy() + self.epsilon
        for idx, error in zip(batch_idx, td_errors):
            priority = (error ** self.alpha)
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

        # 计算加权损失
        loss = (weights * (q_target - q_eval).pow(2)).mean()
        self._optimize_model(loss)

        # 更新计数器
        self.learn_step_counter += 1
        q_eval = self.eval_net(batch_state).gather(1, batch_action).squeeze()
        
        # 计算目标Q值（Double DQN）
        q_eval_next = self.eval_net(batch_next_state).detach()
        eval_act_next = q_eval_next.max(dim=1)[1].unsqueeze(1)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + (1 - done) * self.gamma * q_next.gather(1, eval_act_next).squeeze()

        # 计算TD误差并更新优先级
        td_errors = (q_target - q_eval).abs().detach().numpy() + self.epsilon
        for i in range(len(batch_idx)):
            idx = batch_idx[i]
            self.tree.update(idx, td_errors[i] ** self.alpha)
            self.max_priority = max(self.max_priority, td_errors[i] ** self.alpha)

        # 计算加权损失
        loss = (weights * (q_target - q_eval).pow(2)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.eval_net.parameters(), 0.5)
        self.optimizer.step()
        self.learn_step_counter += 1
        self.decay_epsilon()