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
        
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch_idx.append(idx)
            batch_transitions.append(data)
            
            prob = p / self.tree.total()
            weight = (prob * self.tree.n_entries) ** (-self.beta)
            batch_weights.append(weight)

        batch_weights = np.array(batch_weights) / np.max(batch_weights)
        return batch_idx, batch_transitions, batch_weights

    def learn(self, done):
        if self.tree.n_entries < self.batch_size:
            return

        # 替换目标网络参数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_params()
            print('\ntarget params replaced\n')

        # 采样
        batch_idx, batch_transitions, batch_weights = self._sample()
        batch_memory = np.array(batch_transitions)
        weights = torch.FloatTensor(batch_weights).unsqueeze(1)

        # 正确解析存储的numpy数组结构
        batch_state = torch.FloatTensor(np.vstack(batch_memory[:, :self.feature_dim]))
        batch_action = torch.LongTensor(batch_memory[:, self.feature_dim].astype(int)).unsqueeze(1)
        batch_reward = torch.FloatTensor(batch_memory[:, self.feature_dim+1])
        batch_next_state = torch.FloatTensor(np.vstack(batch_memory[:, -self.feature_dim:]))

        # 计算当前Q值
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