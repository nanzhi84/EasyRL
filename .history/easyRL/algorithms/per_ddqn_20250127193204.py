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

    # 向上传播
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
        self.max_priority = 1.0 

    def store_transition(self, state, action, reward, next_state, done):
        """存储transition到优先级经验回放树中"""
        transition = np.hstack((state, [action, reward], next_state, [done]))
        self.tree.add(self.max_priority, transition)

    def learn(self, done):
        """执行学习过程"""
        # 检查是否有足够样本
        if self.tree.n_entries < self.batch_size:
            return

        # 定期更新目标网络参数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_params()

        # 初始化变量
        batch_idx = np.empty(self.batch_size, dtype=np.int32)
        batch_memory = np.empty((self.batch_size, self.tree.data[0].size), dtype=np.float32)
        weights = np.empty(self.batch_size, dtype=np.float32)
        
        # 计算分段大小并更新beta
        segment = self.tree.total() / self.batch_size
        self.beta = min(1.0, self.beta + self.beta_increment)

        # 采样batch
        for i in range(self.batch_size):
            s = random.uniform(segment*i, segment*(i+1))
            idx, p, data = self.tree.get(s)
            batch_idx[i] = idx
            batch_memory[i] = data
            
            # 计算重要性采样权重
            prob = p / (self.tree.total() + 1e-5)
            weights[i] = (prob * self.tree.n_entries) ** -self.beta

        # 转换数据格式
        weights = torch.FloatTensor(weights / np.max(weights)).unsqueeze(1).to(self.device)
        
        # 解包batch数据
        batch_state = torch.FloatTensor(batch_memory[:, :self.feature_dim]).to(self.device)
        batch_action = torch.LongTensor(batch_memory[:, self.feature_dim].astype(int)).unsqueeze(1).to(self.device)
        batch_reward = torch.FloatTensor(batch_memory[:, self.feature_dim+1]).to(self.device)
        batch_next_state = torch.FloatTensor(batch_memory[:, self.feature_dim+2:-1]).to(self.device)
        batch_done = torch.BoolTensor(batch_memory[:, -1].astype(bool)).to(self.device)

        # 计算当前Q值
        q_eval = self.eval_net(batch_state).gather(1, batch_action).squeeze()
        
        # 计算目标Q值
        with torch.no_grad():
            q_eval_next = self.eval_net(batch_next_state)
            eval_act_next = q_eval_next.max(dim=1)[1].unsqueeze(1)
        batch_state = torch.FloatTensor(batch_memory[:, :self.feature_dim])
        batch_action = torch.LongTensor(batch_memory[:, self.feature_dim].astype(int)).unsqueeze(1)
        batch_reward = torch.FloatTensor(batch_memory[:, self.feature_dim+1])
        batch_next_state = torch.FloatTensor(batch_memory[:, self.feature_dim+2:-1])
        batch_done = torch.BoolTensor(batch_memory[:, -1].astype(bool))

        q_eval = self.eval_net(batch_state).gather(1, batch_action).squeeze()
        
        q_eval_next = self.eval_net(batch_next_state).detach()
        eval_act_next = q_eval_next.max(dim=1)[1].unsqueeze(1)
        
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + ((~batch_done) * self.gamma * q_next.gather(1, eval_act_next).squeeze())

        # 计算优先级并更新
        td_errors = (q_target - q_eval).abs().detach().squeeze().numpy() 
        for idx, error in zip(batch_idx, td_errors):
            priority = (error + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

        loss = (weights * (q_target - q_eval).pow(2)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.eval_net.parameters(), 0.5)
        self.optimizer.step()

        self.learn_step_counter += 1
        self.decay_epsilon()