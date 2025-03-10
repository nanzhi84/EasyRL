import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

class ActorNetwork(nn.Module):
    def __init__(self, feature_dim, action_dim):
        super().__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.mu = nn.Linear(64, action_dim)
        self.sigma = nn.Linear(64, action_dim)
        
        # 参数初始化
        nn.init.uniform_(self.mu.weight, -1e-3, 1e-3)
        nn.init.uniform_(self.sigma.weight, -1e-3, 1e-3)
        
    def forward(self, x):
        x = self.shared_net(x)
        mu = torch.tanh(self.mu(x))
        sigma = torch.nn.functional.softplus(self.sigma(x)) + 1e-4
        return mu, sigma

class CriticNetwork(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.net(x)

class TRPO:
    def __init__(self, 
                 feature_dim,
                 action_dim,
                 learning_rate=3e-4,
                 gamma=0.99,
                 gae_lambda=0.95,
                 max_kl=1e-2,
                 cg_iters=10,
                 damping=1e-1):
        """
        参数说明：
           feature_dim: 状态特征维度
           action_dim: 动作维度
           learning_rate: 学习率（仅用于价值网络更新）
           gamma: 折扣因子
           gae_lambda: GAE lambda参数
           max_kl: 最大KL散度（信赖域约束)
           cg_iters: 共轭梯度迭代步数
           damping: 阻尼系数，用于fisher矩阵向量积的近似计算
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.max_kl = max_kl
        self.cg_iters = cg_iters
        self.damping = damping
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = ActorNetwork(feature_dim, action_dim).to(self.device)
        self.value_net = CriticNetwork(feature_dim).to(self.device)
        
        # TRPO的策略更新是通过共轭梯度求解，不直接使用梯度下降更新策略参数
        # 价值网络仍然采用Adam优化器
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)
        
        # 存储交互数据
        self.states = []
        self.actions = []
        self.raw_actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def choose_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mu, sigma = self.policy_net(state_tensor)
            dist = Normal(mu, sigma)
            raw_action = dist.rsample()  # 使用rsample以便后续计算梯度
            action = torch.tanh(raw_action)
            log_prob = (dist.log_prob(raw_action).sum(dim=-1) -
                        torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1))
            value = self.value_net(state_tensor)
            action = action.squeeze().cpu().numpy()
        return action, raw_action, log_prob.item(), value.item()

    def store_transition(self, state, action, raw_action, log_prob, value, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.raw_actions.append(raw_action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def learn(self, done, step_counter):
        # 仅在达到一定步数后更新
        if step_counter % 100 != 0:
            return

        # 1. 数据准备
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        raw_actions = torch.cat(self.raw_actions, dim=0).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        values = np.array(self.values)

        # 计算GAE优势和回报
        returns, advantages = self._compute_gae(values, rewards, dones)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = torch.FloatTensor(advantages).to(self.device)

        # 2. 计算旧策略输出（用于KL计算）
        with torch.no_grad():
            old_mu, old_sigma = self.policy_net(states)
            old_mu = old_mu.detach()
            old_sigma = old_sigma.detach()

        # 3. 保存当前策略网络参数
        old_params = self._get_flat_params_from(self.policy_net)

        # 4. 定义代理目标函数（surrogate loss），TRPO中目标为：
        #    L(θ) = E[ (πθ(a|s) / πθ_old(a|s)) * A ]
        def surrogate_loss_fn():
            mu, sigma = self.policy_net(states)
            dist = Normal(mu, sigma)
            new_log_probs = (dist.log_prob(raw_actions).sum(dim=-1) -
                             torch.log(1 - actions.pow(2) + 1e-6).sum(dim=-1))
            ratio = torch.exp(new_log_probs - old_log_probs)
            return (ratio * advantages).mean()

        # 5. 计算策略梯度 g
        self.policy_net.zero_grad()
        surrogate_loss = surrogate_loss_fn()
        surrogate_loss.backward()
        g = self._get_flat_grad_from(self.policy_net).detach()

        # 6. 定义 Fisher 矩阵向量积函数
        def fisher_vector_product(v):
            self.policy_net.zero_grad()
            kl = self._kl_divergence(states, old_mu, old_sigma)
            kl_grad = torch.autograd.grad(kl, self.policy_net.parameters(), create_graph=True)
            flat_kl_grad = torch.cat([grad.view(-1) for grad in kl_grad])
            kl_v = (flat_kl_grad * v).sum()
            kl_v_grad = torch.autograd.grad(kl_v, self.policy_net.parameters())
            flat_kl_v_grad = torch.cat([grad.contiguous().view(-1) for grad in kl_v_grad])
            return flat_kl_v_grad + self.damping * v

        # 7. 共轭梯度求解获取步长方向
        step_direction = self.conjugate_gradient(fisher_vector_product, g, nsteps=self.cg_iters)
        step_direction = step_direction.detach()

        # 8. 计算最终步长缩放因子：令 full_step = step_direction * sqrt(2*max_kl / (d^T F d))
        fvp = fisher_vector_product(step_direction)
        step_dir_dot = (step_direction * fvp).sum(0, keepdim=True)
        lagrange_multiplier = torch.sqrt(2 * self.max_kl / (step_dir_dot + 1e-8))
        full_step = step_direction * lagrange_multiplier

        # 9. 回溯线搜索
        old_surrogate = surrogate_loss_fn().item()
        accepted = False
        for stepfrac in [1.0, 0.5, 0.25, 0.125]:
            new_params = old_params + stepfrac * full_step
            self._set_flat_params_to(self.policy_net, new_params)
            new_surrogate = surrogate_loss_fn().item()
            kl = self._kl_divergence(states, old_mu, old_sigma).item()
            if new_surrogate > old_surrogate and kl <= self.max_kl:
                accepted = True
                break
        if not accepted:
            self._set_flat_params_to(self.policy_net, old_params)

        # 10. 更新价值网络（采用均方误差，迭代若干次）
        for _ in range(10):
            value_pred = self.value_net(states).squeeze()
            value_loss = 0.5 * (value_pred - returns).pow(2).mean()
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

        # 11. 清空内存
        self._clear_memory()

    def _compute_gae(self, values, rewards, dones):
        gae = 0
        returns = []
        advantages = []
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0  # 终止状态的后续价值视为0
            else:
                next_value = values[t+1] * (1 - dones[t])
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        return np.array(returns), np.array(advantages)

    def _clear_memory(self):
        self.states = []
        self.actions = []
        self.raw_actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def _get_flat_params_from(self, model):
        return torch.cat([param.data.view(-1) for param in model.parameters()])

    def _set_flat_params_to(self, model, flat_params):
        offset = 0
        for param in model.parameters():
            numel = param.numel()
            param.data.copy_(flat_params[offset:offset+numel].view_as(param))
            offset += numel

    def _get_flat_grad_from(self, model):
        grads = []
        for param in model.parameters():
            if param.grad is not None:
                grads.append(param.grad.data.view(-1))
            else:
                grads.append(torch.zeros_like(param.data.view(-1)))
        return torch.cat(grads)

    def conjugate_gradient(self, f_Ax, b, nsteps, residual_tol=1e-10):
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        for i in range(nsteps):
            Ap = f_Ax(p)
            alpha = rdotr / (torch.dot(p, Ap) + 1e-8)
            x += alpha * p
            r -= alpha * Ap
            new_rdotr = torch.dot(r, r)
            if new_rdotr < residual_tol:
                break
            beta = new_rdotr / (rdotr + 1e-8)
            p = r + beta * p
            rdotr = new_rdotr
        return x

    def _kl_divergence(self, states, old_mu, old_sigma):
        mu, sigma = self.policy_net(states)
        old_dist = Normal(old_mu, old_sigma)
        new_dist = Normal(mu, sigma)
        kl = torch.distributions.kl.kl_divergence(old_dist, new_dist)
        return kl.mean()