import torch
import torch.nn as nn
import numpy as np
import random
from easyRL.algorithms.dqn import DQN

class DDQN(DQN):
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
                 replace_target_iter=300):
        super().__init__(
            feature_dim, action_dim, learning_rate, gamma, 
            epsilon_start, epsilon_end, epsilon_decay, 
            memory_size, batch_size, replace_target_iter
        )

    def learn(self, done):
        if len(self.memory) < self.batch_size:
            return
        
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_params()
        
        # sample batch from memory
        batch_memory = np.array(random.sample(self.memory, self.batch_size))
        
        batch_state = torch.FloatTensor(batch_memory[:, :self.feature_dim]).to(self.device)
        batch_action = torch.LongTensor(batch_memory[:, self.feature_dim].astype(int)).unsqueeze(1).to(self.device)
        batch_reward = torch.FloatTensor(batch_memory[:, self.feature_dim+1]).to(self.device)
        batch_next_state = torch.FloatTensor(batch_memory[:, self.feature_dim+2:-1]).to(self.device)
        batch_done = torch.BoolTensor(batch_memory[:, -1].astype(bool))

        q_eval = self.eval_net(batch_state).gather(1, batch_action).squeeze()
        
        q_eval_next = self.eval_net(batch_next_state).detach()
        eval_act_next = q_eval_next.max(dim=1)[1].unsqueeze(1)
        
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + ((~batch_done) * self.gamma * q_next.gather(1, eval_act_next)).squeeze()

        # train eval network
        loss = self.loss_function(q_target, q_eval)
        self.optimizer.zero_grad()
        loss.backward()
        # Add gradient clipping
        torch.nn.utils.clip_grad_value_(self.eval_net.parameters(), 1)
        self.optimizer.step()

        self.learn_step_counter += 1
        self.decay_epsilon()