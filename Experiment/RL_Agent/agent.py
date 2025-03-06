import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        
        # Increased network capacity
        self.hidden1 = nn.Linear(state_dim, 256)
        self.hidden2 = nn.Linear(256, 256)
        self.hidden3 = nn.Linear(256, 128)
        self.advantage = nn.Linear(128, action_dim)
        self.value = nn.Linear(128, 1)
        
        # Initialize weights with Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.hidden1.weight)
        nn.init.xavier_uniform_(self.hidden2.weight)
        nn.init.xavier_uniform_(self.hidden3.weight)
        nn.init.xavier_uniform_(self.advantage.weight)
        nn.init.xavier_uniform_(self.value.weight)
        
        # Replace batch normalization with layer normalization
        self.ln1 = nn.LayerNorm(256)
        self.ln2 = nn.LayerNorm(256)
        self.ln3 = nn.LayerNorm(128)
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Handle single sample input during inference
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        # Apply layer normalization instead of batch normalization
        x = F.relu(self.ln1(self.hidden1(x)))
        x = self.dropout(x)
        x = F.relu(self.ln2(self.hidden2(x)))
        x = self.dropout(x)
        x = F.relu(self.ln3(self.hidden3(x)))
        
        advantage = self.advantage(x)
        value = self.value(x)
        
        # Dueling DQN architecture
        return value + advantage - advantage.mean(dim=1, keepdim=True)

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling weight
        self.beta_increment = 0.001  # Gradually increase beta to 1
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.buffer = []
        self.position = 0
        self.eps = 1e-5  # Small constant to prevent zero priority

    def add(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return None
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Calculate sampling probabilities
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities = probabilities / probabilities.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-self.beta)
        weights = weights / weights.max()
        
        samples = [self.buffer[idx] for idx in indices]
        batch = list(zip(*samples))
        
        return (
            torch.FloatTensor(np.array(batch[0])),  # states
            torch.LongTensor(batch[1]),             # actions
            torch.FloatTensor(batch[2]),            # rewards
            torch.FloatTensor(np.array(batch[3])),  # next_states
            torch.FloatTensor(batch[4]),            # dones
            torch.FloatTensor(weights),             # importance sampling weights
            indices
        )

    def update_priorities(self, indices, priorities):
        priorities = priorities.detach().cpu().numpy()
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.eps

class ImprovedDQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.0003, gamma=0.99, initial_epsilon=0.5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = initial_epsilon
        self.tau = 0.005  # Soft update parameter
        
        # Initialize networks
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Use Adam optimizer with gradient clipping
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(), 
            lr=lr, 
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            factor=0.5, 
            patience=1000, 
            verbose=True
        )
        
        # Keep track of updates for target network sync
        self.update_count = 0
        self.sync_every = 100  # Sync target network every N updates
        
        # Store last N states for reward normalization
        self.reward_history = deque(maxlen=10000)
        self.reward_mean = 0
        self.reward_std = 1

    def update_reward_stats(self, reward):
        """Update running statistics for reward normalization"""
        self.reward_history.append(reward)
        if len(self.reward_history) > 1:
            self.reward_mean = np.mean(self.reward_history)
            self.reward_std = np.std(self.reward_history) + 1e-5

    def normalize_reward(self, reward):
        """Normalize reward using running statistics"""
        return (reward - self.reward_mean) / self.reward_std

    def select_action(self, state, testing=False):
        # Convert state to tensor
        state = torch.FloatTensor(state)
        
        # Use lower epsilon during testing
        if testing:
            eps_threshold = 0.05
        else:
            eps_threshold = self.epsilon
        
        if random.random() > eps_threshold:
            with torch.no_grad():
                return self.q_network(state).argmax().item()
        else:
            return random.randrange(self.action_dim)

    def update(self, buffer, batch_size):
        # Sample from replay buffer
        batch = buffer.sample(batch_size)
        if batch is None:
            return
        
        states, actions, rewards, next_states, dones, weights, indices = batch
        
        # Normalize rewards
        for reward in rewards:
            self.update_reward_stats(reward.item())
        normalized_rewards = torch.tensor([self.normalize_reward(r.item()) for r in rewards])
        
        # Calculate current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Calculate next Q values with target network (double Q-learning)
        with torch.no_grad():
            # Get actions from policy network
            next_actions = self.q_network(next_states).argmax(1).unsqueeze(1)
            # Get Q-values from target network
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            
        # Calculate expected Q values
        expected_q_values = normalized_rewards.unsqueeze(1) + \
                          (1 - dones.unsqueeze(1)) * self.gamma * next_q_values
        
        # Calculate loss with importance sampling weights
        td_errors = torch.abs(current_q_values - expected_q_values)
        loss = (td_errors * weights.unsqueeze(1)).mean()
        
        # Update priorities in buffer
        buffer.update_priorities(indices, td_errors.squeeze())
        
        # Optimize the policy network
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # Update target network
        self.update_count += 1
        if self.update_count % self.sync_every == 0:
            self.soft_update()
        
        return loss.item()
    
    def soft_update(self):
        """Soft update of target network parameters"""
        for target_param, policy_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(
                self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
            )