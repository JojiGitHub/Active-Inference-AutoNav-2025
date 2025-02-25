import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random

class EnhancedDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(EnhancedDQN, self).__init__()
        
        # Very simple network since we want direct state-action mapping
        self.feature_net = nn.Sequential(
            nn.Linear(17, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, action_dim)  # Direct mapping to actions
        )
        
    def forward(self, state):
        return self.feature_net(state)

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        
    def add(self, state, action, reward, next_state, done):
        max_priority = np.max(self.priorities) if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return None
            
        N = len(self.buffer)
        if N < self.capacity:
            priorities = self.priorities[:N]
        else:
            priorities = self.priorities
            
        # Calculate sampling probabilities
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices and calculate importance weights
        indices = np.random.choice(N, batch_size, p=probs)
        weights = (N * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*samples)
        
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones), indices, weights)
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

class ImprovedDQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.q_network = EnhancedDQN(state_dim, action_dim).to(self.device)
        self.target_network = EnhancedDQN(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        
        # More aggressive exploration strategy
        self.epsilon = 1.0
        self.epsilon_min = 0.05  # Lower minimum epsilon
        self.epsilon_decay = 0.97  # Much faster decay
        self.gamma = 0.9
        
        self.sync_frequency = 10
        self.steps = 0
        self.action_dim = action_dim
        
    def select_action(self, state, testing=False):
        # During testing, use the saved epsilon without modification
        if testing:
            if random.random() < self.epsilon:
                return random.randint(0, self.action_dim - 1)
        # During training, use aggressive exploration early, then exploit
        else:
            if self.steps < 1000 and random.random() < self.epsilon:  # More exploration early
                return random.randint(0, self.action_dim - 1)
            elif random.random() < self.epsilon * 0.5:  # Reduced random actions later
                return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def update(self, replay_buffer, batch_size):
        batch = replay_buffer.sample(batch_size)
        if batch is None:
            return
            
        states, actions, rewards, next_states, dones, indices, weights = batch
        
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # More aggressive Q-learning
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Huber loss for more stable learning
        loss = torch.nn.functional.smooth_l1_loss(current_q_values, target_q_values)
        
        # Update priorities with larger minimum value
        td_errors = abs(current_q_values - target_q_values).detach().cpu().numpy()
        replay_buffer.update_priorities(indices, td_errors + 1e-4)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 0.5)  # Reduced gradient clipping
        self.optimizer.step()
        
        # More frequent target updates
        self.steps += 1
        if self.steps % self.sync_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)