import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

class EnhancedDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(EnhancedDQN, self).__init__()
        
        # Improved architecture with separate vision and coordinate processing paths
        
        # Vision processing - handle the 5x5 grid (25 cells)
        self.vision_encoder = nn.Sequential(
            nn.Linear(25, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(0.3),  # Increased dropout (was 0.2)
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.3)   # Added dropout here too
        )
        
        # Coordinate processing - handle agent and goal coordinates (4 values)
        self.coordinate_encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Dropout(0.2),  # Added dropout
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(0.2)   # Added dropout
        )
        
        # Combined processing for decision making
        self.combined_net = nn.Sequential(
            nn.Linear(128 + 64, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.4),  # Increased dropout (was 0.2)
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.3),  # Added more dropout
            nn.Linear(128, action_dim)
        )
        
        # Initialize weights with proper scaling
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, state):
        # Split state into vision (25 cells) and coordinates (4 values)
        vision = state[:, :25]    # First 25 elements are 5x5 vision grid
        coords = state[:, 25:]    # Last 4 elements are coordinates
        
        # Process vision and coordinates separately
        vision_features = self.vision_encoder(vision)
        coord_features = self.coordinate_encoder(coords)
        
        # Combine features
        combined = torch.cat([vision_features, coord_features], dim=1)
        q_values = self.combined_net(combined)
        
        return q_values


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment  # Gradually increase beta to 1
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
        
        # Gradually increase beta to reduce importance sampling bias
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*samples)
        
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones), indices, weights)
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-5  # Small constant for stability


class ImprovedDQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.0003, initial_epsilon=1.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.q_network = EnhancedDQN(state_dim, action_dim).to(self.device)
        self.target_network = EnhancedDQN(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2000, gamma=0.9)
        
        # More stable exploration strategy
        self.epsilon = initial_epsilon
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.999
        self.gamma = 0.99
        
        self.sync_frequency = 5  # Sync target network every 5 updates
        self.steps = 0
        self.action_dim = action_dim
        self.training_steps = 0
        
        # Store recent rewards for adaptive exploration
        self.recent_rewards = deque(maxlen=100)
        
    def select_action(self, state, testing=False):
        """Select action with balanced exploration and exploitation"""
        if testing:
            epsilon = 0.05  # Low exploration during testing
        else:
            epsilon = self.epsilon
            
        # Pure exploration phase
        if not testing and self.steps < 3000:
            # Start with very high exploration, then gradually reduce
            if random.random() < max(0.95, epsilon):
                return random.randint(0, self.action_dim - 1)
        
        # Standard epsilon-greedy with some heuristics
        if random.random() < epsilon:
            # Guided exploration: Prefer moving towards the goal if coordinates available
            if hasattr(self, 'last_state') and len(state) >= 21:
                agent_x, agent_y = state[17], state[18]
                goal_x, goal_y = state[19], state[20]
                
                # 70% chance of choosing a "good" direction during exploration
                if random.random() < 0.7:
                    if goal_x > agent_x and random.random() < 0.5:
                        return 3  # RIGHT
                    elif goal_x < agent_x and random.random() < 0.5:
                        return 2  # LEFT
                    elif goal_y > agent_y and random.random() < 0.5:
                        return 1  # DOWN
                    elif goal_y < agent_y and random.random() < 0.5:
                        return 0  # UP
            
            # Fall back to pure random
            return random.randint(0, self.action_dim - 1)
        
        # Save state for potential future reference
        self.last_state = state
        
        # Greedy action based on Q-values
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
        
        # Track rewards for adaptive exploration
        self.recent_rewards.extend(rewards.cpu().numpy())
        
        # Double Q-learning with target network
        with torch.no_grad():
            # Get actions that would be selected by current network
            next_actions = self.q_network(next_states).argmax(1)
            
            # Evaluate these actions using the target network
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            
            # Calculate target values (bellman equation)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Get current Q-value estimates
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Calculate loss with importance sampling weights
        loss = F.smooth_l1_loss(current_q_values, target_q_values, reduction='none')
        weighted_loss = (loss * weights).mean()
        
        # Calculate TD errors for prioritized replay
        with torch.no_grad():
            td_errors = abs(current_q_values - target_q_values).detach().cpu().numpy()
            
        # Update replay priorities
        replay_buffer.update_priorities(indices, td_errors)
        
        # Optimize the network
        self.optimizer.zero_grad()
        weighted_loss.backward()
        # Use gradient clipping to prevent instability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()
        
        # Step the learning rate scheduler
        self.training_steps += 1
        if self.training_steps % 100 == 0:
            self.scheduler.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.sync_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Adaptive epsilon decay - decay faster when rewards improve
        if len(self.recent_rewards) >= 10:
            avg_reward = np.mean(self.recent_rewards)
            
            # If recent rewards are good, start decaying epsilon faster
            if avg_reward > 0 and self.steps > 1000:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            # Otherwise, decay epsilon more slowly
            elif self.steps > 3000:
                self.epsilon = max(self.epsilon_min, self.epsilon * 0.9999)

    def save(self, path):
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'training_steps': self.training_steps,
        }, path)
        
    def load(self, path):
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.epsilon = checkpoint['epsilon'] 
        self.steps = checkpoint['steps']
        if 'training_steps' in checkpoint:
            self.training_steps = checkpoint['training_steps']