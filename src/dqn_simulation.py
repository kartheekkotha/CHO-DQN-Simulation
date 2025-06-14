import random
import matplotlib.pyplot as plt
import math
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
import os

# Ensure current directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# --- DQN Agent Implementation ---

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=32):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)  # Add another layer
        self.out = nn.Linear(hidden_size, action_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size=64, 
                 lr=1e-3, gamma=0.95, buffer_size=1000, batch_size=64, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=5000,
                 learning_starts=50, target_update_interval=100, max_grad_norm=10):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = DQN(state_size, action_size, hidden_size).to(self.device)
        self.target_net = DQN(state_size, action_size, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(capacity=buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        
        # Epsilon-greedy parameters
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

        # Training controls
        self.learning_starts = learning_starts
        self.target_update_interval = target_update_interval
        self.max_grad_norm = max_grad_norm
    
    def select_action(self, state):
        self.steps_done += 1
        epsilon = self.epsilon_min + \
                  (self.epsilon - self.epsilon_min) * \
                  np.exp(-1. * self.steps_done / self.epsilon_decay)

        if random.random() < epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state)
                return q_values.max(1)[1].item()
    
    def push_memory(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def update(self):
        # Delay learning until sufficient samples
        if self.steps_done < self.learning_starts or len(self.memory) < self.batch_size:
            return
        
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        
        state      = torch.FloatTensor(state).to(self.device)
        action     = torch.LongTensor(action).unsqueeze(1).to(self.device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done       = torch.FloatTensor(done).unsqueeze(1).to(self.device)
        
        # Current Q values
        q_values = self.policy_net(state).gather(1, action)
        
        # Double DQN: action selection by policy net, evaluation by target net
        next_actions = self.policy_net(next_state).argmax(1, keepdim=True)
        next_q_values = self.target_net(next_state).gather(1, next_actions)
        
        # Compute target Q values
        expected_q_values = reward + (1 - done) * self.gamma * next_q_values
        
        # Use Huber loss for stability
        loss = F.smooth_l1_loss(q_values, expected_q_values)
        
        # Gradient descent step
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Periodically update target network
        if self.steps_done % self.target_update_interval == 0:
            self.update_target_network()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)
