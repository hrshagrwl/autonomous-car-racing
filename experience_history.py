from collections import namedtuple, deque
import random
import numpy as np
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class History:
  def __init__(self, capacity, batch_size, seed):
    self.capacity = capacity
    self.memory = deque(maxlen = capacity)
    self.transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
    self.batch_size = batch_size
    self.seed = random.seed(seed)

  def add(self, state, action, reward, next_state, done):
    """Saves a transition."""
    t = self.transition(state, action, reward, next_state, done)
    self.memory.append(t)

  def sample(self):
    transitions = random.sample(self.memory, self.batch_size)
    
    states = torch.from_numpy(np.vstack([[e.state for e in transitions if e is not None]])).float().to(device)
    actions = torch.from_numpy(np.vstack([e.action for e in transitions if e is not None])).long().to(device)
    rewards = torch.from_numpy(np.vstack([e.reward for e in transitions if e is not None])).float().to(device)
    next_states = torch.from_numpy(np.vstack([[e.next_state for e in transitions if e is not None]])).float().to(device)
    dones = torch.from_numpy(np.vstack([e.done for e in transitions if e is not None]).astype(np.uint8)).float().to(device)
    
    return (states, actions, rewards, next_states, dones)

  def __len__(self):
    return len(self.memory)