from collections import namedtuple, deque
import random
import numpy as np
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class History:
  # def __init__(self, capacity, batch_size, seed):
  #   self.capacity = capacity
  #   self.memory = deque(maxlen = capacity)
  #   self.transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
  #   self.batch_size = batch_size
  #   self.frames = []

  # def add(self, state, action, reward, next_state, done):
  #   """Saves a transition."""
  #   t = self.transition(state, action, reward, next_state, done)
  #   self.memory.append(t)

  # def sample(self):
  #   transitions = random.sample(self.memory, self.batch_size)
    
  #   states = torch.from_numpy(np.vstack([[e.state for e in transitions if e is not None]])).float().to(device)
  #   actions = torch.from_numpy(np.vstack([e.action for e in transitions if e is not None])).long().to(device)
  #   rewards = torch.from_numpy(np.vstack([e.reward for e in transitions if e is not None])).float().to(device)
  #   next_states = torch.from_numpy(np.vstack([[e.next_state for e in transitions if e is not None]])).float().to(device)
  #   dones = torch.from_numpy(np.vstack([e.done for e in transitions if e is not None]).astype(np.uint8)).float().to(device)
    
  #   return (states, actions, rewards, next_states, dones)

  # def __len__(self):
  #   return len(self.memory)

  def __init__(self,
            num_frame_stack=4,
            capacity=int(1e5),
            pic_size=(96, 96)
    ):
    self.num_frame_stack = num_frame_stack
    self.capacity = capacity
    self.pic_size = pic_size
    self.counter = 0
    self.frame_window = None
    self.init_caches()
    self.expecting_new_episode = True

  def add(self, frame, action, reward, done):
    assert self.frame_window is not None, "start episode first"
    self.counter += 1
    frame_idx = self.counter % self.max_frame_cache
    exp_idx = (self.counter - 1) % self.capacity

    self.prev_states[exp_idx] = self.frame_window
    self.frame_window = np.append(self.frame_window[1:], frame_idx)
    self.next_states[exp_idx] = self.frame_window
    self.actions[exp_idx] = action
    self.is_done[exp_idx] = done
    self.frames[frame_idx] = frame
    self.rewards[exp_idx] = reward
    if done:
      self.expecting_new_episode = True

  def start_new_episode(self, frame):
    # it should be okay not to increment counter here
    # because episode ending frames are not used
    assert self.expecting_new_episode, "previous episode didn't end yet"
    frame_idx = self.counter % self.max_frame_cache
    self.frame_window = np.repeat(frame_idx, self.num_frame_stack)
    self.frames[frame_idx] = frame
    self.expecting_new_episode = False

  def sample(self, n):
    count = min(self.capacity, self.counter)
    batchidx = np.random.randint(count, size=n)

    prev_frames = self.frames[self.prev_states[batchidx]]
    next_frames = self.frames[self.next_states[batchidx]]

    states = torch.from_numpy(np.vstack([[e for e in prev_frames if e is not None]])).float().to(device)
    actions = torch.from_numpy(np.vstack([e for e in self.actions[batchidx] if e is not None])).long().to(device)
    rewards = torch.from_numpy(np.vstack([e for e in self.rewards[batchidx] if e is not None])).float().to(device)
    next_states = torch.from_numpy(np.vstack([[e for e in next_frames if e is not None]])).float().to(device)
    dones = torch.from_numpy(np.vstack([e for e in self.is_done[batchidx] if e is not None]).astype(np.uint8)).float().to(device)

    return states, actions, rewards, next_states, dones

  def current_state(self):
    # assert not self.expecting_new_episode, "start new episode first"'
    assert self.frame_window is not None, "do something first"
    return self.frames[self.frame_window]

  def init_caches(self):
    self.rewards = np.zeros(self.capacity, dtype="float32")
    self.prev_states = -np.ones((self.capacity, self.num_frame_stack),
        dtype="int32")
    self.next_states = -np.ones((self.capacity, self.num_frame_stack),
        dtype="int32")
    self.is_done = -np.ones(self.capacity, "int32")
    self.actions = -np.ones(self.capacity, dtype="int32")

    # lazy to think how big is the smallest possible number. At least this is big enough
    self.max_frame_cache = self.capacity + 2 * self.num_frame_stack + 1
    self.frames = -np.ones((self.max_frame_cache,) + self.pic_size, dtype="float32")