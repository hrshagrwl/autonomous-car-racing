from collections import namedtuple, deque
import random
import numpy as np
import torch


class History:
  def __init__(self, num_frame_stack = 3, capacity = int(1e5), pic_size=(96, 96)):
    self.num_frame_stack = num_frame_stack
    self.capacity = capacity
    self.memory = deque(maxlen = capacity)
    self.pic_size = pic_size
    self.counter = 0
    self.frame_window = None
    self.transition = namedtuple('transition', ('state', 'action', 'reward', 'next_state', 'done'))
    self.max_frame_cache = self.capacity + 2 * self.num_frame_stack + 1
    self.frames = -np.ones((self.max_frame_cache,) + self.pic_size, dtype="float32")
    self.expecting_new_episode = True

  def add(self, frame, action, reward, done):
    assert self.frame_window is not None, "start episode first"

    self.counter += 1
    frame_idx = self.counter % self.max_frame_cache
    self.frames[frame_idx] = frame
    prev_states = self.frame_window
    self.frame_window = np.append(self.frame_window[1:], frame_idx)
    next_states = self.frame_window
    t = self.transition(prev_states, action, reward, next_states, done)
    self.memory.append(t)
    if done:
      self.expecting_new_episode = True

  def start_new_episode(self, frame):
    assert self.expecting_new_episode, "previous episode didn't end yet"

    frame_idx = self.counter % self.max_frame_cache
    self.frame_window = np.repeat(frame_idx, self.num_frame_stack)
    self.frames[frame_idx] = frame
    self.expecting_new_episode = False

  def sample(self, n):
    transitions = random.sample(self.memory, n)

    states = np.vstack([[self.frames[e.state] for e in transitions if e is not None]])
    actions = np.array([e.action for e in transitions if e is not None])
    rewards = np.array([e.reward for e in transitions if e is not None])
    next_states = np.vstack([[self.frames[e.next_state] for e in transitions if e is not None]])
    dones = np.array([e.done for e in transitions if e is not None])

    return states, actions, rewards, next_states, dones

  def current_state(self):
    # assert not self.expecting_new_episode, "start new episode first"'
    assert self.frame_window is not None, "do something first"
    return self.frames[self.frame_window]