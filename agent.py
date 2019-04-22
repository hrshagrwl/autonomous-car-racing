import numpy as np
import itertools as it
import random
from skimage import color, transform
from collections import namedtuple, deque
from PIL import Image

from model import DQN
from experience_history import History

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQNAgent:
  def __init__(self, env):
      
    self.env = env
    self.global_counter = 0
    
    print('---------- Initializing -----------')
    # Training Parameters
    self.batch_size = 64
    self.num_frame_stack = 3
    self.image_size = (96, 96)
    self.gamma = 0.95 
    self.initial_epsilon = 1.0
    self.min_epsilon = 0.1
    self.epsilon_decay_steps = int(1e5)
    self.learning_rate = 4e-4
    self.tau = 1e-3

    # Flags
    self.network_update_frequency = int(1e3)
    self.train_freq = 4
    self.frame_skip = 3
    self.min_experience_size = 64
    
    # Enviroment
    self.render = True
    self.seed = 7    # Seed to random 
    
    # Possible Actions and their corresponding weights
    left_right = [-1, 0, 1]
    acceleration = [1, 0]
    brake = [0.3, 0]
    all_actions = np.array([action for action in it.product(left_right, acceleration, brake)])

    # defined_actions = [[-1, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0.3]]
    # all_actions = np.array(defined_actions)

    self.action_map = all_actions
    self.num_actions = len(self.action_map)
    gas_actions = np.array([a[1] == 1 and a[2] == 0 for a in self.action_map])
    # Increase the weight of gas actions for the car.
    self.action_weights = 10.0 * gas_actions + 1.0
    self.action_weights /= np.sum(self.action_weights)

    print('Action Map -> ', len(self.action_map))
    
    # Model (Neural Network)
    self.training_model = DQN(self.num_actions)
    self.target_model = DQN(self.num_actions)

    # Load models to GPU
    if torch.cuda.is_available():
      self.training_model.cuda()
      self.target_model.cuda()
    
    self.optimizer = optim.Adam(self.training_model.parameters(), lr = self.learning_rate)

    print('---------- Model ---------')
    print(self.training_model)
    
    # Negative Reward
    # To check if we want to end the episode earlier
    self.neg_reward_counter = 0
    self.max_neg_rewards = 12
    
    # History
    self.experience_capacity = int(4e4)
    self.memory = History(self.num_frame_stack, self.experience_capacity)
    self.t_step = 0
    self.network_chosen_action = 0
  
  def step(self, state, action, reward, done):
    # Save experience in replay memory
    self.memory.add(state, action, reward, done)
    
    # Learn every UPDATE_EVERY time steps.
    self.t_step = (self.t_step + 1) % self.train_freq
    if self.t_step == 0:
      # If enough samples are available in memory, get random subset and learn
      if self.memory.counter > self.min_experience_size:
        experiences = self.memory.sample(self.batch_size)
        self.learn(experiences)

  def get_action(self, state):
    """Returns actions for given state as per current policy.
    
    Params
    ======
        state (array_like): current state
        eps (float): epsilon, for epsilon-greedy action selection
    """
    # Epsilon-greedy action selection
    if random.random() > self.get_epsilon():
      self.network_chosen_action += 1
      # Add the batch dimension before creating a tensor
      state = state[np.newaxis, np.newaxis, ...]
      state = torch.from_numpy(state).float().to(device)
      self.training_model.eval()
      with torch.no_grad():
        action_values = self.training_model(state)
        
      self.training_model.train()
      action = np.argmax(action_values.cpu().data.numpy())
      return action
    else:
      return self.get_random_action()

  def learn(self, experiences):
    """Update value parameters using given batch of experience tuples.

    Params
    ======
        experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
        gamma (float): discount factor
    """
    states, actions, rewards, next_states, dones = experiences
    
    # Get max predicted Q values (for next states) from target model
    Q_targets_next = self.target_model(next_states).detach().max(1)[0].unsqueeze(1)
    # Compute Q targets for current states 
    Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
    # print(Q_targets)
    # Get expected Q values from local model
    Q_expected = self.training_model(states).gather(1, actions)
    # Compute loss
    # print(Q_expected)
    loss = F.mse_loss(Q_expected, Q_targets)

    # print(loss.item())

    # Minimize the loss
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    # ------------------- update target network ------------------- #
    if self.global_counter % self.network_update_frequency == 0:
      self.soft_update(self.training_model, self.target_model)                     

  def soft_update(self, local_model, target_model):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target

    Params
    ======
        local_model (PyTorch model): weights will be copied from
        target_model (PyTorch model): weights will be copied to
        tau (float): interpolation parameter 
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
      # self.tau * local_param.data + (1.0 - self.tau) * target_param.data
      target_param.data.copy_(local_param.data)
          
  def get_epsilon(self):
    if self.global_counter >= self.epsilon_decay_steps:
      return self.min_epsilon
    else:
      # linear decay
      r = 1.0 - self.global_counter / float(self.epsilon_decay_steps)
      return self.min_epsilon + (self.initial_epsilon - self.min_epsilon) * r
          
  def play_episode(self):
    total_reward = 0
    frames_in_episode = 0
    
    state = self.env.reset()
    state = self.process_image(state)
    self.memory.start_new_episode(state)

    while True:
      self.global_counter += 1
      frames_in_episode += 1
      action_idx = self.get_action(state)
      action = self.action_map[action_idx]
      reward = 0
      if self.render:
        self.env.render()

      for _ in range(self.frame_skip):
        next_state, r, done, _ = self.env.step(action)
        reward += r
        if self.render:
          self.env.render()
        if done:
          break
  
      early_done, punishment = self.check_early_stop(reward, total_reward)

      if early_done:
        reward += punishment

      done = early_done or done

      next_state = self.process_image(next_state)
      self.step(state, action_idx, reward, done)

      state = next_state
      total_reward += reward

      if early_done or done:
        break
  
    return total_reward, frames_in_episode
    
  
  # Convert RGB Image to grayscale and channel dimension
  def process_image(self, img):
    i = 2 * color.rgb2gray(img) - 1
    # return i[np.newaxis, ...]
    return i
    # i = np.swapaxes(img, 0, 2)
    # return (i - 128) / 128.0

  # Returns a random action.
  def get_random_action(self):
    return np.random.choice(self.num_actions, p=self.action_weights)
  
  def check_early_stop(self, reward, total_reward):
    if reward < 0:
      self.neg_reward_counter += 1
      done = (self.neg_reward_counter > self.max_neg_rewards)
      
      if done and total_reward <= 500:
        punishment = -20.0
      else:
        punishment = 0.0
          
      if done:
        self.neg_reward_counter = 0

      return done, punishment
    else:
      self.neg_reward_counter = 0
      return False, 0.0

        
        