import numpy as np
import itertools as it
import random
from skimage import color, transform
from collections import namedtuple, deque

import tensorflow.contrib.slim as slim
import tensorflow as tf

import sys

# from model import DQN
from experience_history import History

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.optimizers import Adam

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
    self.initial_epsilon = 1
    self.min_epsilon = 0.1
    self.epsilon_decay_steps = int(1e5)
    self.learning_rate = 4e-4
    self.tau = 0

    # Flags
    self.network_update_frequency = int(1e3)
    self.train_freq = 4
    self.frame_skip = 3
    self.min_experience_size = int(1e4)
    
    # Enviroment
    self.render = True 
    
    # Possible Actions and their corresponding weights
    left_right = [-1, 0, 1]
    acceleration = [1, 0]
    brake = [0.2, 0]
    all_actions = np.array([action for action in it.product(left_right, acceleration, brake)])

    # defined_actions = [[-1, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0.3]]
    # all_actions = np.array(defined_actions)

    self.action_map = all_actions
    self.num_actions = len(self.action_map)
    gas_actions = np.array([a[1] == 1 and a[2] == 0 for a in self.action_map])
    # Increase the weight of gas actions for the car.
    self.action_weights = 14.0 * gas_actions + 1.0
    self.action_weights /= np.sum(self.action_weights)
    print('Action Map -> ', self.action_map)
    
    # Model (Neural Network)
    # self.training_model = DQN(self.num_actions)
    # self.target_model = DQN(self.num_actions)

    self.model = self._get_model()
    self.target_model = self._get_model()

    # # Load models to GPU
    # if torch.cuda.is_available():
    #   self.training_model.cuda()
    #   self.target_model.cuda()
    
    # self.optimizer = optim.Adam(self.training_model.parameters(), lr = self.learning_rate)

    print('---------- Model ---------')
    print(self.model.summary())
    
    # Negative Reward
    # To check if we want to end the episode earlier
    self.neg_reward_counter = 0
    self.max_neg_rewards = 12
    
    # History
    self.experience_capacity = int(4e4)
    self.memory = History(self.num_frame_stack, self.experience_capacity)
    self.network_chosen_action = 0
  
  def _get_model(self):
    model = Sequential()
    model.add(Conv2D(filters = 8, kernel_size = 7, activation='relu', 
      input_shape=(self.num_frame_stack,) + self.image_size, data_format='channels_first'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2), dim_ordering="th"))
    model.add(Conv2D(filters = 16, kernel_size = 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2), dim_ordering="th"))
    model.add(Flatten())
    # model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(self.num_actions))
    model.compile(optimizer=Adam(lr=self.learning_rate), loss = 'mse')
    return model

  def step(self, state, action, reward, done):
    # Save experience in replay memory
    self.memory.add(state, action, reward, done)
    
    # Learn every UPDATE_EVERY time steps.
    if self.global_counter % self.train_freq == 0:
      # If enough samples are available in memory, get random subset and learn
      if self.memory.counter > self.min_experience_size:
        experiences = self.memory.sample(self.batch_size)
        self.learn(experiences)

  def get_action(self):
    # Epsilon-greedy action selection

    if random.random() > self.get_epsilon():
      # self.network_chosen_action += 1
      # # Add the batch dimension before creating a tensor
      state = self.memory.current_state()[np.newaxis, ...]
      # state = torch.from_numpy(state).float().to(device)
      # self.training_model.eval()
      # with torch.no_grad():
      #   action_values = self.training_model(state)
        
      # self.training_model.train()
      # action = np.argmax(action_values.cpu().data.numpy())
      act_values = self.model.predict(state)
      action = np.argmax(act_values[0])
      self.network_chosen_action += 1
      return action
      # return action
    else:
      return self.get_random_action()

  def get_action_for_state(self, state):
    act_values = self.model.predict(state[np.newaxis, ...])
    action = np.argmax(act_values[0])
    return action

  def learn(self, experiences):
    states, actions, rewards, next_states, dones = experiences
    Q_target_next = np.amax(self.target_model.predict(next_states), axis = 1)
    Q_target = rewards + self.gamma * Q_target_next * (1 - dones)
    Q_expected = self.model.predict(states)
    for i, a_idx in enumerate(actions):
      Q_expected[i, a_idx] = Q_target[i]
    self.model.fit(states, Q_expected, epochs=1, verbose=0)

    # if not done:
    #   # predict the future discounted reward
    # target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
    #   # make the agent to approximately map
    #   # the current state to future discounted reward
    #   # We'll call that target_f
    #   target_f = self.model.predict(state)
    #   target_f[0][action] = target
    #   # Train the Neural Net with the state and target_f
    

    # ------------------- update target network ------------------- #
    if self.global_counter % self.network_update_frequency == 0:
      self.soft_update()
                        

  def soft_update(self):
    weights = self.model.get_weights()
    target_weights = self.target_model.get_weights()
    for i in range(len(target_weights)):
      target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
    self.target_model.set_weights(target_weights)
          
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

    # self.soft_update(self.training_model, self.target_model) 
    
    state = self.env.reset()
    state = self.process_image(state)
    self.memory.start_new_episode(state)

    while True:
      self.global_counter += 1
      frames_in_episode += 1
      action_idx = self.get_action()
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
      self.step(next_state, action_idx, reward, done)

      total_reward += reward

      if done:
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

        
        