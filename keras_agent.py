import sys
import numpy as np
import itertools as it
import random
from skimage import color, transform
from collections import namedtuple, deque

import tensorflow as tf
import tensorflow.contrib.slim as slim

# from model import DQN
from experience_history import History

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
    self.tau = 1
    self.regularization = 1e-6
    self.optimizer_params = dict(learning_rate=0.0004, epsilon=1e-7)

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
    self.action_map = all_actions
    self.num_actions = len(self.action_map)
    gas_actions = np.array([a[1] == 1 and a[2] == 0 for a in self.action_map])
    # Increase the weight of gas actions for the car.
    self.action_weights = 14.0 * gas_actions + 1.0
    self.action_weights /= np.sum(self.action_weights)
    print('Action Map -> ', self.action_map)
    
    # Build the model
    self.build_model()
    self.session = None

    # Negative Reward
    # To check if we want to end the episode earlier
    self.neg_reward_counter = 0
    self.max_neg_rewards = 12
    
    # History
    self.experience_capacity = int(4e4)
    self.memory = History(self.num_frame_stack, self.experience_capacity)
    self.network_chosen_action = 0
  

  def build_model(self):
    input_dim_with_batch = (self.batch_size, self.num_frame_stack) + self.image_size
    input_dim_general = (None, self.num_frame_stack) + self.image_size

    self.input_prev_state = tf.placeholder(tf.float32, input_dim_general, "prev_state")
    self.input_next_state = tf.placeholder(tf.float32, input_dim_with_batch, "next_state")
    self.input_reward = tf.placeholder(tf.float32, self.batch_size, "reward")
    self.input_actions = tf.placeholder(tf.int32, self.batch_size, "actions")
    self.input_done_mask = tf.placeholder(tf.int32, self.batch_size, "done_mask")

    # These are the state action values for all states
    # The target Q-values come from the fixed network
    with tf.variable_scope("fixed"):
      qsa_targets = self.create_network(self.input_next_state, trainable=False)

    with tf.variable_scope("train"):
      qsa_estimates = self.create_network(self.input_prev_state, trainable=True)

    self.best_action = tf.argmax(qsa_estimates, axis=1)

    not_done = tf.cast(tf.logical_not(tf.cast(self.input_done_mask, "bool")), "float32")
    q_target = tf.reduce_max(qsa_targets, -1) * self.gamma * not_done + self.input_reward
    # select the chosen action from each row
    # in numpy this is qsa_estimates[range(batchsize), self.input_actions]
    action_slice = tf.stack([tf.range(0, self.batch_size), self.input_actions], axis=1)
    q_estimates_for_input_action = tf.gather_nd(qsa_estimates, action_slice)

    training_loss = tf.nn.l2_loss(q_target - q_estimates_for_input_action) / self.batch_size

    optimizer = tf.train.AdamOptimizer(**(self.optimizer_params))

    reg_loss = tf.add_n(tf.losses.get_regularization_losses())
    self.train_op = optimizer.minimize(reg_loss + training_loss)

    train_params = self.get_variables("train")
    fixed_params = self.get_variables("fixed")

    self.copy_network_ops = [tf.assign(fixed_v, train_v) for train_v, fixed_v in zip(train_params, fixed_params)]

  def get_variables(self, scope):
    vars = [t for t in tf.global_variables()
        if "%s/" % scope in t.name and "Adam" not in t.name]
    return sorted(vars, key=lambda v: v.name)

  def create_network(self, input, trainable):
    if trainable:
      wr = slim.l2_regularizer(self.regularization)
    else:
      wr = None

    # the input is stack of black and white frames.
    # put the stack in the place of channel (last in tf)
    input_t = tf.transpose(input, [0, 2, 3, 1])
    net = slim.conv2d(input_t, 8, (7, 7), data_format="NHWC", activation_fn=tf.nn.relu, stride=3, weights_regularizer=wr, trainable=trainable)
    net = slim.max_pool2d(net, 2, 2)
    net = slim.conv2d(net, 16, (3, 3), data_format="NHWC", activation_fn=tf.nn.relu, weights_regularizer=wr, trainable=trainable)
    net = slim.max_pool2d(net, 2, 2)
    net = slim.flatten(net)
    net = slim.fully_connected(net, 256, activation_fn=tf.nn.relu, weights_regularizer=wr, trainable=trainable)
    q_state_action_values = slim.fully_connected(net, self.num_actions, activation_fn=None, weights_regularizer=wr, trainable=trainable)
    return q_state_action_values

  def step(self, state, action, reward, done):
    # Save experience in replay memory
    self.memory.add(state, action, reward, done)
    
    # Learn every UPDATE_EVERY time steps.
    if self.global_counter % self.train_freq == 0:
      # If enough samples are available in memory, get random subset and learn
      if self.memory.counter > self.min_experience_size:
        experiences = self.memory.sample(self.batch_size)
        self.learn(experiences)

  def learn(self, experiences):
    states, actions, rewards, next_states, dones = experiences
    fd = {
        self.input_reward: rewards,
        self.input_prev_state: states,
        self.input_next_state: next_states,
        self.input_actions: actions,
        self.input_done_mask: dones
    }
    self.session.run([self.train_op], fd)

  def get_action(self):
    # Epsilon-greedy action selection
    if np.random.rand() > self.get_epsilon():
      self.network_chosen_action += 1
      action_idx = self.session.run(self.best_action,
          { self.input_prev_state: self.memory.current_state()[np.newaxis, ...] })[0]
      return action_idx
    else:
      return np.random.choice(self.num_actions, p=self.action_weights)            
          
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
      self.step(state, action_idx, reward, done)

      state = next_state
      total_reward += reward

      if self.global_counter % self.network_update_frequency:
        self.session.run(self.copy_network_ops)

      if done:
        break
      
    return total_reward, frames_in_episode
    
  # Convert RGB Image to grayscale and channel dimension
  def process_image(self, img):
    return 2 * color.rgb2gray(img) - 1

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

        
        