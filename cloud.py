import os
import gym
import numpy as np
from pyvirtualdisplay import Display
import itertools as it
from keras_agent import DQNAgent

env = gym.make("CarRacing-v0")

if __name__ == '__main__':
  display = Display(visible=0, size=(1400, 900))
  display.start()
  if type(os.environ.get("DISPLAY")) is not str or len(os.environ.get("DISPLAY"))==0:
    os.system('bash ../xvfb start')
  agent = DQNAgent(env)
  agent.render = False
  scores = []
  frames_in_episodes = []
  for i in range(10100):
    score, frames = agent.play_episode()
    scores.append(score)
    frames_in_episodes.append(frames)
    print('Episode -> {} Reward -> {} Frames -> {} Playsteps -> {} Network Actions -> {}'.format(i, score, frames, agent.global_counter, agent.network_chosen_action))
    if i % 500 == 0:
      model_json = agent.model.to_json()
      with open("model.json", "w") as json_file:
          json_file.write(model_json)
      # serialize weights to HDF5
      agent.model.save_weights("model.h5")
      print("Saved model to disk")
    if i % 100 == 0:
      print('---------------------------------------')
      print('Average Score for last 100 episodes -> ', sum(scores[-100:])/100)
      print('---------------------------------------')