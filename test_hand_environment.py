import myosuite
import gym
import numpy as np

env = gym.make('myoHandPoseRandom-v0')
for _ in range(30):
  env.reset()
  ep_rewards = []
  done = False
  obs = env.reset()
  while not done:
      o = env.get_obs()
      # get the next action randomly
      action = env.action_space.sample()
      # take an action based on the current observation
      obs, reward, done, info = env.step(action)
      ep_rewards.append(reward)
      env.mj_render()
  print(np.sum(ep_rewards))
env.close()