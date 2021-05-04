import gym
import numpy as np

env = gym.make("CartPole-v1")
observation = env.reset()

#actions = np.array([1,-1])
actions = []

print('\n')

for i in range(3):
  env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)

  observation, reward, done, info = env.step(action)

  if(action == 0):
    action = -1
  actions.append(np.array([action*0.5]))

  print(observation)

  if done:
    observation = env.reset()
env.close()

print('\n')

env_name = 'CartPoleContinuous-v0'
env = gym.make(env_name)
observation = env.reset()


for action in actions:
  env.render()

  observation, reward, done, info = env.step(action)

  print(observation)

  if done:
    observation = env.reset()
env.close()
