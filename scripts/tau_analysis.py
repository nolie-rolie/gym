import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

env_name = 'Pendulum-v0'
env = gym.make(env_name)
eval_env = gym.make(env_name)


sDim = env.observation_space.shape[0] # number of states

aDim = env.action_space.shape[0] # number of actions
aHigh = env.action_space.high[0] # save for later
aLow = env.action_space.low[0]

time_step = env.reset()

def get_actor():
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(sDim,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)

    # Our upper bound is 2.0 for Pendulum.
    outputs = outputs * aHigh
    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic():
    # State as input
    state_input = layers.Input(shape=(sDim))
    state_out = layers.Dense(16, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(state_out)

    # Action as input
    action_input = layers.Input(shape=(aDim))
    action_out = layers.Dense(32, activation="relu")(action_input)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model

@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))

sitting_actor = get_actor()
sitting_actor.load_weights('./pendulum_actor.h5')
sitting_critic = get_critic()
sitting_critic.load_weights('./pendulum_critic.h5')

target_actor = get_actor()
target_critic = get_critic()

steps = 10000
tau = 0.5

actor_diff_list = []
critic_diff_list = []

for step in range(steps):
    update_target(target_actor.variables, sitting_actor.variables, tau)
    update_target(target_critic.variables, sitting_critic.variables, tau)
    state = np.ones((1,sDim))
    rand_state = np.random.randn(sDim)
    for i in range(sDim):
        state[(0,i)] = state[(0,i)]*rand_state[i]
    state = tf.convert_to_tensor(state)

    action = np.ones((1,aDim))
    rand_action = np.random.randn(aDim)
    for i in range(aDim):
        action[(0,i)] = action[(0,i)]*rand_action[i]
    action = tf.convert_to_tensor(action)

    actor_difference = abs(target_actor(state, training = True) -
                        sitting_actor(state))
    critic_difference = abs(target_critic([state,action], training = True) -
                        sitting_critic([state,action]))

    actor_diff_list.append(actor_difference[0])
    critic_diff_list.append(critic_difference[0])

plot1 = plt.figure(1)
plt.plot(actor_diff_list, 'r', label='Actor Difference')
plt.title("Actor: Difference of main and target output for random state and action")
plt.xlabel("Step")
plt.ylabel("Difference")
plt.savefig('actor_sitting_target.png')

plot2 = plt.figure(2)
plt.plot(critic_diff_list, 'b', label='Critic Difference')
plt.title("Critic: Difference of main and target output for random state and action")
plt.xlabel("Step")
plt.ylabel("Difference")
plt.savefig('critic_sitting_target.png')
