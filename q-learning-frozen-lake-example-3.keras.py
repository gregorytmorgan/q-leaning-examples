#!/usr/bin/python3.6
#
# https://gist.github.com/awjuliani/9024166ca08c489a60994e529484f7fe#file-q-table-learning-clean-ipynb
#
# https://gym.openai.com/envs/FrozenLake-v0
#
# SFFF       (S: starting point, safe)
# FHFH       (F: frozen surface, safe)
# FFFH       (H: hole, fall to your doom)
# HFFG       (G: goal, where the frisbee is located)
#

#import random
import numpy as np
import gym
#import matplotlib.pyplot as plt

from pathlib import Path

from keras.models import Sequential
from keras.layers import Dense # , Activation
#from keras.backend import clear_session
from keras.models import load_model

#%matplotlib inline

np.set_printoptions(precision=4, linewidth=280, suppress=True, threshold=64)

DEBUG_EPISODE_SUMMARY = 1

# train the model. Will save to disk after training
DEBUG_TRAIN = 4

# dump table/model after training
DEBUG_DUMP_TABLE = 8

# play the game. Will load model from disk if exists
DEBUG_PLAY_GAME = 16

DEBUG = DEBUG_EPISODE_SUMMARY | DEBUG_DUMP_TABLE | DEBUG_PLAY_GAME | DEBUG_TRAIN

game_to_play = 3
model_filename = "frozen-lake-model.h5"
report_increment = 100

#
# init the OpenAI environment
#

#from gym.envs.registration import register
#register(
#    id='FrozenLakeNotSlippery-v0',
#    entry_point='gym.envs.toy_text:FrozenLakeEnv',
#    kwargs={'map_name' : '8x8', 'is_slippery': False},
#    max_episode_steps=100,
#    reward_threshold=0.8196, # optimum = .8196, changing this seems have no influence
#)

env = gym.make("FrozenLake-v0")

#
# helper constants/functions
#

action_size = env.action_space.n
state_size = env.observation_space.n
qtable = np.zeros((state_size, action_size))
qtable.shape # (16, 4)

env.render()

# ['W', 'S', 'E', 'N']
Directions = [
    'Left',
    'Down',
    'Right',
    'Up'
]

"""
MAPS = {
    '4x4': [
        'SFFF',
        'FHFH',
        'FFFH',
        'HFFG'
    ],
    '8x8': [
        'SFFFFFFF',
        'FFFFFFFF',
        'FFFHFFFF',
        'FFFFFHFF',
        'FFFHFFFF',
        'FHHFFFHF',
        'FHFFHFHF',
        'FFFHFFFG'
    ]
}
"""

def action_to_state(current_state, action):
    """
    Get the expected state from the current state given an action
    """
    action_size = 4
    state_size = 16

    current_row = current_state % 4
    current_column = current_state % action_size

    lower_rbound = current_row * action_size
    upper_rbound = lower_rbound + action_size - 1

    if action == 0:     # left
        new_state = current_state - 1
        if new_state < lower_rbound or new_state > upper_rbound:
            new_state = current_state
    elif action == 1:   # down
        new_row = (current_row + 1) % state_size
        new_state = (new_row * action_size) + current_column if new_row < state_size else current_state
    elif action == 2:   # right
        new_state = current_state + 1
        if new_state < lower_rbound or new_state > upper_rbound:
            new_state = current_state
    elif action == 3:   # up
        new_row = current_row - 1
        new_state = (new_row * action_size) + current_column if new_row >= 0 else current_state
    else:
        runtimeException("Invalid action: {}".format(action))


#
# train
#
if DEBUG and (DEBUG & DEBUG_TRAIN):

    #model_file = Path(model_filename)
    model_file = False

#    model = Sequential()
#    model.add(Embedding(16, 4, input_length=1))
#    model.add(Reshape((4,)))

    #
    # init the model
    #
    if model_file and model_file.is_file():
        model = load_model(model_filename)
    else:

        # kernel_initializer='RandomUniform'
        # kernel_initializer='uniform', 'RandomUniform'
        # activation='relu'
        # activation='softmax'
        # activation='linear'
        # activation='sigmoid'
        model = Sequential([
            Dense(16, input_shape=(16,), use_bias=False, kernel_initializer='RandomUniform', activation='sigmoid'),
            Dense(4, activation='linear', use_bias=False, kernel_initializer='RandomUniform')
        ])

        # loss='mean_squared_error' or 'mse'
        # optimizer='adam', 'sgd'
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    #
    # hyper params
    #
    num_episodes = 2000       # Total episodes
    #learning_rate = 0.8           # Learning rate
    max_steps = 99                # Max steps per episode
    gamma = 0.98                  # Discounting rate

    # Exploration parameters
    epsilon = 1                # Exploration rate
    max_epsilon = 1.0           # Exploration probability at start
    min_epsilon = 0.01          # Minimum exploration probability
    #decay_rate = 0.005          # Exponential decay rate for exploration prob
    decay_rate = 0.001

    steps = []
    rewards = []

    if DEBUG : print("Train ****************************************************")

    for episode in range(num_episodes):

        state = env.reset()

        step = 0
        done = False

        #print("state:{}".format(state))

        while not done and step < max_steps:

            # get next action
            if np.random.random() < epsilon:
                action = np.random.randint(0, 4)
            else:
                state_oh = np.identity(16)[state:state + 1]
                #print("np.identity(16)[state:state + 1]:{}".format(state_oh))
                p = model.predict(state_oh)
                #print("p:{}".format(p))
                action = np.argmax(p)

            new_state, reward, done, _ = env.step(action)


            if reward:
                print("success {} {}".format(reward, done))

            #if new_statetate != action_to_state(state, action):
            #    print("Requested state {}, slipped to ()".format(action_to_state(state, action), new_state))

            #print("state:{}, r:{}, done:{}".format(new_state, r, done))

            # get loss
            current_value = np.max(model.predict(np.identity(16)[state:state + 1]))
            target_value = reward + gamma * current_value

            #print("target:{}".format(target))

            target_vec = model.predict(np.identity(16)[state:state + 1])[0]
            target_vec[action] = target_value

            #print("update model {}/{} (state/action), {} -> {}".format(state, action, current_value, target_value))

            #print("target_vec:{}".format(target_vec))
            #print("np.identity(16)[state:state + 1]:{}".format(np.identity(16)[state:state + 1]))
            #print("target_vec.reshape(-1, 4):{}".format(target_vec.reshape(-1, 4)))

            # update model
            model.fit(np.identity(16)[state:state + 1], target_vec.reshape(-1, 4), epochs=1, verbose=0)

            state = new_state

            step += 1

            # Reduce epsilon (because we need less and less exploration)
            if reward:
                epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

        steps.append(step)
        rewards.append(reward)

        if DEBUG and (DEBUG & DEBUG_EPISODE_SUMMARY):
            if episode % report_increment == 0:
                print("Episode {:>3d} of {:>4d} epsilon/mean_steps/rewards: {:6.4f} / {:4.1f} / {}".format(episode, num_episodes, epsilon, np.mean(steps), np.sum(rewards))) # np.mean(steps), np.mean(rewards)

    #
    # dump the table/values
    #
    if DEBUG and (DEBUG & DEBUG_DUMP_TABLE):
        for s in range(state_size):
          print("{:<2d}: {}".format(s, model.predict(np.identity(16)[s:s + 1])))

    if model_file and model_file:
        model.save(model_filename)
        del model

#
# play the game
#
if DEBUG and (DEBUG & DEBUG_PLAY_GAME) and np.mean(rewards) >= .5:

    if model_file and model_file.is_file():
        model = load_model(model_filename)
    else:
        runtimeException("No model")

    env.reset()

    for episode in range(game_to_play):
        state = env.reset()
        step = 0
        done = False
        print("\nPlay ****************************************************")
        print("EPISODE ", episode)

        for step in range(max_steps):
            print("\nStep {}".format(step))

            states = model.predict(np.identity(16)[state:state + 1])[0]

            action = np.argmax(states)

            print("Requesting action:{}({})".format(Directions[action], action))

            new_state, reward, done, info = env.step(action)

            if new_state != action_to_state(state, action):
                print("Requested state {}, slipped to ()".format(action_to_state(state, action), new_state))

            if done:
                if reward:
                  print("Success")
                else:
                  print("Fail")
                break

            #print("state:{}, r:{}, done:{}, info".format(new_state, reward, done, info))

            env.render()

            if done:
              # We print the number of step it took.
              print("Number of steps", step)
              break

            state = new_state

env.close()



