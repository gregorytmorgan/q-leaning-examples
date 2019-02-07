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
from keras import optimizers
from keras import backend as K

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

mean_trailing_reward = False

#
# init the OpenAI environment
#

env = gym.make("FrozenLake-v0")

#from gym.envs.registration import register
#register(
#    id='FrozenLakeNotSlippery-v0',
#    entry_point='gym.envs.toy_text:FrozenLakeEnv',
#    kwargs={'map_name' : '4x4', 'is_slippery': False},
#    max_episode_steps=100,
#    reward_threshold=0.8196, # optimum = .8196, changing this seems have no influence
#)
#env = gym.make("FrozenLakeNotSlippery-v0")

print("Map:")
env.render()

#
# helper constants/functions
#

action_size = env.action_space.n
state_size = env.observation_space.n

Directions = ['Left', 'Down', 'Right', 'Up']

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

def dump_table(model, state_size):
    for s in range(state_size):
      print("{:<2d}: {}".format(s, model.predict(np.identity(state_size)[s:s + 1])))

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


    #
    # hyper params
    #
    num_episodes = 1000         # Total episodes
    max_steps = 99              # Max steps per episode

    # learning params
    gamma = 0.98                # Discounting rate
    learning_rate = 0.1         # Learning rate

    # Exploration parameters
    max_epsilon = .1            # Exploration probability at start
    min_epsilon = 0.01          # Minimum exploration probability
    epsilon = max_epsilon       # Exploration rate
    #decay_rate = 0.005         # Exponential decay rate for exploration prob
    decay_rate = 0.005

    #
    # build the model
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
            Dense(16, activation='sigmoid', kernel_initializer='RandomUniform', use_bias=False, input_shape=(16,)),
 #           Dense(24, activation='relu', kernel_initializer='RandomUniform', use_bias=False),
 #           Dense(16, activation='relu', kernel_initializer='RandomUniform', use_bias=False),
            Dense(4, activation='linear',  kernel_initializer='RandomUniform', use_bias=False)
        ])

        # stackoverflow.com/questions/45869939/something-wrong-with-keras-code-q-learning-openai-gym-frozenlake
        def sum_of_sqares(yTrue, yPred):
            return K.sum(K.square(yTrue - yPred))

        # loss='mean_squared_error' or 'mse'
        # loiss=
        # optimizer='adam', 'sgd'
        # optimizers.SGD(lr=0.01, clipvalue=0.5)
        # optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
        # optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        model.compile(loss=sum_of_sqares, optimizer = optimizers.SGD(lr=learning_rate), metrics=['accuracy'])

        #print(model.summary())

        dump_table(model, state_size)

    #
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

            #if done and reward == 0 and episode < 500:
            #    reward = -.001

            #if new_statetate != action_to_state(state, action):
            #    print("Requested state {}, slipped to ()".format(action_to_state(state, action), new_state))

            #print("state:{}, r:{}, done:{}".format(new_state, r, done))

            # get loss
            target_vec = model.predict(np.identity(16)[state:state + 1])[0]

            if reward == 1:
                print("success")
                #print("original vector: {}".format(target_vec))

            current_value = np.max(target_vec)
            target_value = reward + gamma * current_value
            target_vec[action] = target_value

            if reward == 1:
                pass
                #print("reward: {}, gamma: {}".format(reward, gamma))
                #print("update model {}/{} (state/action), {:.4f} -> {:.4f}".format(state, action, current_value, target_value))
                #print("target_vec:{}".format(target_vec))
                #print("np.identity(16)[state:state + 1]:{}".format(np.identity(16)[state:state + 1]))
                #print("target_vec.reshape(-1, 4):{}".format(target_vec.reshape(-1, 4)))
                #dump_table(model, state_size)


            # update model
            model.fit(np.identity(16)[state:state + 1], target_vec.reshape(-1, 4), epochs=1, verbose=0, batch_size=1) # batch_size=None

            state = new_state

            step += 1

            #if step == max_steps:
            #    print("max steps")

            # Reduce epsilon (because we need less and less exploration)
            #if reward = 1:
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

        steps.append(step)
        rewards.append(reward)

        reward_trail_start = 0 if len(rewards) < report_increment else round(len(rewards) * .8)

        mean_trailing_reward = round(np.mean(rewards[reward_trail_start:]), 4) if rewards else 0

        if DEBUG and (DEBUG & DEBUG_EPISODE_SUMMARY):
            if (episode % report_increment == 0 or episode == num_episodes - 1) and episode != 0:
                #print("Episode {:>3d} of {:>4d} epsilon/mean_steps/mean_rewards(trailing {}): {:6.4f} / {:4.1f} / {}".format(episode, num_episodes, len(rewards) - reward_trail_start, epsilon, round(np.mean(steps), 4), mean_trailing_reward)) # np.mean(steps), np.mean(rewards)
                print("Episode {:>3d} of {:>4d} e/s/r: {:.4f} / {:4.1f} / {}".format(episode, num_episodes, epsilon, round(np.mean(steps), 4), round(np.mean(rewards), 4)))

    #
    # dump the table/values
    #
    if DEBUG and (DEBUG & DEBUG_DUMP_TABLE):
        dump_table(model, state_size)

    midpoint = round(len(rewards)/2)
    success_count = sum(rewards[midpoint:])/midpoint if midpoint else 0
    print("Percent of succesful episodes (2nd half of training): {:4f}%".format(success_count))

    #
    # save the model
    #
    if model_file and model_file:
        model.save(model_filename)
        del model

#
# play the game
#
if DEBUG and (DEBUG & DEBUG_PLAY_GAME):

    rewards_game_threshold = .5

    # mean_trailing_reward will be False if we didn't train
    if mean_trailing_reward is False or mean_trailing_reward >= rewards_game_threshold:

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
    else:
        print("Aborted game, trailing {} rewards is {}. Threshold is {}".format(reward_trail_start, mean_trailing_reward, rewards_game_threshold))

env.close()



