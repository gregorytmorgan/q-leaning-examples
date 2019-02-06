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

import math
import gym
import numpy as np
#import random
import tensorflow as tf
import matplotlib.pyplot as plt

# 1 = dump each episode, 2 dump each step, 4: break at first success, 8: plotm 16: play game . e.g. DEBUG = 7, dump all steps,episodes,break on success
DEBUG = 1

games_to_play = 3
report_increment = 100

np.set_printoptions(precision=4, linewidth=280, suppress=True, threshold=64)

Directions = ['Left', 'Down', 'Right', 'Up']

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

    return new_state

#%matplotlib inline

#
# init the OpenAI environment
#

env = gym.make('FrozenLake-v0')

#from gym.envs.registration import register
#register(
#    id='FrozenLakeNotSlippery-v0',
#    entry_point='gym.envs.toy_text:FrozenLakeEnv',
#    kwargs={'map_name' : '4x4', 'is_slippery': False},
#    max_episode_steps=100,
#    reward_threshold=0.8196, # optimum = .8196, changing this seems have no influence
#)
#env = gym.make('FrozenLakeNotSlippery-v0')

print("Map:")
env.render()

tf.reset_default_graph()

#These lines establish the feed-forward part of the network used to choose actions
inputs1 = tf.placeholder(shape=[1,16],dtype=tf.float32)
W = tf.Variable(tf.random_uniform([16,4], 0, 0.01))
Qout = tf.matmul(inputs1, W)
predict = tf.argmax(Qout, 1)

#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[1, 4],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)

#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()

#
# Hyper params
#

# Exploration parameters
min_epsilon = 0.01          # Minimum exploration probability
max_epsilon = 1.0           # Exploration probability at start
epsilon = .2      # Exploration rate
#decay_rate = 0.005          # Exponential decay rate for exploration prob
decay_rate = 0.005          # Exponential decay rate for exploration prob
#decay_rate = epsilon/1000   # Exponential decay rate for exploration prob

# learning params
y = .98

#num_episodes = 2000 original value
num_episodes = 1

#create lists to contain total rewards and steps per episode
stepList = []
rewardList = []

with tf.Session() as sess:
    sess.run(init)
    for episode in range(num_episodes):
        if DEBUG & 1:
            if episode % report_increment == 0:
                print("\nEpisode {}, epsilon:{:.2f}\n========".format(episode, epsilon))

        #Reset environment and get first new observation
        s = env.reset()
        d = False
        step = 0
        max_steps = 100
        path = []
        random_action_count = 0
        solved = False

        #The Q-Network
        while step < max_steps:
            step += 1

            #Choose an action by greedily (with epsilon chance of random action) from the Q-network
            a, allQ = sess.run([predict, Qout], feed_dict={inputs1:np.identity(16)[s:s+1]})

            if DEBUG & 2: print("a:{}, allQ:{}".format(a, allQ))

            if np.random.rand(1) < epsilon:
                random_action_count += 1
                a[0] = env.action_space.sample()

            #Get new state and reward from environment
            s1, r, d, _ = env.step(a[0])

            path.append((a[0], Directions[a[0]]))

#            if r and random_action_count == 0:
#                env.render
#                print(path)
#                solved = True

            if DEBUG & 2: print("step result:{} --> s1:{}, r:{}, d:{}".format(Directions[a[0]], s1,r, d))

            #Obtain the Q' values by feeding the new state through our network
            Q1 = sess.run(Qout, feed_dict={inputs1:np.identity(16)[s1:s1 + 1]})

            if DEBUG & 2: print("Q1:{}".format(Q1))

            #Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0, a[0]] = r + y * maxQ1

            #Train our network using target and predicted Q values
            if not solved:
                _, W1 = sess.run([updateModel, W], feed_dict={inputs1:np.identity(16)[s:s+1], nextQ:targetQ})

            s = s1

            if d == True:
                #Reduce chance of random action as we train the model.
                #e = 1./((episode/50.) + 10)
                epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
                break

        stepList.append(step)
        rewardList.append(r)

        if solved:
            break

        # dump each step iteration
        #if DEBUG & 1: print("steplist: {}".format(stepList))
        #if DEBUG & 1: print("rewardList:{}".format(rewardList))

        # run until first success
        if DEBUG & 4 and r:
            break

        if DEBUG & 1:
            if episode % report_increment == 0:
                nStepList = len(stepList)
                nRewardList = len(rewardList)


                startpoint = nRewardList - math.ceil(nRewardList/10)
                #startpoint = episode - report_increment + 1

                if nStepList and nRewardList:
                    print("trailing {}, mean steps: {:.4f}, mean reward: {:.4f} rsum:{}, rcnt:{}".format(nStepList - startpoint, np.mean(stepList[startpoint:]), np.mean(rewardList[startpoint:]), np.sum(rewardList[startpoint:]), len(rewardList[startpoint:])))

    # dump the table
    if (episode == num_episodes - 1) or (DEBUG & 4):
        for s in range(15):
            q = sess.run(Qout, feed_dict={inputs1:np.identity(16)[s:s+1]})
            print(q)

    midpoint = round(len(rewardList)/2)
    success_count = sum(rewardList[midpoint:])/midpoint if midpoint else 0
    print("Percent of succesful episodes (2nd half of training): {:4f}%".format(success_count))

    if DEBUG & 8:
        # rewardList is a list of success/failure. 1's represent success, 1's become more
        # frequent over time.
        plt.gca().set_title("Reward/Episode".format())
        plt.plot(rewardList)
        plt.show()

        # stepList is a list of episode durations. Episode length greather than chance
        # repreent 'knowledge'
        plt.gca().set_title("Steps/Episode".format())
        plt.plot(stepList)
        plt.show()

    #
    # play the game
    #
    if DEBUG & 16:

        env.reset()

        for episode in range(games_to_play):
            state = env.reset()
            step = 0
            done = False
            print("\nPlay ****************************************************")
            print("EPISODE ", episode)

            for step in range(max_steps):
                print("\nStep {}".format(step))

                #Obtain the Q' values by feeding the new state through our network
                action, allQ = sess.run([predict, Qout], feed_dict={inputs1:np.identity(16)[state:state + 1]})

                print("Requesting action: {}({})".format(Directions[action[0]], action[0]))

                new_state, reward, done, info = env.step(action[0])

                env.render()

                if new_state != action_to_state(state, action[0]):
                    print("Requested state: {}, slipped to {}".format(action_to_state(state, action[0]), new_state))

                if done:
                  if reward:
                    print("Success")
                  else:
                    print("Fail")
                  break

                if done:
                  # We print the number of step it took.
                  print("Number of steps", step)
                  break

                state = new_state

env.close()




