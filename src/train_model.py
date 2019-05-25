import os
import pickle
import numpy as np

import gym

from utils.choose_action import choose_action

### Hyperparameters ###
# How much new information affects current reward estimates
learning_rate = 1
# Exploration
epsilon_max = 1.0
epsilon_min = 0.1
fixed_eps = True
# How much to discount into the future
gamma = 0.9
# How long we run the training for
max_episodes = 2000
max_steps = 20
###

def train_model(game_name, run_name, render=False):
    """
    Trains a standard Qnetwork on given game, for given run.
    """

    if game_name == 'FrozenLake-v0':
        from gym.envs.registration import register
        register(
            id='FrozenLakeNotSlippery-v0',
            entry_point='gym.envs.toy_text:FrozenLakeEnv',
            kwargs={'map_name' : '8x8', 'is_slippery': False},
            max_episode_steps=100,
            reward_threshold=0.78,
        )
        game_name = 'FrozenLakeNotSlippery-v0'

    env = gym.make(game_name)
    n_actions = env.action_space.n

    Q = []
    for i in range(8*8):
        Q.append([])
        for _ in range(4):
            Q[i].append(0.5)

    # Initialize progress
    total_episodes = 0
    total_steps = 0

    # Set up run-specific information
    run_path = './data/run_' + run_name
    prg_path = run_path + '/progress.pickle'

    if not os.path.exists(run_path): # If this is the first time for this run
        os.mkdir(run_path)

    epsilon = epsilon_min
    # Start playing game for given number of episodes
    while total_episodes < max_episodes:
        total_episodes += 1
        state = env.reset()

        # Start playing episode
        epi_reward = 0
        epi_steps = 1
        done = False
        while epi_steps < max_steps and not done:

            # Select and perform action
            action = choose_action(state, Q, epsilon, n_actions)
            n_state, reward, done, _ = env.step(action)
            epi_reward += reward

            if done:
                Q[state][action] = Q[state][action] + learning_rate * (reward - Q[state][action])                
            else:
                max_next = max(Q[n_state])
                Q[state][action] = Q[state][action] + learning_rate * (reward + gamma * max_next - Q[state][action])                

            state = n_state

            epi_steps += 1
            total_steps += 1

        with open(prg_path, 'ab') as pickle_handle:
            pickle.dump((total_steps, total_episodes, epi_steps, epi_reward), pickle_handle)

        if total_episodes % 10 == 0:
            print("episode " + str(total_episodes) + " reward: " + str(epi_reward))

    print("exploiting")
    interp = {1: "Down", 2: "Right", 3: "Up", 4: "Left"}
    state = env.reset()
    env.render()
    tot_rew = 0
    done = False
    while not done:
        action = np.argmax(Q[state])
        print(interp[action])
        state, reward, done, _ = env.step(action)
        tot_rew += reward

    print(tot_rew)
