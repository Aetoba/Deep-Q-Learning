import os
import shutil
import pickle
import warnings

import numpy as np
import tensorflow as tf

from PIL import Image

from utils.process_frame import phi
from utils.choose_action import choose_action
from model.qnetwork import Q_cnn
from model.experiences import Experiences
from utils.run_network import max_action_preds, train
from utils.exploit_learning import Exploit

warnings.filterwarnings("ignore")

### Hyperparameters ###
# How much new information affects current reward estimates
learning_rate = 0.00025
# Exploration
epsilon_max = 1.0
epsilon_min = 0.1
fixed_eps = False
# How much to discount into the future
gamma = 0.9
# We stack frames to have some sense of the past leading up to this state
frame_stack_size = 4
# For experience replay
mem_size = 500000
minibatch_size = 32
# How long we run the training for
max_episodes = 5000
max_steps = 4500 # (60 * 60 * 5) / 4 means 5 minutes at 60fps taking every 4th frame
###

def train_model(game_name, run_name, use_retro, capture=False, render=False, exp_replay_size=mem_size, cpu=False):
    """
    Trains a standard Qnetwork on given game, for given run.
    """

    # Set up environment
    if use_retro:
        import retro
        env = retro.make(game=game_name)
    else:
        import gym
        env = gym.make(game_name)
    n_actions = env.action_space.n

    # Set up experience replay and Q network
    experiences = Experiences(exp_replay_size, minibatch_size)
    tf.reset_default_graph()
    Q = Q_cnn(learning_rate, n_actions)
    saver = tf.train.Saver()

    # Initialize network session
    if cpu:
        config = tf.ConfigProto(device_count={'GPU': 0})
    else:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
        config=tf.ConfigProto(gpu_options=gpu_options)
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # Set up system to test current ability
    exploit = Exploit(sess, Q, env, frame_stack_size, max_steps)

    # Set up run-specific information
    run_path = './data/run_' + run_name
    prg_path = run_path + '/progress.pickle'
    mta_path = run_path + '/metadata.pickle'
    mod_path = run_path + '/model.ckpt'
    prf_path = run_path + '/performance.pickle'
    cap_dir = './data/not_runs/captures'

    # Loads model if it already exists (but still acts as starting a new run)
    if not os.path.exists(run_path):
        os.mkdir(run_path)
        saver.save(sess, mod_path)
    else:
        print("Loading exisiting model...")
        saver.restore(sess, mod_path)

    # Initialize progress
    total_episodes = 0
    total_steps = 0

    # Start playing game for given number of episodes
    while total_episodes < max_episodes:
        total_episodes += 1
        init_screen = env.reset()
        state, stack_state = phi(init_screen, frame_stack_size, new_episode=True)

        # Start playing episode
        epi_reward = 0
        epi_steps = 0
        losses = []
        vals = []
        done = False
        while epi_steps < max_steps and not done:

            # Linear annealement over first 1,000,000 steps
            if total_steps > 1000000 or fixed_eps:
                epsilon = epsilon_min
            else:
                epsilon = epsilon_max - total_steps * ((epsilon_max - epsilon_min)/1000000)

            # Select and perform action
            action, p_val = choose_action(stack_state, Q, sess, epsilon, n_actions)
            vals.append(p_val)
            if use_retro:
                n_screen, reward, done, _ = env.step(action)
            else:
                n_screen, reward, done, _ = env.step(np.argmax(action))
            epi_reward += reward

            # Capture frames for video or render live
            if capture:
                img = Image.fromarray(n_screen, 'RGB')
                img.save(cap_dir + '/e' + str(total_episodes) + 's' + str(epi_steps) + '.png')
            if render:
                env.render()

            # Clip rewards
            reward = np.sign(reward)

            # Pre-process and save transition
            state, n_stack_state = phi(n_screen, frame_stack_size, curr_state=state)
            experiences.append((stack_state, action, reward, n_stack_state, done))
            stack_state = n_stack_state

            # Gather experiences for experience replay
            b_stack_states, b_actions, b_rewards, b_n_stack_states, b_dones, b_size = experiences.sample()

            # Calculate the new target for Q(s,a)
            _, next_vals = max_action_preds(sess, Q, b_n_stack_states)
            next_vals[b_dones] = 0
            targets = b_rewards + (gamma * next_vals)

            # Take a step towards the targets
            loss = train(sess, Q, b_stack_states, targets, b_actions)

            losses.append(loss)

            epi_steps += 1
            total_steps += 1

            if done:
                print("Done with episode " + str(total_episodes) + " in " + str(epi_steps) + " steps")


        # End of Episode
        # Check performance with exploit policy
        perf, perf_steps = exploit.run_exploit()

        # Log relevant information
        loss_average = np.average(losses)
        loss_max = np.max(losses)
        val_average = np.average(vals)
        val_max = np.max(vals)
        with open(prg_path, 'ab') as pickle_handle:
            pickle.dump((total_steps, total_episodes, epi_steps, epi_reward), pickle_handle)
        with open(mta_path, 'ab') as pickle_handle:
            pickle.dump((loss_average, loss_max, val_average, val_max), pickle_handle)
        with open(prf_path, 'ab') as pickle_handle:
            pickle.dump((perf, epsilon), pickle_handle)
        saver.save(sess, mod_path)

        # Print update on performance
        print("Episode " + str(total_episodes) + " reward: " + str(epi_reward) + " in " + str(epi_steps) + " steps")
        print("Performance reward: " + str(perf) + " in " + str(perf_steps) + " steps")
        print("Experience replay has size " + str(experiences.len()))
        print("Loss average of: " + str(loss_average))
        print("Loss max of: " + str(loss_max))
        print("Average chosen value of: " + str(val_average))
        print("Max chosen value of: " + str(val_max))
        print("Epsilon was at: " + str(epsilon))
        print()
