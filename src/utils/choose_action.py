import random

import numpy as np

from utils.run_network import max_action_preds

def choose_action(stack_state, Q, sess, epsilon, n_actions):
    """
    Choose random action with prob epislon, or the best action according to the network.
    """

    if random.uniform(0, 1) < epsilon:
        p_val = 0
        action = random.randint(0, n_actions-1)
    else:
        action, p_val = max_action_preds(sess, Q, stack_state.reshape((1, *stack_state.shape)))
        action, p_val = action[0], p_val[0]

    action_vec = np.zeros(n_actions, dtype=np.int)
    action_vec[action] = 1
    return action_vec, p_val
