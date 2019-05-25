import random

import numpy as np

def choose_action(state, Q, epsilon, n_actions):
    """
    Choose random action with prob epislon, or the best action according to the Q-table.
    """

    if random.uniform(0, 1) < epsilon:
        action = random.randint(0, n_actions-1)
    else:
        vals = Q[state]
        action = np.argmax(vals)

    return action
