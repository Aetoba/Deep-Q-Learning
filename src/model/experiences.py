from collections import deque
# import pickle
import numpy as np

class Experiences:
    """
    Stores past transitions for experiences replay.
    """

    def __init__(self, mem_size, minibatch_size):
        self.memory = deque(maxlen=mem_size)
        self.mem_size = mem_size
        self.minibatch_size = minibatch_size

    def append(self, experience):
        self.memory.append(experience)

    def sample(self):
        curr_size = len(self.memory)
        b_size = min([curr_size, self.minibatch_size])
        indices = np.random.choice(curr_size, b_size, replace=False)
        batch = [self.memory[i] for i in indices]
        b_stack_states = [exp[0] for exp in batch]
        b_actions = [exp[1] for exp in batch]
        b_rewards = [exp[2] for exp in batch]
        b_n_stack_states = [exp[3] for exp in batch]
        b_dones = [exp[4] for exp in batch]
        return b_stack_states, b_actions, b_rewards, b_n_stack_states, b_dones, b_size

    def len(self):
        return len(self.memory)

    # def load(self, path):
    #     with open(path, 'rb') as pickle_handle:
    #         self.memory = pickle.load(pickle_handle)

    # def store(self, path):
    #     if self.mem_size > 0:
    #         with open(path, 'wb') as pickle_handle:
    #             pickle.dump(self.memory, pickle_handle)
