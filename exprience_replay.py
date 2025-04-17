from collections import deque
import random

class ReplayMemory():
    def __init__(self, maxlen, seed=None):
        self.memory = deque([], maxlen=maxlen)

        if seed is not None: # if we want to control the randomness
            random.seed(seed)

    def __len__(self):
        return len(self.memory)
    
    def append(self, transition):
        self.memory.append(transition)
    
    def sample(self, sample_size):
        return random.sample(self.memory, k=sample_size)
