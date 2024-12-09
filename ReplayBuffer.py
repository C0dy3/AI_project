from collections import deque
import random


class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)
    def add(self, experience):
        state, action, reward, next_state, done = experience
        if state.shape != (96, 96, 3) or next_state.shape != (96, 96, 3):
            raise ValueError(
                f"State or next state must have this format (96, 96, 3), but has {state.shape} a {next_state.shape}")
        self.buffer.append(experience)
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    def size(self):
        return len(self.buffer)


