from collections import deque
import random
import torch

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        data = (list(state), action, reward, list(next_state), done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)#データは消える？

        # state: array with length 4
        # action: int
        # reward: float
        # next_state: array with length 4
        # done: 0 or 1
        state = torch.tensor([x[0] for x in data])
        action = torch.tensor([x[1] for x in data], dtype=torch.int64)
        reward = torch.tensor([x[2] for x in data])
        next_state = torch.tensor([x[3] for x in data])
        done = torch.tensor([x[4] for x in data], dtype=torch.int32)
        return state, action, reward, next_state, done

class ReplayBuffer_Dict:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        data = {"state": list(state), "action": action, "reward": reward, "next_state": list(next_state), "done": done}
        # arrayになってしまう なぜ？
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)#データは消える？

        # state: array with length 4
        # action: int
        # reward: float
        # next_state: array with length 4
        # done: 0 or 1
        state = torch.tensor([x["state"] for x in data])
        action = torch.tensor([x["action"] for x in data], dtype=torch.int64)
        reward = torch.tensor([x["reward"] for x in data])
        next_state = torch.tensor([x["next_state"] for x in data])
        done = torch.tensor([x["done"] for x in data], dtype=torch.int32)
        return state, action, reward, next_state, done