import pickle
import numpy as np


# code based on baselines: https://github.com/openai/baselines/blob/master/baselines/ddpg/memory.py

class RingBuffer(object):
    def __init__(self, maxlen, shape, dtype='float32'):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = np.zeros((maxlen,) + shape).astype(dtype)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def get_batch(self, idxs):
        return self.data[(self.start + idxs) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v

    #def resize(self, new_maxlen):
    #    self.maxlen = new_maxlen
    #    self.data.resize(new_maxlen,)
    #    self.start = 0


# observation_before_action, action, reward, isdone, observation

class Memory(object):
    def __init__(self, limit, action_shape, observation_shape):
        self.limit = limit

        self.observations0 = RingBuffer(limit, shape=observation_shape)
        self.actions = RingBuffer(limit, shape=action_shape)
        self.rewards = RingBuffer(limit, shape=(1,))
        self.terminals1 = RingBuffer(limit, shape=(1,))
        self.observations1 = RingBuffer(limit, shape=observation_shape)

    def sample_batch(self, batch_size):
        # Draw such that we always have a proceeding element.
        batch_idxs = np.random.random_integers(self.nb_entries - 2, size=batch_size)

        obs0_batch = self.observations0.get_batch(batch_idxs)
        action_batch = self.actions.get_batch(batch_idxs)
        reward_batch = self.rewards.get_batch(batch_idxs)
        terminal1_batch = self.terminals1.get_batch(batch_idxs)
        obs1_batch = self.observations1.get_batch(batch_idxs)

        result = [obs0_batch, action_batch, reward_batch, terminal1_batch, obs1_batch]
        return result

    def append(self, obs0, action, reward, terminal1, obs1):
        self.observations0.append(obs0)
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminals1.append(terminal1)
        self.observations1.append(obs1)

    @property
    def nb_entries(self):
        return len(self.observations0)

    def size(self):
        return len(self.observations0)

    def save(self, pathname):
        print('Dumping memory...')
        with open(pathname, 'wb') as f:
            # protocol=4 supports > 4Gb files
            # splitting save up to lower memory use during saving
            pickle.dump(self.limit, f, protocol=4)
            pickle.dump(self.observations0, f, protocol=4)
            pickle.dump(self.actions, f, protocol=4)
            pickle.dump(self.rewards, f, protocol=4)
            pickle.dump(self.terminals1, f, protocol=4)
            pickle.dump(self.observations1, f, protocol=4)
        print('memory dumped into', pathname, " limit:", self.limit, "size: ", self.size())

    def load(self, pathname):
        with open(pathname, 'rb') as f:
            self.limit = pickle.load(f)
            self.observations0 = pickle.load(f)
            self.actions = pickle.load(f)
            self.rewards = pickle.load(f)
            self.terminals1 = pickle.load(f)
            self.observations1 = pickle.load(f)
        print('memory loaded from', pathname, " limit:", self.limit, "size: ", self.size())
        print('memory start:', self.observations0.start, "length:", self.observations0.length)
