import numpy as np

class one_fsq_noise(object):
    def __init__(self):
        self.buffer = np.array([0.])
        self.state = np.random.RandomState()
        self.skip = 1
        self.skipcounter = 0

    def one(self,size,noise_level=1.):
        # draw one gaussian
        g = self.state.normal(loc=0.,scale=noise_level,size=size)

        if self.buffer.shape != size:
            self.buffer = np.zeros(size,dtype='float32')

        # skip for frequency adjustment to suit different envs.
        self.skipcounter+=1
        if self.skipcounter >= self.skip-1:
            self.skipcounter=0

            self.buffer += g

            # high pass a little
            self.buffer *= .9

        return self.buffer.copy()

    def ask(self):
        return self.buffer.copy()

# 1/f^2 noise: http://hal.in2p3.fr/in2p3-00024797/document
