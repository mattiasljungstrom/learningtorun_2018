# environment wrapper

import math

from observation_2018 import process_observation

class fastenv:
    def __init__(self, real_env, skipcount):
        # real env might be a remote env
        # or a local instance during test()
        self.real_env = real_env
        self.stepcount = 0
        self.skipcount = skipcount

    def get_observation(self, plain_obs):
        return process_observation(plain_obs)

    def step(self, action, project=False):        
        # validate action for NaN
        for i in range(len(action)):
            if math.isnan(action[i]):
                action[i] = 0

        sr = 0
        for _ in range(self.skipcount):
            self.stepcount += 1
            oo, r, d, i = self.real_env.step(action, project=False)
            o = self.get_observation(oo)
            sr += r
            if d == True:
                break

        return o, sr, d, i

    def reset(self, project=False):
        self.stepcount=0
        oo = self.real_env.reset(project=False)
        o = self.get_observation(oo)
        return o

    # some gym env need this
    def render(self):
        self.real_env.render()
