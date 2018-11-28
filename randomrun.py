import numpy as np
import random

#from osim.env import ProstheticsEnv
from create_env import create_env
from observation_2018 import process_observation
from observation_2018 import normalize_dir

from reward_mod import calc_extra_reward
from reward_mod import bind_alt_reward
from reward_mod import bind_alt_reset
from reward_mod import print_target_changes


seed = 5656324
random.seed(seed)


env = create_env(train=False, render=True)

# testing
bind_alt_reward(env)
#bind_alt_reset(env)

state_desc = env.reset(project=False)
#print(observation)
#observation = env.reset()

observation = np.array( process_observation(state_desc) )

N_S = observation.shape  #env.observation_space.shape[0]
N_A = env.action_space.shape
A_BOUND = [env.action_space.low, env.action_space.high]
print("obs shape: ", N_S)
print("action shape: ", N_A)
#print("action bounds: ", A_BOUND)


for i in range(100):
    observation_flat = np.array( process_observation(state_desc) )
    #print("\n\nstate:\n", state_desc)
    #print("\nflatten:\n", observation_flat)
    #print("\nobs dims:", len(observation_flat))
    #print("real obs shape: ", observation_flat.shape)

    #reward_extra = calc_extra_reward(state_desc)
    #print("extra reward: ", reward_extra)
    #if reward_extra==0.0:
    #    import time
    #    time.sleep(10)

    action = [0.0] * 19
    #action[2] = 1.0
    #action[8] = 1.0
    action[13] = 1.0

    state_desc, reward, done, info = env.step(action, project=False)

    print("reward: ", reward)
    #if reward <= -11:
    #    import time
    #    time.sleep(100)

    if done:
        env.reset()
        break
