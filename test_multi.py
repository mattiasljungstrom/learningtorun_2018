
import numpy as np
import tensorflow as tf
import random

from IPython.display import clear_output
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from create_env import create_env
from observation_2018 import process_observation
from observation_2018 import processed_dims

from ddpg_multi_agent import Agent



def set_all_seeds(seed):
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def blend_color(t, c1, c2):
    t = np.clip(t, 0, 1)
    c1 = colors.rgb_to_hsv(c1)
    c2 = colors.rgb_to_hsv(c2)
    c = ((c1 * t) + (c2 * (1-t)))
    c = np.clip(c, 0.0, 1.0)
    return colors.hsv_to_rgb(c)
    
def draw_rewards(rewards, xv, zv):
    clear_output(True)
    plt.figure(figsize=(16, 8*2))
    # rewards
    plt.subplot(2, 1, 1)   # nrows, ncols, index
    plt.grid(True)
    plt.title('reward')
    plt.plot(rewards)
    # path
    xp = np.cumsum(xv)
    zp = np.cumsum(zv)
    plt.subplot(2, 1, 2)
    plt.grid(True)
    plt.axis([0, 2000, -500, 500])
    plt.plot(xp,zp*-1.0)                 # z axis is other way in 3d view        
    
    # display
    plt.tight_layout()
    plt.show()
    

def draw_test_rewards(test_stats):
    plt.figure(figsize=(8, 4), dpi=150, facecolor='w', edgecolor='k')

    # test_stats = [[rewards, pv_x, pv_z, total_reward, step_nr] ... ]
    plt.grid(True)
    scale = 0.8
    plt.axis([0, 2000*scale, -500*scale, 500*scale])
    for i, stats in enumerate(test_stats):        
        xv = stats[1]
        zv = stats[2]
        xp = np.cumsum(xv)
        zp = np.cumsum(zv)
        reward = stats[3]
        print(i+1, "reward", reward, "in steps", stats[4])
        
        c1 = colors.rgb_to_hsv((0.0, 0.8, 0.0))
        c2 = colors.rgb_to_hsv((1.0, 0.3, 0.3))
        t = np.clip((reward-9800) / 150, 0, 1)
        c = ((c1 * t) + (c2 * (1-t)))
        c = np.clip(c, 0.0, 1.0)
        c = colors.hsv_to_rgb(c)
        plt.plot(xp, zp*-1.0, color=c, linewidth=1)
    
    # display
    plt.tight_layout()
    plt.show()


# this doesn't reset env, need to do it before
def test_debug(agent, env_test, plain_obs, visualize=True):
    step_nr = 0
    total_reward = 0
    
    if visualize:
        clear_output(True)
    rewards = []
    pv_x = []
    pv_z = []
    
    while True:
        # get observation
        observation = np.array( process_observation(plain_obs) )
        
        # get action
        action = agent.get_max_action(observation)

        # get pelvis vel
        state_desc = env_test.get_state_desc()
        pv_x.append(state_desc["body_vel"]["pelvis"][0])
        pv_z.append(state_desc["body_vel"]["pelvis"][2])

        # do step
        action_out = [float(action[i]) for i in range(len(action))]
        plain_obs, reward, done, _info = env_test.step(action_out, project=False)

        # update stats
        step_nr += 1
        total_reward += reward
        rewards.append(reward)

        if visualize:
            draw_rewards(rewards, pv_x, pv_z)
        else:
            if (step_nr+1) % 50 == 0:
                print('.', end='', flush=True)
        
        if done:            
            break
            
    if visualize:
        print('Done after', step_nr, 'got reward', total_reward)
    else:
        print('')
        return rewards, pv_x, pv_z, total_reward, step_nr

# run tests and gather statistics
def run_tests(agent, env_test, start_seed, test_nr):
    seed = start_seed
    test_stats = []
    total_reward_sum = 0
    low_reward = 10000
    high_reward = 0
    
    for _ in range(test_nr):
        set_all_seeds(seed)
        plain_obs = env_test.reset(project=False)
        rewards, pv_x, pv_z, total_reward, step_nr = test_debug(agent, env_test, plain_obs, visualize=False)
        test_stats.append([rewards, pv_x, pv_z, total_reward, step_nr])
        total_reward_sum += total_reward
        if total_reward > high_reward:
            high_reward = total_reward
        if total_reward < low_reward:
            low_reward = total_reward
        print(seed, ": Test done with reward:", total_reward, "after", step_nr, "steps")
        seed += 1

    total_reward_sum /= test_nr
    print("Average reward:", total_reward_sum, "Lowest reward:", low_reward, "Highest reward:", high_reward)

    return test_stats


if __name__=='__main__':

    if False:
        # create test env
        env = create_env(train=False, render=False)

        # create agent
        model_path = './model_prosthetics_v2'
        agent = Agent(processed_dims, env.action_space, discount_factor=.96, model_path=model_path)

        model_version = 'model-20000'
        agent.load_weights(model_version)

        # run tests, takes a while
        print("Running tests, hold on...")
        test_stats = run_tests(agent, env, start_seed=1, test_nr=60)

        # save to disk
        pathname = model_version + '_tests_60.dat'
        with open(pathname, 'wb') as f:
            import pickle
            pickle.dump(test_stats, f, protocol=4)

    else:
        # load from disk
        model_version = 'model-10000'
        pathname = model_version + '_tests_60.dat'
        with open(pathname, 'rb') as f:
            import pickle
            test_stats = pickle.load(f)


    # display plot
    draw_test_rewards(test_stats)
