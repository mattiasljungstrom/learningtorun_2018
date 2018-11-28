import os
import random
import time

import threading as th
import numpy as np
import tensorflow as tf

from triggerbox import TriggerBox

from farmer import farmer as farmer_class

from create_env import create_env
from fastenv import fastenv
from observation_2018 import processed_dims
from ddpg_multi_agent import Agent
from hyper_params import hyper


if __name__=='__main__':

    # make sure save dir exists
    if not os.path.exists(hyper.model_path):
        os.makedirs(hyper.model_path)

    # init random seed
    tf.set_random_seed(hyper.seed)
    np.random.seed(hyper.seed)
    random.seed(hyper.seed)

    # create test env
    env_test = create_env(train=False, render=False)

    # create agent
    agent = Agent(processed_dims, env_test.action_space,
                  discount_factor=hyper.discount_factor, model_path=hyper.model_path)

    # noise
    noise_level = hyper.noise_start

    # remote setup
    farmer = None
    def setup_farmer():
        global farmer
        if farmer==None:
            farmer = farmer_class()

    # stop button
    stop_signal = False
    def stop():
        global stop_signal
        stop_signal = True
    stop_button = TriggerBox('Stop Training', ['stop training'], [stop])


    def train_episode(noise_level, env, ac_id):
        try:
            agent.run_episode(env, training=True, render=False, max_steps=-1, noise_level=noise_level, ac_id=ac_id)
        except Exception as ex:
            pass
        if env!=None:
            env.rel()

    def train_if_available(noise_level, ac_id):
        while True:
            remote_env = farmer.acq_env()
            if remote_env == False:
                # no free environment
                time.sleep(0.5)
            else:
                break
        try:
            t = th.Thread(target=train_episode, args=(noise_level, remote_env, ac_id), daemon=True)
            t.start()
        except Exception as ex:
            pass


    def test_episode():
        # want to run with visualize=False here
        # this doesn't run on a background process (but still in a thread)
        fenv = fastenv(env_test, 1)
        agent.run_episode(fenv, training=False, render=False, max_steps=-1, noise_level=0.0, ac_id=-1)
        del fenv

    def test_background():
        t = th.Thread(target=test_episode, args=(), daemon=True)
        t.start()
        

    def run_training(episodes):
        global noise_level, stop_signal

        episode_start = agent.global_step
        start_time = time.time()

        active_id = 0

        for i in range(episodes):
            if stop_signal:
                stop_signal = False
                print('Stop signal received, stopping...')
                break

            noise_level *= (1.0 - hyper.noise_decay)
            noise_level = max(hyper.noise_floor, noise_level)
            noise = noise_level if np.random.uniform()>0.10 else hyper.no_noise

            print(episode_start+i+1, ': Starting episode: ', i+1 , '/' , episodes, ' noise_level: ', noise)
            train_if_available(noise, active_id)

            active_id += 1
            if active_id >= agent.nr_networks:
                active_id = 0

            time.sleep(0.5)

            if (i+1) % hyper.swap_ac_interval == 0:
                # switch AC pairs around
                print("Swapping actors")
                agent.swap_actors()

            if (i+1) % hyper.test_interval == 0:
                test_background()

            if (i+1) % hyper.save_interval == 0:
                # save the model.
                save()

            if (i+1) % hyper.plot_interval == 0:
                plot()

        train_time = time.time()-start_time
        print("Completed in:", train_time / (60*60), "hours")

    # if we have UI running the updates need be on main thread, run this
    # or thing will crash, losing results
    # note: memory is not saved out automatically because it's too damn slow
    def r(episodes):
        setup_farmer()
        agent.setup_memory()
        run_training(episodes)

    # if you're running on a server without showing buttons/ui this enables you
    # to stop training gracefully by calling stop()
    # you can also show stats by running agent.history.print_last_tests(25)
    def rb(episodes):
        setup_farmer()
        agent.setup_memory()
        t = th.Thread(target=run_training, args=(episodes,))
        t.start()

    # collect experience without training and then start training normally
    def collect_data_and_train(collect, episodes):
        setup_farmer()
        agent.setup_memory()
        agent.block_training = True
        run_training(collect)
        agent.block_training = False
        run_training(episodes)

    # only train, no new experience
    def run_train(episodes):
        global stop_signal
        for i in range(episodes):
            if stop_signal:
                stop_signal = False
                print('Stop signal received, stopping...')
                break
            print(agent.global_step, ': episode:', i)
            for k in range(500):
                print('.', end='', flush=True)
                if (k+1)%80==0:
                    print('')
                for j in range(agent.nr_networks):
                    agent.train_batch(j)
            print('')
            agent.global_step += 1

            if (i+1) % 100 == 0:
                save()


    def plot():
        agent.history.plot()


    def save():
        agent.save_weights()
        agent.history.save()

    def save_memory():
        agent.memory.save(hyper.model_path + '/memory.pickle')


    def load_model(model=""):
        agent.load_weights(model)

    def load():
        global noise_level
        agent.load_weights()
        agent.history.load()
        noise_level = agent.history.get_last_noise_level(hyper.no_noise)
        agent.global_step = agent.history.get_train_count()
        print('noise_level:', noise_level)
        print('global_step:', agent.global_step)
        agent.history.plot()
        agent.setup_memory()
        agent.memory.load(hyper.model_path + '/memory.pickle')

