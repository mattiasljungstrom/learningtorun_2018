
from osim.env import ProstheticsEnv
from fastenv import fastenv
from reward_mod import bind_alt_reward
from reward_mod import bind_alt_reset
from hyper_params import hyper

difficulty = 1

def create_env(train=False, render=False):
    if train:
        renv = ProstheticsEnv(visualize=False)
        renv.change_model(model='3D', prosthetic=True, difficulty=difficulty, seed=None)
        bind_alt_reward(renv)
        bind_alt_reset(renv)
        env = fastenv(renv, skipcount=hyper.step_skip)
        return env
    else:
        if render:
            # create visual env
            env_render = ProstheticsEnv(visualize=True)
            env_render.change_model(model='3D', prosthetic=True, difficulty=difficulty, seed=None)
            return env_render
        else:
            # create test env
            env_test = ProstheticsEnv(visualize=False)
            env_test.change_model(model='3D', prosthetic=True, difficulty=difficulty, seed=None)
            return env_test
