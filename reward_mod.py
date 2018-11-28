import numpy as np

from osim.env import ProstheticsEnv

from observation_2018 import normalize_dir
from hyper_params import hyper


def calc_ang_to_target(rot_y, target_v):
    pv = np.array([np.cos(rot_y), -np.sin(rot_y)])
    cos_ang = np.dot(target_v, pv)
    ang = np.arccos(cos_ang)
    return ang


# femur = lårben
# tibia = smalben
# talus = fotled
# calcn_l = hälben 
# pros = protes

def calc_extra_reward(state_desc):
    reward_extra = 0

    # penalty for not bending knees
    for joint in ["knee_l","knee_r"]:
        # make this a penalty instead of a reward: range approx [-4, 0.3] * X
        # negative is bending correct way
        penalty = (-1.0 * state_desc["joint_pos"][joint][0] - 0.5) * 0.5
        if penalty > 0.0:
            penalty = 0.0
        reward_extra += penalty

    # reduce points if the femur is far away from foot in z-axis
    # depend on movement direction
    # target vector
    tv = np.array([3.0, 0.0])
    difficulty = 0
    if "target_vel" in state_desc:
        difficulty = 1
        tv[0] = state_desc["target_vel"][0]
        tv[1] = state_desc["target_vel"][2]
    # normalize
    tv_len = np.linalg.norm(tv)
    if tv_len > 0.0:
        tv = tv / tv_len
    else:
        tv = np.array(1.0, 0.0)

    side = np.array([tv[1]*-1.0 , tv[0]])     # rotate 90 CCW
    #print("side:", side)

    # keep left foot under upper leg
    femur_l_pos = np.array([state_desc["body_pos"]["femur_l"][i] for i in (0,2)])
    toes_l_pos = np.array([state_desc["body_pos"]["toes_l"][i] for i in (0,2)])
    diff_l_v = femur_l_pos - toes_l_pos
    diff_l = np.dot(side, diff_l_v)
    reward_extra -= (diff_l ** 2) * 5.0                                # max is around -3.0 with * 10.0
    #print("left diff:", diff_l)

    # keep right foot under upper leg
    femur_r_pos = np.array([state_desc["body_pos"]["femur_r"][i] for i in (0,2)])
    foot_r_pos = np.array([state_desc["body_pos"]["pros_foot_r"][i] for i in (0,2)])
    diff_r_v = femur_r_pos - foot_r_pos
    diff_r = np.dot(side, diff_r_v)
    reward_extra -= (diff_r ** 2) * 5.0                                # max is around -4.0 with * 10.0
    #print("right diff:", diff_r)

    # orient pelvis forward => avoid weird sideways running
    if difficulty==0:
        pelvis_rot_y = state_desc["body_pos_rot"]["pelvis"][1]
        reward_extra -= (pelvis_rot_y ** 2) * 10.0                      # max is around -4.0 with * 10.0 (90 sideways)
    else:
        pelvis_rot_y = state_desc["body_pos_rot"]["pelvis"][1]
        ang = calc_ang_to_target(pelvis_rot_y, tv)
        pelvis_dir_penalty = (ang ** 2) * 20.0
        #print(ang, pelvis_dir_penalty)
        reward_extra -= pelvis_dir_penalty


    # feet direction in target direction => encourage actual turning instead of sideways running
    if difficulty > 0:
        toes_l_rot_y = state_desc["body_pos_rot"]["toes_l"][1]
        tl_ang = calc_ang_to_target(toes_l_rot_y, tv)
        tl_penalty = (tl_ang ** 2) * 20.0
        reward_extra -= tl_penalty

        foot_r_rot_y = state_desc["body_pos_rot"]["pros_foot_r"][1]
        fr_ang = calc_ang_to_target(foot_r_rot_y, tv)
        fr_penalty = (fr_ang ** 2) * 20.0
        reward_extra -= fr_penalty
        #print("tv:", tv, "feet dir penalty (left):", tl_ang, tl_penalty, "(right):", fr_ang, fr_penalty)


    # make feets be on correct side of each other
    # get distance projected on side vector
    feet_v = foot_r_pos - toes_l_pos
    diff_feet = np.dot(side, feet_v)
    #print("diff feet:", diff_feet)
    if diff_feet < 0.176:
        # crossing feet, this is extra bad
        reward_extra -= (diff_feet - 0.176) ** 2 * 100.0
    else:        
        reward_extra -= (diff_feet - 0.176) ** 2 * 10.0


    '''
    # make left foot flat to the ground as the foot is set down
    #"calcn_l","talus_l","toes_l"
    foot_l1_y = state_desc["body_pos"]["calcn_l"][1]
    #foot_l1_yb = state_desc["body_pos"]["talus_l"][1]
    foot_l2_y = state_desc["body_pos"]["toes_l"][1]
    # the initial diff is 0.002
    foot_l_y_diff = (foot_l1_y - foot_l2_y) - 0.002
    #print("foot_l1_y", foot_l1_y, "foot_l1_yb", foot_l1_yb, "foot_l2_y", foot_l2_y, "foot_l_y_diff", foot_l_y_diff)
    # make penalty smaller when foot is in air
    foot_l_y_min = max(0.0, min(foot_l1_y, foot_l2_y))      # >= 0
    foot_l_mod = max(0.0, 1.0 - foot_l_y_min*8.0)          # >= 0
    foot_l_penalty = (foot_l_y_diff ** 2) * 100 * foot_l_mod
    #print("foot_l_y_min", foot_l_y_min, "foot_l_mod", foot_l_mod, "foot_l_penalty", foot_l_penalty)
    #print("foot_l_penalty", foot_l_penalty)
    reward_extra -= foot_l_penalty
    '''

    # cap penalty, trying to keep final reward in a range of [-11,10]
    max_penalty = -9
    #if difficulty > 0:
    #    max_penalty = -10

    if reward_extra < max_penalty:
        reward_extra = max_penalty
    if reward_extra > 0.0:
        reward_extra = 0.0

    #print("reward_extra", reward_extra)
    return reward_extra


def bind_alt_reward(renv):
    def reward_round2(self):
        state_desc = self.get_state_desc()
        prev_state_desc = self.get_prev_state_desc()
        penalty = 0
        # Small penalty for too much activation (cost of transport)
        penalty += np.sum(np.array(self.osim_model.get_activations())**2) * 0.001
        # Big penalty for not matching the vector on the X,Z projection.
        # No penalty for the vertical axis
        penalty += (state_desc["body_vel"]["pelvis"][0] - state_desc["target_vel"][0])**2
        penalty += (state_desc["body_vel"]["pelvis"][2] - state_desc["target_vel"][2])**2
        # Reward for not falling
        #print("penalty", penalty, state_desc["body_vel"]["pelvis"][0], state_desc["target_vel"][0])
        #reward = 1.25**2
        reward = 10
        # the penalty above isn't punishing enough, the agent is ok with scoring positive
        # balance it around zero for starting speed
        penalty *= 8
        if penalty > 12:
            penalty = 12
        #print("penalty", penalty)
        return reward - penalty

    def reward(self):
        state_desc = self.get_state_desc()
        extra_reward = calc_extra_reward(state_desc)
        if self.difficulty == 0:
            return self.reward_round1() + extra_reward
        return self.reward_round2() + extra_reward
    import types
    renv.reward_round2 = types.MethodType(reward_round2, renv)
    renv.reward = types.MethodType(reward, renv)


def bind_alt_reset(renv):
    def reset(self, project = True):
        # make episodes shorter
        self.time_limit = hyper.env_time_step_limit
        self.spec.timestep_limit = self.time_limit
        # turn every 60s step => 4 turns in 300 steps
        # turn every 75s step => 3 turns 
        self.generate_new_targets(poisson_lambda = hyper.env_poisson_lambda)
        return super(ProstheticsEnv, self).reset(project = project)
    import types
    renv.reset = types.MethodType(reset, renv)


def print_target_changes(env):
    t_speed_last = 0
    t_dir_last = 0
    for i, t in enumerate(env.targets):
        tvx = t[0]
        tvz = t[2]
        t_speed = np.sqrt( tvx**2 + tvz**2 )
        t_dir = normalize_dir( np.arctan2(-tvz, tvx) )
        if t_speed_last!=t_speed or t_dir_last!=t_dir:
            t_speed_last = t_speed
            t_dir_last = t_dir
            print(i, ':', t_speed, ',', t_dir, ',', t_dir/(2.0*np.pi)*360.0)


def get_target_speed_heading(env):
    nsteps = len(env.targets)
    speed = np.zeros(nsteps)
    heading = np.zeros(nsteps)
    for i, t in enumerate(env.targets):
        tvx = t[0]
        tvz = t[2]
        t_speed = np.sqrt( tvx**2 + tvz**2 )
        t_heading = normalize_dir( np.arctan2(-tvz, tvx) )      # z is other way
        speed[i] = t_speed
        heading[i] = t_heading
    return speed, heading
