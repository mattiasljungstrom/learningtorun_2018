import numpy as np



processed_dims = 319 + 2       # with muscles, no calcn_l
#processed_dims = 282            # without muscles data
#processed_dims = 339       # with muscles, with calcn_l


def normalize_dir(rad):
    if rad > np.pi:
        rad -= np.pi * 2.0
    elif rad < -np.pi:
        rad += np.pi * 2.0
    return rad

# expect a dictionary to come in here
# output a np array of X values
def process_observation(state_desc, print_debug=False):

    #print(state_desc)
    scale_vel = 1.0
    scale_acc = 0.0001

    res = []

    # get pelvic position, velocity, acceleration
    pelvis_pos = state_desc["body_pos"]["pelvis"][0:3]
    pelvis_vel = [state_desc["body_vel"]["pelvis"][i] * scale_vel for i in range(3)] 
    pelvis_acc = [state_desc["body_acc"]["pelvis"][i] * scale_acc for i in range(3)]

    pelvis = []
    pelvis += pelvis_pos
    pelvis += pelvis_vel
    pelvis += pelvis_acc
    pelvis += state_desc["body_pos_rot"]["pelvis"][0:3]
    pelvis += [state_desc["body_vel_rot"]["pelvis"][i] * scale_vel for i in range(3)]
    pelvis += [state_desc["body_acc_rot"]["pelvis"][i] * scale_acc for i in range(3)]


    # add pelvis
    #
    res += pelvis_pos[1:2]      # only add pelvis y-pos
    res += pelvis[3:]           # the rest


    # add body parts
    #
    # because names are hard
    # femur = lårben
    # tibia = smalben
    # talus = fotled
    # calcn_l = hälben 
    # pros = protes
    #
    for body_part in ["head","torso",
                      #"femur_l","tibia_l","calcn_l","talus_l","toes_l", 
                      "femur_l","tibia_l","talus_l","toes_l", 
                      "femur_r","pros_tibia_r","pros_foot_r"]:
        cur = []
        cur += state_desc["body_pos"][body_part][0:3]
        cur += [state_desc["body_vel"][body_part][i] * scale_vel for i in range(3)]
        cur += [state_desc["body_acc"][body_part][i] * scale_acc for i in range(3)]
        cur += state_desc["body_pos_rot"][body_part][0:3]
        cur += [state_desc["body_vel_rot"][body_part][i] * scale_vel for i in range(3)]
        cur += [state_desc["body_acc_rot"][body_part][i] * scale_acc for i in range(3)]
        # make relative to pelvis
        for i in range(len(pelvis)):
            cur[i] = cur[i] - pelvis[i]
        res += cur

    # add joints
    #
    # the joints have different nr of values, most only flexion == ~böjning
    for joint in ["ankle_l","ankle_r","back","hip_l","hip_r","knee_l","knee_r"]:
        res += state_desc["joint_pos"][joint]
        res += [(x * scale_vel) for x in state_desc["joint_vel"][joint]]
        res += [(x * scale_acc) for x in state_desc["joint_acc"][joint]]

    # add muscles
    scale_muscle_vel = 0.001
    for muscle in ['abd_l', 'abd_r', 'add_l', 'add_r', 'bifemsh_l', 'bifemsh_r', 
                    'gastroc_l', 'glut_max_l', 'glut_max_r', 'hamstrings_l', 'hamstrings_r', 
                    'iliopsoas_l', 'iliopsoas_r', 'rect_fem_l', 'rect_fem_r', 'soleus_l', 'tib_ant_l', 
                    'vasti_l', 'vasti_r']:
        res += [state_desc["muscles"][muscle]["activation"]]
        res += [state_desc["muscles"][muscle]["fiber_length"]]
        res += [(x * scale_muscle_vel) for x in [state_desc["muscles"][muscle]["fiber_velocity"]]]

    # add foot forces
    scale_forces = 0.001
    for force in ["foot_l", "pros_foot_r_0"]:
        res += [(x * scale_forces) for x in state_desc["forces"][force]]

    # add center of mass
    cm_pos = state_desc["misc"]["mass_center_pos"][0:3]
    cm_vel = state_desc["misc"]["mass_center_vel"][0:3]
    cm_acc = state_desc["misc"]["mass_center_acc"][0:3]
    res += [cm_pos[i] - pelvis_pos[i] for i in range(3)]
    res += [cm_vel[i] * scale_vel - pelvis_vel[i] for i in range(3)]
    res += [cm_acc[i] * scale_acc - pelvis_acc[i] for i in range(3)]
    

    if "target_vel" in state_desc:
        # difficulty == 1
        # get vector
        tvx = state_desc["target_vel"][0]
        tvz = state_desc["target_vel"][2]
        # need to change this to [speed, relative direction]
        target_speed = np.sqrt( tvx**2 + tvz**2 )
        target_dir = normalize_dir( np.arctan2(-tvz, tvx) )

        # use rotation instead of velocity direction
        pelvis_dir = normalize_dir( state_desc["body_pos_rot"]["pelvis"][1] )

        # calculate relative dir
        rel_dir = normalize_dir( target_dir - pelvis_dir )

        if print_debug:
            print("speed", target_speed, "dir:", target_dir, "p_dir:", pelvis_dir, "rel_dir:", rel_dir)

        # scale direction signal
        # max change would be PI/8=0.39, *5 = 1.96, *10 = 3.9
        rel_dir *= 5.0

        # make sure nothing funky happens when speed is slow
        if target_speed < 0.1:
            rel_dir = 0.0

        # use relative speed?
        res += [target_speed, rel_dir]
    else:
        # difficulty == 0
        # add wanted velocity vector
        # speed, relative direction
        res += [3.0, 0.0]

    return res

