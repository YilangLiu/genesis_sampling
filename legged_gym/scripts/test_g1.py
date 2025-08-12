import genesis as gs 
from rsl_rl.rsl_rl.algorithms.predictive_sampling_g1 import PS_Genesis
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
import time
import copy
from tqdm import tqdm
from legged_gym.envs.g1 import g1

from legged_gym.envs.g1.g1_config import G1Cfg, G1CfgPPO

def play(args):
    gs.init(
        backend=gs.cpu if args.cpu else gs.gpu,
        logging_level='warning',
    )

    sim_env = g1.G1Env(1, dt=0.01, offscreen_cam=True)
    
    rollout_env = g1.G1Env(G1CfgPPO.planner.num_samples, dt=G1Cfg.control.dt)
    
    # Initialize Planner 
    planner = PS_Genesis(rollout_env, G1Cfg, G1CfgPPO)
    
    # since we are using sync version, the shift time will always be control dt 
    shift_time = 0.01

    initialized = False

    sim_env.reset()
    priv_obs = sim_env.get_states()

    dt = sim_env.dt
    cam = sim_env.get_offscreen_cam()
    cam.start_recording()
    with torch.inference_mode():
        for i in tqdm(range(400)):
            if not initialized:
                actions = planner.update(shift_time, priv_obs, iterations=10)
                initialized = True
            else:
                actions = planner.update(shift_time, priv_obs, iterations=2)
            sim_env.step(actions[:, 0, :].detach())
            cam.render()
            priv_obs = sim_env.get_states()

    cam.stop_recording(save_to_filename="g1.mp4", fps=int(1/dt))

if __name__ == '__main__':
    args = get_args()
    play(args)