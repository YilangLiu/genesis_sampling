import genesis as gs 
from rsl_rl.rsl_rl.algorithms.mppi import MPPI_Genesis
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
import time
import copy
from tqdm import tqdm


def play(args):
    gs.init(
        backend=gs.cpu if args.cpu else gs.gpu,
        logging_level='warning',
    )
    
    # create the agent environment
    env_cfg, planner_cfg = task_registry.get_cfgs(name=args.task)

    # creating environment
    # args.headless = False
    sim_env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    priv_obs = sim_env.get_privileged_observations()
    
    # since we are using sync version, the shift time will always be control dt 
    shift_time = env_cfg.control.dt 

    early_termination = False
    initialized = False
    with torch.inference_mode():
        sim_env.start_recording(record_internal=False)
        for i in tqdm(range(400)):
            # actions = torch.repeat_interleave(sim_env.default_dof_pos.unsqueeze(0), planner_cfg.planner.num_samples, dim=0)
            actions = sim_env.default_dof_pos.unsqueeze(0)
            obs, priv_obs, rews, dones, infos = sim_env.step(actions)
            # import pdb; pdb.set_trace()
            if dones[0] == True:
                print("env reset")
                sim_env.stop_recording("walking.mp4")
                print("early termination")
                early_termination = True
                break
        if not early_termination:
            sim_env.stop_recording("walking.mp4")

if __name__ == '__main__':
    args = get_args()
    play(args)