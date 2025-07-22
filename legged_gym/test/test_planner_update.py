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

    # create planner environment
    rollout_env_cfg = copy.deepcopy(env_cfg)
    rollout_env_cfg.env.num_envs = env_cfg.env.num_envs * planner_cfg.planner.num_samples
    rollout_env_cfg.env.force_reset = False
    rollout_env_cfg.control.decimation = 1

    # creating environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    priv_obs = env.get_privileged_observations()

    # set planner environment
    rollout_env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=rollout_env_cfg)
    
    # Initialize Planner 
    planner = MPPI_Genesis(rollout_env, env_cfg, planner_cfg)
    
    # since we are using sync version, the shift time will always be control dt 
    shift_time = env_cfg.control.dt

    actions = planner.update(shift_time, priv_obs)
    

    




if __name__ == '__main__':
    args = get_args()
    play(args)