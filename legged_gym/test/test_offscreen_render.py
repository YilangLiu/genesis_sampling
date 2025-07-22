import genesis as gs
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

def play(args):
    gs.init(
        backend=gs.cpu if args.cpu else gs.gpu,
        logging_level='warning',
    )
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()


    actions = env.default_dof_pos
    feet_names = ["FL", "FR", "RL", "RR"]
    x = np.linspace(0, 100 * env.dt, 100)
    y = []
    env.start_recording(record_internal=False)
    for i in tqdm(range(100)):
        # actions = planner.U_nom
        obs, priv_obs, rews, dones, infos = env.step(actions.unsqueeze(0))
    env.stop_recording("walking.mp4")
    exit()
    
if __name__ == '__main__':
    args = get_args()
    
    play(args)
