import genesis as gs
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
import time
from tqdm import tqdm

def play(args):
    gs.init(
        backend=gs.cpu if args.cpu else gs.gpu,
        logging_level='warning',
    )
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    actions = torch.repeat_interleave(env.default_dof_pos.unsqueeze(0), env_cfg.env.num_envs, dim=0)
    actions = torch.rand(actions.shape)
    for i in tqdm(range(10*int(env.max_episode_length))): 
        obs, priv_obs, rews, dones, infos = env.step(actions)
        import pdb; pdb.set_trace()
        
if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False  # only record frames in extra camera view
    MOVE_CAMERA   = False
    FOLLOW_ROBOT  = False
    assert not (MOVE_CAMERA and FOLLOW_ROBOT), "Cannot move camera and follow robot at the same time"
    args = get_args()
    
    play(args)
