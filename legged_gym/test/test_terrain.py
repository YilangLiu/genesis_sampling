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
    env_1, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env_1.get_observations()

    env_cfg.env.num_envs *= 2

    env_2, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env_2.get_observations()
    
    import pdb; pdb.set_trace()

    torch.testing.assert_close(env_1.height_samples, env_2.height_samples)
    torch.testing.assert_close(env_1.utils_terrain.height_field_raw,  env_2.utils_terrain.height_field_raw)

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False  # only record frames in extra camera view
    MOVE_CAMERA   = False
    FOLLOW_ROBOT  = False
    assert not (MOVE_CAMERA and FOLLOW_ROBOT), "Cannot move camera and follow robot at the same time"
    args = get_args()
    
    play(args)
