import genesis as gs
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
import time


def play(args):
    gs.init(
        backend=gs.cpu if args.cpu else gs.gpu,
        logging_level='warning',
    )
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 2)
    env_cfg.env.debug_viz = True
    env_cfg.viewer.add_camera = False  # use a extra camera for moving
    env_cfg.depth.render_window = False
    env_cfg.terrain.plane_length = 5
    env_cfg.terrain.border_size = 5
    env_cfg.terrain.num_rows = 2
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = True
    env_cfg.terrain.selected = False
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.asset.fix_base_link = False
    # initial state randomization
    env_cfg.init_state.yaw_angle_range = [0., 0.]
    # velocity range
    env_cfg.commands.ranges.lin_vel_x = [0.5, 1.0]
    env_cfg.commands.ranges.lin_vel_y = [0., 0.]
    env_cfg.commands.ranges.ang_vel_yaw = [0., 0.]
    env_cfg.commands.ranges.heading = [0, 0]

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    actions = torch.zeros((env_cfg.env.num_envs, 12), device="cuda")
    for i in range(10*int(env.max_episode_length)):
        obs, _, rews, dones, infos = env.step(actions.detach())

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False  # only record frames in extra camera view
    MOVE_CAMERA   = False
    FOLLOW_ROBOT  = False
    assert not (MOVE_CAMERA and FOLLOW_ROBOT), "Cannot move camera and follow robot at the same time"
    args = get_args()
    
    play(args)
