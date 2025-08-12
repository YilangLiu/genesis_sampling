from legged_gym.envs import * 
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgSample
from scipy.interpolate import make_interp_spline
import torch 
import numpy as np 
import matplotlib.pyplot as plt
from abc import ABC
from torch.distributions.multivariate_normal import MultivariateNormal
from rsl_rl.storage import RolloutStorage
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline
import time

class dial_Genesis(ABC):
    def __init__(self, env: LeggedRobot, env_cfg: LeggedRobotCfg, planner_cfg: LeggedRobotCfgSample):
        self.rollout_envs = env
        self.env_cfg = env_cfg
        self.planner_cfg = planner_cfg
        self.sample_method = planner_cfg.planner.sampling_method
        self.num_samples_per_env = planner_cfg.planner.num_samples
        self.num_knots = planner_cfg.planner.num_knots
        self.num_agent_envs =  env_cfg.env.num_envs
        self.num_planner_envs = self.num_agent_envs * self.num_samples_per_env # Important
        self.device = planner_cfg.planner.device
        self.T = planner_cfg.planner.horizon
        self.nu = env_cfg.env.num_actions
        self.U_nom = torch.zeros((self.num_agent_envs, self.num_knots, self.nu), device=self.device) 
        # self.U_nom += self.rollout_envs.default_dof_pos.reshape((1, 1, self.nu))
        self.action_sequence = torch.zeros((self.num_agent_envs, self.T, self.nu), device=self.device) 
        self.action_sequence += self.rollout_envs.default_dof_pos.reshape((1, 1, self.nu))
        self.noise_sigma = torch.eye(self.nu, device=self.device)
        self.noise_mu = torch.zeros(self.nu, device=self.device)
        self.noise_dist = MultivariateNormal(self.noise_mu, self.noise_sigma)
        self.num_dof = self.rollout_envs.num_actions
        self.storage = None
        self.transition = RolloutStorage.Transition()
        self.init_storage(
            self.num_planner_envs,
            self.T,
            [self.rollout_envs.num_obs],
            [None],
            [self.rollout_envs.num_actions]
        )
        self.ctrl_dt = env_cfg.control.dt 
        self.ctrl_steps = torch.linspace(0, self.T * self.ctrl_dt, self.T, dtype=torch.float32, device=self.device)
        self.knots_steps = torch.linspace(0, self.T * self.ctrl_dt, self.num_knots, dtype=torch.float32, device=self.device)

    def init_storage(self, num_envs, num_horizon_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_horizon_per_env, actor_obs_shape,  critic_obs_shape, action_shape, self.device)

    def reset_plan(self):
        # reset environment to the nominal plan
        self.U_nom = torch.zeros((self.num_agent_envs, self.num_knots, self.nu), device=self.device) 
        # self.U_nom += self.rollout_envs.default_dof_pos.reshape((1, 1, self.nu))

    def control_interpolations(self, x: torch.Tensor, y: torch.Tensor, interp_array: torch.Tensor):
        # torch_array (num_samples, num_knots, num_actions)
        coeffs = natural_cubic_spline_coeffs(x, y)
        spline = NaturalCubicSpline(coeffs)
        return spline.evaluate(interp_array) 

    # def reverse_once(self, )

    def update(self, shift_time, priv_obs, iterations = 3):
        """
        Get the optimized action of all environments.

        Args:
            actions (torch.Tensor): A 3d Tensor from the agent environments capturing the current action plan 
            for all agent environments. Shape [num_agent_envs, T, nu]
            
            priv_state (torch.Tensor): A 2d Tensor from the agent environment including the XYZ base position, 
            base orientation in quaternion, base_lin_velocities, base_ang_velocities, joint positions in radian, 
            joint velocities in radian/second. Shape [num_agent_envs, 37]

        Returns:
            actions (Tuple[torch.Tensor, torch.Tensor]):
                The optimized action plan for the agent environments to apply. Shape [num_agent_envs, T, nu]
        """
        
        # convert shift time
        shift_time = torch.tensor(shift_time, dtype=torch.float32, device=self.device)
        
        # Shift actions forward in sync mode 
        self.U_nom = self.control_interpolations(self.knots_steps, self.U_nom, self.knots_steps + shift_time)

        for _ in range(iterations):
            # set rollout environment according to the agent environment
            self.rollout_envs.set_sim(
                torch.repeat_interleave(priv_obs, self.num_samples_per_env, dim=0)
                )
        
            # Repeat the current action plan from the agent environments since we need to sample
            actions_sampled = torch.repeat_interleave(self.U_nom, self.num_samples_per_env, dim=0)

            # Sample Gaussian Noise and clip this to [-1, 1]
            # noise = torch.clip(self.sample_noise() * self.planner_cfg.planner.sample_noise, -1, 1)
            noise = self.sample_noise() * self.planner_cfg.planner.sample_noise

            # Add noise to the sampled action plan
            U_sampled = torch.clip(actions_sampled + noise, -1, 1)

            # Upsample the sampled actions to match the actual control sequences 
            U_sampled = self.control_interpolations(self.knots_steps, U_sampled, self.ctrl_steps)

            # Clip U_sampled to the action range
            # U_sampled = torch.clip(U_sampled, self.rollout_envs.default_dof_pos_low, self.rollout_envs.default_dof_pos_high)

            # apply rollout 
            # now = time.time()
            rewards = self.rollout(U_sampled).view(-1, self.num_samples_per_env)
            # print("Rollout time: ", time.time() - now)

            # For every consecutive sampled environments, find the idxes that maximize the cumulative reward
            best_idxs = torch.argmax(rewards, dim=1) + torch.arange(0, self.num_planner_envs, 
                                                                    self.num_samples_per_env, 
                                                                    device=self.device)

            # Clear the storage to for next iteration
            self.storage.clear()

            # Downsample back to the U_nom for next iteration
            self.U_nom[:] = self.control_interpolations(self.ctrl_steps, U_sampled[best_idxs], self.knots_steps)  

        # return the optimized action sequence for the agent environments
        return U_sampled[best_idxs]

    def rollout(self, actions):
        """
        Rollout the envs given the actions sampled. Notice the planner environment will not reset automatically. 
        In other words, we need to record the reset buff and stop adding rewards after environments are reset 
        Additionally, the paralleled environments are stacked as a blcok

        Args:
            actions (torch.Tensor): sampled action plan. Shape [num_planner_envs, T, nu]
        
        Returns:
            rewards (torch.Tensor): the cumulative reward after rolling out the system dynamics. shape [num_planner_envs]
        """
        
        curr_reset_env_ids = torch.zeros(self.num_planner_envs, device= self.device, dtype=torch.int64)
        for i in range(self.T):
            obs, privileged_obs, rewards, dones, infos = self.rollout_envs.step(actions[:, i, :])
            obs, rewards, dones = obs.to(self.device), rewards.to(self.device), dones.to(self.device)
            curr_reset_env_ids |= dones.to(torch.int64)
            self.transition.observations = obs
            self.transition.critic_observations = privileged_obs
            self.transition.actions = actions[:, i, :]
            self.transition.actions_log_prob = torch.zeros((self.num_planner_envs, 1), device = self.device)
            self.transition.action_mean = actions[:, i, :]
            self.transition.action_sigma = torch.ones_like(actions[:, i, :])
            self.transition.rewards = rewards.clone()
            self.transition.dones = curr_reset_env_ids
            self.transition.values = torch.zeros((self.num_planner_envs, 1), device = self.device)
            self.storage.add_transitions(self.transition)
            self.transition.clear()

        self.storage.compute_returns(last_values=torch.zeros((self.num_planner_envs, 1), device = self.device),
                                    gamma=1,
                                    lam=1)
        
        # Return the cumulative sum of rewards at step 0 
        return self.storage.returns[0].to(self.device)
    
    def sample_noise(self):
        """
        Sample Gaussian Noise with mean zeros and variances are specified by the config file. Notice 
        here num_envs = env_cfg.env.num_envs * self.num_samples_per_env

        Returns:
            noises (torch.Tensor): the sampled Gaussian noise with shape [num_planner_envs, num_knots, self.nu]. 
        """
        noise = self.noise_dist.sample((self.num_planner_envs, self.num_knots))
        # the first noise will zeros
        noise[0] = torch.zeros((self.num_knots, self.nu), dtype=torch.float32, device= self.device)
        return noise
    