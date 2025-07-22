import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat, rotvec_to_R
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver
from genesis.engine.solvers.avatar_solver import AvatarSolver
from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
import numpy as np
import os

import torch, torchvision
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.math import wrap_to_pi, torch_rand_sqrt_float, quat_apply_yaw
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.gs_utils import *
from .legged_robot_config import LeggedRobotCfg
from legged_gym.utils.gait_scheduler import get_foot_step

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.height_samples = None
        self.debug_viz = self.cfg.env.debug_viz
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_device, headless)

        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True
        self.global_counter = 0
        self.resize_transform = torchvision.transforms.Resize((self.cfg.depth.resized[1], self.cfg.depth.resized[0]), 
                                                              interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.post_physics_step()
        

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        if self.cfg.sim.use_implicit_controller: # use embedded pd controller
            target_dof_pos = self._compute_target_dof_pos(exec_actions)
            self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
            self.scene.step()
        if self.cfg.sim.use_dial_controller:
            for _ in range(self.cfg.control.decimation):
                self.torques = self.act2tau(exec_actions)
                if self.num_build_envs == 0:
                    torques = self.torques.squeeze()
                    self.robot.control_dofs_force(torques, self.motor_dofs)
                else:
                    self.robot.control_dofs_force(self.torques, self.motor_dofs)
                self.scene.step()
                self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
                self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)
        else:
            for _ in range(self.cfg.control.decimation): # use self-implemented pd controller
                self.torques = self._compute_torques(exec_actions)
                if self.num_build_envs == 0:
                    torques = self.torques.squeeze()
                    self.robot.control_dofs_force(torques, self.motor_dofs)
                else:
                    self.robot.control_dofs_force(self.torques, self.motor_dofs)
                self.scene.step()
                self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
                self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)
        
        self.global_counter += 1
        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.post_physics_step()
        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        # update depth image if specified
        if self.cfg.depth.use_camera and self.global_counter % self.cfg.depth.update_interval == 0:
            self.extras["depth"] = self.depth_buffer[:, -2]
        else:
            self.extras["depth"] = None 


        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """

        # prepare quantities
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        base_quat_rel = gs_quat_mul(self.base_quat, gs_inv_quat(self.base_init_quat.reshape(1, -1).repeat(self.num_envs, 1)))
        self.base_euler = gs_quat2euler(base_quat_rel)
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat) # trasform to base frame
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)
        self.roll, self.pitch, self.yaw = self.base_euler[:,0], self.base_euler[:, 1], self.base_euler[:,2]
        
        self.foot_positions[:] = self.rigid_solver.get_links_pos(self.feet_link_indices_world_frame) - self.foot_radius
        self.foot_quaternions[:] = self.rigid_solver.get_links_quat(self.feet_link_indices_world_frame)
        self.foot_velocities[:] = self.rigid_solver.get_links_vel(self.feet_link_indices_world_frame)

        self.link_contact_forces[:] = torch.tensor(
            self.robot.get_links_net_contact_force(),
            device=self.device,
            dtype=gs.tc_float,
        )

        # rampup the lin_vel command gradually from the initial episode
        self.lin_vel_tar[:] = torch.minimum(
            self.commands[:, :2] * self.episode_length_buf.unsqueeze(1) * self.dt / self.cfg.commands.ramp_up_time,
            self.commands[:, :2]
            )
        
        self._update_goals()
        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_base_pos_out_of_bound()
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if self.num_build_envs > 0 and self.cfg.env.force_reset:
            self.reset_idx(env_ids)

        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)

        self.update_depth_buffer()
        
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.llast_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        
        if gs.platform != 'macOS':
            self._render_headless()
        if self.debug_viz:
            self._draw_debug_vis()
            
    
    def check_base_pos_out_of_bound(self):
        """ Check if the base position is out of the terrain bounds
        """
        x_out_of_bound = (self.base_pos[:, 0] >= self.terrain_x_range[1]) | (self.base_pos[:, 0] <= self.terrain_x_range[0])
        y_out_of_bound = (self.base_pos[:, 1] >= self.terrain_y_range[1]) | (self.base_pos[:, 1] <= self.terrain_y_range[0])
        out_of_bound_buf = x_out_of_bound | y_out_of_bound
        envs_idx = out_of_bound_buf.nonzero(as_tuple=False).flatten()
        # reset base position to initial position
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_pos[envs_idx] += self.env_origins[envs_idx]
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.link_contact_forces[:, self.termination_indices, :], dim=-1)> 1.0, dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        roll_cutoff = torch.abs(self.roll) > 1.5
        pitch_cutoff = torch.abs(self.pitch) > 1.5

        self.reset_buf |= self.time_out_buf
        self.reset_buf |= roll_cutoff
        self.reset_buf |= pitch_cutoff
        
    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """

        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        # if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
        #     self.update_command_curriculum(env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        # domain randomization
        if self.cfg.domain_rand.randomize_friction:
            self._randomize_friction(env_ids)
        if self.cfg.domain_rand.randomize_base_mass:
            self._randomize_base_mass(env_ids)
        if self.cfg.domain_rand.randomize_com_displacement:
            self._randomize_com_displacement(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.cur_goal_idx[env_ids] = 0
        self.reach_goal_timer[env_ids] = 0
        
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,                    # 3
                                    self.base_ang_vel  * self.obs_scales.ang_vel,                   # 3
                                    self.projected_gravity,                                         # 3
                                    self.commands[:, :3] * self.commands_scale,                     # 3
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,# num_dofs
                                    self.dof_vel * self.obs_scales.dof_vel,                         # num_dofs
                                    self.actions                                                    # num_actions
                                    ),dim=-1)
        
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.base_pos[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
        
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf[:] = torch.cat(
                (   self.base_pos,  # 3
                    self.base_quat, # 4
                    self.base_lin_vel, # 3
                    self.base_ang_vel, # 3
                    self.dof_pos, # 12    
                    self.dof_vel, # 12   
                    self.commands, # 4  
                    self.last_actions, # 12  
                    self.last_dof_vel, # 12
                    self.feet_air_time, # 4
                    torch.tensor(self.episode_length_buf, device=self.device, dtype=gs.tc_float).unsqueeze(1), # 1
                    torch.tensor(self.reset_buf, device=self.device, dtype=gs.tc_float).unsqueeze(1), # 1
                    torch.tensor(self.cur_goal_idx, device=self.device, dtype=gs.tc_float).unsqueeze(1),
                    self.reach_goal_timer.unsqueeze(1),
                    self.terrain_levels.unsqueeze(1).float(), # 1
                    self.terrain_types.unsqueeze(1).float()
                ),
                dim=-1,
            )
        
    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.sim_dt, 
                substeps=self.sim_substeps),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(1 / self.dt * self.cfg.control.decimation),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx = list(range(min(self.cfg.viewer.num_rendered_envs, self.num_envs)))),
            rigid_options=gs.options.RigidOptions(
                dt=self.sim_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                enable_self_collision=self.cfg.asset.self_collisions,
            ),
            show_viewer= not self.headless,
        )
        # query rigid solver
        for solver in self.scene.sim.solvers:
            if not isinstance(solver, RigidSolver):
                continue
            elif isinstance(solver, AvatarSolver):
                continue
            self.rigid_solver = solver
            
        # add camera if needed
        if self.cfg.viewer.add_camera:
            self._setup_camera()

        # add depth camera if needed 
        if self.cfg.depth.use_camera:
            self._setup_depth_cameras()
        
        # add terrain
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type=='plane':
            self.terrain = self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
        elif mesh_type=='heightfield':
            self.utils_terrain = Terrain(self.cfg.terrain, self.num_envs)
            self._create_heightfield()

        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self.terrain.set_friction(self.cfg.terrain.friction)
        # specify the boundary of the heightfield
        self.terrain_x_range = torch.zeros(2, device=self.device)
        self.terrain_y_range = torch.zeros(2, device=self.device)
        if self.cfg.terrain.mesh_type=='heightfield':
            self.terrain_x_range[0] = -self.cfg.terrain.border_size + 1.0 # give a small margin(1.0m)
            self.terrain_x_range[1] = self.cfg.terrain.border_size + self.cfg.terrain.num_rows * self.cfg.terrain.terrain_length - 1.0
            self.terrain_y_range[0] = -self.cfg.terrain.border_size + 1.0
            self.terrain_y_range[1] = self.cfg.terrain.border_size + self.cfg.terrain.num_cols * self.cfg.terrain.terrain_width - 1.0
        elif self.cfg.terrain.mesh_type=='plane': # the plane used has limited size, 
                                                  # and the origin of the world is at the center of the plane
            self.terrain_x_range[0] = -self.cfg.terrain.plane_length/2+1
            self.terrain_x_range[1] = self.cfg.terrain.plane_length/2-1
            self.terrain_y_range[0] = -self.cfg.terrain.plane_length/2+1 # the plane is a square
            self.terrain_y_range[1] = self.cfg.terrain.plane_length/2-1

        self._create_envs()

        self._attach_depth_cameras()

    def normalize_depth_image(self, depth_image):
        depth_image = depth_image * -1
        depth_image = (depth_image - self.cfg.depth.near_clip) / (self.cfg.depth.far_clip - self.cfg.depth.near_clip)  - 0.5
        return depth_image
    
    def process_depth_image(self, depth_image):
        # These operations are replicated on the hardware
        depth_image = self.crop_depth_image(depth_image)
        depth_image += self.cfg.depth.dis_noise * 2 * (torch.rand(1)-0.5)[0]
        depth_image = torch.clip(depth_image, -self.cfg.depth.far_clip, -self.cfg.depth.near_clip)
        depth_image = self.resize_transform(depth_image[None, :]).squeeze()
        depth_image = self.normalize_depth_image(depth_image)
        return depth_image
    
    def crop_depth_image(self, depth_image):
        # crop 30 pixels from the left and right and and 20 pixels from bottom and return croped image
        return depth_image[:-2, 4:-4]


    def update_depth_buffer(self):
        if not self.cfg.depth.use_camera:
            return
        
        if self.global_counter % self.cfg.depth.update_interval != 0:
            return 

        for i, depth_cam in enumerate(self.depth_cameras):
            depth_image_ = depth_cam.render(rgb=True, depth=True)[1]
            depth_image = torch.tensor(depth_image_, dtype=gs.tc_float, device=self.device, requires_grad=False)
            depth_image = self.process_depth_image(depth_image)
            init_flat = self.episode_length_buf <= 1
            if init_flat[i]:
                self.depth_buffer[i] = torch.stack([depth_image] * self.cfg.depth.buffer_len, dim=0)
            else:
                self.depth_buffer[i] = torch.cat([self.depth_buffer[i, 1:], depth_image.to(self.device).unsqueeze(0)], dim=0)
    
    def set_sim(self, priv_state):
        # set robot state. refer to compute obersevations for state format
        self.base_pos[:] = priv_state[:, :3]
        self.robot.set_pos(self.base_pos, zero_velocity=False)
        self.base_quat[:] = priv_state[:, 3:7]
        # set base orientation
        self.robot.set_quat(self.base_quat, zero_velocity=False)
        # set base linear and angular velocities
        # self.robot.zero_all_dofs_velocity()

        # update projected gravity
        inv_base_quat = gs_inv_quat(self.base_quat)
        self.projected_gravity = gs_transform_by_quat(self.global_gravity, inv_base_quat)

        # set base linear and angular velocities
        self.base_lin_vel[:] = priv_state[:, 7:10]
        self.base_ang_vel[:] = priv_state[:, 10:13]
        base_vel = torch.concat([self.base_lin_vel, self.base_ang_vel], dim=1)
        self.robot.set_dofs_velocity(velocity=base_vel, dofs_idx_local=[0, 1, 2, 3, 4, 5])
        
        # set dof positions and velocities
        self.dof_pos[:] = priv_state[:, 13:25]
        self.dof_vel[:] = priv_state[:, 25:37]
        self.robot.set_dofs_position(position=self.dof_pos, dofs_idx_local=self.motor_dofs, zero_velocity=False)
        self.robot.set_dofs_velocity(velocity=self.dof_vel, dofs_idx_local=self.motor_dofs)

        # set commands 
        self.commands[:] = priv_state[:, 37:41]

        # set buffers for reward computation 
        self.last_actions[:] = priv_state[:, 41:53]
        self.last_dof_vel[:] = priv_state[:, 53:65]
        self.feet_air_time[:] = priv_state[:, 65:69]
        self.episode_length_buf[:] = torch.tensor(priv_state[:, 69], device=self.device, dtype=gs.tc_int)
        self.reset_buf[:] = torch.tensor(priv_state[:, 70], device=self.device, dtype=gs.tc_int)
        self.cur_goal_idx[:] = torch.tensor(priv_state[:, 71], device=self.device, dtype=torch.long)
        self.reach_goal_timer[:] = priv_state[:, 72]
        # set terrain goals 
        self.terrain_levels[:] = torch.tensor(priv_state[:, 73], device=self.device, dtype=torch.int32)
        self.terrain_types[:] = torch.tensor(priv_state[:, 74], device=self.device, dtype=torch.int32)
        
        self.env_origins[:] = self.terrain_origins[self.terrain_levels[:], self.terrain_types[:]]
        temp = self.terrain_goals[self.terrain_levels, self.terrain_types]
        last_col = temp[:, -1].unsqueeze(1)
        self.env_goals[:] = torch.cat((temp, last_col.repeat(1, self.cfg.env.num_future_goal_obs, 1)), dim=1)[:]
        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)

    def act2joint(self, act):
        act_normalized = (
            act + 1.0
        ) / 2.0
        joint_targets = self.default_dof_pos_low + act_normalized * (
            self.default_dof_pos_high - self.default_dof_pos_low
        )

        return joint_targets
    
    def act2tau(self, act):
        joint_target = self.act2joint(act)

        torques = (
            self.batched_p_gains * (joint_target - self.dof_pos)
            - self.batched_d_gains * self.dof_vel
        )
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    #------------- Callbacks --------------
    def _setup_camera(self):
        ''' Set camera position and direction
        '''
        self._floating_camera = self.scene.add_camera(
            res = (1280, 960),
            pos=np.array(self.cfg.viewer.pos),
            lookat=np.array(self.cfg.viewer.lookat),
            fov=40,
            GUI=False,
        )

        self._recording = False
        self._recorded_frames = []
    
    def _render_headless(self):
        if self._recording:
            robot_pos = np.array(self.base_pos[self.lookat_id].cpu())
            self._floating_camera.set_pose(pos=robot_pos + np.array(self.cfg.viewer.pos), lookat=robot_pos + np.array(self.cfg.viewer.lookat))
            frame, _, _, _ = self._floating_camera.render()
            self._recorded_frames.append(frame)

    def get_recorded_frames(self):
        if len(self._recorded_frames) == 150:
            frames = self._recorded_frames
            self._recorded_frames = []
            self._recording = False
            return frames
        else:
            return None
    
    def start_recording(self, record_internal=True):
        self._recorded_frames = []
        self._recording = True
        if record_internal:
            self._record_frames = True
        else:
            self._floating_camera.start_recording()

    def stop_recording(self, save_path=None):
        self._recorded_frames = []
        self._recording = False
        if save_path is not None:
            print("fps", int(1 / self.dt))
            self._floating_camera.stop_recording(save_path, fps = int(1 / self.dt))

    def _attach_depth_cameras(self):
        for nenv, depth_camera in enumerate(self.depth_cameras):
            # Initialize the camera position and orientation 
            T = np.eye(4)
            angle_x, angle_y, angle_z = self.cfg.depth.angle
            Rot_z = rotvec_to_R(np.array([0, 0, np.radians(angle_z)]))
            Rot_y = rotvec_to_R(np.array([0, np.radians(angle_y), 0]))
            T[:3, 3] = np.array(self.cfg.depth.position)
            T[:3, :3] = Rot_y @ Rot_z
            depth_camera.attach(self.robot.get_link("base"), T, nenv)

    def _setup_depth_cameras(self):
        ''' Attach cameras to the head of each go2 
        '''
        # width = self.cfg.depth.original[0]
        # height = self.cfg.depth.original[1]
        fov = self.cfg.depth.horizontal_fov

        if self.cfg.depth.use_camera:
            for nenv in range(self.num_envs):
                
                depth_camera = self.scene.add_camera(
                    GUI=self.cfg.depth.render_window,
                    fov=fov
                )
                
                self.depth_cameras.append(depth_camera)

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = gs_transform_by_quat(self.forward_vec, self.base_quat)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.8*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)
            self.commands[:, 2] *= torch.abs(self.commands[:, 2]) > self.cfg.commands.ang_vel_clip

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots:
            self._push_robots()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = gs_rand_float(*self.cfg.commands.ranges.lin_vel_x, (len(env_ids),), self.device)
        self.commands[env_ids, 1] = gs_rand_float(*self.cfg.commands.ranges.lin_vel_y, (len(env_ids),), self.device)
        self.commands[env_ids, 2] = gs_rand_float(*self.cfg.commands.ranges.ang_vel_yaw, (len(env_ids),), self.device)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _compute_torques(self, actions):
        # control_type = 'P'
        actions_scaled = actions * self.cfg.control.action_scale
        # torques = (
        #     self.batched_p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos)
        #     - self.batched_d_gains * self.dof_vel
        # )
        torques = (
            self.batched_p_gains * (actions_scaled - self.dof_pos)
            - self.batched_d_gains * self.dof_vel
        )
        return torch.clip(torques, -self.torque_limits, self.torque_limits)
    
    def _compute_target_dof_pos(self, actions):
        # control_type = 'P'
        actions_scaled = actions * self.cfg.control.action_scale
        target_dof_pos = actions_scaled # + self.default_dof_pos

        return target_dof_pos

    def _reset_dofs(self, envs_idx):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        
        self.dof_pos[envs_idx] = (self.default_dof_pos) + gs_rand_float(-0.3, 0.3, (len(envs_idx), self.num_actions), self.device)
        
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )
        self.robot.zero_all_dofs_velocity(envs_idx)
        
    def _reset_root_states(self, envs_idx):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """

        # import pdb; pdb.set_trace()
        # base pos: xy [-1, 1]
        if self.custom_origins:   
            self.base_pos[envs_idx] = self.base_init_pos
            self.base_pos[envs_idx] += self.env_origins[envs_idx]
            # self.base_pos[envs_idx, :2] += gs_rand_float(-0.1, 0.1, (len(envs_idx), 2), self.device)
        else:
            self.base_pos[envs_idx] = self.base_init_pos
            self.base_pos[envs_idx] += self.env_origins[envs_idx]
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        
        # base quat
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        base_euler = gs_rand_float(-0.1, 0.1, (len(envs_idx), 3), self.device) # roll, pitch [-0.1, 0.1]
        base_euler[:, 2] = gs_rand_float(*self.cfg.init_state.yaw_angle_range, (len(envs_idx),), self.device) # yaw angle
        self.base_quat[envs_idx] = gs_quat_mul(gs_euler2quat(base_euler), self.base_quat[envs_idx],)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.zero_all_dofs_velocity(envs_idx)

        # update projected gravity
        inv_base_quat = gs_inv_quat(self.base_quat)
        self.projected_gravity = gs_transform_by_quat(self.global_gravity, inv_base_quat)
        
        # reset root states - velocity
        self.base_lin_vel[envs_idx] = (gs_rand_float(-0.5, 0.5, (len(envs_idx), 3), self.device))
        self.base_ang_vel[envs_idx] = (gs_rand_float(-0.5, 0.5, (len(envs_idx), 3), self.device))
        base_vel = torch.concat([self.base_lin_vel[envs_idx], self.base_ang_vel[envs_idx]], dim=1)
        self.robot.set_dofs_velocity(velocity=base_vel, dofs_idx_local=[0, 1, 2, 3, 4, 5], envs_idx=envs_idx)

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        if self.push_interval_s > 0 and not self.debug:
            max_push_vel_xy = self.cfg.domain_rand.max_push_vel_xy
            # in Genesis, base link also has DOF, it's 6DOF if not fixed.
            dofs_vel = self.robot.get_dofs_velocity() # (num_envs, num_dof) [0:3] ~ base_link_vel
            push_vel = gs_rand_float(-max_push_vel_xy, max_push_vel_xy, (self.num_envs, 2), self.device)
            push_vel[((self.common_step_counter + self.env_identities) % int(self.push_interval_s / self.dt) != 0)] = 0
            dofs_vel[:, :2] += push_vel
            self.robot.set_dofs_velocity(dofs_vel)

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.base_pos[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.utils_terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        temp = self.terrain_goals[self.terrain_levels, self.terrain_types]
        last_col = temp[:, -1].unsqueeze(1)
        self.env_goals[:] = torch.cat((temp, last_col.repeat(1, self.cfg.env.num_future_goal_obs, 1)), dim=1)[:]
        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)

    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)


    def _get_noise_scale_vec(self):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[24:36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[36:48] = 0. # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[48:235] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec()
        self.forward_vec = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.forward_vec[:, 0] = 1.0
        self.base_init_pos = torch.tensor(
            self.cfg.init_state.pos, device=self.device
        )
        self.base_init_quat = torch.tensor(
            self.cfg.init_state.rot, device=self.device
        )
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.lin_vel_tar = torch.zeros((self.num_envs, 2), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.cfg.commands.num_commands), device=self.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], 
            device=self.device,
            dtype=gs.tc_float, 
            requires_grad=False,) # TODO change this
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.llast_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False) # last last actions
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.feet_air_time = torch.zeros((self.num_envs, len(self.feet_indices)), device=self.device, dtype=gs.tc_float)
        self.last_contacts = torch.zeros((self.num_envs, len(self.feet_indices)), device=self.device,dtype=gs.tc_int)
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.reach_goal_timer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.link_contact_forces = torch.zeros(
            (self.num_envs, self.robot.n_links, 3), device=self.device, dtype=gs.tc_float
        )
        self.continuous_push = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.env_identities = torch.arange(
            self.num_envs,
            device=self.device,
            dtype=gs.tc_int, 
        )
        self.terrain_heights = torch.zeros(
            (self.num_envs,),
            device=self.device,
            dtype=gs.tc_float,
        )
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        self.default_dof_pos = torch.tensor(
            [self.cfg.init_state.default_joint_angles[name] for name in self.cfg.asset.dof_names],
            device=self.device,
            dtype=gs.tc_float,
        )

        self.default_dof_pos_low = torch.tensor(
            [self.cfg.init_state.joint_angles_range_low[name] for name in self.cfg.asset.dof_names],
            device=self.device,
            dtype=gs.tc_float,
        )

        self.default_dof_pos_high = torch.tensor(
            [self.cfg.init_state.joint_angles_range_high[name] for name in self.cfg.asset.dof_names],
            device=self.device,
            dtype=gs.tc_float,
        )

        # gait control
        self.foot_positions = torch.ones(
            self.num_envs, len(self.feet_indices), 3, device=self.device, dtype=gs.tc_float,
        )
        self.foot_quaternions = torch.ones(
            self.num_envs, len(self.feet_indices), 4, device=self.device, dtype=gs.tc_float,
        )
        self.foot_velocities = torch.ones(
            self.num_envs, len(self.feet_indices), 3, device=self.device, dtype=gs.tc_float,
        )

        # PD control
        stiffness = self.cfg.control.stiffness
        damping = self.cfg.control.damping
        
        self.p_gains, self.d_gains = [], []
        for dof_name in self.cfg.asset.dof_names:
            for key in stiffness.keys():
                if key in dof_name:
                    self.p_gains.append(stiffness[key])
                    self.d_gains.append(damping[key])
        self.p_gains = torch.tensor(self.p_gains, device=self.device)
        self.d_gains = torch.tensor(self.d_gains, device=self.device)
        self.batched_p_gains = self.p_gains[None, :].repeat(self.num_envs, 1)
        self.batched_d_gains = self.d_gains[None, :].repeat(self.num_envs, 1)
        # PD control params
        self.robot.set_dofs_kp(self.p_gains, self.motor_dofs)
        self.robot.set_dofs_kv(self.d_gains, self.motor_dofs)

        if self.cfg.depth.use_camera:
            self.depth_buffer = torch.zeros(self.num_envs,
                                            self.cfg.depth.buffer_len,
                                            self.cfg.depth.resized[1],
                                            self.cfg.depth.resized[0]).to(self.device)

        self.get_foot_step_vmap = torch.vmap(get_foot_step, in_dims=(None, None, None, None, 0))
        self.gait = self.cfg.rewards.gait
        self._gait_params = {
            #                  ratio, cadence, amplitude
            "stand": torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32, device=self.device),
            "walk": torch.tensor([0.75, 1.0, 0.08], dtype=torch.float32, device=self.device),
            "trot": torch.tensor([0.45, 2, 0.08], dtype=torch.float32, device=self.device),
            "canter": torch.tensor([0.4, 4, 0.06], dtype=torch.float32, device=self.device),
            "gallop": torch.tensor([0.3, 3.5, 0.10], dtype=torch.float32, device=self.device),
        }
        self._gait_phase = {
            "stand": torch.zeros(4, dtype=torch.float32, device=self.device),
            "walk": torch.tensor([0.0, 0.5, 0.75, 0.25], dtype=torch.float32, device=self.device),
            "trot": torch.tensor([0.0, 0.5, 0.5, 0.0], dtype=torch.float32, device=self.device),
            "canter": torch.tensor([0.0, 0.33, 0.33, 0.66], dtype=torch.float32, device=self.device),
            "gallop": torch.tensor([0.0, 0.05, 0.4, 0.35], dtype=torch.float32, device=self.device),
        }
        self.duty_ratio, self.cadence, self.amplitude = self._gait_params[self.gait]
        self.phase = self._gait_phase[self.gait]
        self.foot_radius = 0.0175

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=gs.tc_float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        self.terrain = self.scene.add_entity(
            gs.morphs.Terrain(
                pos=(-self.cfg.terrain.border_size, -self.cfg.terrain.border_size, 0.0),
                horizontal_scale=self.cfg.terrain.horizontal_scale,
                vertical_scale=self.cfg.terrain.vertical_scale,
                height_field=self.utils_terrain.height_field_raw,
            )
        )
        self.height_samples = torch.tensor(self.utils_terrain.heightsamples).view(
            self.utils_terrain.tot_rows, self.utils_terrain.tot_cols).to(self.device)
        self.x_edge_mask = torch.tensor(self.utils_terrain.x_edge_mask).view(self.utils_terrain.tot_rows, self.utils_terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset, create entity
             2. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=os.path.join(asset_root, asset_file),
                merge_fixed_links = True,  # if merge_fixed_links is True, then one link may have multiple geometries, which will cause error in set_friction_ratio
                links_to_keep = self.cfg.asset.links_to_keep,
                pos= np.array(self.cfg.init_state.pos),
                quat=np.array(self.cfg.init_state.rot),
                fixed = self.cfg.asset.fix_base_link,
            ),
            visualize_contact=self.debug,
        )

        # build
        self.scene.build(n_envs=self.num_envs)

        self._get_env_origins()
        
        # name to indices
        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.dof_names]
        
        # find link indices, termination links, penalized links, and feet
        def find_link_indices(names):
            link_indices = list()
            for link in self.robot.links:
                flag = False
                for name in names:
                    if name in link.name:
                        flag = True
                if flag:
                    link_indices.append(link.idx - self.robot.link_start)
            return link_indices
        self.termination_indices = find_link_indices(self.cfg.asset.terminate_after_contacts_on)
        all_link_names = [link.name for link in self.robot.links]
        print(f"all link names: {all_link_names}")
        print("termination link indices:", self.termination_indices)
        self.penalized_indices = find_link_indices(self.cfg.asset.penalize_contacts_on)
        print(f"penalized link indices: {self.penalized_indices}")
        self.feet_indices = find_link_indices(self.cfg.asset.foot_name)
        print(f"feet link indices: {self.feet_indices}")
        assert len(self.termination_indices) > 0
        assert len(self.feet_indices) > 0
        self.feet_link_indices_world_frame = [i+1 for i in self.feet_indices]
        
        hip_names = ["FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint"]
        self.hip_indices = torch.zeros(len(hip_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(hip_names):
            self.hip_indices[i] = self.dof_names.index(name)

        # dof position limits
        self.dof_pos_limits = torch.stack(self.robot.get_dofs_limit(self.motor_dofs), dim=1)
        self.torque_limits = self.robot.get_dofs_force_range(self.motor_dofs)[1]
        for i in range(self.dof_pos_limits.shape[0]):
            # soft limits
            m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
            r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
            self.dof_pos_limits[i, 0] = (
                m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
            )
            self.dof_pos_limits[i, 1] = (
                m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        )
            
        # randomize friction
        if self.cfg.domain_rand.randomize_friction:
            self._randomize_friction(np.arange(self.num_envs))
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            self._randomize_base_mass(np.arange(self.num_envs))
        # randomize COM displacement
        if self.cfg.domain_rand.randomize_com_displacement:
            self._randomize_com_displacement(np.arange(self.num_envs))
    
    def _randomize_friction(self, env_ids=None):
        ''' Randomize friction of all links'''
        min_friction, max_friction = self.cfg.domain_rand.friction_range

        solver = self.rigid_solver

        ratios = gs.rand((len(env_ids), 1), dtype=float).repeat(1, solver.n_geoms) \
                 * (max_friction - min_friction) + min_friction
        solver.set_geoms_friction_ratio(ratios, torch.arange(0, solver.n_geoms), env_ids)
    
    def _randomize_base_mass(self, env_ids=None):
        ''' Randomize base mass'''
        min_mass, max_mass = self.cfg.domain_rand.added_mass_range
        base_link_id = 1
        added_mass = gs.rand((len(env_ids), 1), dtype=float) * (max_mass - min_mass) + min_mass
        self.rigid_solver.set_links_mass_shift(added_mass, [base_link_id, ], env_ids)
    
    def _randomize_com_displacement(self, env_ids):

        min_displacement, max_displacement = self.cfg.domain_rand.com_displacement_range
        base_link_id = 1

        com_displacement = gs.rand((len(env_ids), 1, 3), dtype=float) \
                            * (max_displacement - min_displacement) + min_displacement
        # com_displacement[:, :, 0] -= 0.02

        self.rigid_solver.set_links_COM_shift(com_displacement, [base_link_id,], env_ids)

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.dt
        if self.cfg.sim.use_implicit_controller: # use embedded PD controller
            self.sim_dt = self.dt
            self.sim_substeps = self.cfg.control.decimation
        else: # use explicit PD controller
            self.sim_dt = self.dt / self.cfg.control.decimation
            self.sim_substeps = 1
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.push_interval_s = self.cfg.domain_rand.push_interval_s

        self.dof_names = self.cfg.asset.dof_names
        self.simulate_action_latency = self.cfg.domain_rand.simulate_action_latency
        self.debug = self.cfg.env.debug
        
    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height points
        if not self.cfg.terrain.measure_heights:
            return
        self.scene.clear_debug_objects()
        height_points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points)
        height_points[0, :, 0] += self.base_pos[0, 0]
        height_points[0, :, 1] += self.base_pos[0, 1]
        height_points[0, :, 2] = self.measured_heights[0, :]
        # self.scene.draw_debug_spheres(height_points[0, :], radius=0.03, color=(0, 0, 1, 0.7)) # only draw for the first env
        # print(f"shape of height_points: ", height_points.shape) # (num_envs, num_points, 3)
        # lin_vel_vec = self.commands[0, :2] / torch.norm(self.commands[0,:2])
        # lin_vel_vec = lin_vel_vec.cpu()
        # self.scene.draw_debug_arrow(self.base_pos[0], (lin_vel_vec[0], lin_vel_vec[1], 0), radius=0.03, color=(0, 0, 1, 0.7))
        self._draw_goals(draw_target=False)
        self._draw_commands()
        self._draw_feet()

    def _draw_feet(self):
        # feet_pos_xy = ((self.foot_positions[:, :, :2] + self.terrain.cfg.border_size) / self.cfg.terrain.horizontal_scale).round().long()
        self.scene.draw_debug_spheres(self.foot_positions[self.lookat_id], radius = 0.02, color=(0, 0, 1, 1))
    
    def _draw_commands(self):
        norm = torch.norm(self.commands[:, :2], dim=-1, keepdim=True)
        target_vec_norm = self.commands[:, :2] / (norm + 1e-5)
        lin_vel_arrow = target_vec_norm[self.lookat_id, :2].cpu().numpy()
        pose_robot = self.base_pos[self.lookat_id, :3].cpu().numpy()
        self.scene.draw_debug_arrow(pose_robot, (lin_vel_arrow[0], lin_vel_arrow[1], 0), radius=0.01, color=(1, 0.35, 0.25, 0.7))
        return 
    
    def _draw_goals(self, draw_target = True):
        # draw first goals 
        goals = self.terrain_goals[self.terrain_levels[self.lookat_id], self.terrain_types[self.lookat_id]].cpu().numpy()
        for i, goal in enumerate(goals):
            goal_xy = goal[:2] + self.utils_terrain.cfg.border_size
            pts = (goal_xy/self.utils_terrain.cfg.horizontal_scale).astype(int)
            goal_z = self.height_samples[pts[0], pts[1]].cpu().item() * self.utils_terrain.cfg.vertical_scale   
            pose = np.array([goal[0], goal[1], goal_z])
            if i == self.cur_goal_idx[self.lookat_id].cpu().item():
                self.scene.draw_debug_sphere(pose, radius=0.10, color=(0, 0, 1, 0.7))
            else:
                self.scene.draw_debug_spheres(pose, radius = 0.10, color = (0, 1, 0, 0.7))
                # if self.reached_goal_ids[self.lookat_id]:
                #     self.scene.draw_debug_sphere(goals[i, :], radius=0.20, color=(1, 0, 0, 0.7))

        if draw_target:
            pose_robot = self.base_pos[self.lookat_id, :3].cpu().numpy()
            norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
            target_vec_norm = self.target_pos_rel / (norm + 1e-5)
            pose_arrow =  target_vec_norm[self.lookat_id, :2].cpu().numpy()
            self.scene.draw_debug_arrow(pose_robot, (pose_arrow[0], pose_arrow[1], 0), radius=0.01, color=(1, 0.35, 0.25, 0.7))

            norm = torch.norm(self.next_target_pos_rel, dim=-1, keepdim=True)
            target_vec_norm = self.next_target_pos_rel / (norm + 1e-5)
            pose_arrow =  target_vec_norm[self.lookat_id, :2].cpu().numpy() 
            self.scene.draw_debug_arrow(pose_robot, (pose_arrow[0], pose_arrow[1], 0), radius=0.01, color=(0, 1, 0.5, 0.7))

          
    def _update_goals(self):
        next_flag = self.reach_goal_timer > self.cfg.env.reach_goal_delay / self.dt
        self.cur_goal_idx[next_flag] += 1
        self.reach_goal_timer[next_flag] = 0

        # import pdb; pdb.set_trace()
        self.reached_goal_ids = torch.norm(self.base_pos[:, :2] - self.cur_goals[:, :2], dim=1) < self.cfg.env.next_goal_threshold
        self.reach_goal_timer[self.reached_goal_ids] += 1

        self.target_pos_rel = self.cur_goals[:, :2] - self.base_pos[:, :2]
        self.next_target_pos_rel = self.next_goals[:, :2] - self.base_pos[:, :2]

        norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.target_pos_rel / (norm + 1e-5)
        self.target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])

        norm = torch.norm(self.next_target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.next_target_pos_rel / (norm + 1e-5)
        self.next_target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])


    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.utils_terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]

            self.terrain_goals = torch.from_numpy(self.utils_terrain.goals).to(self.device).to(torch.float)
            self.env_goals = torch.zeros(self.num_envs, self.cfg.terrain.num_goals + self.cfg.env.num_future_goal_obs, 3, device=self.device, requires_grad=False)
            self.cur_goal_idx = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
            temp = self.terrain_goals[self.terrain_levels, self.terrain_types]
            last_col = temp[:, -1].unsqueeze(1)
            self.env_goals[:] = torch.cat((temp, last_col.repeat(1, self.cfg.env.num_future_goal_obs, 1)), dim=1)[:]
            self.cur_goals = self._gather_cur_goals()
            self.next_goals = self._gather_cur_goals(future=1)
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            # plane has limited size, we need to specify spacing base on num_envs, to make sure all robots are within the plane
            # restrict envs to a square of [plane_length/2, plane_length/2]
            spacing = min((self.cfg.terrain.plane_length / 2) / (num_rows-1), (self.cfg.terrain.plane_length / 2) / (num_cols-1))
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.
            self.env_origins[:, 0] -= self.cfg.terrain.plane_length / 4
            self.env_origins[:, 1] -= self.cfg.terrain.plane_length / 4
    
    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points
    
    def _gather_cur_goals(self, future=0):
        return self.env_goals.gather(1, (self.cur_goal_idx[:, None, None]+future).expand(-1, -1, self.env_goals.shape[-1])).squeeze(1)
    
    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.base_pos[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.base_pos[:, :3]).unsqueeze(1)

        points += self.cfg.terrain.border_size
        points = (points/self.cfg.terrain.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.cfg.terrain.vertical_scale

    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
    
    def _reward_tracking_yaw(self):
        rew = torch.exp(-torch.abs(self.target_yaw - self.yaw))
        return rew
    
    def _reward_tracking_goal_position(self):
        rew = torch.sum(torch.square(self.base_pos[:, :2] - self.cur_goals[:, :2]), dim=1)
        return rew
    
    def _reward_tracking_goal_vel(self):
        norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.target_pos_rel / (norm + 1e-5)
        cur_vel = self.base_lin_vel[:, :2]
        rew = torch.minimum(torch.sum(target_vec_norm * cur_vel, dim=-1), self.commands[:, 0]) / (self.commands[:, 0] + 1e-5)
        return rew

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.base_pos[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        rew = torch.square(base_height - self.cfg.rewards.base_height_target)
        return rew
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.link_contact_forces[:, self.penalized_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:, self.hip_indices] - self.default_dof_pos[self.hip_indices]), dim=1)
                         
    # def _reward_dof_vel_limits(self):
    #     # Penalize dof velocities too close to the limit
    #     # clip to max error = 1 rad/s per joint to avoid huge penalties
    #     return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        contact = self.link_contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_dof_close_to_default(self):
        # Penalize dof position deviation from default
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_action_close_default(self):
        return torch.sum(torch.square(self.actions - self.default_dof_pos), dim=1)
    
    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        rew = torch.any(torch.norm(self.link_contact_forces[:, self.feet_indices, :2], dim=2) >\
             4 *torch.abs(self.link_contact_forces[:, self.feet_indices, 2]), dim=1)
        return rew.float()

    def _reward_feet_edge(self):
        feet_pos_xy = ((self.foot_positions[:, :, :2] + self.terrain.cfg.border_size) / self.cfg.terrain.horizontal_scale).round().long()  # (num_envs, 4, 2)
        feet_pos_xy[..., 0] = torch.clip(feet_pos_xy[..., 0], 0, self.x_edge_mask.shape[0]-1)
        feet_pos_xy[..., 1] = torch.clip(feet_pos_xy[..., 1], 0, self.x_edge_mask.shape[1]-1)
        feet_at_edge = self.x_edge_mask[feet_pos_xy[..., 0], feet_pos_xy[..., 1]]
        contact = self.link_contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.feet_at_edge = contact_filt & feet_at_edge
        rew = (self.terrain_levels > 3) * torch.sum(self.feet_at_edge, dim=-1)
        return rew
    
    def _reward_dial_upright(self):
        vec_tar = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        vec = gs_transform_by_quat(vec_tar, self.base_quat)
        rew = -torch.sum(torch.square(vec - vec_tar), dim=1)
        return rew
    
    def _reward_dial_gaits(self):
        z_feet_tar = self.get_foot_step_vmap(self.duty_ratio, self.cadence, self.amplitude, self.phase, self.episode_length_buf * self.dt)
        rew = -torch.sum(((z_feet_tar - self.foot_positions[:,:,2]) / 0.05) ** 2, dim=1)
        return rew
    
    def _reward_dial_yaw(self):
        d_yaw = self.yaw - self.target_yaw
        rew = -torch.square(torch.arctan2(torch.sin(d_yaw), torch.cos(d_yaw)))
        return rew

    def _reward_dial_vel(self):
        rew = -torch.sum((self.base_lin_vel[:, :2] - self.lin_vel_tar)**2, dim=1)
        return rew
    
    def _reward_dial_ang_vel(self):
        # rew = -(self.base_ang_vel[:, 2] - self.commands[:, 2]) **2
        rew = -(self.base_ang_vel[:, 2]) **2
        return rew
    
    def _reward_dial_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.base_pos[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        rew = -torch.square(base_height - self.cfg.rewards.base_height_target)
        return rew