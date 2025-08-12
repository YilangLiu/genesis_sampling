import genesis as gs
import os
import torch
from legged_gym.envs.g1.base_env import BaseEnv

class G1Env(BaseEnv):
    def __init__(self, num_envs, dt = 0.01, show_viewer=False, show_FPS=False, offscreen_cam=False):
        
        self._num_envs = num_envs
        self._num_actions = 29
        self._dt = dt
        self._num_states = 71 # 36 + 35
        self._t = torch.zeros(self._num_envs, device=gs.device, dtype=gs.tc_float)
        
        self.motor_joint_names = ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", 
                                "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
                                
                                "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
                                "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
                                
                                "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
                                
                                "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
                                "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
                                
                                "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                                "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]

        self.default_motor_joints_pos = torch.tensor([[0, 0, 0, 0, 0, 0,
                                                0, 0, 0, 0, 0, 0,
                                                0, 0, 0,
                                                0.2, 0.2, 0, 1.28, 0, 0, 0,
                                                0.2, -0.2, 0, 1.28, 0, 0, 0]], device=gs.device, dtype=gs.tc_float).repeat((num_envs, 1))
        self.default_pos = torch.tensor([[0, 0, 0.79]], device=gs.device, dtype=gs.tc_float).repeat((num_envs, 1))
        self.default_quat = torch.tensor([[2**0.5/2, 0, -2**0.5/2, 0]], device=gs.device, dtype=gs.tc_float).repeat((num_envs, 1))
        
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self._dt),
            show_viewer=show_viewer,
            show_FPS=show_FPS
        )
        self.plane = self.scene.add_entity(
            gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True)
        )
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(file=os.path.join("/home/yilang/research/genesis_sampling/resources/robots/g1/g1.xml"))
        )

        if offscreen_cam:
            self._offscreen_cam = self.scene.add_camera(
            res    = (640, 480),
            pos    = (3.5, 0.0, 2.5),
            lookat = (0, 0, 0.5),
            fov    = 30,
            GUI    = False)
        else:
            self._offscreen_cam = None

        self.scene.build(n_envs=num_envs)

        self.motors_dof_idx = [self.robot.get_joint(name).dof_start for name in self.motor_joint_names]            
        self.base_dof_idx = self.robot.base_joint.dofs_idx_local

        # Using the position control, the action_limits equal to the joints limit
        self._actions_limits = self.robot.get_dofs_limit(dofs_idx_local=self.motors_dof_idx)
        
        # TODO Set the force range for each motors. Actuatorfrcrange in the joint tag is not parse correctly by mujoco?

    def step(self, actions):
        actions = torch.clip(actions, self._actions_limits[0], self._actions_limits[1])
        self.robot.control_dofs_position(actions, self.motors_dof_idx)
        self.scene.step()
        self._t += self._dt

    def reset(self):
        self.robot.set_pos(self.default_pos, zero_velocity=True)
        self.robot.set_quat(self.default_quat, zero_velocity=True)
        self.robot.set_dofs_position(
            position=self.default_motor_joints_pos,
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True
        )
        self._t = torch.zeros(self._num_envs, device=gs.device, dtype=gs.tc_float)
        self.scene.visualizer.update()

    def reset_with_states(self, states):
        pos = states[..., :3]
        quat = states[..., 3:7]
        joints_pos = states[..., 7:36]
        vel_and_ang = states[..., 36:42]
        joints_vel = states[..., 42:]

        self.robot.set_pos(pos)
        self.robot.set_quat(quat)
        self.robot.set_dofs_position(position=joints_pos, dofs_idx_local=self.motors_dof_idx)
        self.robot.set_dofs_velocity(velocity=vel_and_ang, dofs_idx_local=self.base_dof_idx)
        self.robot.set_dofs_velocity(velocity=joints_vel, dofs_idx_local=self.motors_dof_idx)

    def compute_rewards(self, states, actions):
        pos = states[...,:3]
        quat = states[...,3:7]
        joints_pos = states[...,7:36]

        height_reward = -10. * (pos[..., -1] - 0.79)**2
        if quat.ndim == 1:
            up_dir = torch.tensor([0, 0, 1], device=gs.device, dtype=gs.tc_float)
        else:
            up_dir = torch.tensor([0, 0, 1], device=gs.device, dtype=gs.tc_float).repeat(quat.shape[:-1] + (1,))
        orientation_reward = 1. * gs.transform_by_quat(up_dir, quat)[..., -1] 
        joints_reward = -0.1 * torch.sum((joints_pos - self.default_motor_joints_pos)**2)
        return height_reward + orientation_reward + joints_reward

    def get_states(self):
        pos = self.robot.get_pos()
        quat = self.robot.get_quat()
        joints_pos = self.robot.get_dofs_position(self.motors_dof_idx)
        vel = self.robot.get_vel()
        ang = self.robot.get_ang()
        joints_vel = self.robot.get_dofs_velocity(self.motors_dof_idx)
        return torch.concat([pos, quat, joints_pos, vel, ang, joints_vel], dim=-1)