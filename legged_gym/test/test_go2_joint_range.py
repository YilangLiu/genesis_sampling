import numpy as np

import genesis as gs

########################## init ##########################
gs.init(backend=gs.gpu)

########################## create a scene ##########################

scene = gs.Scene(
    show_viewer=True,
    rigid_options=gs.options.RigidOptions(
        dt=0.01,
        constraint_solver=gs.constraint_solver.Newton,
        enable_self_collision = False,
    ),
)

########################## entities ##########################
scene.add_entity(
    gs.morphs.Plane(),
)
robot = scene.add_entity(
    gs.morphs.URDF(
        file="urdf/go2/urdf/go2.urdf",
        pos=(0, 0, 0.8),
    ),
)
########################## build ##########################
n_envs = 1
scene.build(n_envs=n_envs)

joints_name = (
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
)

joint_angles_range_low = {
            'FL_hip_joint': -0.5,   # [rad]
            'RL_hip_joint': -0.5,   # [rad]
            'FR_hip_joint': -0.5 ,  # [rad]
            'RR_hip_joint': -0.5,   # [rad]

            'FL_thigh_joint': 0.4,     # [rad]
            'RL_thigh_joint': 0.4,   # [rad]
            'FR_thigh_joint': 0.4,     # [rad]
            'RR_thigh_joint': 0.4,   # [rad]

            'FL_calf_joint': -2.3,   # [rad]
            'RL_calf_joint': -2.3,    # [rad]
            'FR_calf_joint': -2.3,  # [rad]
            'RR_calf_joint': -2.3,    # [rad]
        }
joint_angles_range_high = {
            'FL_hip_joint': 0.5,   # [rad]
            'RL_hip_joint': 0.5,   # [rad]
            'FR_hip_joint': 0.5 ,  # [rad]
            'RR_hip_joint': 0.5,   # [rad]

            'FL_thigh_joint': 1.4,     # [rad]
            'RL_thigh_joint': 1.4,   # [rad]
            'FR_thigh_joint': 1.4,     # [rad]
            'RR_thigh_joint': 1.4,   # [rad]

            'FL_calf_joint': -1.3,   # [rad]
            'RL_calf_joint': -1.3,    # [rad]
            'FR_calf_joint': -1.3,  # [rad]
            'RR_calf_joint': -1.3,    # [rad]
        }
motors_dof_idx = [robot.get_joint(name).dofs_idx_local[0] for name in joints_name]

default_dof_pos_low = np.array(
            [joint_angles_range_low[name] for name in joints_name]
        )

default_dof_pos_high = np.array(
            [joint_angles_range_high[name] for name in joints_name]
        )

robot.set_dofs_position(position=default_dof_pos_low, dofs_idx_local=motors_dof_idx, zero_velocity=False)

# Speed: 14.4M FPS
import time
now = time.time()
for i in range(1000):
    scene.step()
    import pdb; pdb.set_trace()
print((time.time() - now)/1000)