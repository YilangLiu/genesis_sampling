import numpy as np
import torch
import genesis as gs
import time

def find_link_indices(robot, names):
    link_indices = list()
    for link in robot.links:
        flag = False
        for name in names:
            if name in link.name:
                flag = True
        if flag:
            link_indices.append(link.idx - robot.link_start)
    return link_indices

########################## init ##########################
gs.init(backend=gs.gpu)

########################## create a scene ##########################
scene = gs.Scene(
    viewer_options = gs.options.ViewerOptions(
        camera_pos    = (0, -3.5, 2.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 30,
        max_FPS       = 60,
    ),
    sim_options = gs.options.SimOptions(
        dt = 0.01,
    ),
    show_viewer = False,
    rigid_options=gs.options.RigidOptions(
                dt=0.01,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                enable_self_collision=False
            ),
)

########################## entities ##########################
plane = scene.add_entity(
    gs.morphs.Plane(),
)

# when loading an entity, you can specify its pose in the morph.
robot = scene.add_entity(
    gs.morphs.MJCF(
        file  = 'g1/g1.xml',
        pos   = (1.0, 1.0, 0.0),
        euler = (0, 0, 0),
    ),
)

# actions = torch.zeros(29, device="cuda", dtype=torch.float32)
########################## build ##########################
scene.build()

import pdb; pdb.set_trace()