import genesis as gs
import numpy as np

def main():
    
    gs.init()
    
    scene = gs.Scene(show_viewer=True)
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(gs.morphs.Box(pos=(0.5, 0.0, 0.0), size=(0.05, 0.05, 0.05)))
    franka = scene.add_entity(gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'))
    
    n_envs = 4
    env_spacing = (2.0, 2.0)
    cams = [scene.add_camera(GUI=True, fov=70) for _ in range(n_envs)]
    scene.build(n_envs=n_envs, env_spacing=env_spacing)

    T = np.eye(4)
    T[:3, :3] = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ])
    T[:3, 3] = np.array([0.1, 0.0, 0.1])
    for nenv, cam in enumerate(cams):
        cam.attach(franka.get_link("hand"), T, nenv)

    target_quat = np.tile(np.array([0, 1, 0, 0]), [n_envs, 1]) # pointing downwards
    center = np.tile(np.array([0.5, 0.0, 0.5]), [n_envs, 1])
    angular_speed = np.random.uniform(-10, 10, n_envs)
    r = 0.1

    ee_link = franka.get_link('hand')

    for i in range(1000):
        target_pos = np.zeros([n_envs, 3])
        target_pos[:, 0] = center[:, 0] + np.cos(i/360*np.pi*angular_speed) * r
        target_pos[:, 1] = center[:, 1] + np.sin(i/360*np.pi*angular_speed) * r
        target_pos[:, 2] = center[:, 2]

        q = franka.inverse_kinematics(
            link     = ee_link,
            pos      = target_pos,
            quat     = target_quat,
            rot_mask = [False, False, True], # for demo purpose: only restrict direction of z-axis
        )

        franka.set_qpos(q)
        scene.step()
        for cam in cams:
            cam.render(rgb=False, depth=True)
        
if __name__ == "__main__":
    main()