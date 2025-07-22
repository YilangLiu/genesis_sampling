import genesis as gs
import numpy as np

def main():
    
    gs.init()
    
    scene = gs.Scene(show_viewer=True)
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(gs.morphs.Box(pos=(0.5, 0.0, 0.0), size=(0.05, 0.05, 0.05)))
    franka = scene.add_entity(gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'))
    
    cam = scene.add_camera(GUI=True, fov=70)
    scene.build()

    T = np.eye(4)
    T[:3, :3] = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ])
    T[:3, 3] = np.array([0.1, 0.0, 0.1])
    cam.attach(franka.get_link("hand"), T)
    
    target_quat = np.array([0, 1, 0, 0]) # pointing downwards
    center = np.array([0.5, 0.0, 0.5])
    angular_speed = np.random.random() * 10.0
    r = 0.1

    ee_link = franka.get_link('hand')

    for i in range(1000):
        target_pos = center.copy()
        target_pos[0] += np.cos(i/360*np.pi*angular_speed) * r
        target_pos[1] += np.sin(i/360*np.pi*angular_speed) * r

        q = franka.inverse_kinematics(
            link     = ee_link,
            pos      = target_pos,
            quat     = target_quat,
            rot_mask = [False, False, True], # for demo purpose: only restrict direction of z-axis
        )

        franka.set_qpos(q)
        scene.step()
        cam.render()
        
if __name__ == "__main__":
    main()