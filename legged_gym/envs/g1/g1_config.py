from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class G1Cfg( LeggedRobotCfg ):
    
    class env( LeggedRobotCfg.env ):
        num_envs = 1
        num_observations = 71
        num_actions = 29
        env_spacing = 3.  # not used with heightfields/trimeshes
    
    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
        friction = 1.0
        restitution = 0.
        
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 1.08] # x,y,z [m]
        default_joint_angles = {
            "left_hip_pitch_joint" : 0.0, 
            "left_hip_roll_joint" : 0.0, 
            "left_hip_yaw_joint":0.0, 
            "left_knee_joint":0.0, 
            "left_ankle_pitch_joint":0.0, 
            "left_ankle_roll_joint":0.0,
            "right_hip_pitch_joint":0.0, 
            "right_hip_roll_joint":0.0, 
            "right_hip_yaw_joint":0.0,
            "right_knee_joint":0.0, 
            "right_ankle_pitch_joint":0.0, 
            "right_ankle_roll_joint":0.0,
            "waist_yaw_joint":0.0, 
            "waist_roll_joint":0.0, 
            "waist_pitch_joint":0.0,
            "left_shoulder_pitch_joint":0.2, 
            "left_shoulder_roll_joint":0.2, 
            "left_shoulder_yaw_joint":0.0,
            "left_elbow_joint":1.28, 
            "left_wrist_roll_joint":0.0, 
            "left_wrist_pitch_joint":0.0, 
            "left_wrist_yaw_joint":0.0,
            "right_shoulder_pitch_joint":0.2, 
            "right_shoulder_roll_joint":-0.2, 
            "right_shoulder_yaw_joint":0.0,
            "right_elbow_joint":1.28, 
            "right_wrist_roll_joint":0.0, 
            "right_wrist_pitch_joint":0.0, 
            "right_wrist_yaw_joint":0.0
            }
        
        # initial state randomization
        yaw_angle_range = [0., 3.14] # min max [rad]

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        # control_type = 'P'
        stiffness = {'hip_joint_saggital': 100.0, 'hip_joint_frontal': 100.0,
                     'hip_joint_transversal': 200., 'knee_joint': 200., 'ankle_joint': 200.}   # [N*m/rad]
        damping = { 'hip_joint_saggital': 3.0, 'hip_joint_frontal': 3.0,
                    'hip_joint_transversal': 6., 'knee_joint': 6., 'ankle_joint': 10.}     # [N*m*s/rad]
        action_scale = 0.25 # action scale: target angle = actionScale * action + defaultAngle
        dt =  0.02  # control frequency 50Hz
        decimation = 4 # decimation: Number of control action updates @ sim DT per policy DT

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/bipedal_walker/urdf/walker3d_hip3d.urdf'
        dof_names = [        # specify yhe sequence of actions
            "left_hip_pitch_joint"
            "left_hip_roll_joint"
            "left_hip_yaw_joint"
            "left_knee_joint"
            "left_ankle_pitch_joint"
            "left_ankle_roll_joint"
            "right_hip_pitch_joint"
            "right_hip_roll_joint"
            "right_hip_yaw_joint"
            "right_knee_joint"
            "right_ankle_pitch_joint"
            "right_ankle_roll_joint"
            "waist_yaw_joint"
            "waist_roll_joint"
            "waist_pitch_joint"
            "left_shoulder_pitch_joint"
            "left_shoulder_roll_joint"
            "left_shoulder_yaw_joint"
            "left_elbow_joint"
            "left_wrist_roll_joint"
            "left_wrist_pitch_joint"
            "left_wrist_yaw_joint"
            "right_shoulder_pitch_joint"
            "right_shoulder_roll_joint"
            "right_shoulder_yaw_joint"
            "right_elbow_joint"
            "right_wrist_roll_joint"
            "right_wrist_pitch_joint"
            "right_wrist_yaw_joint"]
        foot_name = ["foot"]
        penalize_contacts_on = []
        terminate_after_contacts_on = ["torso", 'thigh','shank']
        links_to_keep = []
        self_collisions = True
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 1.08
        only_positive_rewards = False
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = -200.0
            # limitation
            dof_pos_limits = -5.0
            collision = -0.0
            # command tracking
            tracking_lin_vel = 1.0
            tracking_ang_vel = 1.0
            # smooth
            lin_vel_z = -2.0
            base_height = -1.0
            ang_vel_xy = -0.05
            orientation = -0.0
            dof_vel = -0.0
            dof_acc = -2.e-7
            action_rate = -0.01
            torques = -1.e-5
            # gait
            feet_air_time = 1.0
            no_fly = 0.25
            # dof_close_to_default = -0.1
    
    class commands( LeggedRobotCfg.commands ):
        curriculum = True
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges( LeggedRobotCfg.commands.ranges ):
            lin_vel_x = [-0.5, 0.5] # min max [m/s]
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]
    
    class domain_rand:
        randomize_friction = True
        friction_range = [0.2, 1.7]
        randomize_base_mass = True
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.
        simulate_action_latency = False # 1 step delay
        randomize_com_displacement = False
        com_displacement_range = [-0.01, 0.01]
    
    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]       # [m]
        lookat = [11., 5, 3.]  # [m]
        num_rendered_envs = 10  # number of environments to be rendered
        add_camera = False

class G1CfgPPO( LeggedRobotCfgPPO ):
    runner_class_name = "OnPolicyRunner"
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'bipedal_walker'
        save_interval = 100
        load_run = "Dec24_21-23-47_"
        checkpoint = -1
        max_iterations = 2000
        
    class planner:
        name: str = 'MPPI'
        num_samples: int = 64  # Reduced from 100 to 20 to reduce memory requirements
        horizon: int = 20
        num_knots: int = 4
        sampling_method: str = 'gaussian'
        device: str = 'cuda:0'
        sample_noise: float = 1.0
