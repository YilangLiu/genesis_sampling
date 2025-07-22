from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgSample

class GO2RoughCfgSampling( LeggedRobotCfg ):
    
    class env( LeggedRobotCfg.env ):
        num_envs = 1
        num_observations = 48 + 121 # robot_state + terrain_heights
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes
        debug_viz = False
        next_goal_threshold = 0.2
    
    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = "heightfield" # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale =  0.05 # 0.2 # [m]. if use smaller horizontal scale, need to decrease terrain_length and terrain_width, or it will compile very slowly.
        vertical_scale = 0.005 # [m]
        border_size = 5 # [m]. implemented a out_of_bound detection, so border_size can be smaller
        curriculum = True
        friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-1.0, -0.8, -0.6, -0.4, -0.2, 0., 0.2, 0.4, 0.6, 0.8, 1.0]
        measured_points_y = [-1.0, -0.8, -0.6, -0.4, -0.2, 0., 0.2, 0.4, 0.6, 0.8, 1.0]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 1 # starting curriculum state
        terrain_length = 18. # 6.0
        terrain_width = 4.0  # 6.0
        num_rows = 2  # number of terrain rows (levels)
        num_cols = 5  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete, parkour hurtle]
        terrain_dict = {"smooth slope": 0., 
                        "rough slope up": 0.0,
                        "rough slope down": 0.0,
                        "rough stairs up": 0., 
                        "rough stairs down": 0., 
                        "discrete": 0., 
                        "stepping stones": 0.0,
                        "gaps": 0., 
                        "smooth flat": 0,
                        "pit": 0.0,
                        "wall": 0.0,
                        "platform": 0.,
                        "large stairs up": 0.,
                        "large stairs down": 0.,
                        "parkour": 0.2,
                        "parkour_hurdle": 0.2,
                        "parkour_flat": 0.2,
                        "parkour_step": 0.2,
                        "parkour_gap": 0.2,
                        "demo": 0.0,}
        terrain_proportions = list(terrain_dict.values())
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces
        num_goals = 8
        downsampled_scale = 0.075
        
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0., 0.38] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.0,   # [rad]
            'RL_hip_joint': 0.0,   # [rad]
            'FR_hip_joint': 0.0 ,  # [rad]
            'RR_hip_joint': 0.0,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }
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

            'FL_calf_joint': -0.85,   # [rad]
            'RL_calf_joint': -1.3,    # [rad]
            'FR_calf_joint': -0.85,  # [rad]
            'RR_calf_joint': -1.3,    # [rad]
        }
        # initial state randomization
        yaw_angle_range = [0., 0] # [0., 3.14] # min max [rad]
    
    class noise( LeggedRobotCfg.noise):
        add_noise = False
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        # control_type = 'P'
        stiffness = {'joint': 30.}   # [N*m/rad]
        damping = {'joint': 0.0}     # [N*m*s/rad]
        action_scale = 1.0 # 0.25 # action scale: target angle = actionScale * action + defaultAngle
        dt =  0.02  # control frequency 50Hz
        decimation = 4 # decimation: Number of control action updates @ sim DT per policy DT

    class sim(LeggedRobotCfg.sim):
        use_implicit_controller = False
        use_dial_controller = True

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        dof_names = [        # specify yhe sequence of actions
            'FR_hip_joint',
            'FR_thigh_joint',
            'FR_calf_joint',
            'FL_hip_joint',
            'FL_thigh_joint',
            'FL_calf_joint',
            'RR_hip_joint',
            'RR_thigh_joint',
            'RR_calf_joint',
            'RL_hip_joint',
            'RL_thigh_joint',
            'RL_calf_joint',]
        foot_name = ["foot"]
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        links_to_keep = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
        self_collisions = True
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.30
        only_positive_rewards = False
        termination_if_roll_greater_than= 0.4
        termination_if_pitch_greater_than= 0.4
        termination_if_height_lower_than= 0.0
        gait = "trot"
        class scales( LeggedRobotCfg.rewards.scales ):
            # limitation
            # dof_pos_limits = 0.0
            # collision = 0.0
            # # command tracking
            # tracking_lin_vel = 0.0
            # tracking_ang_vel = 0.0
            # tracking_yaw = 0.0
            # # smooth
            # lin_vel_z = 0.0
            # base_height = -1.0
            # ang_vel_xy = 0.0
            # orientation = -1.0
            # dof_vel = 0.0
            # dof_acc = 0.0
            # action_rate = 0.0
            # torques = -0.0
            # # gait
            # feet_air_time = 0.0
            # action_close_default = -0.0
            # dof_close_to_default = -0.25
            # # Goal position
            # tracking_goal_position = -5.0
            

            # Extreme Parkour Rewards
            # command tracking
            # tracking_goal_vel = 1.5
            # tracking_yaw = 0.5
            # # smooth
            # lin_vel_z = -1.0
            # ang_vel_xy = -0.05
            # base_height = -1.0
            # orientation = -1.0
            # dof_acc = -2.e-7
            # collision = -10.0
            # # dof_vel = -5.e-4
            # action_rate = -0.1
            # torques = -2.e-4
            # hip_pos = -0.5
            # dof_close_to_default = -0.04
            # # gait
            # feet_air_time = 1.0
            

            # # genesis RL rewards
            # dof_pos_limits = -10.0
            # collision = -1.0
            # # command tracking
            # tracking_goal_vel = 1.5
            # tracking_yaw = 0.5
            # # smooth
            # lin_vel_z = -2.0
            # base_height = -10.0
            # ang_vel_xy = -0.05
            # orientation = -5.0
            # dof_vel = -5.e-4
            # dof_acc = -2.e-7
            # action_rate = -0.01
            # torques = -2.e-4
            # # gait
            # feet_air_time = 1.0
            # dof_close_to_default = -0.05

            dial_gaits = 0.1
            dial_upright = 0.5
            dial_yaw = 0.3
            dial_vel = 1.0
            dial_ang_vel = 1.0
            dial_height = 1.0

    class commands( LeggedRobotCfg.commands ):
        curriculum = True
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error

        lin_vel_clip = 0.2
        ang_vel_clip = 0.4

        class ranges( LeggedRobotCfg.commands.ranges ):
            lin_vel_x = [0., 1.5] # min max [m/s]
            lin_vel_y = [0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [0, 0]    # min max [rad/s]
            heading = [0, 0]
    
    class domain_rand:
        randomize_friction = False
        friction_range = [0.2, 1.7]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        push_robots = False
        push_interval_s = 15
        max_push_vel_xy = 1.
        simulate_action_latency = False # 1 step delay
        randomize_com_displacement = False
        com_displacement_range = [-0.01, 0.01]
    
    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [-1, -1, 0.5]       # [m]
        lookat = [0, 0, -0.1]  # [m]
        num_rendered_envs = 4  # number of environments to be rendered
        add_camera = True

class Go2SamplingCfg( LeggedRobotCfgSample ):
    class planner( LeggedRobotCfgSample.planner ):
        num_samples = 2000
        sample_noise = 1.0
        horizon = 16
        num_knots = 4

    class rollout_env( LeggedRobotCfgSample.rollout_env ):
        dt = 0.02
        substeps = 4