from .base_config import BaseConfig

class LeggedRobotCfg(BaseConfig):
    class env:
        num_envs = 4096
        num_observations = 48
        num_privileged_obs = 75 # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 12
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds
        debug = False # if debugging, visualize contacts, 
        debug_viz = False # draw debug visualizations
        force_reset = True # force reset the environment if the environment is done
        num_future_goal_obs = 2
        next_goal_threshold = 0.2
        reach_goal_delay = 0.1

    class terrain:
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield
        plane_length = 200.0 # [m]. plane size is 200x200x10 by default
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 5 # [m]
        curriculum = False
        friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = False
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 1 # starting curriculum state
        terrain_length = 6.0
        terrain_width = 6.0
        num_rows = 4  # number of terrain rows (levels)
        num_cols = 4  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces
        num_goals = 10
        height = [0.01, 0.03]
        downsampled_scale = 0.075
        y_range = [-0.4, 0.4]

        origin_zero_z = True
        edge_width_thresh = 0.05
        
    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 6. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        ramp_up_time = 1.0
        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state:
        pos = [0.0, 0.0, 1.] # x,y,z [m]
        rot = [1.0, 0.0, 0.0, 0.0] # w, x, y, z [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        # initial state randomization
        yaw_angle_range = [0.0, 0.0] # min max [rad]
        default_joint_angles = { # target angles when action = 0.0
            "joint_a": 0., 
            "joint_b": 0.}
        
        joint_angles_range_low = {
            'FL_hip_joint': -1.0472,   # [rad]
            'RL_hip_joint': -1.0472,   # [rad]
            'FR_hip_joint': -1.0472 ,  # [rad]
            'RR_hip_joint': -1.0472,   # [rad]

            'FL_thigh_joint': -1.5708,     # [rad]
            'RL_thigh_joint': -0.5236,   # [rad]
            'FR_thigh_joint': -1.5708,     # [rad]
            'RR_thigh_joint': -0.5236,   # [rad]

            'FL_calf_joint': -2.7227,   # [rad]
            'RL_calf_joint': -2.7227,    # [rad]
            'FR_calf_joint': -2.7227,  # [rad]
            'RR_calf_joint': -2.7227,    # [rad]
        }
        joint_angles_range_high = {
            'FL_hip_joint': 1.0472,   # [rad]
            'RL_hip_joint': 1.0472,   # [rad]
            'FR_hip_joint': 1.0472 ,  # [rad]
            'RR_hip_joint': 1.0472,   # [rad]

            'FL_thigh_joint': 3.4907,     # [rad]
            'RL_thigh_joint': 4.5379,   # [rad]
            'FR_thigh_joint': 3.4907,     # [rad]
            'RR_thigh_joint': 4.5379,   # [rad]

            'FL_calf_joint': -0.83776,   # [rad]
            'RL_calf_joint': -0.83776,    # [rad]
            'FR_calf_joint': -0.8376,  # [rad]
            'RR_calf_joint': -0.83776,    # [rad]
        }

    class control:
        control_type = 'P' # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        dt =  0.02 # 0.02 # control frequency 50Hz
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        
    class asset:
        dof_names = ["joint_a", "joint_b"]
        file = ""
        links_to_keep = []     # links that are not merged because of fixed joints
        foot_name = "None"     # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        links_to_keep = []
        self_collisions = True   # enable self collisions by default
        fix_base_link = False    # fix base link to the world
        
    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.
        randomize_com_displacement = True
        com_displacement_range = [-0.01, 0.01]
        simulate_action_latency = True # 1 step delay
    
    class depth:
        use_camera = False
        render_window = False
        camera_num_envs = 192
        camera_terrain_num_rows = 10
        camera_terrain_num_cols = 20

        position = [0.4, 0, 0.1]  # front camera
        angle = [0, -60, -90]  # positive pitch down

        update_interval = 5  # 5 works without retraining, 8 worse

        original = (106, 60)
        resized = (87, 58)
        horizontal_fov = 87
        buffer_len = 2
        
        near_clip = 0
        far_clip = 2
        dis_noise = 0.0
        
        scale = 1
        invert = True

    class rewards:
        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            tracking_yaw = 1.0
            tracking_goal_position = 1.0
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.
            torques = -0.00001
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0. 
            feet_air_time = 1.0
            collision = -1.
            feet_stumble = -0.0 
            action_rate = -0.01
            stand_still = -0.
            action_close_default = 0.0
            dof_close_to_default = 0.0
        
        only_positive_rewards = True
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 1.
        termination_if_roll_greater_than= 0.4
        termination_if_pitch_greater_than= 0.4
        termination_if_height_lower_than= 0.2

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]       # [m]
        lookat = [11., 5, 3.]  # [m]
        num_rendered_envs = 1  # number of environments to be rendered
        add_camera = False

    class sim:
        use_implicit_controller = False
        gravity = [0., 0. ,-9.81]  # [m/s^2]


class LeggedRobotCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration
        max_iterations = 1500 # number of policy updates

        # logging
        save_interval = 50 # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt

class LeggedRobotCfgSample(BaseConfig):
    seed: int = 1
    class planner:
        name: str = 'MPPI'
        num_samples: int = 20  # Reduced from 100 to 20 to reduce memory requirements
        horizon: int = 30
        num_knots: int = 4
        sampling_method: str = 'gaussian'
        device: str = 'cuda:0'
        sample_noise: float = 0.1

    class rollout_env:
        dt: float = 0.02
        substeps: int = 2