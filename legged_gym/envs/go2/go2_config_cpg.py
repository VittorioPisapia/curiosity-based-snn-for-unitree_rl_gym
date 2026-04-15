from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class GO2CPGCfg( LeggedRobotCfg ):

    class env( LeggedRobotCfg.env ): 
        num_envs = 4096
        num_observations = 68
        num_actions = 12


    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.32] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'CPG_OFFSETX'
        stiffness = {'joint':100.}  # [N*m/rad]
        damping = {'joint': 2.}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.32
        only_positive_rewards = True   
        tracking_sigma = 0.25
        max_contact_force = 150.0

        class scales(LeggedRobotCfg.rewards.scales):

            # ---- TASK ----
            tracking_lin_vel = 2.0
            tracking_ang_vel = 0.0
            forward_vel = 0.0

            # ---- STABILITY ----
            orientation = -1.0

            # ---- EFFICIENCY ----
            energy = -0.002

            # ---- CONTACT QUALITY ----
            feet_contact_forces = -0.02
            slip = -0.1

            # ---- GAIT SHAPING ----
            feet_air_time = 0.3

            # ---- LIGHT REGULARIZATION ----
            action_rate = -0.01
            torques = -0.0001
            dof_pos_limits = -5.0

            # ---- DISABLED (important for CPG) ----
            dof_acc = 0.0
            dof_vel = 0.0
            base_height = 0.0
            collision = 0.0

class GO2CPGCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_go2'

  
