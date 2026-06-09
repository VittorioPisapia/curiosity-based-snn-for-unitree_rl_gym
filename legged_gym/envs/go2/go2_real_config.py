from .go2_snn_config import GO2RoughSNNCfgPPO
from .go2_config import GO2RoughCfg

class GO2RoughRealCfg( GO2RoughCfg ):
    class env ( GO2RoughCfg.env):
        num_envs = 8000
        num_observations = 45 # add 187 for height measurements
        num_privileged_obs = 48 # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_obs_hist = 1
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds
        test = False
  
    class rewards( GO2RoughCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.3
        class scales( GO2RoughCfg.rewards.scales ):
            torques = -0.0002
            dof_pos_limits = -10.0

            # New
            base_height = -10
            feet_air_time = 0.01
            action_rate = -0.02
            slip = -0.01
            cost_of_transport = -0.1 # -0.05 ~ -0.2
            reward_feet_distance = -1.0

class GO2RoughRealCfgPPO( GO2RoughSNNCfgPPO):

    runner_class_name = "RndRunner"
    class policy ( GO2RoughSNNCfgPPO.policy ):
        class snn:
            snn_threshold = 0.4 # 0.5
            snn_lens = 0.3 # 0.3 Gaussian width
            snn_st = 1  # 1
            neuron_type = "Gaussian" # Gaussian, BPTT
            num_neurons = [384, 384] # 256, 384, 512

    class runner ( GO2RoughSNNCfgPPO.runner ):
        experiment_name = "rough_go2_real"
        algorithm_class_name = 'PPO_Rnd'

    class algorithm ( GO2RoughSNNCfgPPO.algorithm ):
        use_rnd = False
        class rnd:
            num_obs = 27 
            num_outputs = 4
            predictor_hidden_dims = [32, 32]
            target_hidden_dims = [32]
            learning_rate = 1.e-4
            weight = 0.00015

        use_spike_loss = False
        spike_loss_coeff = 0.05
        spike_rate_target = [0.10]

        use_symmetry = True
        class symmetry:
            use_data_augmentation = False
            use_mirror_loss = True
            mirror_loss_coeff = 1