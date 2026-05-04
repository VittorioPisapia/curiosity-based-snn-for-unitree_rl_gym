from .go2_snn_config import GO2RoughSNNCfgPPO

class GO2RoughICMCfgPPO( GO2RoughSNNCfgPPO):

    runner_class_name = "RndRunner"
    class policy ( GO2RoughSNNCfgPPO.policy ):
        class icm:

            use_icm = False
            icm_beta = 0.3
            icm_intrinsic_coeff = 0.02
            icm_reward_clamp = 0.05
            icm_epochs = 5
            icm_num_mini_batches = 4

            use_rnd = True
            rnd_intrinsic_coeff = 0.01
            rnd_reward_clamp = 0.05
            rnd_num_mini_batches = 4
            rnd_epochs = 5

    class runner ( GO2RoughSNNCfgPPO.runner ):
        experiment_name = "rough_go2_icm"

    class algorithm ( GO2RoughSNNCfgPPO.algorithm ):
        use_rnd = True

        class rnd:
            num_obs = 48 - 3
            num_outputs = 32
            predictor_hidden_dims = [128, 128]
            target_hidden_dims = [128, 128]
            learning_rate = 1.e-4
            weight = 100