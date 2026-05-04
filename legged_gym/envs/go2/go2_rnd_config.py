from .go2_snn_config import GO2RoughSNNCfgPPO

class GO2RoughRNDCfgPPO( GO2RoughSNNCfgPPO):

    runner_class_name = "RndRunner"
    class policy ( GO2RoughSNNCfgPPO.policy ):

    class runner ( GO2RoughSNNCfgPPO.runner ):
        experiment_name = "rough_go2_rnd"

    class algorithm ( GO2RoughSNNCfgPPO.algorithm ):
        use_rnd = True

        class rnd:
            num_obs = 48 - 3
            num_outputs = 24
            predictor_hidden_dims = [24, 24]
            target_hidden_dims = [24]
            learning_rate = 1.e-4
            weight = 100