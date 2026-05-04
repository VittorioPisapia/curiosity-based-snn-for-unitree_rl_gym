from .go2_snn_config import GO2RoughSNNCfgPPO

class GO2RoughRNDCfgPPO( GO2RoughSNNCfgPPO):

    runner_class_name = "RndRunner"

    class runner ( GO2RoughSNNCfgPPO.runner ):
        experiment_name = "rough_go2_rnd"

    class algorithm ( GO2RoughSNNCfgPPO.algorithm ):
        use_rnd = True

        class rnd:
            num_obs = 48 - 3 - 12
            num_outputs = 64
            predictor_hidden_dims = [128, 128]
            target_hidden_dims = [128]
            learning_rate = 1.e-4
            weight = 0.00008