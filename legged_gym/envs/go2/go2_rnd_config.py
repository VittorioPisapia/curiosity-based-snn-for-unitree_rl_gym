from .go2_snn_config import GO2RoughSNNCfgPPO

class GO2RoughRNDCfgPPO( GO2RoughSNNCfgPPO):

    runner_class_name = "RndRunner"

    class runner ( GO2RoughSNNCfgPPO.runner ):
        experiment_name = "rough_go2_rnd"
        algorithm_class_name = 'PPO_Rnd'

    class algorithm ( GO2RoughSNNCfgPPO.algorithm ):
        use_rnd = True
        class rnd:
            num_obs = 27 
            num_outputs = 4
            predictor_hidden_dims = [32, 32]
            target_hidden_dims = [32]
            learning_rate = 1.e-4
            weight = 0.00015

        use_spike_loss = True
        spike_loss_coeff = 0.05
        spike_rate_target = [0.10]

        use_symmetry = True
        class symmetry:
            use_data_augmentation = False
            use_mirror_loss = True
            mirror_loss_coeff = 0.5