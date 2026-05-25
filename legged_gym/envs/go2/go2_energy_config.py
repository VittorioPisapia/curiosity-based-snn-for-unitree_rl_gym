from .go2_config import GO2RoughCfg, GO2RoughCfgPPO

class GO2RoughEnergyCfgPPO( GO2RoughCfgPPO):

    runner_class_name = "SnnRunner_energy"
    class policy ( GO2RoughCfgPPO.policy ):
        class snn:
            snn_threshold = 0.4 # 0.5
            snn_lens = 0.3 # 0.3 Gaussian width
            snn_st = 1  # 1
            neuron_type = "Gaussian" # Gaussian, BPTT
            num_neurons = [384, 384] # 256, 384, 512

    class runner ( GO2RoughCfgPPO.runner ):
        policy_class_name = "ActorCriticSNN"
        experiment_name = "rough_go2_energy"
        algorithm_class_name = 'PPO_energy'

    class algorithm ( GO2RoughCfgPPO.algorithm):
        use_symmetry = True

        use_spike_loss = True
        spike_loss_coeff = 0.05
        spike_rate_target = [0.15, 0.10]

        use_rnd = True

        class rnd:
            num_obs = 27 
            num_outputs = 4
            predictor_hidden_dims = [32, 32]
            target_hidden_dims = [32]
            learning_rate = 1.e-4
            weight = 0.00013

        class symmetry:
            use_data_augmentation = False
            use_mirror_loss = True
            mirror_loss_coeff = 0.5

        class energy:
            forward_loss_coeff = 0.5
            energy_loss_coeff = 0.1


        