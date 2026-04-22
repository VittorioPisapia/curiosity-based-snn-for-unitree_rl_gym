from .go2_snn_config import GO2RoughSNNCfgPPO

class GO2RoughICMCfgPPO( GO2RoughSNNCfgPPO):

    runner_class_name = "SnnRunner"
    class policy:

        class snn:
            snn_threshold = 0.3
            snn_lens = 0.3
            snn_st = 1
            neuron_type = "Gaussian"

        class icm:
            icm_beta = 0.3
            icm_intrinsic_coeff = 0.02
            icm_reward_clamp = 0.05
            use_icm = False
            use_rnd = False
            rnd_intrinsic_coeff = 0.007
            rnd_reward_clamp = 0.05

        class runner:
            policy_class_name = "ActorCriticSnn"
            experiment_name = "rough_go2_snn"
