from .go2_config import GO2RoughCfg, GO2RoughCfgPPO

class GO2RoughSNNCfgPPO( GO2RoughCfgPPO):

    runner_class_name = "SnnRunner"
    class policy:

        class snn:

            snn_threshold = 0.3
            snn_lens = 0.3
            snn_st = 1
            neuron_type = "Gaussian"

        class runner:
            policy_class_name = "ActorCriticSNN"
            experiment_name = "rough_go2_snn"

        