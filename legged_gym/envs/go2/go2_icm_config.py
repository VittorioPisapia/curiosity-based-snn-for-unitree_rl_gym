from .go2_snn_config import GO2RoughSNNCfgPPO

class GO2RoughICMCfgPPO( GO2RoughSNNCfgPPO):

    runner_class_name = "IcmRunner"
    class policy ( GO2RoughSNNCfgPPO.policy ):
        class icm:
            use_icm = True
            icm_beta = 0.3
            icm_intrinsic_coeff = 0.02
            icm_reward_clamp = 0.05
            icm_epochs = 1
            icm_num_mini_batches = 32
            icm_decay = 0.9999
            
            use_rnd = False
            rnd_intrinsic_coeff = 0.007
            rnd_reward_clamp = 0.05
            rnd_decay = 0.9999

    class runner ( GO2RoughSNNCfgPPO.runner ):
        experiment_name = "rough_go2_icm"