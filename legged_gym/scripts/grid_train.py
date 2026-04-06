import os
import numpy as np
from datetime import datetime
import sys
import json

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.task_registry import LEGGED_GYM_ROOT_DIR
import torch

lens_search = [0.3]
threshold_search = [0.5]
st_search = [1]
eta_search = [0.001, 0.005, 0.01]
beta_search = [0.1, 0.2, 0.3]
intrinsic_coeff_search = [0.001, 0.01, 0.1]


results = {}

def train(args, lens, threshold, st, eta, beta, intrinsic_coeff, log_root):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    _, train_cfg = task_registry.get_cfgs(args.task)
    train_cfg.policy.snn_lens = lens
    train_cfg.policy.snn_threshold = threshold
    train_cfg.policy.snn_st = st
    train_cfg.policy.icm_eta = eta
    train_cfg.policy.icm_beta = beta
    train_cfg.policy.icm_intrinsic_coeff = intrinsic_coeff
    ppo_runner, _ = task_registry.make_alg_runner(env=env, train_cfg=train_cfg, args=args, log_root=log_root)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
    _, average_episode_reward = ppo_runner.alg.storage.get_statistics()
    env.gym.destroy_sim(env.sim)  
    return ppo_runner.log_dir, average_episode_reward.item()

if __name__ == '__main__':
    args = get_args()
    grid_log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 'grid_search_' + datetime.now().strftime('%b%d_%H-%M-%S'))
    os.makedirs(grid_log_root, exist_ok=True)
    results = {}
    for lens in lens_search:
        for threshold in threshold_search:
            for st in st_search:
                for eta in eta_search:
                    for beta in beta_search:
                        for intrinsic_coeff in intrinsic_coeff_search:
                            print(f"Running with lens={lens}, threshold={threshold}, st={st}, eta={eta}, beta={beta}, intrinsic_coeff={intrinsic_coeff}")
                            log_dir, avg_reward = train(args, lens, threshold, st, eta, beta, intrinsic_coeff, grid_log_root)
                            results[f"lens_{lens}_threshold_{threshold}_st_{st}_eta_{eta}_beta_{beta}_intrinsic_coeff_{intrinsic_coeff}"] = {"log_dir": log_dir, "avg_reward": avg_reward}

    with open(os.path.join(grid_log_root, 'grid_search_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Grid search completed. Logs saved in {grid_log_root}")
