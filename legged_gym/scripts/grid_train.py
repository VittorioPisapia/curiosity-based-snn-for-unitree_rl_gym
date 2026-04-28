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

from itertools import product

def set_nested_attr(obj, attr, value):
    parts = attr.split(".")
    for p in parts[:-1]:
        if not hasattr(obj, p):
            raise AttributeError(f"{p} not found in {obj}")
        obj = getattr(obj, p)
    if not hasattr(obj, parts[-1]):
        raise AttributeError(f"{parts[-1]} not found in {obj}")
    setattr(obj, parts[-1], value)

search_space = {

    # "policy.snn.snn_lens": [],
    # "policy.snn.snn_threshold": [],
    # "policy.snn.snn_st": [],
    # "policy.snn.neuron_type": [],
    # "policy.snn.num_neurons": [],
    
    # "policy.icm.use_icm": [],
    # "policy.icm.icm_beta": [],
    # "policy.icm.icm_reward_clamp": [],
    # "policy.icm.icm_epochs": [],
    # "policy.icm.num_mini_batches": [],

    # "policy.icm.use_rnd": [],
    # "policy.icm.rnd_intrinsic_coeff": [],
    # "policy.icm.rnd_reward_clamp": []

}

def train(args, params, log_root):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    _, train_cfg = task_registry.get_cfgs(args.task)

    for k, v in params.items():
        set_nested_attr(train_cfg, k, v)

    ppo_runner, _ = task_registry.make_alg_runner(
        env=env, train_cfg=train_cfg, args=args, log_root=log_root
    )

    ppo_runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True
    )

    _, avg_reward = ppo_runner.alg.storage.get_statistics()

    env.gym.destroy_sim(env.sim)
    torch.cuda.empty_cache()

    return ppo_runner.log_dir, avg_reward.item()

if __name__ == '__main__':
    args = get_args()
    args.headless = True
    grid_log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 'grid_search_' + datetime.now().strftime('%b%d_%H-%M-%S'))
    os.makedirs(grid_log_root, exist_ok=True)

    results_path = os.path.join(grid_log_root, 'grid_search_results.json')

    # Load existing results if they exist
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
        print(f"Loaded existing results with {len(results)} entries.")
    else:
        results = {}

    keys = list(search_space.keys())
    values = list(search_space.values())

    combos = list(product(*values))
    total = len(combos)

    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))

        key = "_".join(
            f"{k.replace('.', '-')}_{v:.4g}" if isinstance(v, float)
            else f"{k.replace('.', '-')}_{v}"
            for k, v in params.items()
        )

        # Skip already completed runs
        if key in results:
            print(f"[{i}/{total}] Skipping {key}, already done.")
            continue

        print(f"[{i}/{total}] Running with: {params}")

        try:
            log_dir, avg_reward = train(args, params, grid_log_root)

            results[key] = {
                "log_dir": log_dir,
                "avg_reward": avg_reward
            }

        except Exception as e:
            print(f"Failed for {params}: {e}")
            results[key] = {
                "error": str(e)
            }

        # Save progressively
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
    print(f"Grid search completed. Logs saved in {grid_log_root}")