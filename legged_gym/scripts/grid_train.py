import os
import numpy as np
from datetime import datetime
import sys
import json
import multiprocessing as mp # Importante: aggiunto multiprocessing

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

def class_to_dict(obj):
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key, val in obj.__class__.__dict__.items():
        if not key.startswith("_"):
            result[key] = class_to_dict(val)
    for key, val in obj.__dict__.items():
        if not key.startswith("_"):
            result[key] = class_to_dict(val)
    return result

search_space = {
    "seed": [1, 2, 3, 4, 5]
}

def train(args, params, log_root):
    _, train_cfg = task_registry.get_cfgs(args.task)

    for k, v in params.items():
        set_nested_attr(train_cfg, k, v)

    if 'seed' in params:
        args.seed = train_cfg.seed
        
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, _ = task_registry.make_alg_runner(
        env=env, train_cfg=train_cfg, args=args, log_root=log_root
    )

    config_dict = {
        "grid_search_params": params,
        "env_config": class_to_dict(env_cfg),
        "train_config": class_to_dict(train_cfg)
    }

    os.makedirs(ppo_runner.log_dir, exist_ok=True)
    
    with open(os.path.join(ppo_runner.log_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=4, default=str)

    ppo_runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True
    )

    _, avg_reward = ppo_runner.alg.storage.get_statistics()
    env.gym.destroy_sim(env.sim)
    
    # Pulizia
    torch.cuda.empty_cache()

    return ppo_runner.log_dir, avg_reward.item()

def train_worker(args, params, log_root, return_dict):
    try:
        log_dir, avg_reward = train(args, params, log_root)
        return_dict['log_dir'] = log_dir
        return_dict['avg_reward'] = avg_reward
        return_dict['status'] = 'success'
    except Exception as e:
        return_dict['error'] = str(e)
        return_dict['status'] = 'error'
# --------------------------------------------------

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True) 

    args = get_args()
    args.headless = True
    
    grid_log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 'grid_search_' + datetime.now().strftime('%b%d_%H-%M-%S'))
    os.makedirs(grid_log_root, exist_ok=True)

    results_path = os.path.join(grid_log_root, 'grid_search_results.json')

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

    manager = mp.Manager()

    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))

        key = "_".join(
            f"{k.replace('.', '-')}_{v:.4g}" if isinstance(v, float)
            else f"{k.replace('.', '-')}_{v}"
            for k, v in params.items()
        )

        if key in results:
            print(f"[{i}/{total}] Skipping {key}, already done.")
            continue

        print(f"[{i}/{total}] Running with: {params}")

        return_dict = manager.dict()

        p = mp.Process(target=train_worker, args=(args, params, grid_log_root, return_dict))
        p.start()
        p.join() 


        if return_dict.get('status') == 'success':
            results[key] = {
                "log_dir": return_dict['log_dir'],
                "avg_reward": return_dict['avg_reward']
            }
        else:
            error_msg = return_dict.get('error', 'Unknown Error / Segfault')
            print(f"Failed for {params}: {error_msg}")
            results[key] = {
                "error": error_msg
            }

            if p.exitcode != 0 and 'status' not in return_dict:
                print("Segfault o crash a basso livello rilevato nel processo figlio. Procedo col prossimo seed.")
                results[key]["error"] = f"Process terminated with exit code {p.exitcode}"

            if "CUDA out of memory" in error_msg:
                print("CUDA OOM rilevato. Interruzione preventiva per evitare corruzione dati.")
                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=4)
                sys.exit(1)

        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
            
    print(f"Grid search completed. Logs saved in {grid_log_root}")