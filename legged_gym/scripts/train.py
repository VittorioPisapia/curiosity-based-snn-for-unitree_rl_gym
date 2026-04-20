import os
import numpy as np
from datetime import datetime
import sys

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

import json

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

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)

    config_dict = {
        "env_config": class_to_dict(env_cfg),
        "train_config": class_to_dict(train_cfg)
    }

    with open(os.path.join(ppo_runner.log_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=4, default=str)

    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)


if __name__ == '__main__':
    args = get_args()
    train(args)
