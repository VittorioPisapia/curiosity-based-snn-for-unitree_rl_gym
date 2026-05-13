from legged_gym import LEGGED_GYM_ROOT_DIR
import os


import isaacgym
from isaacgym import gymapi
import random

from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger, get_load_path, set_seed
from legged_gym.utils.helpers import  configure_commands
from datetime import datetime

import numpy as np


from legged_gym.utils.recorder import VideoRecorder
from legged_gym.utils.play_logger import RobotLogger
from legged_gym.utils.plotting import plot_run, show_plot


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # override some parameters for testing
    if args.seed is not None:
        env_cfg.seed = args.seed
        np.random.seed(args.seed)
    if args.num_envs is None:
        env_cfg.env.num_envs = 1
    else:
        env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    
    command_mode = args.cmd_type
    timestamp = datetime.now().strftime('%b%d_%H-%M-%S')

    robot_idx = 0 
    logger = RobotLogger()
    recorder = None

    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = args.push
    env_cfg.domain_rand.push_interval_s=5
    env_cfg.domain_rand.max_push_vel_xy=1

    configure_commands(env_cfg, command_mode)

    env_cfg.env.test = True

    if hasattr(env_cfg.env, "enable_camera_sensors"):
        env_cfg.env.enable_camera_sensors = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    if args.record:

        experiment_root = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            'logs',
            train_cfg.runner.experiment_name
        )
        model_path = get_load_path(
            root=experiment_root,
            load_run=args.load_run,
            checkpoint=args.checkpoint
        )
        experiment_dir = os.path.dirname(model_path)
        video_dir = os.path.join(experiment_dir, 'videos')
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(
            video_dir,
            f"{timestamp}_seed_{env_cfg.seed}.mp4"
        )
        recorder = VideoRecorder(env, video_path)

    try:
        for _ in range(700):

            actions = policy(obs.detach())
            obs, _, rews, dones, infos = env.step(actions.detach())
            env.gym.refresh_rigid_body_state_tensor(env.sim)

            logger.log_step(env, robot_idx)

            if recorder:
                recorder.capture_frame()
    
    except KeyboardInterrupt:
        
        print("\nSimulation stopped by user! Generating plots...")
    

    if logger.has_data():

        print(f"Plotting {logger.num_steps} steps of simulation...")
        fig = plot_run(logger, env, env_cfg)

        if args.plot:
            experiment_root = os.path.join(
                LEGGED_GYM_ROOT_DIR,
                'logs',
                train_cfg.runner.experiment_name
            )
            model_path = get_load_path(
                root=experiment_root,
                load_run=args.load_run,
                checkpoint=args.checkpoint
            )
            experiment_dir = os.path.dirname(model_path)
            plots_dir = os.path.join(experiment_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            plot_path = os.path.join(
                plots_dir,
                f"{timestamp}_seed_{env_cfg.seed}.png"
            )
            fig.savefig(plot_path, dpi=300)
            print(f"Plots saved in : {plot_path}")

        show_plot()

    else:
        print("No data collected.")

    if recorder:
        recorder.close()

if __name__ == '__main__':
    EXPORT_POLICY = True
    MOVE_CAMERA = False
    args = get_args()
    play(args)
