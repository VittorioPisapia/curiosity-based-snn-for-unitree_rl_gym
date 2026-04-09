import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import imageio
import os
from datetime import datetime


import isaacgym # type: ignore
from isaacgym import gymapi # Required for camera setup
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger, get_load_path

import numpy as np
import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s=5
    env_cfg.domain_rand.max_push_vel_xy=1.5

    env_cfg.commands.ranges.lin_vel_x=[-1,1]
    env_cfg.commands.ranges.lin_vel_y=[-1, 1]
    env_cfg.commands.ranges.ang_vel_yaw=[-1, 1]
    env_cfg.commands.ranges.heading=[-3.14,3.14]

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

        experiment_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)     
        model_path = get_load_path(root=experiment_root, 
                                   load_run=args.load_run, 
                                   checkpoint=args.checkpoint)
        experiment_dir = os.path.dirname(model_path)
        video_dir = os.path.join(experiment_dir, 'videos')
        os.makedirs(video_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%b%d_%H-%M-%S')
        video_path = os.path.join(video_dir, f"{timestamp}_headless_eval.mp4")

        camera_props = gymapi.CameraProperties()
        camera_props.width = 1280
        camera_props.height = 720
        camera_handle = env.gym.create_camera_sensor(env.envs[0], camera_props)
        
        camera_position = gymapi.Vec3(4.0, 4.0, 3.0)
        camera_target = gymapi.Vec3(0.0, 0.0, 0.5)
        env.gym.set_camera_location(camera_handle, env.envs[0], camera_position, camera_target)
        
        video_writer = imageio.get_writer(video_path, fps=50)
        print("Starting headless video recording...")
        print(f"Video will be saved to: {video_path}")

    for i in range(10*int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

        if args.record:
            robot_pos = env.root_states[0, :3].cpu().numpy()
            env_origin = env.gym.get_env_origin(env.envs[0])
            local_x = robot_pos[0] - env_origin.x
            local_y = robot_pos[1] - env_origin.y
            local_z = robot_pos[2] - env_origin.z
            
            cam_pos = gymapi.Vec3(local_x + 2.0, local_y + 2.0, local_z + 1.0)
            cam_target = gymapi.Vec3(local_x, local_y, local_z)
            env.gym.set_camera_location(camera_handle, env.envs[0], cam_pos, cam_target)
            
            env.gym.fetch_results(env.sim, True)

            env.gym.step_graphics(env.sim)
            env.gym.render_all_camera_sensors(env.sim)
            
            image = env.gym.get_camera_image(env.sim, env.envs[0], camera_handle, gymapi.IMAGE_COLOR)
            image_np = image.reshape((camera_props.height, camera_props.width, 4))
            rgb_image = image_np[..., :3] 
            
            video_writer.append_data(rgb_image)

    if args.record:
        video_writer.close()
        print("Saved video successfully to headless_eval.mp4!")


if __name__ == '__main__':
    EXPORT_POLICY = True
    MOVE_CAMERA = False
    args = get_args()
    play(args)
