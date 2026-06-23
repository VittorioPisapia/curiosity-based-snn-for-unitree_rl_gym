from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from isaacgym import gymapi, gymtorch
import torch
import numpy as np

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
def camera_target_from_transform(cam, distance=1.0):

    qx = cam.r.x
    qy = cam.r.y
    qz = cam.r.z
    qw = cam.r.w

    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)]
    ])

    forward = R[:, 2]

    target = np.array([
        cam.p.x,
        cam.p.y,
        cam.p.z
    ]) + distance * forward

    return target

def reset_robot(env):

    env.root_states[:] = env.base_init_state
    env.root_states[:, :3] += env.env_origins

    env.dof_pos[:] = env.default_dof_pos
    env.dof_vel[:] = 0.

    env.dof_state[:, :, 0] = env.dof_pos
    env.dof_state[:, :, 1] = env.dof_vel

    env.gym.set_actor_root_state_tensor(
        env.sim,
        gymtorch.unwrap_tensor(env.root_states)
    )

    env.gym.set_dof_state_tensor(
        env.sim,
        gymtorch.unwrap_tensor(env.dof_state)
    )

    env.reset_buf[:] = 0
    env.progress_buf[:] = 0


def play(args):

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    env_cfg.env.num_envs = 1
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.commands.heading_command = False
    env_cfg.env.episode_length_s = 1000000
    env, _ = task_registry.make_env(
        name=args.task,
        args=args,
        env_cfg=env_cfg
    )

    env.set_camera(
        [1.041, -3.016, 1.260],
        [1.075, -2.043, 1.031]
    )
    
    obs = env.get_observations()

    train_cfg.runner.resume = True

    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg
    )

    policy = ppo_runner.get_inference_policy(
        device=env.device
    )

    ########################################################
    # TELEOP COMMANDS
    ########################################################

    vx = 0.0
    vy = 0.0
    yaw = 0.0

    gym = env.gym
    viewer = env.viewer

    gym.subscribe_viewer_keyboard_event(
        viewer,
        gymapi.KEY_W,
        "vx_plus"
    )

    gym.subscribe_viewer_keyboard_event(
        viewer,
        gymapi.KEY_S,
        "vx_minus"
    )

    gym.subscribe_viewer_keyboard_event(
        viewer,
        gymapi.KEY_Q,
        "yaw_plus"
    )

    gym.subscribe_viewer_keyboard_event(
        viewer,
        gymapi.KEY_E,
        "yaw_minus"
    )

    gym.subscribe_viewer_keyboard_event(
        viewer,
        gymapi.KEY_A,
        "vy_plus"
    )

    gym.subscribe_viewer_keyboard_event(
        viewer,
        gymapi.KEY_D,
        "vy_minus"
    )

    gym.subscribe_viewer_keyboard_event(
        viewer,
        gymapi.KEY_SPACE,
        "zero_cmd"
    )

    gym.subscribe_viewer_keyboard_event(
        viewer,
        gymapi.KEY_R,
        "reset_robot"
    )

    gym.subscribe_viewer_keyboard_event(
        viewer,
        gymapi.KEY_C,
        "print_camera"
    )

    print()
    print("========== TELEOP ==========")
    print("W      : vx += 0.1")
    print("S      : vx -= 0.1")
    print("A      : vy += 0.1")
    print("D      : vy -= 0.1")
    print("Q      : yaw += 0.1")
    print("E      : yaw -= 0.1")
    print("SPACE  : zero commands")
    print("R      : reset robot")
    print("============================")
    print()

    while not gym.query_viewer_has_closed(viewer):

        ####################################################
        # PROCESS KEYBOARD EVENTS
        ####################################################

        for evt in gym.query_viewer_action_events(viewer):

            if evt.value <= 0:
                continue

            if evt.action == "vx_plus":
                vx += 0.1
                print(f"vx = {vx:.2f}")

            elif evt.action == "vx_minus":
                vx -= 0.1
                print(f"vx = {vx:.2f}")

            if evt.action == "vy_plus":
                vy += 0.1
                print(f"vy = {vy:.2f}")

            elif evt.action == "vy_minus":
                vy -= 0.1
                print(f"vy = {vy:.2f}")

            elif evt.action == "yaw_plus":
                yaw += 0.1
                print(f"yaw = {yaw:.2f}")

            elif evt.action == "yaw_minus":
                yaw -= 0.1
                print(f"yaw = {yaw:.2f}")

            elif evt.action == "zero_cmd":
                vx = 0.0
                vy = 0.0
                yaw = 0.0
                print("commands reset")

            elif evt.action == "reset_robot":
                env.reset_idx(torch.tensor([0], device=env.device))
                print("robot reset")

            elif evt.action == "print_camera":

                cam = gym.get_viewer_camera_transform(viewer, None)

                target = camera_target_from_transform(cam)

                print("\n===== CAMERA =====")

                print(
                    f"env.set_camera(\n"
                    f"    gymapi.Vec3({cam.p.x:.3f}, {cam.p.y:.3f}, {cam.p.z:.3f}),\n"
                    f"    gymapi.Vec3({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f})\n"
                    f")"
                )

                print("==================\n")

        ####################################################
        # APPLY COMMANDS
        ####################################################

        env.commands[:, 0] = vx
        env.commands[:, 1] = vy
        env.commands[:, 2] = yaw

        ####################################################
        # STEP POLICY
        ####################################################

        actions = policy(obs.detach())

        obs, _, _, _, _ = env.step(
            actions.detach()
        )


if __name__ == "__main__":

    args = get_args()

    play(args)

