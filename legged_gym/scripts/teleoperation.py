from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from isaacgym import gymapi, gymtorch
import torch
import numpy as np
import pygame

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.play_logger import RobotLogger
from legged_gym.utils.plotting import plot_run, show_plot
from legged_gym.utils.recorder import VideoRecorder

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
    env_cfg.domain_rand.push_interval_s=5
    env_cfg.domain_rand.max_push_vel_xy=1000
    env, _ = task_registry.make_env(
        name=args.task,
        args=args,
        env_cfg=env_cfg
    )

    env.set_camera(
        [-1.213, -2.786, 0.916],
        [-1.179, -1.813, 0.687]
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

    pygame.init()
    pygame.joystick.init()

    joystick = None

    if pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()

        print(f"Controller connected: {joystick.get_name()}")
    else:
        print("No controller detected.")

    ########################################################
    # TELEOP COMMANDS
    ########################################################

    vx = 0.0
    vy = 0.0
    yaw = 0.0

    max_vx = 0.5
    max_vy = 0.5
    max_yaw = 1.0

    gym = env.gym
    viewer = env.viewer

    recording = False

    logger = None
    recorder = None

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

    gym.subscribe_viewer_keyboard_event(
        viewer,
        gymapi.KEY_L,
        "toggle_recording"
    )

    print()
    print("========== TELEOP ==========")
    print("KEYBOARD")
    print("  W/S : vx +/-")
    print("  A/D : vy +/-")
    print("  Q/E : yaw +/-")
    print("  SPACE : stop")
    print("  R : reset")
    print()
    print("XBOX")
    print("  Left stick  : vx / vy")
    print("  Right stick : yaw")
    print("  A : reset")
    print("  B : stop")
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
                vx += 0.5
                print(f"vx = {vx:.2f}")

            elif evt.action == "vx_minus":
                vx -= 0.5
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

            elif evt.action == "toggle_recording":

                if not recording:

                    logger = RobotLogger()

                    recorder = VideoRecorder(
                        env=env,
                        video_path="run.mp4",
                        fps=round(1.0 / env.dt)
                    )

                    recording = True

                    print("Recording started.")

                else:

                    recording = False

                    recorder.close()

                    plot_run(
                        logger,
                        env,
                        env_cfg
                    )

                    show_plot()

                    sim_time = logger.num_steps * env.dt
                    video_time = logger.num_steps / recorder.fps

                    print(f"Simulation time : {sim_time:.2f}s")
                    print(f"Video duration  : {video_time:.2f}s")

        ####################################################
        # GAMEPAD
        ####################################################

        if joystick is not None:

            pygame.event.pump()

            left_x = joystick.get_axis(0)
            left_y = joystick.get_axis(1)

            try:
                right_x = joystick.get_axis(3)
            except:
                right_x = 0.0

            deadzone = 0.10

            if abs(left_x) < deadzone:
                left_x = 0.0

            if abs(left_y) < deadzone:
                left_y = 0.0

            if abs(right_x) < deadzone:
                right_x = 0.0

            vx = -left_y * max_vx
            vy = left_x * max_vy
            yaw = right_x * max_yaw

            ################################################
            # Buttons
            ################################################

            if joystick.get_button(0):  # A
                env.reset_idx(
                    torch.tensor([0], device=env.device)
                )
                print("robot reset")

            if joystick.get_button(1):  # B
                vx = 0.0
                vy = 0.0
                yaw = 0.0

            if joystick.get_button(4):  # LB
                max_vx = max(0.2, max_vx - 0.01)

            if joystick.get_button(5):  # RB
                max_vx = min(3.0, max_vx + 0.01)
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
        if recording:

            logger.log_step(env)

            recorder.capture_frame()

    if recording:

        recorder.close()

        plot_run(
            logger,
            env,
            env_cfg
        )

        show_plot()
if __name__ == "__main__":

    args = get_args()

    play(args)

