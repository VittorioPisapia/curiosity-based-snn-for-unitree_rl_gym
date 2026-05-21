import isaacgym
from legged_gym.envs import *
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from legged_gym.utils.recorder import VideoRecorder
from legged_gym.utils.plotting import plot_run

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils import task_registry, get_args, set_seed, get_load_path
from legged_gym.utils.play_logger import RobotLogger

VELOCITIES = [0.75, 1, 1.25]
SEEDS = [0,1,2,3]

EPISODE_STEPS = 600 

def compute_metrics(logger, env):
    """
    Returns:
        dict with CoT, tracking error, etc.
    """

    # =========================
    # Global temporal mask
    # =========================

    time_axis = np.arange(logger.num_steps) * env.dt

    mask = time_axis >= 3.0

    # =========================
    # Apply mask to all signals
    # =========================

    cmd = np.array(logger.cmd_vel_x)[mask]
    act = np.array(logger.act_vel_x)[mask]

    cot = np.array(logger.cot)[mask]

    contacts = np.array(logger.contacts)[mask]

    # =========================
    # Metrics
    # =========================

    vel_err = np.abs(cmd - act)

    rmse = np.sqrt(np.mean((cmd - act) ** 2))
    mae = np.mean(vel_err)

    cot_mean = np.mean(cot)
    cot_std = np.std(cot)

    contact_variance = np.var(contacts)

    return {
        "rmse": rmse,
        "mae": mae,
        "cot": cot_mean,
        "cot_std": cot_std,
        "contact_var": contact_variance
    }


def run_episode(env, policy, speed, seed, env_cfg, save_dir=None, record_video=True, make_plot=True):

    set_seed(seed)
    env_cfg.env.test = True

    obs = env.get_observations()
    logger = RobotLogger()

    recorder = None

    if record_video and save_dir is not None:
        video_path = os.path.join(save_dir, "video.mp4")
        recorder = VideoRecorder(env, video_path)


    env.reset()

    env.commands[:] = 0.0
    env.commands[:] = 0
    env.commands[:, 0] = speed
    env.commands[:, 1] = 0.0
    env.commands[:, 2] = 0.0
    env_cfg.commands.resampling_time = 100000
    
    try:
        for _ in range(EPISODE_STEPS):

            actions = policy(obs.detach())
            obs, _, _, _, _ = env.step(actions.detach())

            logger.log_step(env, robot_idx=0)

            if recorder:
                recorder.capture_frame()

    finally:
        if recorder:
            recorder.close()

    if make_plot and save_dir is not None:

        fig = plot_run(logger, env, env_cfg)

        plot_path = os.path.join(save_dir, "run_plot.png")
        fig.savefig(plot_path, dpi=300)
        plt.close(fig)

    return logger

def run_benchmark(env, policy, env_cfg, out_dir):

    results = {}

    for v in VELOCITIES:
        results[v] = {}

        for seed in SEEDS:

            run_dir = os.path.join(out_dir, f"v_{v}_seed_{seed}")
            os.makedirs(run_dir, exist_ok=True)

            print(f"[Benchmark] v={v}, seed={seed}")

            is_video_run = (seed == 0)

            logger = run_episode(
                env,
                policy,
                v,
                seed,
                env_cfg,
                save_dir=run_dir if is_video_run else None,
                record_video=is_video_run,
                make_plot=is_video_run
            )

            metrics = compute_metrics(logger, env)

            with open(os.path.join(run_dir, "metrics.txt"), "w") as f:
                f.write(str(metrics))

            results[v][seed] = metrics

    return results

def aggregate(results):

    vel_list = []
    cot_mean, cot_std = [], []
    rmse_mean, rmse_std = [], []

    for v, seed_dict in results.items():

        cots = [seed_dict[s]["cot"] for s in seed_dict]
        rmses = [seed_dict[s]["rmse"] for s in seed_dict]

        vel_list.append(v)

        cot_mean.append(np.mean(cots))
        cot_std.append(np.std(cots))

        rmse_mean.append(np.mean(rmses))
        rmse_std.append(np.std(rmses))

    return {
        "vel": np.array(vel_list),
        "cot_mean": np.array(cot_mean),
        "cot_std": np.array(cot_std),
        "rmse_mean": np.array(rmse_mean),
        "rmse_std": np.array(rmse_std),
    }


def plot_results(agg, out_dir):

    os.makedirs(out_dir, exist_ok=True)

    # CoT
    plt.figure()
    plt.errorbar(
        agg["vel"],
        agg["cot_mean"],
        yerr=agg["cot_std"],
        marker="o"
    )
    plt.xlabel("Velocity (m/s)")
    plt.ylabel("CoT")
    plt.title("Cost of Transport vs Velocity")
    plt.grid()
    plt.savefig(os.path.join(out_dir, "cot_curve.png"), dpi=300)

    # tracking error
    plt.figure()
    plt.errorbar(
        agg["vel"],
        agg["rmse_mean"],
        yerr=agg["rmse_std"],
        marker="o",
        color="r"
    )
    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Velocity RMSE")
    plt.title("Tracking Error vs Velocity")
    plt.grid()
    plt.savefig(os.path.join(out_dir, "tracking_curve.png"), dpi=300)

    plt.show()

import csv

def save_csv(results, out_dir):

    csv_path = os.path.join(out_dir, "results.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["velocity", "seed", "rmse", "mae", "cot"])

        for v, seed_dict in results.items():
            for s, m in seed_dict.items():
                writer.writerow([v, s, m["rmse"], m["mae"], m["cot"]])

def main():

    args = get_args()

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    env_cfg.env.num_envs = 1
    env_cfg.env.test = True
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False

    env, _ = task_registry.make_env(args.task, args, env_cfg)

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env, args.task, args, train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    experiment_root = os.path.join(
    LEGGED_GYM_ROOT_DIR,
        "logs",
        train_cfg.runner.experiment_name
    )

    model_path = get_load_path(
        root=experiment_root,
        load_run=args.load_run,
        checkpoint=args.checkpoint
    )

    run_dir = os.path.dirname(model_path) 
    out_dir = os.path.join(run_dir, "benchmark")

    os.makedirs(out_dir, exist_ok=True)

    results = run_benchmark(env, policy, env_cfg, out_dir)

    np.save(os.path.join(out_dir, "results.npy"), results, allow_pickle=True)

    agg = aggregate(results)

    np.save(os.path.join(out_dir, "aggregated.npy"), agg)

    plot_results(agg, out_dir)

    print("Benchmark completed:", out_dir)
    save_csv(results, out_dir)


if __name__ == "__main__":
    main()