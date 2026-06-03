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

VELOCITIES = [0.5, 0.75, 1, 1.2, 1.3]
SEEDS = [0,1,2,3,4,5]

#VELOCITIES = [0.5, 0.75, 1, 1.2, 1.3]
#SEEDS = [0]

EPISODE_STEPS = 600 

def safe_corr(a, b):
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return np.nan
    return np.corrcoef(a, b)[0, 1]

def compute_metrics(logger, env):
    """
    Returns:
        dict containing:
            - tracking metrics
            - CoT metrics
            - gait metrics
    """

    time_axis = np.arange(logger.num_steps) * env.dt
    mask = time_axis >= 3.0

    cmd = np.array(logger.cmd_vel_x)[mask]
    act = np.array(logger.act_vel_x)[mask]

    cot = np.array(logger.cot)[mask]

    contacts = np.array(logger.contacts)[mask]

    vel_err = np.abs(cmd - act)

    rmse = np.sqrt(np.mean((cmd - act) ** 2))
    mae = np.mean(vel_err)

    cot_mean = np.mean(cot)
    cot_std = np.std(cot)

    contact_variance = np.var(contacts)

    FL = contacts[:, 0].astype(float)
    FR = contacts[:, 1].astype(float)
    RL = contacts[:, 2].astype(float)
    RR = contacts[:, 3].astype(float)

    df_fl = np.mean(FL)
    df_fr = np.mean(FR)
    df_rl = np.mean(RL)
    df_rr = np.mean(RR)

    duty_factor = np.mean([
        df_fl,
        df_fr,
        df_rl,
        df_rr
    ])

    sym_front = abs(df_fl - df_fr)
    sym_rear = abs(df_rl - df_rr)

    stance_symmetry = 0.5 * (
        sym_front +
        sym_rear
    )

    left_stance = FL.sum() + RL.sum()
    right_stance = FR.sum() + RR.sum()

    lr_balance = (
        left_stance /
        (right_stance + 1e-8)
    )

    diag_corr_1 = safe_corr(FL, RR)
    diag_corr_2 = safe_corr(FR, RL)

    diag_sync = np.nanmean([
        diag_corr_1,
        diag_corr_2
    ])

    lat_corr_1 = safe_corr(FL, FR)
    lat_corr_2 = safe_corr(RL, RR)

    lat_sync = np.nanmean([
        lat_corr_1,
        lat_corr_2
    ])

    support_legs = contacts.sum(axis=1)

    mean_support = np.mean(support_legs)
    std_support = np.std(support_legs)

    switches = []

    for foot in [FL, FR, RL, RR]:

        transitions = np.sum(
            np.abs(np.diff(foot))
        )

        switches.append(
            transitions /
            len(foot)
        )

    switch_frequency = np.mean(switches)

    return {

        # Tracking
        "rmse": rmse,
        "mae": mae,

        # CoT
        "cot": cot_mean,
        "cot_std": cot_std,

        # Generic contacts
        "contact_var": contact_variance,

        # Duty factor
        "df_fl": df_fl,
        "df_fr": df_fr,
        "df_rl": df_rl,
        "df_rr": df_rr,
        "duty_factor": duty_factor,

        # Symmetry
        "sym_front": sym_front,
        "sym_rear": sym_rear,
        "stance_symmetry": stance_symmetry,

        # Left-right balance
        "lr_balance": lr_balance,

        # Diagonal synchronization
        "diag_corr_1": diag_corr_1,
        "diag_corr_2": diag_corr_2,
        "diag_sync": diag_sync,

        # Lateral synchronization
        "lat_corr_1": lat_corr_1,
        "lat_corr_2": lat_corr_2,
        "lat_sync": lat_sync,

        # Support
        "mean_support": mean_support,
        "std_support": std_support,

        # Switching
        "switch_frequency": switch_frequency,
    }


def run_episode(env, policy, speed, seed, env_cfg, save_dir=None, record_video=True, make_plot=True):

    set_seed(seed)
    env_cfg.env.test = True 

    logger = RobotLogger()
    recorder = None

    if record_video and save_dir is not None:
        video_path = os.path.join(save_dir, "video.mp4")
        recorder = VideoRecorder(env, video_path)

    env.reset()

    env.commands[:] = 0.0
    env.commands[:, 0] = speed  
    env.commands[:, 1] = 0.0   
    env.commands[:, 2] = 0.0    

    env_cfg.terrain.num_rows = 3
    env_cfg.terrain.num_cols = 3
    env_cfg.terrain.max_init_terrain_level = 6
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s=5
    env_cfg.domain_rand.max_push_vel_xy=1
    
    if hasattr(env, 'command_substeps'):
        env.command_substeps = 1000000000000000000
    elif hasattr(env, 'cfg'):
        env.cfg.commands.resampling_time = 100000000000000000

    obs = env.get_observations()

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

            time_axis = np.arange(logger.num_steps) * env.dt
            mask = time_axis >= 3.0

            contacts = np.array(
                logger.contacts,
                dtype=np.uint8
            )

            contacts = contacts[mask]

            metrics["binary_pattern"] = contacts.T
            
            np.save(
                os.path.join(run_dir, "binary_pattern.npy"),
                contacts.T.astype(np.uint8)
            )
            with open(os.path.join(run_dir, "metrics.txt"), "w") as f:
                tmp = metrics.copy()
                tmp.pop("binary_pattern")
                f.write(str(tmp))

            results[v][seed] = metrics

    return results

def aggregate(results):

    vel_list = []

    cot_mean = []
    cot_std = []

    rmse_mean = []
    rmse_std = []

    duty_factor_mean = []
    diag_sync_mean = []
    stance_symmetry_mean = []
    support_mean = []
    lr_balance_mean = []
    switch_mean = []

    for v, seed_dict in results.items():

        cots = [seed_dict[s]["cot"] for s in seed_dict]
        rmses = [seed_dict[s]["rmse"] for s in seed_dict]

        dfs = [seed_dict[s]["duty_factor"] for s in seed_dict]
        diags = [seed_dict[s]["diag_sync"] for s in seed_dict]
        syms = [seed_dict[s]["stance_symmetry"] for s in seed_dict]
        supports = [seed_dict[s]["mean_support"] for s in seed_dict]
        balances = [seed_dict[s]["lr_balance"] for s in seed_dict]
        switches = [seed_dict[s]["switch_frequency"] for s in seed_dict]

        vel_list.append(v)

        cot_mean.append(np.mean(cots))
        cot_std.append(np.std(cots))

        rmse_mean.append(np.mean(rmses))
        rmse_std.append(np.std(rmses))

        duty_factor_mean.append(np.mean(dfs))
        diag_sync_mean.append(np.mean(diags))
        stance_symmetry_mean.append(np.mean(syms))
        support_mean.append(np.mean(supports))
        lr_balance_mean.append(np.mean(balances))
        switch_mean.append(np.mean(switches))

    return {

        "vel": np.array(vel_list),

        "cot_mean": np.array(cot_mean),
        "cot_std": np.array(cot_std),

        "rmse_mean": np.array(rmse_mean),
        "rmse_std": np.array(rmse_std),

        "duty_factor": np.array(duty_factor_mean),
        "diag_sync": np.array(diag_sync_mean),
        "stance_symmetry": np.array(stance_symmetry_mean),
        "mean_support": np.array(support_mean),
        "lr_balance": np.array(lr_balance_mean),
        "switch_frequency": np.array(switch_mean),
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
    
    # Duty Factor
    plt.figure()
    plt.plot(
        agg["vel"],
        agg["duty_factor"],
        marker="o"
    )
    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Duty Factor")
    plt.grid()
    plt.savefig(
        os.path.join(out_dir, "duty_factor.png"),
        dpi=300
    )

    # Diagonal Sync
    plt.figure()
    plt.plot(
        agg["vel"],
        agg["diag_sync"],
        marker="o"
    )
    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Diagonal Synchronization")
    plt.grid()
    plt.savefig(
        os.path.join(out_dir, "diag_sync.png"),
        dpi=300
    )

    # Support Legs
    plt.figure()
    plt.plot(
        agg["vel"],
        agg["mean_support"],
        marker="o"
    )
    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Mean Supporting Legs")
    plt.grid()
    plt.savefig(
        os.path.join(out_dir, "support_legs.png"),
        dpi=300
    )

    # Stance Symmetry
    plt.figure()
    plt.plot(
        agg["vel"],
        agg["stance_symmetry"],
        marker="o"
    )
    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Stance Symmetry")
    plt.grid()
    plt.savefig(
        os.path.join(out_dir, "stance_symmetry.png"),
        dpi=300
    )

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