import os
import numpy as np
import matplotlib.pyplot as plt


def load_config(seed_paths):
    """
    seed_paths:
        lista di path agli aggregated.npy
    """

    data = [
        np.load(p, allow_pickle=True).item()
        for p in seed_paths
    ]

    vel = data[0]["vel"]

    metrics = {}

    for key in data[0].keys():

        if key == "vel":
            continue

        values = np.stack(
            [d[key] for d in data],
            axis=0
        )

        metrics[key] = {
            "mean": np.mean(values, axis=0),
            "std": np.std(values, axis=0)
        }

    return vel, metrics


def compare_metric(
    vel,
    snn_metrics,
    rnd_metrics,
    metric_name,
    ylabel=None
):

    plt.figure(figsize=(6,4))

    plt.errorbar(
        vel,
        snn_metrics[metric_name]["mean"],
        yerr=snn_metrics[metric_name]["std"],
        marker="o",
        capsize=4,
        label="SNN"
    )

    plt.errorbar(
        vel,
        rnd_metrics[metric_name]["mean"],
        yerr=rnd_metrics[metric_name]["std"],
        marker="s",
        capsize=4,
        label="RND"
    )

    plt.xlabel("Velocity [m/s]")
    plt.ylabel(ylabel or metric_name)
    plt.grid(True)
    plt.legend()

    plt.tight_layout()


# ==========================
# PATHS
# ==========================

snn_paths = [
    "/home/vittorio/Desktop/curiosity-based-snn-for-unitree_rl_gym/logs/rough_go2_snn/snn_spike_symm/benchmark/aggregated.npy",
    "/home/vittorio/Desktop/curiosity-based-snn-for-unitree_rl_gym/logs/rough_go2_snn/snn_spike_symm_10/benchmark/aggregated.npy",
    "/home/vittorio/Desktop/curiosity-based-snn-for-unitree_rl_gym/logs/rough_go2_snn/snn_spike_symm_67/benchmark/aggregated.npy",
    "/home/vittorio/Desktop/curiosity-based-snn-for-unitree_rl_gym/logs/rough_go2_snn/snn_spike_symm_1000/benchmark/aggregated.npy",
]

rnd_paths = [
    "/home/vittorio/Desktop/curiosity-based-snn-for-unitree_rl_gym/logs/rough_go2_rnd/rnd_spike_symm/benchmark/aggregated.npy",
    "/home/vittorio/Desktop/curiosity-based-snn-for-unitree_rl_gym/logs/rough_go2_rnd/rnd_spike_symm_10/benchmark/aggregated.npy",
    "/home/vittorio/Desktop/curiosity-based-snn-for-unitree_rl_gym/logs/rough_go2_rnd/rnd_spike_symm_67/benchmark/aggregated.npy",
    "/home/vittorio/Desktop/curiosity-based-snn-for-unitree_rl_gym/logs/rough_go2_rnd/rnd_spike_symm_1000/benchmark/aggregated.npy"
]


# ==========================
# LOAD
# ==========================

vel_snn, snn = load_config(snn_paths)
vel_rnd, rnd = load_config(rnd_paths)

assert np.allclose(vel_snn, vel_rnd)

vel = vel_snn


# ==========================
# PLOTS
# ==========================

compare_metric(
    vel,
    snn,
    rnd,
    "cot_mean",
    "CoT"
)

compare_metric(
    vel,
    snn,
    rnd,
    "rmse_mean",
    "Velocity RMSE"
)

# compare_metric(
#     vel,
#     snn,
#     rnd,
#     "duty_factor",
#     "Duty Factor"
# )

# compare_metric(
#     vel,
#     snn,
#     rnd,
#     "diag_sync",
#     "Diagonal Sync"
# )

# compare_metric(
#     vel,
#     snn,
#     rnd,
#     "stance_symmetry",
#     "Stance Symmetry"
# )

# compare_metric(
#     vel,
#     snn,
#     rnd,
#     "mean_support",
#     "Mean Support Legs"
# )

plt.show()