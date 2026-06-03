import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_aggregate(benchmark_dir):

    return np.load(
        os.path.join(
            benchmark_dir,
            "aggregated.npy"
        ),
        allow_pickle=True
    ).item()


def get_velocities(benchmark_dir):

    velocities = set()

    for name in os.listdir(benchmark_dir):

        if not name.startswith("v_"):
            continue

        try:

            vel_str = name.split("_seed_")[0][2:]

            velocities.add(float(vel_str))

        except Exception:
            pass

    return sorted(list(velocities))


def find_pattern_file(
    benchmark_dir,
    velocity,
    seed
):

    for folder in os.listdir(benchmark_dir):

        if not folder.endswith(
            f"_seed_{seed}"
        ):
            continue

        if not folder.startswith("v_"):
            continue

        try:

            vel = float(
                folder.split("_seed_")[0][2:]
            )

        except Exception:
            continue

        if abs(vel - velocity) < 1e-6:

            path = os.path.join(
                benchmark_dir,
                folder,
                "binary_pattern.npy"
            )

            if os.path.exists(path):
                return path

    return None


def load_binary_pattern(
    benchmark_dir,
    velocity,
    seed
):

    path = find_pattern_file(
        benchmark_dir,
        velocity,
        seed
    )

    if path is None:
        return None

    return np.load(path)

def plot_performance(
    agg1,
    label1,
    agg2=None,
    label2=None
):

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    axes[0].errorbar(
        agg1["vel"],
        agg1["cot_mean"],
        yerr=agg1["cot_std"],
        marker="o",
        linewidth=2,
        label=label1
    )

    if agg2 is not None:

        axes[0].errorbar(
            agg2["vel"],
            agg2["cot_mean"],
            yerr=agg2["cot_std"],
            marker="s",
            linewidth=2,
            label=label2
        )

    axes[0].set_title("Cost of Transport")
    axes[0].set_xlabel("Velocity [m/s]")
    axes[0].set_ylabel("CoT")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].errorbar(
        agg1["vel"],
        agg1["rmse_mean"],
        yerr=agg1["rmse_std"],
        marker="o",
        linewidth=2,
        label=label1
    )

    if agg2 is not None:

        axes[1].errorbar(
            agg2["vel"],
            agg2["rmse_mean"],
            yerr=agg2["rmse_std"],
            marker="s",
            linewidth=2,
            label=label2
        )

    axes[1].set_title(
        "Velocity Tracking RMSE"
    )

    axes[1].set_xlabel(
        "Velocity [m/s]"
    )

    axes[1].set_ylabel(
        "RMSE"
    )

    axes[1].grid(True)
    axes[1].legend()

    fig.suptitle(
        "Performance Comparison"
    )

    fig.tight_layout()

    return fig

def plot_gait_metrics(
    agg1,
    label1,
    agg2=None,
    label2=None
):

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10, 8)
    )

    axes = axes.flatten()

    metrics = [

        ("duty_factor",
         "Duty Factor"),

        #("diag_sync",
        # "Diagonal Synchronization"),

        ("mean_support",
         "Mean Support Legs"),

        ("stance_symmetry",
         "Stance Symmetry"),

        ("lr_balance",
         "Left-Right Balance"),

        #("switch_frequency",
        # "Switch Frequency")
    ]

    for ax, (key, title) in zip(
        axes,
        metrics
    ):

        ax.plot(
            agg1["vel"],
            agg1[key],
            marker="o",
            linewidth=2,
            label=label1
        )

        if agg2 is not None:

            ax.plot(
                agg2["vel"],
                agg2[key],
                marker="s",
                linewidth=2,
                label=label2
            )

        ax.set_title(title)
        ax.set_xlabel(
            "Velocity [m/s]"
        )

        ax.grid(True)
        ax.legend()

    fig.suptitle(
        "Gait Analysis"
    )

    fig.tight_layout()

    return fig

def plot_contact_map(
    pattern,
    title,
    ax,
    max_steps=150
):

    pattern = pattern[:, :max_steps]

    ax.imshow(
        pattern.astype(float),
        aspect="auto",
        interpolation="nearest",
        cmap="binary",
        origin="upper"
    )

    ax.set_yticks(
        [0, 1, 2, 3]
    )

    ax.set_yticklabels(
        ["FL", "FR", "RL", "RR"]
    )

    ax.set_xlabel(
        "Time Step"
    )

    ax.set_title(title)


def plot_binary_patterns(
    benchmark_dir_1,
    label1,
    benchmark_dir_2=None,
    label2=None,
    seed=0
):

    velocities = get_velocities(
        benchmark_dir_1
    )

    n_vel = len(velocities)

    cols = (
        2
        if benchmark_dir_2 is not None
        else 1
    )

    fig, axes = plt.subplots(
        n_vel,
        cols,
        figsize=(
            14,
            2.5 * n_vel
        ),
        squeeze=False
    )

    for row, v in enumerate(
        velocities
    ):

        pattern1 = load_binary_pattern(
            benchmark_dir_1,
            v,
            seed
        )

        if pattern1 is not None:

            plot_contact_map(
                pattern1,
                f"{label1} | v={v}",
                axes[row, 0]
            )

        if benchmark_dir_2 is not None:

            pattern2 = load_binary_pattern(
                benchmark_dir_2,
                v,
                seed
            )

            if pattern2 is not None:

                plot_contact_map(
                    pattern2,
                    f"{label2} | v={v}",
                    axes[row, 1]
                )

    fig.suptitle(
        f"Binary Contact Patterns (seed={seed})",
        fontsize=16
    )

    fig.tight_layout()

    return fig

def print_summary(
    agg1,
    label1,
    agg2=None,
    label2=None
):

    print()
    print("=" * 80)
    print("AVERAGE METRICS")
    print("=" * 80)

    metrics = [

        ("cot_mean", "CoT"),
        ("rmse_mean", "RMSE"),
        ("duty_factor", "Duty Factor"),
        ("diag_sync", "Diagonal Sync"),
        ("mean_support", "Mean Support"),
        ("stance_symmetry", "Stance Symmetry"),
        ("lr_balance", "LR Balance"),
        ("switch_frequency", "Switch Frequency")
    ]

    for key, name in metrics:

        v1 = np.mean(
            agg1[key]
        )

        if agg2 is None:

            print(
                f"{name:20s}"
                f"{label1:15s}: "
                f"{v1:8.4f}"
            )

        else:

            v2 = np.mean(
                agg2[key]
            )

            print(
                f"{name:20s}"
                f"{label1:15s}: "
                f"{v1:8.4f}    "
                f"{label2:15s}: "
                f"{v2:8.4f}"
            )

def main():

    parser = argparse.ArgumentParser(
        description="Compare benchmarks"
    )

    parser.add_argument(
        "--bench1",
        required=True
    )

    parser.add_argument(
        "--bench2",
        default=None
    )

    parser.add_argument(
        "--label1",
        default="Policy 1"
    )

    parser.add_argument(
        "--label2",
        default="Policy 2"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0
    )

    args = parser.parse_args()

    agg1 = load_aggregate(
        args.bench1
    )

    agg2 = None

    if args.bench2 is not None:

        agg2 = load_aggregate(
            args.bench2
        )

    print_summary(
        agg1,
        args.label1,
        agg2,
        args.label2
    )

    plot_performance(
        agg1,
        args.label1,
        agg2,
        args.label2
    )

    plot_gait_metrics(
        agg1,
        args.label1,
        agg2,
        args.label2
    )

    plot_binary_patterns(
        args.bench1,
        args.label1,
        args.bench2,
        args.label2,
        seed=args.seed
    )

    plt.show()


if __name__ == "__main__":
    main()