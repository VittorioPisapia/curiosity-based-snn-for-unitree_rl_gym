import pandas as pd
import matplotlib.pyplot as plt


def moving_average(values, window=50):

    return (
        pd.Series(values)
        .rolling(window, min_periods=1)
        .mean()
        .to_numpy()
    )


def plot_tensorboard_csvs(
        csv_files,
        title,
        ylabel,
        smooth_window=50,
        save_path=None):

    plt.figure(figsize=(10, 5))

    for label, csv_path in csv_files.items():

        df = pd.read_csv(csv_path)

        steps = df["Step"].to_numpy()

        values = df["Value"].to_numpy()

        values = moving_average(
            values,
            smooth_window
        )

        plt.plot(
            steps,
            values,
            linewidth=2,
            label=label
        )

    plt.title(title)

    plt.xlabel("Training Step")

    plt.ylabel(ylabel)

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=300
        )

    plt.show()


def main():

    csv_files = {

        "SNN":
            r"/home/vittorio/Desktop/curiosity-based-snn-for-unitree_rl_gym/plot_tensorboard/rough_go2_snn_baseline_snn.csv",

        "RND":
            r"/home/vittorio/Desktop/curiosity-based-snn-for-unitree_rl_gym/plot_tensorboard/rough_go2_rnd_rnd_baseline_0.00015.csv",

        "SNN SPIKE SYMM":
            r"/home/vittorio/Desktop/curiosity-based-snn-for-unitree_rl_gym/plot_tensorboard/rough_go2_snn_snn_spike_symm.csv",

        "RND SPIKE SYMM":
            r"/home/vittorio/Desktop/curiosity-based-snn-for-unitree_rl_gym/plot_tensorboard/rough_go2_rnd_rnd_spike_symm.csv",
    }

    plot_tensorboard_csvs(
        csv_files=csv_files,
        title="Spike Rate (First layer)",
        ylabel="Spike Rate",
        smooth_window=50,
        save_path="spike_rate_comparison.png"
    )


if __name__ == "__main__":
    main()