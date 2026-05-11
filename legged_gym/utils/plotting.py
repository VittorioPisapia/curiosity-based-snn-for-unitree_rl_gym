# legged_gym/utils/plotting.py

import numpy as np
import matplotlib.pyplot as plt


def _style_axis(ax, title, ylabel=None, xlabel=None):

    ax.set_title(title)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    ax.grid(True)


def plot_linear_velocities(ax, logger, time_axis):

    ax.plot(
        time_axis,
        logger.cmd_vel_x,
        'r--',
        label='Cmd Lin X'
    )

    ax.plot(
        time_axis,
        logger.act_vel_x,
        'r',
        label='Act Lin X'
    )

    ax.plot(
        time_axis,
        logger.cmd_vel_y,
        'g--',
        label='Cmd Lin Y'
    )

    ax.plot(
        time_axis,
        logger.act_vel_y,
        'g',
        label='Act Lin Y'
    )

    _style_axis(
        ax,
        title='Commanded vs Actual Linear Velocities',
        ylabel='Velocity (m/s)'
    )

    ax.legend(loc='upper right', ncol=2)


def plot_angular_velocities(ax, logger, time_axis):

    ax.plot(
        time_axis,
        logger.cmd_yaw,
        'b--',
        label='Cmd Ang Z (Yaw)'
    )

    ax.plot(
        time_axis,
        logger.act_yaw,
        'b',
        label='Act Ang Z (Yaw)'
    )

    _style_axis(
        ax,
        title='Commanded vs Actual Angular Velocities',
        ylabel='Angular Velocity (rad/s)'
    )

    ax.legend(loc='upper right')


def plot_base_height(ax, logger, time_axis):

    ax.plot(
        time_axis,
        logger.target_z,
        'k--',
        label='Target Height'
    )

    ax.plot(
        time_axis,
        logger.actual_z,
        'k',
        label='Actual Height'
    )

    _style_axis(
        ax,
        title='Base Height (Z)',
        ylabel='Height (m)'
    )

    ax.legend(loc='upper right')


def plot_foot_xy(ax, logger, time_axis):

    ax.plot(
        time_axis,
        logger.foot_FL_x,
        'r',
        label='Foot X'
    )

    ax.plot(
        time_axis,
        logger.foot_FL_y,
        'g',
        label='Foot Y'
    )

    _style_axis(
        ax,
        title='Global Position of Front-Left Foot',
        ylabel='Position (m)',
        xlabel='Time (s)'
    )

    ax.legend(loc='upper right')


def plot_foot_heights(ax, logger, time_axis):

    ax.plot(
        time_axis,
        logger.foot_FL_z,
        'r',
        label='FL'
    )

    ax.plot(
        time_axis,
        logger.foot_FR_z,
        'g',
        label='FR'
    )

    ax.plot(
        time_axis,
        logger.foot_RL_z,
        'b',
        label='RL'
    )

    ax.plot(
        time_axis,
        logger.foot_RR_z,
        'y',
        label='RR'
    )

    _style_axis(
        ax,
        title='Foot Heights',
        ylabel='Height (m)',
        xlabel='Time (s)'
    )

    ax.legend(loc='upper right', ncol=4)


def plot_cot(ax, logger, time_axis):

    cot_array = np.array(logger.cot)

    mask_3s = time_axis >= 3.0

    if np.any(mask_3s):
        mean_cot_3s = np.mean(cot_array[mask_3s])
    else:
        mean_cot_3s = np.nan

    mean_cot = np.mean(cot_array)

    ax.plot(
        time_axis,
        logger.cot,
        'm',
        label='COT'
    )

    ax.set_title(
        f'COT '
        f'(mean total: {mean_cot:.4f} | '
        f'mean t≥3s: {mean_cot_3s:.4f})'
    )

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('COT')

    ax.grid(True)

    ax.legend(loc='upper right')


def plot_gait(ax, logger, env, time_axis):

    gait_matrix = np.array(logger.contacts).T

    ax.imshow(
        gait_matrix,
        aspect='auto',
        cmap='binary',
        interpolation='nearest',
        extent=[0, len(logger.contacts) * env.dt, 4, 0]
    )

    ax.set_yticks([0.5, 1.5, 2.5, 3.5])

    ax.set_yticklabels([
        'FL',
        'FR',
        'RL',
        'RR'
    ])

    ax.set_title('Gait Diagram')

    ax.set_xlabel('Time (s)')

    ax.set_ylabel('Feet')


def plot_run(logger, env, env_cfg):

    time_axis = (
        np.arange(logger.num_steps) * env.dt
    )

    fig, axs = plt.subplots(
        7,
        1,
        figsize=(12, 12)
    )

    fig.suptitle(
        f"Run Seed: {env_cfg.seed}",
        fontsize=14
    )

    plot_linear_velocities(
        axs[0],
        logger,
        time_axis
    )

    plot_angular_velocities(
        axs[1],
        logger,
        time_axis
    )

    plot_base_height(
        axs[2],
        logger,
        time_axis
    )

    plot_foot_xy(
        axs[3],
        logger,
        time_axis
    )

    plot_foot_heights(
        axs[4],
        logger,
        time_axis
    )

    plot_cot(
        axs[5],
        logger,
        time_axis
    )

    plot_gait(
        axs[6],
        logger,
        env,
        time_axis
    )

    plt.tight_layout()

    return fig