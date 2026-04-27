<div align="center">
  <h1 align="center">Curiosity killed the cat or made it more efficient?</h1>
  <p align="center">
  </p>
</div>

I am a master student from Sapienza, Universty of Rome currently researching about SNN at the NeuroRobotics Lab. of Tohoku University. 
> **⚠️ WORK IN PROGRESS FORK ⚠️**
>
> This repository is a fork of the original Unitree RL GYM. It is currently a **Work In Progress (WIP)** and introduces an implementation of **Spiking Neural Networks (SNN)** and the **Intrinsic Curiosity Module (ICM)**.
>
> *Note: The Chinese translation (`README_zh.md`) is currently out of date and does not reflect these new modifications.*

---

---

## 📦 Installation and Configuration

Please refer to [setup.md](/doc/setup_en.md) for installation and configuration steps. (NEED UPDATE)

## 🛠️ User Guide

### 1. Training

Run the following command to start training:

```bash
python legged_gym/scripts/train.py --task=xxx
```

#### ⚙️ Parameter Description
- `--task`: Required parameter; values can be (go2, go2_snn, go2_icm, g1, h1, h1_2).
- `--headless`: Defaults to starting with a graphical interface; set to true for headless mode (higher efficiency).
- `--resume`: Resume training from a checkpoint in the logs.
- `--experiment_name`: Name of the experiment to run/load.
- `--run_name`: Name of the run to execute/load.
- `--load_run`: Name of the run to load; defaults to the latest run.
- `--checkpoint`: Checkpoint number to load; defaults to the latest file.
- `--num_envs`: Number of environments for parallel training.
- `--seed`: Random seed.
- `--max_iterations`: Maximum number of training iterations.
- `--sim_device`: Simulation computation device; specify CPU as `--sim_device=cpu`.
- `--rl_device`: Reinforcement learning computation device; specify CPU as `--rl_device=cpu`.

**Default Training Result Directory**: `logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`

---

### 2. Play

To visualize the training results in Gym, run the following command:

```bash
python legged_gym/scripts/play.py --task=xxx
```

**Description**:

- Play’s parameters are the same as Train’s except:
- - `--plot`: Display and save some useful plots during play.py
- `--record`: Record one environment during play.py
- By default, it loads the latest model from the experiment folder’s last run.
- You can specify other models using `load_run` and `checkpoint`.

## 🎉 Acknowledgments

This repository is built upon the support and contributions of the following open-source projects. Special thanks to:

- [legged\_gym](https://github.com/leggedrobotics/legged_gym): The foundation for training and running codes.
- [rsl\_rl](https://github.com/leggedrobotics/rsl_rl.git): Reinforcement learning algorithm implementation.
- [mujoco](https://github.com/google-deepmind/mujoco.git): Providing powerful simulation functionalities.
- [unitree\_sdk2\_python](https://github.com/unitreerobotics/unitree_sdk2_python.git): Hardware communication interface for physical deployment.
- [unitree-rl-gym](https://github.com/unitreerobotics/unitree_rl_gym) : Original implementation of Unitree's robots in Legged Gym

---

## 🔖 License

This project is licensed under the [BSD 3-Clause License](./LICENSE):
1. The original copyright notice must be retained.
2. The project name or organization name may not be used for promotion.
3. Any modifications must be disclosed.

For details, please read the full [LICENSE file](./LICENSE).

