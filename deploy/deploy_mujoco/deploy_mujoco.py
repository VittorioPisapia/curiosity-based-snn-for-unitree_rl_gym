import time
import argparse
import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml
from legged_gym import LEGGED_GYM_ROOT_DIR


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands safely"""
    return (target_q - q) * kp + (target_dq - dq) * kd


if __name__ == "__main__":
    # 1. Parsing degli argomenti da terminale
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    
    # 2. Caricamento della configurazione YAML
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)
        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    # 3. Inizializzazione delle variabili di simulazione e di contesto
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0

    # 4. Caricamento del modello fisico in MuJoCo
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # 5. Caricamento della Policy TorchScript
    policy = torch.jit.load(policy_path)
    lin_vel_scale = config.get("lin_vel_scale", 2.0)

    # === CONFIGURAZIONI CINEMATICHE CRITICHE (Allineamento Isaac -> MuJoCo) ===
    # L'ordine dei giunti coincide [FL, FR, RL, RR]
    isaac_to_mujoco_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    
    # Moltiplicatori per correggere gli assi invertiti delle zampe destre (FR e RR)
    axis_signs = np.array([1,  1,  1,    # FL (Sinistra anteriore)
                          -1, -1, -1,    # FR (Destra anteriore - Specchiata)
                           1,  1,  1,    # RL (Sinistra posteriore)
                          -1, -1, -1],   # RR (Destra posteriore - Specchiata)
                          dtype=np.float32)

    # 6. Avvio del simulatore con visualizzatore passivo
    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()

        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()

            # --- PARTE A: CONTROLLO ATTUATORI PD (Eseguito ad ogni step di fisica, es. 500Hz) ---
            qj_mujoco = d.qpos[7:].copy()
            dqj_mujoco = d.qvel[6:].copy()

            # Riordina il target generato dalla policy seguendo l'ordine di MuJoCo
            target_dof_pos_mujoco = np.zeros_like(target_dof_pos)
            for isaac_idx, mujoco_idx in enumerate(isaac_to_mujoco_indices):
                target_dof_pos_mujoco[mujoco_idx] = target_dof_pos[isaac_idx]

            # Correzione algebrica dei segni degli assi prima del calcolo dell'errore PD
            qj_corrected = qj_mujoco * axis_signs
            dqj_corrected = dqj_mujoco * axis_signs
            target_corrected = target_dof_pos_mujoco * axis_signs

            # Calcolo delle coppie stabili
            tau = pd_control(target_corrected, qj_corrected, kps, np.zeros_like(kds), dqj_corrected, kds)
            
            # Applica le coppie finali ripristinando il segno geometrico di MuJoCo
            d.ctrl[:] = tau * axis_signs

            # Avanzamento della fisica dello stimatore
            mujoco.mj_step(m, d)
            counter += 1

            # --- PARTE B: INFERENZA DELLA RETE NEURALE (Frequenza decimata, es. 50Hz) ---
            if counter % control_decimation == 0:
                quat = d.qpos[3:7]   # Orientamento scocca [w, x, y, z]
                omega = d.qvel[3:6]  # Velocità angolare locale della scocca

                # Estrazione e correzione cinematica dei sensori per la rete
                qj_isaac = qj_mujoco[isaac_to_mujoco_indices] * axis_signs
                dqj_isaac = dqj_mujoco[isaac_to_mujoco_indices] * axis_signs

                # Costruzione della matrice di rotazione solida
                rot_matrix = np.zeros(9, dtype=np.float64)
                mujoco.mju_quat2Mat(rot_matrix, quat)
                rot_matrix = rot_matrix.reshape(3, 3)

                # Calcolo della velocità lineare locale nel frame del robot (Body Frame)
                lin_vel_global = d.qvel[0:3]
                lin_vel_local = rot_matrix.T @ lin_vel_global

                # Calcolo della gravità proiettata (Identico a Isaac Gym/PhysX)
                gravity_orientation = rot_matrix[2, :]

                # Normalizzazione matematica degli input tramite i fattori di scala
                lin_vel_scaled = lin_vel_local * lin_vel_scale
                omega_scaled = omega * ang_vel_scale
                cmd_scaled = cmd * cmd_scale
                qj_scaled = (qj_isaac - default_angles) * dof_pos_scale
                dqj_scaled = dqj_isaac * dof_vel_scale

                # Assemblaggio del vettore Observation (48 elementi totali richiesti dal Go2)
                obs[:3] = lin_vel_scaled                        # 0, 1, 2
                obs[3:6] = omega_scaled                         # 3, 4, 5
                obs[6:9] = gravity_orientation                  # 6, 7, 8
                obs[9:12] = cmd_scaled                          # 9, 10, 11
                obs[12:24] = qj_scaled                          # 12 a 23
                obs[24:36] = dqj_scaled                         # 24 a 35
                obs[36:48] = action                             # 36 a 47 (Azione precedente)

                # Conversione esplicita in Float Tensor (32-bit float standard di PyTorch)
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)

                # =====================================================================
                # BIAS DI TEST DIAGNOSTICO:
                # Per far camminare il robot con l'intelligenza artificiale:
                # scommenta la riga 'action = policy...' e commenta 'action = np.zeros...'
                # =====================================================================
                # action = policy(obs_tensor).detach().numpy().squeeze()
                action = np.zeros(num_actions, dtype=np.float32)

                # Calcolo della posizione articolare target (Giunti in ordine Isaac)
                target_dof_pos = action * action_scale + default_angles

            # Sincronizzazione della viewport grafica di MuJoCo
            viewer.sync()

            # Gestione rigorosa del tempo reale per evitare accelerazioni grafiche
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)