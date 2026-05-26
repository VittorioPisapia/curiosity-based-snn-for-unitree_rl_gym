import time

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation

def express_in_local_frame(quat, global_vector):
    """Ruota un vettore dal world frame al body frame usando il quaternione [w, x, y, z]"""
    qw, qx, qy, qz = quat
    
    R = np.array([
        [1 - 2*(qy**2 + qz**2),   2*(qx*qy + qw*qz),   2*(qx*qz - qw*qy)],
        [  2*(qx*qy - qw*qz), 1 - 2*(qx**2 + qz**2),   2*(qy*qz + qw*qx)],
        [  2*(qx*qz + qw*qy),   2*(qy*qz - qw*qx), 1 - 2*(qx**2 + qy**2)]
    ])
    
    return R @ global_vector

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
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

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # load policy
    policy = torch.jit.load(policy_path)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # Apply control signal here.

                # create observation
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]

                lin_vel_global = d.qvel[:3]
                ang_vel_global = d.qvel[3:6]

                base_lin_vel_local = express_in_local_frame(quat, lin_vel_global)
                base_ang_vel_local = express_in_local_frame(quat, ang_vel_global)

                base_lin_vel_scaled = base_lin_vel_local * config.get("lin_vel_scale", 2.0) 
                base_ang_vel_scaled = base_ang_vel_local * ang_vel_scale

                qj_scaled = (qj - default_angles) * dof_pos_scale
                dqj_scaled = dqj * dof_vel_scale

                gravity_orientation = get_gravity_orientation(quat)
                omega = omega * ang_vel_scale

                idx = 0
                # Velocità lineare della base (3)
                obs[idx:idx+3] = base_lin_vel_scaled
                idx += 3
                
                # Velocità angolare della base (3)
                obs[idx:idx+3] = base_ang_vel_scaled
                idx += 3
                
                # Gravità (3)
                obs[idx:idx+3] = gravity_orientation
                idx += 3
                
                # Comandi (3)
                obs[idx:idx+3] = cmd * cmd_scale
                idx += 3
                
                # Posizione giunti (num_actions)
                obs[idx : idx + num_actions] = qj_scaled
                idx += num_actions
                
                # Velocità giunti (num_actions)
                obs[idx : idx + num_actions] = dqj_scaled
                idx += num_actions
                
                # Azioni precedenti (num_actions)
                obs[idx : idx + num_actions] = action
                idx += num_actions

                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                # policy inference
                action = policy(obs_tensor).detach().numpy().squeeze()
                # transform action to target_dof_pos
                target_dof_pos = action * action_scale + default_angles

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)