import numpy as np


class RobotLogger:

    def __init__(self):

        self.cmd_vel_x = []
        self.act_vel_x = []

        self.cmd_vel_y = []
        self.act_vel_y = []

        self.cmd_yaw = []
        self.act_yaw = []

        self.target_z = []
        self.actual_z = []

        self.foot_FL_x = []
        self.foot_FL_y = []

        self.foot_FL_z = []
        self.foot_FR_z = []
        self.foot_RL_z = []
        self.foot_RR_z = []

        self.cot = []

        self.contacts = []

    def log_step(self, env, robot_idx=0):

        contacts = (
            env.contact_forces[
                robot_idx,
                env.feet_indices,
                2
            ] > 1.0
        ).cpu().numpy()

        self.contacts.append(contacts)

        self.cmd_vel_x.append(
            env.commands[robot_idx, 0].item()
        )

        self.act_vel_x.append(
            env.base_lin_vel[robot_idx, 0].item()
        )

        self.cmd_vel_y.append(
            env.commands[robot_idx, 1].item()
        )

        self.act_vel_y.append(
            env.base_lin_vel[robot_idx, 1].item()
        )

        self.cmd_yaw.append(
            env.commands[robot_idx, 2].item()
        )

        self.act_yaw.append(
            env.base_ang_vel[robot_idx, 2].item()
        )

        self.target_z.append(
            env.cfg.rewards.base_height_target
        )

        self.actual_z.append(
            env.root_states[robot_idx, 2].item()
        )

        foot_idx_FL = env.feet_indices[0]
        foot_idx_FR = env.feet_indices[1]
        foot_idx_RL = env.feet_indices[2]
        foot_idx_RR = env.feet_indices[3]

        foot_pos = env.rigid_body_states[
            robot_idx,
            foot_idx_FL,
            0:3
        ].cpu().numpy()

        self.foot_FL_x.append(foot_pos[0])
        self.foot_FL_y.append(foot_pos[1])
        self.foot_FL_z.append(foot_pos[2])

        self.foot_FR_z.append(
            env.rigid_body_states[
                robot_idx,
                foot_idx_FR,
                2
            ].cpu().numpy()
        )

        self.foot_RL_z.append(
            env.rigid_body_states[
                robot_idx,
                foot_idx_RL,
                2
            ].cpu().numpy()
        )

        self.foot_RR_z.append(
            env.rigid_body_states[
                robot_idx,
                foot_idx_RR,
                2
            ].cpu().numpy()
        )

        self.cot.append(
            env.current_cot[0].item()
        )

    @property
    def num_steps(self):
        return len(self.cmd_vel_x)

    def has_data(self):
        return self.num_steps > 0