import os
import imageio

from isaacgym import gymapi


class VideoRecorder:

    def __init__(
        self,
        env,
        video_path,
        width=1280,
        height=720,
        fps=50,
        camera_offset=(1.0, 1.0, 0.5),
    ):

        self.env = env
        self.video_path = video_path
        self.camera_offset = camera_offset

        self.camera_props = gymapi.CameraProperties()
        self.camera_props.width = width
        self.camera_props.height = height

        self.camera_handle = self.env.gym.create_camera_sensor(
            self.env.envs[0],
            self.camera_props
        )

        camera_position = gymapi.Vec3(4.0, 4.0, 3.0)
        camera_target = gymapi.Vec3(0.0, 0.0, 0.5)

        self.env.gym.set_camera_location(
            self.camera_handle,
            self.env.envs[0],
            camera_position,
            camera_target
        )

        self.video_writer = imageio.get_writer(video_path, fps=fps)

        print("Starting video recording...")
        print(f"Video will be saved to: {video_path}")

    def capture_frame(self):

        robot_pos = self.env.root_states[0, :3].cpu().numpy()

        env_origin = self.env.gym.get_env_origin(self.env.envs[0])

        local_x = robot_pos[0] - env_origin.x
        local_y = robot_pos[1] - env_origin.y
        local_z = robot_pos[2] - env_origin.z

        dx, dy, dz = self.camera_offset

        cam_pos = gymapi.Vec3(
            local_x + dx,
            local_y + dy,
            local_z + dz
        )

        cam_target = gymapi.Vec3(
            local_x,
            local_y,
            local_z
        )

        self.env.gym.set_camera_location(
            self.camera_handle,
            self.env.envs[0],
            cam_pos,
            cam_target
        )

        self.env.gym.fetch_results(self.env.sim, True)

        self.env.gym.step_graphics(self.env.sim)
        self.env.gym.render_all_camera_sensors(self.env.sim)

        image = self.env.gym.get_camera_image(
            self.env.sim,
            self.env.envs[0],
            self.camera_handle,
            gymapi.IMAGE_COLOR
        )

        image_np = image.reshape(
            (
                self.camera_props.height,
                self.camera_props.width,
                4
            )
        )

        rgb_image = image_np[..., :3]

        self.video_writer.append_data(rgb_image)

    def close(self):

        self.video_writer.close()

        print("Saved video successfully")