import math

import numpy as np
import torch

try:
    import warp as wp
except ImportError as exc:
    raise ImportError(
        "Warp depth camera requires warp-lang. Install it in the legged-gym "
        "environment with: pip install warp-lang==1.4.2"
    ) from exc


NO_HIT_DEPTH = wp.constant(1000.0)


@wp.kernel
def render_depth_kernel(
    mesh_ids: wp.array(dtype=wp.uint64),
    camera_positions: wp.array(dtype=wp.vec3),
    camera_orientations: wp.array(dtype=wp.quat),
    inverse_intrinsics: wp.mat44,
    far_plane: float,
    pixels: wp.array(dtype=wp.float32, ndim=3),
    center_x: int,
    center_y: int,
):
    env_id, pixel_y, pixel_x = wp.tid()
    camera_position = camera_positions[env_id]
    camera_orientation = camera_orientations[env_id]

    pixel = wp.vec3(
        float(pixel_x) + 0.5,
        float(pixel_y) + 0.5,
        1.0,
    )
    principal_pixel = wp.vec3(float(center_x), float(center_y), 1.0)
    camera_ray = wp.transform_vector(inverse_intrinsics, pixel)
    principal_ray = wp.transform_vector(inverse_intrinsics, principal_pixel)

    ray_direction = wp.normalize(wp.quat_rotate(camera_orientation, camera_ray))
    principal_direction = wp.normalize(wp.quat_rotate(camera_orientation, principal_ray))
    depth_projection = wp.dot(ray_direction, principal_direction)

    distance = NO_HIT_DEPTH
    hit_distance = float(0.0)
    hit_u = float(0.0)
    hit_v = float(0.0)
    hit_sign = float(0.0)
    hit_normal = wp.vec3()
    hit_face = int(0)
    if wp.mesh_query_ray(
        mesh_ids[0],
        camera_position,
        ray_direction,
        far_plane / depth_projection,
        hit_distance,
        hit_u,
        hit_v,
        hit_sign,
        hit_normal,
        hit_face,
    ):
        distance = depth_projection * hit_distance

    pixels[env_id, pixel_y, pixel_x] = distance


class WarpDepthCamera:
    """GPU terrain-mesh depth camera using NVIDIA Warp ray queries."""

    def __init__(
        self,
        terrain_vertices,
        terrain_triangles,
        terrain_offset,
        num_envs,
        width,
        height,
        horizontal_fov_deg,
        far_plane,
        device,
    ):
        if not str(device).startswith("cuda"):
            raise ValueError("Warp depth camera requires a CUDA simulation device")

        wp.init()
        self.device = str(device)
        self.num_envs = int(num_envs)
        self.width = int(width)
        self.height = int(height)
        self.far_plane = float(far_plane)

        vertices = np.asarray(terrain_vertices, dtype=np.float32).copy()
        vertices += np.asarray(terrain_offset, dtype=np.float32)
        triangles = np.asarray(terrain_triangles, dtype=np.int32)
        self.mesh = wp.Mesh(
            points=wp.array(vertices, dtype=wp.vec3, device=self.device),
            indices=wp.array(triangles.reshape(-1), dtype=wp.int32, device=self.device),
        )
        self.mesh_ids = wp.array([self.mesh.id], dtype=wp.uint64, device=self.device)

        self.camera_positions = torch.zeros(
            self.num_envs, 3, dtype=torch.float32, device=self.device
        )
        self.camera_orientations = torch.zeros(
            self.num_envs, 4, dtype=torch.float32, device=self.device
        )
        self.depth = torch.zeros(
            self.num_envs, self.height, self.width, dtype=torch.float32, device=self.device
        )

        self.wp_camera_positions = wp.from_torch(self.camera_positions, dtype=wp.vec3)
        self.wp_camera_orientations = wp.from_torch(self.camera_orientations, dtype=wp.quat)
        self.wp_depth = wp.from_torch(self.depth, dtype=wp.float32)

        center_x = self.width / 2.0
        center_y = self.height / 2.0
        focal_length = center_x / math.tan(math.radians(horizontal_fov_deg) / 2.0)
        vertical_fov = 2.0 * math.atan(self.height / (2.0 * focal_length))
        alpha_x = center_x / math.tan(math.radians(horizontal_fov_deg) / 2.0)
        alpha_y = center_y / math.tan(vertical_fov / 2.0)
        intrinsics = wp.mat44(
            alpha_x, 0.0, center_x, 0.0,
            0.0, alpha_y, center_y, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        )
        self.inverse_intrinsics = wp.inverse(intrinsics)
        self.center_x = int(center_x)
        self.center_y = int(center_y)
        self.graph = None

    def _launch(self):
        wp.launch(
            kernel=render_depth_kernel,
            dim=(self.num_envs, self.height, self.width),
            inputs=[
                self.mesh_ids,
                self.wp_camera_positions,
                self.wp_camera_orientations,
                self.inverse_intrinsics,
                self.far_plane,
                self.wp_depth,
                self.center_x,
                self.center_y,
            ],
            device=self.device,
        )

    def render(self, positions, orientations):
        self.camera_positions.copy_(positions)
        self.camera_orientations.copy_(orientations)
        if self.graph is None:
            wp.capture_begin(device=self.device)
            self._launch()
            self.graph = wp.capture_end(device=self.device)
        wp.capture_launch(self.graph)
        return self.depth
