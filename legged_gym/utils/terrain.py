# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg


def _cfg_to_dict(cfg):
    """Convert nested config objects used by BaseConfig into plain dictionaries."""
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return dict(cfg)

    result = {}
    for key in dir(cfg):
        if key.startswith("__"):
            continue
        value = getattr(cfg, key)
        if callable(value):
            continue
        result[key] = value
    return result


class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:
        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width

        if cfg.num_rows < 2:
            raise ValueError("cfg.terrain.num_rows must be at least 2: one rough row plus at least one ladder row")
        self.num_ladder_rows = cfg.num_rows - 1
        self.num_cols = cfg.num_cols
        self.cfg.num_sub_terrains = cfg.num_rows * self.num_cols
        self.env_origins = np.zeros((cfg.num_rows, self.num_cols, 3))
        self.ladder_origins = np.zeros((cfg.num_rows, self.num_cols, 3), dtype=np.float32)
        self.platform_centers = np.zeros((cfg.num_rows, self.num_cols, 3), dtype=np.float32)
        self.ladder_mask = np.ones((cfg.num_rows, self.num_cols), dtype=np.bool_)
        self.ladder_bar_spacing = np.zeros((cfg.num_rows, self.num_cols), dtype=np.float32)
        self.ladder_angles = np.zeros((cfg.num_rows, self.num_cols), dtype=np.float32)

        self.obs_horizontal_scale = getattr(cfg, "obs_horizontal_scale", cfg.horizontal_scale)

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)
        self.obs_width_per_env_pixels = int(self.env_width / self.obs_horizontal_scale)
        self.obs_length_per_env_pixels = int(self.env_length / self.obs_horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.obs_border = int(cfg.border_size / self.obs_horizontal_scale)
        self.tot_cols = int(self.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border
        self.obs_tot_cols = int(self.num_cols * self.obs_width_per_env_pixels) + 2 * self.obs_border
        self.obs_tot_rows = int(cfg.num_rows * self.obs_length_per_env_pixels) + 2 * self.obs_border

        self.height_field_raw = np.zeros((self.obs_tot_rows, self.obs_tot_cols), dtype=np.int16)
        self.physics_height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        self.vertices = None
        self.triangles = None

        terrain_kwargs = _cfg_to_dict(self.cfg.terrain_kwargs)
        self._selected_ladder_bars_terrain(**terrain_kwargs)

        self.heightsamples = self.height_field_raw
        self.physics_heightsamples = self.physics_height_field_raw

    def _selected_ladder_bars_terrain(self,
                                      bar_mesh_file,
                                      bar_spacing=0.3,
                                      bar_count=10,
                                      ladder_angle=0.0,
                                      bar_x_scale=(3.0, 1.0),
                                      bar_x_scale_min_level=None,
                                      platform_length=1.0,
                                      platform_width=1.2,
                                      platform_gap=0.1,
                                      ladder_x_offset=0.0,
                                      rough_probability=0.0,
                                      low_difficulty_probability=0.0,
                                      rough_height_range=(0.02, 0.08),
                                      rough_grid_size=0.1,
                                      difficulty=1.0,
                                      ):
        bar_vertices, bar_triangles = load_stl_mesh(bar_mesh_file)
        all_vertices = []
        all_triangles = []
        vertex_offset = 0
        self.rough_probability = float(np.clip(rough_probability, 0.0, 1.0))

        for k in range(self.cfg.num_sub_terrains):
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.num_cols))

            is_rough = i == 0
            self.ladder_mask[i, j] = not is_rough

            if is_rough:
                self.ladder_angles[i, j] = 0.0
                local_physics_height_field = np.zeros(
                    (self.length_per_env_pixels, self.width_per_env_pixels),
                    dtype=np.int16)
                local_height_field = np.zeros(
                    (self.obs_length_per_env_pixels, self.obs_width_per_env_pixels),
                    dtype=np.int16)
                platform_center = np.asarray([self.env_length * 0.85, self.env_width * 0.5, 0.0], dtype=np.float32)
                ladder_origin = np.asarray([self.env_length * 0.5, self.env_width * 0.5, 0.0], dtype=np.float32)
                local_vertices = np.zeros((0, 3), dtype=np.float32)
                local_triangles = np.zeros((0, 3), dtype=np.uint32)
            else:
                row_difficulty = ((i - 1) / (self.cfg.num_rows - 2)
                                  if self.cfg.curriculum and self.cfg.num_rows > 2
                                  else difficulty)
                tile_bar_spacing = _sample_range(bar_spacing)
                tile_ladder_angle = _lerp_range(ladder_angle, row_difficulty)
                self.ladder_bar_spacing[i, j] = tile_bar_spacing
                self.ladder_angles[i, j] = tile_ladder_angle
                bar_centers, prepared_bar_vertices, local_vertices, local_triangles, platform_center = generate_ladder_bar_mesh(
                    env_length=self.env_length,
                    env_width=self.env_width,
                    difficulty=row_difficulty,
                    ladder_level=i,
                    max_ladder_level=self.cfg.num_rows - 1,
                    bar_spacing=tile_bar_spacing,
                    bar_count=bar_count,
                    ladder_angle=tile_ladder_angle,
                    bar_x_scale=bar_x_scale,
                    bar_x_scale_min_level=bar_x_scale_min_level,
                    platform_length=platform_length,
                    platform_width=platform_width,
                    platform_gap=platform_gap,
                    x_offset=ladder_x_offset,
                    bar_vertices=bar_vertices,
                    bar_triangles=bar_triangles)
                ladder_origin = np.asarray(
                    [self.env_length * 0.5 + ladder_x_offset, self.env_width * 0.5, 0.0],
                    dtype=np.float32,
                )

                # Height scan is intentionally simple: inside a bar/platform XY
                # range means returning the bar center/platform top height.
                local_height_field = rasterize_ladder_bars(
                    horizontal_scale=self.obs_horizontal_scale,
                    vertical_scale=self.cfg.vertical_scale,
                    num_rows=self.obs_length_per_env_pixels,
                    num_cols=self.obs_width_per_env_pixels,
                    bar_centers=bar_centers,
                    bar_vertices=prepared_bar_vertices,
                    platform_length=platform_length,
                    platform_width=platform_width,
                    platform_gap=platform_gap)
                local_physics_height_field = np.zeros(
                    (self.length_per_env_pixels, self.width_per_env_pixels),
                    dtype=np.int16)

            ground_vertices, ground_triangles = make_ground_mesh(self.env_length, self.env_width)
            if local_vertices.shape[0] == 0:
                local_vertices = ground_vertices
                local_triangles = ground_triangles
            else:
                local_triangles = np.concatenate(
                    (ground_triangles, local_triangles + ground_vertices.shape[0]),
                    axis=0)
                local_vertices = np.concatenate((ground_vertices, local_vertices), axis=0)

            start_x = self.border + i * self.length_per_env_pixels
            start_y = self.border + j * self.width_per_env_pixels
            obs_start_x = self.obs_border + i * self.obs_length_per_env_pixels
            obs_start_y = self.obs_border + j * self.obs_width_per_env_pixels
            self.height_field_raw[
                obs_start_x:obs_start_x + self.obs_length_per_env_pixels,
                obs_start_y:obs_start_y + self.obs_width_per_env_pixels] = local_height_field
            self.physics_height_field_raw[
                start_x:start_x + self.length_per_env_pixels,
                start_y:start_y + self.width_per_env_pixels] = local_physics_height_field

            env_origin_x = (i + 0.5) * self.env_length
            env_origin_y = (j + 0.5) * self.env_width
            env_origin_z = 0.0
            self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
            self.platform_centers[i, j] = [
                platform_center[0] + start_x * self.cfg.horizontal_scale - self.cfg.border_size,
                platform_center[1] + start_y * self.cfg.horizontal_scale - self.cfg.border_size,
                platform_center[2],
            ]
            self.ladder_origins[i, j] = [
                ladder_origin[0] + start_x * self.cfg.horizontal_scale - self.cfg.border_size,
                ladder_origin[1] + start_y * self.cfg.horizontal_scale - self.cfg.border_size,
                ladder_origin[2],
            ]

            # Move the local ladder tile into its global terrain-grid position.
            world_vertices = np.copy(local_vertices)
            world_vertices[:, 0] += start_x * self.cfg.horizontal_scale
            world_vertices[:, 1] += start_y * self.cfg.horizontal_scale
            all_vertices.append(world_vertices)
            all_triangles.append(local_triangles + vertex_offset)
            vertex_offset += local_vertices.shape[0]

        self.vertices = np.concatenate(all_vertices, axis=0).astype(np.float32)
        self.triangles = np.concatenate(all_triangles, axis=0).astype(np.uint32)


def rasterize_ladder_bars(horizontal_scale,
                          vertical_scale,
                          num_rows,
                          num_cols,
                          bar_centers,
                          bar_vertices,
                          platform_length=1.0,
                          platform_width=1.2,
                          platform_gap=0.1,
                          base_height=0.0):
    # Coarse scan map: use the imported bar's XY bounds and return each bar's
    # center height instead of tracing the exact round STL surface.
    height_field = np.full((num_rows, num_cols), int(np.round(base_height / vertical_scale)), dtype=np.int16)
    bar_mins = bar_vertices.min(axis=0)
    bar_maxs = bar_vertices.max(axis=0)
    for center in bar_centers:
        mins = bar_mins + center
        maxs = bar_maxs + center
        min_x = max(0, int(np.floor(mins[0] / horizontal_scale)))
        max_x = min(num_rows - 1, int(np.floor(maxs[0] / horizontal_scale)))
        min_y = max(0, int(np.floor(mins[1] / horizontal_scale)))
        max_y = min(num_cols - 1, int(np.floor(maxs[1] / horizontal_scale)))
        if min_x > max_x or min_y > max_y:
            continue

        height_value = int(np.round(center[2] / vertical_scale))
        height_field[min_x:max_x + 1, min_y:max_y + 1] = height_value

    _fill_platform_height(
        height_field,
        horizontal_scale=horizontal_scale,
        vertical_scale=vertical_scale,
        bar_centers=bar_centers,
        platform_length=platform_length,
        platform_width=platform_width,
        platform_gap=platform_gap)
    return height_field


def generate_ladder_bar_mesh(env_length,
                             env_width,
                             difficulty,
                             ladder_level,
                             max_ladder_level,
                             bar_spacing=0.3,
                             bar_count=10,
                             ladder_angle=0.0,
                             bar_x_scale=(3.0, 1.0),
                             bar_x_scale_min_level=None,
                             platform_length=1.0,
                             platform_width=1.2,
                             platform_gap=0.1,
                             x_offset=0.0,
                             bar_vertices=None,
                             bar_triangles=None
                             ):

    center_x = env_length * 0.5 + x_offset
    center_y = env_width * 0.5

    bar_spacing = _lerp_range(bar_spacing, difficulty)
    bar_count = int(round(_lerp_range(bar_count, difficulty)))
    ladder_angle = _lerp_range(ladder_angle, difficulty)
    bar_x_scale = _lerp_range_by_level(
        bar_x_scale,
        ladder_level=ladder_level,
        max_ladder_level=max_ladder_level,
        min_value_level=bar_x_scale_min_level,
    )

    vertices = []
    triangles = []

    # Center the imported STL once. Every rung is just this mesh translated.
    prepared_bar_vertices = _prepare_bar_mesh(bar_vertices)
    prepared_bar_vertices[:, 0] *= bar_x_scale
    bar_centers = _compute_bar_centers(
        center_x=center_x,
        center_y=center_y,
        bar_spacing=bar_spacing,
        bar_count=bar_count,
        ladder_angle=ladder_angle)
    for center in bar_centers:
        _append_prepared_bar_mesh(
            vertices,
            triangles,
            bar_vertices=prepared_bar_vertices,
            bar_triangles=bar_triangles,
            center=center)
    _append_platform_mesh(
        vertices,
        triangles,
        bar_centers=bar_centers,
        platform_length=platform_length,
        platform_width=platform_width,
        platform_gap=platform_gap)

    platform_corners_xy, platform_top_z = _platform_corners(bar_centers, platform_length, platform_width, platform_gap)
    platform_center = np.asarray([
        np.mean(platform_corners_xy[:, 0]),
        np.mean(platform_corners_xy[:, 1]),
        platform_top_z,
    ], dtype=np.float32)

    return bar_centers, prepared_bar_vertices, np.asarray(vertices, dtype=np.float32), np.asarray(triangles, dtype=np.uint32), platform_center


def generate_random_rough_height_fields(env_length,
                                        env_width,
                                        physics_horizontal_scale,
                                        obs_horizontal_scale,
                                        vertical_scale,
                                        difficulty,
                                        rough_height_range=(0.02, 0.08),
                                        rough_grid_size=0.1):
    rough_grid_size = max(float(rough_grid_size), min(physics_horizontal_scale, obs_horizontal_scale))
    grid_rows = max(2, int(np.ceil(env_length / rough_grid_size)) + 1)
    grid_cols = max(2, int(np.ceil(env_width / rough_grid_size)) + 1)

    if np.isscalar(rough_height_range):
        min_height = 0.0
        max_height = float(rough_height_range)
    else:
        min_height = float(min(rough_height_range))
        max_height = float(max(rough_height_range))

    height_grid = np.random.uniform(min_height, max_height, size=(grid_rows, grid_cols))
    height_grid = np.round(height_grid / vertical_scale).astype(np.int16)

    physics_height_field = _rasterize_rough_height_grid(
        height_grid=height_grid,
        num_rows=int(env_length / physics_horizontal_scale),
        num_cols=int(env_width / physics_horizontal_scale),
        env_length=env_length,
        env_width=env_width,
        horizontal_scale=physics_horizontal_scale)
    obs_height_field = _rasterize_rough_height_grid(
        height_grid=height_grid,
        num_rows=int(env_length / obs_horizontal_scale),
        num_cols=int(env_width / obs_horizontal_scale),
        env_length=env_length,
        env_width=env_width,
        horizontal_scale=obs_horizontal_scale)
    platform_center = np.asarray([env_length * 0.85, env_width * 0.5, 0.0], dtype=np.float32)
    return physics_height_field, obs_height_field, platform_center


def generate_random_rough_mesh(env_length,
                               env_width,
                               horizontal_scale,
                               vertical_scale,
                               difficulty,
                               rough_height_range=(0.02, 0.08),
                               rough_grid_size=0.1):
    rough_height = _lerp_range(rough_height_range, difficulty)
    rough_grid_size = max(float(rough_grid_size), horizontal_scale)
    grid_rows = max(2, int(np.ceil(env_length / rough_grid_size)) + 1)
    grid_cols = max(2, int(np.ceil(env_width / rough_grid_size)) + 1)

    height_grid = np.random.uniform(0.0, rough_height, size=(grid_rows, grid_cols))
    height_grid = np.round(height_grid / vertical_scale).astype(np.int16)

    center_col = grid_cols // 2
    spawn_rows = max(1, int(np.ceil(1.0 / rough_grid_size)))
    target_row = min(grid_rows - 1, int(round(0.85 * (grid_rows - 1))))
    target_radius = max(1, int(np.ceil(0.5 / rough_grid_size)))
    target_col_min = max(0, center_col - target_radius)
    target_col_max = min(grid_cols, center_col + target_radius + 1)
    height_grid[:spawn_rows, :] = 0
    height_grid[max(0, target_row - target_radius):min(grid_rows, target_row + target_radius + 1),
                target_col_min:target_col_max] = 0

    height_field = _rasterize_rough_height_grid(
        height_grid=height_grid,
        num_rows=int(env_length / horizontal_scale),
        num_cols=int(env_width / horizontal_scale),
        env_length=env_length,
        env_width=env_width,
        horizontal_scale=horizontal_scale)
    vertices, triangles = _height_grid_to_mesh(
        height_grid=height_grid,
        env_length=env_length,
        env_width=env_width,
        vertical_scale=vertical_scale)
    platform_center = np.asarray([env_length * 0.85, env_width * 0.5, 0.0], dtype=np.float32)
    return vertices, triangles, height_field, platform_center


def _rasterize_rough_height_grid(height_grid,
                                 num_rows,
                                 num_cols,
                                 env_length,
                                 env_width,
                                 horizontal_scale):
    sample_x = np.clip(
        np.floor(np.arange(num_rows) * horizontal_scale / env_length * (height_grid.shape[0] - 1)).astype(np.int64),
        0,
        height_grid.shape[0] - 1)
    sample_y = np.clip(
        np.floor(np.arange(num_cols) * horizontal_scale / env_width * (height_grid.shape[1] - 1)).astype(np.int64),
        0,
        height_grid.shape[1] - 1)
    return height_grid[np.ix_(sample_x, sample_y)].astype(np.int16)


def _height_grid_to_mesh(height_grid, env_length, env_width, vertical_scale):
    x = np.linspace(0.0, env_length, height_grid.shape[0], dtype=np.float32)
    y = np.linspace(0.0, env_width, height_grid.shape[1], dtype=np.float32)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    vertices = np.stack(
        (xx, yy, height_grid.astype(np.float32) * vertical_scale),
        axis=-1).reshape(-1, 3)

    triangles = []
    num_cols = height_grid.shape[1]
    for row in range(height_grid.shape[0] - 1):
        for col in range(height_grid.shape[1] - 1):
            idx = row * num_cols + col
            triangles.append([idx, idx + 1, idx + num_cols + 1])
            triangles.append([idx, idx + num_cols + 1, idx + num_cols])
    return vertices.astype(np.float32), np.asarray(triangles, dtype=np.uint32)


def _compute_bar_centers(center_x, center_y, bar_spacing, bar_count, ladder_angle):
    angle_rad = np.deg2rad(ladder_angle)
    ground_spacing = bar_spacing * np.cos(angle_rad)
    height_spacing = bar_spacing * np.sin(angle_rad)

    centers = []
    for bar_idx in range(bar_count):
        # Extending the ladder line backward intersects the ground at center_x.
        step = bar_idx + 1
        centers.append([center_x + step * ground_spacing, center_y, step * height_spacing])
    return np.asarray(centers, dtype=np.float32)


def _append_ground_mesh(vertices, triangles, env_length, env_width, z=0.0):
    base_idx = len(vertices)
    vertices.extend([
        [0.0, 0.0, z],
        [env_length, 0.0, z],
        [env_length, env_width, z],
        [0.0, env_width, z],
    ])
    triangles.extend([
        [base_idx + 0, base_idx + 1, base_idx + 2],
        [base_idx + 0, base_idx + 2, base_idx + 3],
    ])


def make_ground_mesh(env_length, env_width, z=0.0):
    vertices = []
    triangles = []
    _append_ground_mesh(vertices, triangles, env_length, env_width, z=z)
    return np.asarray(vertices, dtype=np.float32), np.asarray(triangles, dtype=np.uint32)


def _append_platform_mesh(vertices, triangles, bar_centers, platform_length, platform_width, platform_gap):
    if platform_length <= 0.0 or platform_width <= 0.0:
        return

    corners_xy, top_z = _platform_corners(bar_centers, platform_length, platform_width, platform_gap)
    base_idx = len(vertices)
    vertices.extend([[xy[0], xy[1], 0.0] for xy in corners_xy])
    vertices.extend([[xy[0], xy[1], top_z] for xy in corners_xy])

    # Top face plus four side faces. No bottom face because it sits on ground.
    triangles.extend([
        [base_idx + 4, base_idx + 5, base_idx + 6],
        [base_idx + 4, base_idx + 6, base_idx + 7],
        [base_idx + 0, base_idx + 1, base_idx + 5],
        [base_idx + 0, base_idx + 5, base_idx + 4],
        [base_idx + 1, base_idx + 2, base_idx + 6],
        [base_idx + 1, base_idx + 6, base_idx + 5],
        [base_idx + 2, base_idx + 3, base_idx + 7],
        [base_idx + 2, base_idx + 7, base_idx + 6],
        [base_idx + 3, base_idx + 0, base_idx + 4],
        [base_idx + 3, base_idx + 4, base_idx + 7],
    ])


def _fill_platform_height(height_field,
                          horizontal_scale,
                          vertical_scale,
                          bar_centers,
                          platform_length,
                          platform_width,
                          platform_gap):
    if platform_length <= 0.0 or platform_width <= 0.0:
        return

    corners_xy, top_z = _platform_corners(bar_centers, platform_length, platform_width, platform_gap)

    min_x = max(0, int(np.floor(np.min(corners_xy[:, 0]) / horizontal_scale)))
    max_x = min(height_field.shape[0] - 1, int(np.floor(np.max(corners_xy[:, 0]) / horizontal_scale)))
    min_y = max(0, int(np.floor(np.min(corners_xy[:, 1]) / horizontal_scale)))
    max_y = min(height_field.shape[1] - 1, int(np.floor(np.max(corners_xy[:, 1]) / horizontal_scale)))
    if min_x <= max_x and min_y <= max_y:
        height_field[min_x:max_x + 1, min_y:max_y + 1] = int(np.round(top_z / vertical_scale))


def _platform_corners(bar_centers, platform_length, platform_width, platform_gap):
    last_bar = bar_centers[-1]
    previous_bar = bar_centers[-2] if len(bar_centers) > 1 else last_bar - np.array([1.0, 0.0, 0.0], dtype=np.float32)
    direction = last_bar[:2] - previous_bar[:2]
    direction_norm = np.linalg.norm(direction)
    direction = np.array([1.0, 0.0], dtype=np.float32) if direction_norm <= 1e-6 else direction / direction_norm

    side = np.array([-direction[1], direction[0]], dtype=np.float32)
    center_xy = last_bar[:2] + direction * (platform_gap + platform_length * 0.5)
    half_length = platform_length * 0.5
    half_width = platform_width * 0.5
    corners_xy = np.asarray([
        center_xy - direction * half_length - side * half_width,
        center_xy + direction * half_length - side * half_width,
        center_xy + direction * half_length + side * half_width,
        center_xy - direction * half_length + side * half_width,
    ], dtype=np.float32)
    return corners_xy, last_bar[2]


def _append_prepared_bar_mesh(vertices,
                              triangles,
                              bar_vertices,
                              bar_triangles,
                              center):
    transformed = np.copy(bar_vertices)
    transformed += np.asarray(center, dtype=np.float32).reshape(1, 3)

    base_idx = len(vertices)
    vertices.extend(transformed.tolist())
    triangles.extend((bar_triangles + base_idx).tolist())


def _prepare_bar_mesh(vertices):
    prepared = np.copy(vertices).astype(np.float32)
    mins = prepared.min(axis=0)
    maxs = prepared.max(axis=0)
    prepared[:, 0] -= (mins[0] + maxs[0]) * 0.5
    prepared[:, 1] -= (mins[1] + maxs[1]) * 0.5
    prepared[:, 2] -= (mins[2] + maxs[2]) * 0.5
    return prepared


def _lerp_range(value_range, difficulty):
    if np.isscalar(value_range):
        return float(value_range)
    if len(value_range) != 2:
        raise ValueError("range values must be scalars or two-element sequences")
    return float(value_range[0] + (value_range[1] - value_range[0]) * difficulty)


def _lerp_range_by_level(value_range, ladder_level, max_ladder_level, min_value_level=None):
    if np.isscalar(value_range):
        return float(value_range)
    if len(value_range) != 2:
        raise ValueError("range values must be scalars or two-element sequences")

    ladder_level = int(ladder_level)
    max_ladder_level = max(1, int(max_ladder_level))
    if min_value_level is None:
        effective_difficulty = float(ladder_level - 1) / max(1, max_ladder_level - 1)
    else:
        min_value_level = max(1, min(int(min_value_level), max_ladder_level))
        effective_difficulty = float(ladder_level - 1) / max(1, min_value_level - 1)
        effective_difficulty = min(effective_difficulty, 1.0)
    return float(value_range[0] + (value_range[1] - value_range[0]) * effective_difficulty)


def _sample_range(value_range):
    if np.isscalar(value_range):
        return float(value_range)
    if len(value_range) != 2:
        raise ValueError("range values must be scalars or two-element sequences")
    low = min(value_range[0], value_range[1])
    high = max(value_range[0], value_range[1])
    return float(np.random.uniform(low, high))


def load_stl_mesh(mesh_file):
    # Binary STL loader. It keeps the STL units as-is, so the file should
    # already be exported in meters for Isaac Gym.
    mesh_file = mesh_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
    vertices = []
    triangles = []
    vertex_map = {}

    with open(mesh_file, "rb") as f:
        f.seek(80)
        tri_count = int(np.frombuffer(f.read(4), dtype=np.uint32)[0])
        for _ in range(tri_count):
            data = f.read(50)
            coords = np.frombuffer(data[12:48], dtype=np.float32).reshape(3, 3)
            tri = []
            for vertex in coords:
                key = tuple(np.round(vertex, 9))
                if key not in vertex_map:
                    vertex_map[key] = len(vertices)
                    vertices.append(vertex.tolist())
                tri.append(vertex_map[key])
            triangles.append(tri)

    vertices = np.asarray(vertices, dtype=np.float32)
    triangles = np.asarray(triangles, dtype=np.uint32)
    return vertices, triangles
