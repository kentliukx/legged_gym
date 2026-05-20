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

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        self.platform_centers = np.zeros((cfg.num_rows, cfg.num_cols, 3), dtype=np.float32)

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        self.vertices = None
        self.triangles = None

        terrain_kwargs = _cfg_to_dict(self.cfg.terrain_kwargs)
        self._selected_ladder_bars_terrain(**terrain_kwargs)

        self.heightsamples = self.height_field_raw

    def _selected_ladder_bars_terrain(self,
                                      bar_mesh_file,
                                      bar_spacing=0.3,
                                      bar_count=10,
                                      ladder_angle=0.0,
                                      bar_x_scale=(3.0, 1.0),
                                      platform_length=1.0,
                                      platform_width=1.2,
                                      platform_gap=0.1,
                                      difficulty=1.0,
                                      ):
        bar_vertices, bar_triangles = load_stl_mesh(bar_mesh_file)
        all_vertices = []
        all_triangles = []
        row_cache = {}
        vertex_offset = 0

        # Add one ground mesh for the full terrain grid. The ladder mesh is
        # tiled per environment, but the flat floor does not need duplicates.
        ground_vertices = []
        ground_triangles = []
        _append_ground_mesh(
            ground_vertices,
            ground_triangles,
            env_length=self.tot_rows * self.cfg.horizontal_scale,
            env_width=self.tot_cols * self.cfg.horizontal_scale)
        all_vertices.append(np.asarray(ground_vertices, dtype=np.float32))
        all_triangles.append(np.asarray(ground_triangles, dtype=np.uint32))
        vertex_offset += 4

        for k in range(self.cfg.num_sub_terrains):
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            # Curriculum only changes by row, so all columns in the same row
            # reuse one generated ladder and one height field.
            if i not in row_cache:
                row_difficulty = i / (self.cfg.num_rows - 1) if self.cfg.curriculum and self.cfg.num_rows > 1 else difficulty
                bar_centers, prepared_bar_vertices, local_vertices, local_triangles, platform_center = generate_ladder_bar_mesh(
                    env_length=self.env_length,
                    env_width=self.env_width,
                    difficulty=row_difficulty,
                    bar_spacing=bar_spacing,
                    bar_count=bar_count,
                    ladder_angle=ladder_angle,
                    bar_x_scale=bar_x_scale,
                    platform_length=platform_length,
                    platform_width=platform_width,
                    platform_gap=platform_gap,
                    bar_vertices=bar_vertices,
                    bar_triangles=bar_triangles)

                # Height scan is intentionally simple: inside a bar/platform XY
                # range means returning the bar center/platform top height.
                local_height_field = rasterize_ladder_bars(
                    horizontal_scale=self.cfg.horizontal_scale,
                    vertical_scale=self.cfg.vertical_scale,
                    num_rows=self.length_per_env_pixels,
                    num_cols=self.width_per_env_pixels,
                    bar_centers=bar_centers,
                    bar_vertices=prepared_bar_vertices,
                    platform_length=platform_length,
                    platform_width=platform_width,
                    platform_gap=platform_gap)
                row_cache[i] = (local_vertices, local_triangles, local_height_field, platform_center)

            local_vertices, local_triangles, local_height_field, platform_center = row_cache[i]

            start_x = self.border + i * self.length_per_env_pixels
            start_y = self.border + j * self.width_per_env_pixels
            self.height_field_raw[
                start_x:start_x + self.length_per_env_pixels,
                start_y:start_y + self.width_per_env_pixels] = local_height_field

            env_origin_x = (i + 0.5) * self.env_length
            env_origin_y = (j + 0.5) * self.env_width
            x1 = int((self.env_length / 2. - 1) / self.cfg.horizontal_scale)
            x2 = int((self.env_length / 2. + 1) / self.cfg.horizontal_scale)
            y1 = int((self.env_width / 2. - 1) / self.cfg.horizontal_scale)
            y2 = int((self.env_width / 2. + 1) / self.cfg.horizontal_scale)
            env_origin_z = np.max(local_height_field[x1:x2, y1:y2]) * self.cfg.vertical_scale
            self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
            self.platform_centers[i, j] = [
                platform_center[0] + start_x * self.cfg.horizontal_scale - self.cfg.border_size,
                platform_center[1] + start_y * self.cfg.horizontal_scale - self.cfg.border_size,
                platform_center[2],
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
                             bar_spacing=0.3,
                             bar_count=10,
                             ladder_angle=0.0,
                             bar_x_scale=(3.0, 1.0),
                             platform_length=1.0,
                             platform_width=1.2,
                             platform_gap=0.1,
                             bar_vertices=None,
                             bar_triangles=None
                             ):

    center_x = env_length * 0.5
    center_y = env_width * 0.5

    bar_spacing = _lerp_range(bar_spacing, difficulty)
    bar_count = int(round(_lerp_range(bar_count, difficulty)))
    ladder_angle = _lerp_range(ladder_angle, difficulty)
    bar_x_scale = _lerp_range(bar_x_scale, difficulty)

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
