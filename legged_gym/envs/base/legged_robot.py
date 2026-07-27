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

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply, quat_apply_yaw, quat_multiply_xyzw, quat_rotate_inverse, wrap_to_pi, torch_rand_float, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg


class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = True
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            delayed_actions = self._get_delayed_actions(self.actions)
            self.torques = self._compute_torques(delayed_actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self._apply_rigid_body_force_disturbances()
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self._update_effector_ladder_distances()
        self._update_feet_air_time_state()
        self._update_symmetry_torque_state()

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self._update_depth_camera_observations()
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        attitude_failure = self.projected_gravity[:, 2] > 0.1
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf = self.reset_buf | attitude_failure | self.time_out_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        self._debug_log_reset_event(env_ids)
        episode_ladder_progress = self._get_ladder_progress(env_ids).mean()
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        elif self.custom_origins and hasattr(self, "terrain_origins"):
            self._sample_terrain_levels_and_types(env_ids)
            self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
            if hasattr(self, "terrain_platform_centers"):
                self.goal_targets[env_ids] = self.terrain_platform_centers[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
                self.goal_targets[env_ids, 2] += self.cfg.rewards.base_height_target
                self._randomize_rough_targets(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)
        if hasattr(self, "goal_targets"):
            self._reset_goal_progress(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        if hasattr(self, "action_delay_buffer"):
            self.action_delay_buffer[:, env_ids] = 0.
            if self.max_action_delay > 0:
                self.action_delay_steps[env_ids] = torch.randint(
                    low=self.min_action_delay,
                    high=self.max_action_delay + 1,
                    size=(len(env_ids),),
                    dtype=torch.long,
                    device=self.device,
                )
            else:
                self.action_delay_steps[env_ids] = 0
        self.last_dof_vel[env_ids] = 0.
        self.proprioception_history_reset_buf[env_ids] = True
        self.feet_air_time[env_ids] = 0.
        self.feet_first_contact[env_ids] = False
        self.feet_last_air_time[env_ids] = 0.
        self.min_feet_last_air_time[env_ids] = 0.
        self.feet_ground_time[env_ids] = 0.
        self.feet_first_air[env_ids] = False
        self.feet_last_ground_time[env_ids] = 0.
        self.min_feet_last_ground_time[env_ids] = 0.
        self.foot_contacts[env_ids] = False
        self.last_contacts[env_ids] = False
        self.phase_feet_air_time[env_ids] = 0.
        self.phase_feet_first_contact[env_ids] = False
        self.phase_feet_last_air_time[env_ids] = 0.
        self.min_phase_feet_last_air_time[env_ids] = 0.
        self.phase_feet_ground_time[env_ids] = 0.
        self.phase_feet_first_air[env_ids] = False
        self.phase_feet_last_ground_time[env_ids] = 0.
        self.min_phase_feet_last_ground_time[env_ids] = 0.
        self.phase_foot_contacts[env_ids] = False
        self.last_phase_contacts[env_ids] = False
        if hasattr(self, "filtered_symmetry_torque_abs_diff"):
            self.filtered_symmetry_torque_abs_diff[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        self._record_slow_reward_episode_returns(env_ids)
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.ladder_levels[env_ids].float())
            self.extras["episode"]["ladder_progress"] = episode_ladder_progress
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def _debug_terrain_sampling_enabled(self):
        return bool(getattr(self.cfg.env, "debug_terrain_sampling", False))

    def _debug_log_reset_event(self, env_ids):
        if not self._debug_terrain_sampling_enabled():
            return
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if len(env_ids) == 0:
            return

        terrain_kinds = []
        for env_id in env_ids.tolist():
            is_ladder = True
            if hasattr(self, "terrain_ladder_mask"):
                is_ladder = bool(self.terrain_ladder_mask[self.terrain_levels[env_id], self.terrain_types[env_id]].item())
            terrain_kinds.append("ladder" if is_ladder else "rough")

        print(
            f"[terrain-debug] reset step={self._debug_step_label()} "
            f"env_ids={env_ids.tolist()} "
            f"prev_terrain={terrain_kinds} "
            f"saved_ladder_levels={self.ladder_levels[env_ids].tolist()} "
            f"time_outs={self.time_out_buf[env_ids].tolist()} "
            f"reached_goal={self.reached_goal[env_ids, 0].tolist()}",
            flush=True,
        )

    def _debug_step_label(self):
        if hasattr(self, "common_step_counter"):
            return str(int(self.common_step_counter))
        return "init"

    def _debug_log_sampling_event(self, env_ids, event_name, rough_prob=None, low_difficulty_prob=None,
                                  sample_draws=None, sample_modes=None):
        if not self._debug_terrain_sampling_enabled():
            return
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if len(env_ids) == 0:
            return

        terrain_kinds = []
        for env_id in env_ids.tolist():
            is_ladder = True
            if hasattr(self, "terrain_ladder_mask"):
                is_ladder = bool(self.terrain_ladder_mask[self.terrain_levels[env_id], self.terrain_types[env_id]].item())
            terrain_kinds.append("ladder" if is_ladder else "rough")

        print(
            f"[terrain-debug] {event_name} step={self._debug_step_label()} "
            f"env_ids={env_ids.tolist()} "
            f"sampled_terrain={terrain_kinds} "
            f"terrain_levels={self.terrain_levels[env_ids].tolist()} "
            f"saved_ladder_levels={self.ladder_levels[env_ids].tolist()} "
            f"terrain_types={self.terrain_types[env_ids].tolist()} "
            f"rough_prob={rough_prob if rough_prob is not None else 'n/a'} "
            f"low_difficulty_prob={low_difficulty_prob if low_difficulty_prob is not None else 'n/a'} "
            f"sample_draws={sample_draws if sample_draws is not None else 'n/a'} "
            f"sample_modes={sample_modes if sample_modes is not None else 'n/a'}",
            flush=True,
        )
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        slow_reward_coeff = self._update_slow_reward_coeff()
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            if name in self.slow_reward_names:
                rew *= slow_reward_coeff
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
        self.episode_return_sums += self.rew_buf

    def _update_slow_reward_coeff(self):
        """Linearly anneal slow-movement penalties from the global mean episode return."""
        if self.slow_reward_episode_return_count == 0:
            return self.slow_reward_coeff_buf
        low, high = self.slow_reward_coeff
        mean_episode_reward = torch.mean(
            self.slow_reward_episode_returns[:self.slow_reward_episode_return_count]
        )
        reward_span = self.slow_reward_coeff_upper_reward_limit - self.slow_reward_coeff_lower_reward_limit
        progress = torch.clamp(
            (mean_episode_reward - self.slow_reward_coeff_lower_reward_limit) / reward_span,
            min=0.0,
            max=1.0,
        )
        target_coeff = low + (high - low) * progress
        self.slow_reward_coeff_buf.mul_(1.0 - self.slow_reward_coeff_lpf_k).add_(
            self.slow_reward_coeff_lpf_k * target_coeff
        )
        return self.slow_reward_coeff_buf

    def _record_slow_reward_episode_returns(self, env_ids):
        """Append completed returns to the same 100-episode window used by runner logging."""
        if not hasattr(self, "episode_return_sums"):
            return
        if self.common_step_counter <= 1:
            self.episode_return_sums[env_ids] = 0.0
            return
        completed_returns = self.episode_return_sums[env_ids]
        history_size = self.slow_reward_episode_returns.numel()
        if completed_returns.numel() > history_size:
            completed_returns = completed_returns[-history_size:]
        count = completed_returns.numel()
        indices = (
            self.slow_reward_episode_return_write_index
            + torch.arange(count, device=self.device)
        ) % history_size
        self.slow_reward_episode_returns[indices] = completed_returns
        self.slow_reward_episode_return_write_index = (
            self.slow_reward_episode_return_write_index + count
        ) % history_size
        self.slow_reward_episode_return_count = min(
            history_size,
            self.slow_reward_episode_return_count + count,
        )
        self.episode_return_sums[env_ids] = 0.0
    
    def compute_observations(self):
        """ Computes observations
        """
        curr_proprio_clean = self._get_proprioception_obs()
        curr_proprio_noisy = self._add_uniform_noise(
            curr_proprio_clean,
            self._get_proprioception_noise_scale().unsqueeze(0),
        )
        self._update_proprioception_history(curr_proprio_noisy)
        proprioception_history = self.proprioception_history_buf.reshape(self.num_envs, -1)
        foot_contact = self._get_foot_contacts().float()
        height_scan = torch.clip(
                self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1) * self.obs_scales.height_measurements
        ladder_obs = self._get_ladder_observations()
        command_obs = self.commands * self.commands_scale
        if hasattr(self, "terrain_ladder_mask") and command_obs.shape[1] > 1:
            is_ladder = self.terrain_ladder_mask[self.terrain_levels, self.terrain_types]
            if torch.count_nonzero(is_ladder).item() > 0:
                command_obs = command_obs.clone()
                ladder_goal_x_sign = torch.where(
                    self.commands[is_ladder, 0] < 0.,
                    -torch.ones_like(self.commands[is_ladder, 0]),
                    torch.ones_like(self.commands[is_ladder, 0]),
                )
                command_obs[is_ladder, 0] = (1. - self.reached_goal[is_ladder, 0]) * ladder_goal_x_sign
                flat_mask = self._get_flat_terrain_mask() > 0.5
                command_obs[is_ladder & (~flat_mask), 1] = 0.
        obs_parts = [
            # goal
            command_obs,
            self.reached_goal,
            # current prioproception
            curr_proprio_clean,
            curr_proprio_noisy,

            # prioproception buffer
            proprioception_history,

            # privileged information to be reconstructed
            self.base_lin_vel * self.obs_scales.lin_vel,
            foot_contact,
            self.friction_coeffs,
            self.base_added_mass,
            self.base_force_local * self.obs_scales.applied_wrench,
            self.base_torque_local * self.obs_scales.applied_wrench,
            self.effector_ladder_plane_distance,
            self.effector_nearest_bar_distance,

            # height scan
            height_scan,
            # ladder obs to be reconstructed
            ladder_obs,

            # forward depth
            self.depth_image_noisy_buf,
        ]

        self.obs_buf = torch.cat(obs_parts, dim=-1)

    def _get_proprioception_obs(self):
        return torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
            ),
            dim=-1,
        )

    def _get_proprioception_noise_scale(self):
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_scale = torch.zeros(3 + 3 + 3 * self.num_actions, dtype=torch.float, device=self.device)
        noise_scale[0:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_scale[3:6] = noise_scales.gravity * noise_level
        dof_pos_start = 6
        noise_scale[dof_pos_start:dof_pos_start + self.num_actions] = (
            noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        )
        dof_vel_start = dof_pos_start + self.num_actions
        noise_scale[dof_vel_start:dof_vel_start + self.num_actions] = (
            noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        )
        return noise_scale

    def _add_uniform_noise(self, tensor, noise_scale):
        if not self.add_noise:
            return tensor
        return tensor + (2 * torch.rand_like(tensor) - 1) * noise_scale

    def _add_depth_speckle_noise(self, depth, edge_mask=None):
        if not self.add_noise:
            return depth

        dropout_prob = float(getattr(self.cfg.noise.noise_scales, "depth_dropout_prob", 0.0))
        outlier_prob = float(getattr(self.cfg.noise.noise_scales, "depth_outlier_prob", 0.0))
        edge_dropout_range = getattr(self.cfg.noise.noise_scales, "depth_edge_dropout_prob", 0.0)
        if isinstance(edge_dropout_range, (tuple, list)):
            if len(edge_dropout_range) != 2:
                raise ValueError("depth_edge_dropout_prob must be a scalar or a [min, max] range")
            edge_dropout_min, edge_dropout_max = map(float, edge_dropout_range)
        else:
            edge_dropout_min = edge_dropout_max = float(edge_dropout_range)
        if (not 0.0 <= dropout_prob <= 1.0
                or not 0.0 <= outlier_prob <= 1.0
                or not 0.0 <= edge_dropout_min <= edge_dropout_max <= 1.0):
            raise ValueError("Depth dropout, edge dropout, and outlier probabilities must be within [0, 1]")
        if dropout_prob + outlier_prob > 1.0:
            raise ValueError("Depth dropout and outlier probabilities must sum to at most 1")

        sample = torch.rand_like(depth)
        dropout_mask = sample < dropout_prob
        if edge_dropout_max > 0.0:
            if edge_mask is None:
                edge_mask = self._get_depth_edge_mask(depth)
            edge_dropout_prob = edge_dropout_min + torch.rand(
                self.num_envs,
                1,
                device=depth.device,
                dtype=depth.dtype,
            ) * (edge_dropout_max - edge_dropout_min)
            dropout_mask |= edge_mask & (torch.rand_like(depth) < edge_dropout_prob)
        outlier_mask = (sample >= dropout_prob) & (sample < dropout_prob + outlier_prob)
        min_depth = float(self.cfg.sensor.depth_min)
        max_depth = float(self.cfg.sensor.depth_max)
        noisy_depth = torch.where(
            dropout_mask,
            torch.full_like(depth, max_depth),
            depth,
        )
        random_depth = min_depth + torch.rand_like(depth) * (max_depth - min_depth)
        return torch.where(outlier_mask, random_depth, noisy_depth)

    def _get_depth_edge_mask(self, depth):
        height = int(self.cfg.sensor.depth_height)
        width = int(self.cfg.sensor.depth_width)
        depth_image = depth.view(self.num_envs, height, width)
        edge_threshold = float(getattr(self.cfg.noise.noise_scales, "depth_edge_threshold", 0.08))

        edge = torch.zeros_like(depth_image, dtype=torch.bool)
        near_dilation = int(getattr(self.cfg.noise.noise_scales, "depth_edge_dilation_near", 0))
        far_dilation = int(getattr(self.cfg.noise.noise_scales, "depth_edge_dilation_far", 0))
        if near_dilation == 0 and far_dilation == 0:
            return edge.view(self.num_envs, -1)
        if not near_dilation <= far_dilation <= 0:
            raise ValueError("Depth edge dilation must satisfy near <= far <= 0 for inward masking")
        dilation_distance_range = getattr(
            self.cfg.noise.noise_scales,
            "depth_edge_dilation_distance_range",
            [self.cfg.sensor.depth_min, self.cfg.sensor.depth_max],
        )
        if len(dilation_distance_range) != 2:
            raise ValueError("depth_edge_dilation_distance_range must contain [near, far] depths")
        near_distance, far_distance = map(float, dilation_distance_range)
        if not near_distance < far_distance:
            raise ValueError("Depth edge dilation distance range must satisfy near < far")
        dilation_exponent = float(getattr(
            self.cfg.noise.noise_scales,
            "depth_edge_dilation_distance_exponent",
            1.0,
        ))
        if dilation_exponent <= 0.0:
            raise ValueError("Depth edge dilation distance exponent must be positive")

        def dilation_width(edge_depth):
            distance_ratio = torch.clamp(
                (edge_depth - near_distance) / (far_distance - near_distance),
                0.0,
                1.0,
            )
            near_strength = torch.pow(1.0 - distance_ratio, dilation_exponent)
            dilation = torch.round(
                far_dilation + (near_dilation - far_dilation) * near_strength
            ).long()
            return -dilation

        ladder_hits = self.depth_camera_ladder_hits.view(self.num_envs, height, width)
        x_ladder_boundary = ladder_hits[:, :, 1:] != ladder_hits[:, :, :-1]
        y_ladder_boundary = ladder_hits[:, 1:, :] != ladder_hits[:, :-1, :]
        x_diff = (torch.abs(depth_image[:, :, 1:] - depth_image[:, :, :-1]) > edge_threshold) & x_ladder_boundary
        y_diff = (torch.abs(depth_image[:, 1:, :] - depth_image[:, :-1, :]) > edge_threshold) & y_ladder_boundary

        x_left_is_nearer = depth_image[:, :, :-1] < depth_image[:, :, 1:]
        y_top_is_nearer = depth_image[:, :-1, :] < depth_image[:, 1:, :]
        x_use_left = x_diff & x_left_is_nearer
        x_use_right = x_diff & (~x_left_is_nearer)
        y_use_top = y_diff & y_top_is_nearer
        y_use_bottom = y_diff & (~y_top_is_nearer)
        x_width = dilation_width(torch.minimum(depth_image[:, :, :-1], depth_image[:, :, 1:]))
        y_width = dilation_width(torch.minimum(depth_image[:, :-1, :], depth_image[:, 1:, :]))

        for offset in range(-near_dilation):
            if offset < width - 1:
                edge[:, :, :width - 1 - offset] |= (x_use_left & (x_width > offset))[:, :, offset:]
                edge[:, :, 1 + offset:] |= (x_use_right & (x_width > offset))[:, :, :width - 1 - offset]
            if offset < height - 1:
                edge[:, :height - 1 - offset, :] |= (y_use_top & (y_width > offset))[:, offset:, :]
                edge[:, 1 + offset:, :] |= (y_use_bottom & (y_width > offset))[:, :height - 1 - offset, :]
        return edge.view(self.num_envs, -1)

    def _update_proprioception_history(self, proprioception):
        self.proprioception_history_buf = torch.roll(self.proprioception_history_buf, shifts=-1, dims=1)
        self.proprioception_history_buf[:, -1, :] = proprioception
        reset_envs = self.proprioception_history_reset_buf
        if torch.any(reset_envs):
            self.proprioception_history_buf[reset_envs] = proprioception[reset_envs].unsqueeze(1)
            self.proprioception_history_reset_buf[reset_envs] = False

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device=self.device)
                self.friction_coeffs = friction_buckets[bucket_ids.squeeze(-1)]
        elif env_id == 0:
            self.friction_coeffs = torch.full((self.num_envs, 1), self.cfg.terrain.static_friction, dtype=torch.float, device=self.device)

        for s in range(len(props)):
            props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        if hasattr(self.cfg.control, "dof_friction"):
            props["friction"][:] = self.cfg.control.dof_friction
        return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            added_mass = np.random.uniform(rng[0], rng[1])
            props[0].mass += added_mass
            self.base_added_mass[env_id, 0] = added_mass
        else:
            self.base_added_mass[env_id, 0] = 0.0
        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        self._update_platform_commands()

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()
        if (self.cfg.domain_rand.push_robot_foot
                and self.cfg.domain_rand.push_foot_interval > 0
                and self.common_step_counter % self.cfg.domain_rand.push_foot_interval == 0):
            self._push_robot_feet()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        if len(env_ids) == 0:
            return
        self._update_platform_commands(env_ids)

    def _update_platform_commands(self, env_ids=None):
        if not hasattr(self, "goal_targets"):
            return

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return

        rel_world = self.goal_targets[env_ids, :3] - self.root_states[env_ids, :3]
        base_yaw_inv = self.base_quat[env_ids].clone()
        base_yaw_inv[:, :3] *= -1
        rel_yaw = quat_apply_yaw(base_yaw_inv, rel_world)
        self.commands[env_ids] = 0.
        self.commands[env_ids, :2] = rel_yaw[:, :2]
        self.goal_dist[env_ids] = torch.norm(rel_world, dim=1)
        self.reached_goal[env_ids, 0] = (self.goal_dist[env_ids] < self.cfg.rewards.goal_radius).float()

    def _reset_goal_progress(self, env_ids=None):
        if not hasattr(self, "goal_targets"):
            return

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return

        self._update_platform_commands(env_ids)
        self.min_goal_dist[env_ids] = self.goal_dist[env_ids]
        self.last_goal_dist[env_ids] = self.goal_dist[env_ids]

    def _get_ladder_progress(self, env_ids=None):
        if not hasattr(self, "goal_targets"):
            if env_ids is None:
                return torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
            return torch.zeros(len(env_ids), device=self.device, dtype=torch.float)

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            progress_origins = self.terrain_ladder_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        else:
            progress_origins = self.env_origins[env_ids, :3]
        target_vec = self.goal_targets[env_ids, :3] - progress_origins
        robot_vec = self.root_states[env_ids, :3] - progress_origins
        target_dist = torch.norm(target_vec, dim=1).clamp(min=1e-6)
        progress = torch.sum(robot_vec * target_vec, dim=1) / torch.square(target_dist)
        return torch.clamp(progress, 0.0, 1.0)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        elif control_type=="UNITREE":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
            same_direction = (self.dof_vel * torques) > 0.0
            max_torque = torch.where(
                same_direction,
                torch.full_like(torques, self.cfg.control.motor_torque_y1),
                torch.full_like(torques, self.cfg.control.motor_torque_y2),
            )
            speed = torch.abs(self.dof_vel)
            x1 = self.cfg.control.motor_velocity_x1
            x2 = self.cfg.control.motor_velocity_x2
            decay_torque = max_torque * (x2 - speed) / max(x2 - x1, 1e-6)
            torque_limit = torch.where(speed < x1, max_torque, torch.clamp(decay_torque, min=0.0))
            torque_limit = torch.minimum(torque_limit, self.torque_limits.unsqueeze(0))
            torques = torch.clip(torques, -torque_limit, torque_limit)
            friction = (
                self.cfg.control.motor_static_friction
                * torch.tanh(self.dof_vel / self.cfg.control.motor_friction_activation_velocity)
                + self.cfg.control.motor_dynamic_friction * self.dof_vel
            )
            return torch.clip(torques - friction, -self.torque_limits, self.torque_limits)
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _get_delayed_actions(self, actions):
        if self.max_action_delay <= 0:
            return actions

        self.action_delay_buffer = torch.roll(self.action_delay_buffer, shifts=1, dims=0)
        self.action_delay_buffer[0] = actions
        return self.action_delay_buffer[self.action_delay_steps, self.action_delay_env_ids]

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            # self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]

        if hasattr(self, "init_state_pos_lower"):
            init_pos = self.init_state_pos_lower + torch.rand(
                len(env_ids), 3, device=self.device
            ) * (self.init_state_pos_upper - self.init_state_pos_lower)
            self.root_states[env_ids, :3] += init_pos - self.base_init_state[:3]
        self._randomize_rough_spawn_yaw(env_ids)
        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.projected_gravity[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.gravity_vec[env_ids])
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _randomize_rough_spawn_yaw(self, env_ids):
        if not hasattr(self, "terrain_ladder_mask"):
            return
        is_ladder = self.terrain_ladder_mask[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        rough_env_ids = env_ids[~is_ladder]
        if len(rough_env_ids) == 0:
            return

        yaw = torch_rand_float(
            -np.deg2rad(30.0),
            np.deg2rad(30.0),
            (len(rough_env_ids), 1),
            device=self.device,
        ).squeeze(-1)
        half_yaw = 0.5 * yaw
        self.root_states[rough_env_ids, 3:7] = torch.stack(
            (
                torch.zeros_like(half_yaw),
                torch.zeros_like(half_yaw),
                torch.sin(half_yaw),
                torch.cos(half_yaw),
            ),
            dim=1,
        )

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _push_robot_feet(self):
        """Randomly push one foot per robot with a one-step velocity-equivalent impulse."""
        if len(self.feet_indices) == 0:
            return
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        max_vel = self.cfg.domain_rand.max_push_foot_vel_xy
        foot_ids = torch.randint(len(self.feet_indices), (self.num_envs,), device=self.device)
        self.foot_push_body_indices[:] = self.feet_indices[foot_ids]
        env_ids = torch.arange(self.num_envs, device=self.device)
        target_vel_xy = torch_rand_float(
            -max_vel,
            max_vel,
            (self.num_envs, 2),
            device=self.device,
        )
        current_vel_xy = self.rigid_body_states[env_ids, self.foot_push_body_indices, 7:9]
        body_mass = self.rigid_body_masses[self.foot_push_body_indices].unsqueeze(1)
        self.foot_push_force.zero_()
        self.foot_push_force[:, :2] = body_mass * (target_vel_xy - current_vel_xy) / self.sim_params.dt
        self.foot_push_steps_remaining = 1

    def _sample_symmetric_body_vector(self, max_values):
        max_tensor = torch.as_tensor(max_values, dtype=torch.float, device=self.device)
        if max_tensor.numel() == 1:
            max_tensor = max_tensor.repeat(3)
        elif max_tensor.numel() != 3:
            raise ValueError("Base disturbance maxima must be a scalar or a 3-element sequence.")
        return (2.0 * torch.rand(self.num_envs, 3, device=self.device) - 1.0) * max_tensor.view(1, 3)

    def _resample_pd_gains(self, env_ids):
        if len(env_ids) == 0:
            return

        base_p = self.base_p_gains.unsqueeze(0)
        base_d = self.base_d_gains.unsqueeze(0)
        if not self.cfg.domain_rand.randomize_pd_gains:
            self.p_gain_multipliers[env_ids] = 1.0
            self.d_gain_multipliers[env_ids] = 1.0
        else:
            p_low, p_high = self.cfg.domain_rand.stiffness_multiplier_range
            d_low, d_high = self.cfg.domain_rand.damping_multiplier_range
            self.p_gain_multipliers[env_ids] = torch_rand_float(p_low, p_high, (len(env_ids), 1), device=self.device)
            self.d_gain_multipliers[env_ids] = torch_rand_float(d_low, d_high, (len(env_ids), 1), device=self.device)
        self.p_gains[env_ids] = base_p * self.p_gain_multipliers[env_ids]
        self.d_gains[env_ids] = base_d * self.d_gain_multipliers[env_ids]

    def _apply_rigid_body_force_disturbances(self):
        self.base_force_tensors.zero_()
        self.base_torque_tensors.zero_()
        self._accumulate_base_force_torque_disturbance()
        self._accumulate_foot_push_disturbance()
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.base_force_tensors),
            gymtorch.unwrap_tensor(self.base_torque_tensors),
            gymapi.ENV_SPACE,
        )
        self.sim_step_counter += 1

    def _accumulate_base_force_torque_disturbance(self):
        cfg = self.cfg.domain_rand
        if not cfg.apply_base_force_torque:
            return

        if self.base_force_duration_steps <= 0 or self.base_force_interval_steps <= 0:
            return

        if self.base_force_torque_steps_remaining <= 0:
            if self.sim_step_counter > 0 and (self.sim_step_counter % self.base_force_interval_steps == 0):
                self.base_force_local[:] = self._sample_symmetric_body_vector(cfg.max_base_force)
                self.base_torque_local[:] = self._sample_symmetric_body_vector(cfg.max_base_torque)
                self.base_force_torque_steps_remaining = self.base_force_duration_steps
            else:
                self.base_force_local.zero_()
                self.base_torque_local.zero_()
                return

        world_force = quat_apply(self.base_quat, self.base_force_local)
        world_torque = quat_apply(self.base_quat, self.base_torque_local)

        self.base_force_tensors[:, self.base_body_index, :] = world_force
        self.base_torque_tensors[:, self.base_body_index, :] = world_torque

        self.base_force_torque_steps_remaining -= 1

    def _accumulate_foot_push_disturbance(self):
        if self.foot_push_steps_remaining <= 0:
            return
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.base_force_tensors[env_ids, self.foot_push_body_indices, :] += self.foot_push_force
        self.foot_push_steps_remaining -= 1

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done or self.common_step_counter == 0:
            # don't change on initial reset
            return
        current_is_ladder = torch.ones(len(env_ids), dtype=torch.bool, device=self.device)
        if hasattr(self, "terrain_ladder_mask"):
            current_is_ladder = self.terrain_ladder_mask[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            progress_origins = self.terrain_ladder_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        else:
            progress_origins = self.env_origins[env_ids, :3]
        distance = torch.norm(self.root_states[env_ids, :3] - progress_origins, dim=1)
        if hasattr(self, "goal_targets"):
            target_vec = self.goal_targets[env_ids, :3] - progress_origins
            robot_vec = self.root_states[env_ids, :3] - progress_origins
            nominal_distance = torch.norm(target_vec, dim=1).clamp(min=1e-6)
            progress = torch.sum(robot_vec * target_vec, dim=1) / nominal_distance
            goal_dist = torch.norm(self.root_states[env_ids, :3] - self.goal_targets[env_ids, :3], dim=1)
            move_up_ratio = float(self.cfg.terrain.curriculum_move_up_ratio)
            move_down_ratio = float(self.cfg.terrain.curriculum_move_down_ratio)
            move_up = (goal_dist < self.cfg.rewards.goal_radius) | (progress > nominal_distance * move_up_ratio)
            move_down = (progress < nominal_distance * move_down_ratio) & ~move_up
        else:
            # robots that walked far enough progress to harder terains
            move_up = distance > self.terrain.env_length / 2
            # robots that walked less than half of their required distance go to simpler terrains
            move_down = ~move_up
        move_up = move_up & current_is_ladder
        move_down = move_down & current_is_ladder

        review_failed = self.review_episode_mask[env_ids] & current_is_ladder & ~move_up
        self.ladder_levels[env_ids] = torch.where(
            review_failed,
            self.terrain_levels[env_ids],
            self.ladder_levels[env_ids],
        )
        move_down = move_down & ~review_failed

        self.ladder_levels[env_ids] += 1 * move_up - 1 * move_down
        self.ladder_levels[env_ids] = torch.clamp(self.ladder_levels[env_ids], min=1, max=self.max_terrain_level)
        self._sample_terrain_levels_and_types(env_ids)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        if hasattr(self, "terrain_platform_centers"):
            self.goal_targets[env_ids] = self.terrain_platform_centers[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
            self.goal_targets[env_ids, 2] += self.cfg.rewards.base_height_target
            self._randomize_rough_targets(env_ids)
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        return


    def _get_ladder_observations(self):
        ladder_obs = torch.zeros(self.num_envs, 5, device=self.device, dtype=torch.float)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            return ladder_obs
        if (not hasattr(self, "terrain_ladder_bar_spacing")
                or not hasattr(self, "terrain_ladder_angles")
                or not hasattr(self, "terrain_ladder_bar_y_scales")):
            return ladder_obs

        bar_spacing = self.terrain_ladder_bar_spacing[self.terrain_levels, self.terrain_types]
        ladder_angle_deg = self.terrain_ladder_angles[self.terrain_levels, self.terrain_types]
        bar_y_scale = self.terrain_ladder_bar_y_scales[self.terrain_levels, self.terrain_types]
        ladder_angle_rad = torch.deg2rad(ladder_angle_deg)
        is_ladder = self.terrain_ladder_mask[self.terrain_levels, self.terrain_types]
        ladder_origins = self.terrain_ladder_origins[self.terrain_levels, self.terrain_types]
        base_lateral_offset = self.root_states[:, 1] - ladder_origins[:, 1]
        base_lateral_offset = torch.where(
            is_ladder, base_lateral_offset, torch.zeros_like(base_lateral_offset)
        )

        forward = quat_apply(self.base_quat, self.forward_vec)
        base_heading = torch.atan2(forward[:, 1], forward[:, 0])
        ladder_up_yaw_rel = wrap_to_pi(-base_heading)
        ladder_up_yaw_rel = torch.where(is_ladder, ladder_up_yaw_rel, torch.zeros_like(ladder_up_yaw_rel))

        ladder_obs[:, 0] = bar_spacing
        ladder_obs[:, 1] = bar_y_scale
        ladder_obs[:, 2] = ladder_angle_rad
        ladder_obs[:, 3] = base_lateral_offset
        ladder_obs[:, 4] = ladder_up_yaw_rel
        return ladder_obs

    def _update_effector_ladder_distances(self):
        """Update effector distances in the ladder-plane coordinate frame.

        A foot is on the ladder only when its world x lies between the ladder
        ground intersection and its topmost rung. Else, plane distance means
        world-z height above the ground/platform support map.
        """
        self.effector_ladder_plane_distance.zero_()
        self.effector_nearest_bar_distance.zero_()
        self.effector_on_ladder_plane.zero_()
        if not hasattr(self, "terrain_ladder_bar_along_distances"):
            return

        effector_positions = self.rigid_body_states[:, self.distance_effector_indices, :3]
        ladder_origins = self.terrain_ladder_origins[self.terrain_levels, self.terrain_types]
        ladder_angles = torch.deg2rad(
            self.terrain_ladder_angles[self.terrain_levels, self.terrain_types]
        )
        offset_x = effector_positions[..., 0] - ladder_origins[:, None, 0]
        offset_z = effector_positions[..., 2] - ladder_origins[:, None, 2]
        cos_angle = torch.cos(ladder_angles)[:, None]
        sin_angle = torch.sin(ladder_angles)[:, None]

        along_ladder = offset_x * cos_angle + offset_z * sin_angle
        bar_along_distances = self.terrain_ladder_bar_along_distances[
            self.terrain_levels, self.terrain_types
        ]
        bar_counts = self.terrain_ladder_bar_counts[self.terrain_levels, self.terrain_types]
        valid_bars = torch.arange(
            bar_along_distances.shape[1], device=self.device
        )[None, :] < bar_counts[:, None]
        top_bar_along = bar_along_distances.masked_fill(~valid_bars, float("-inf")).amax(dim=1)
        top_bar_x = ladder_origins[:, 0] + top_bar_along * cos_angle[:, 0]
        ladder_x_min = torch.minimum(ladder_origins[:, 0], top_bar_x)
        ladder_x_max = torch.maximum(ladder_origins[:, 0], top_bar_x)
        on_ladder_plane = (
            (bar_counts[:, None] > 0)
            & (effector_positions[..., 0] >= ladder_x_min[:, None])
            & (effector_positions[..., 0] <= ladder_x_max[:, None])
        )
        self.effector_on_ladder_plane[:] = on_ladder_plane
        nearest_bar_distance = torch.abs(
            along_ladder[:, :, None] - bar_along_distances[:, None, :]
        ).masked_fill(~valid_bars[:, None, :], float("inf")).amin(dim=2)
        support_height = self._sample_height_map_at_world_points(
            effector_positions, self.support_height_samples
        )
        self.effector_nearest_bar_distance[:] = torch.where(
            on_ladder_plane,
            nearest_bar_distance,
            torch.zeros_like(nearest_bar_distance),
        )
        self.effector_ladder_plane_distance[:] = torch.where(
            on_ladder_plane,
            offset_x * sin_angle - offset_z * cos_angle,
            effector_positions[..., 2] - support_height,
        )

    def _lerp_cfg_range(self, value_range, difficulty):
        if np.isscalar(value_range):
            return torch.full_like(difficulty, float(value_range))
        low, high = value_range
        return float(low) + (float(high) - float(low)) * difficulty

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, self.num_bodies, 13)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.sim_step_counter = 0
        self.extras = {}
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.base_p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.base_d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gain_multipliers = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gain_multipliers = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.proprioception_dim = 3 + 3 + 3 * self.num_actions
        self.proprioception_history_len = int(self.cfg.env.proprioception_history_len)
        self.proprioception_history_buf = torch.zeros(
            self.num_envs,
            self.proprioception_history_len,
            self.proprioception_dim,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.proprioception_history_reset_buf = torch.ones(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )
        self.depth_image_buf = torch.zeros(
            self.num_envs,
            int(self.cfg.sensor.depth_height) * int(self.cfg.sensor.depth_width),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.depth_image_noisy_buf = torch.zeros_like(self.depth_image_buf)
        self.depth_image_latency_buf = torch.zeros(
            self.depth_camera_latency_steps + 1,
            self.num_envs,
            self.depth_image_buf.shape[1],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.depth_image_latency_valid = torch.zeros(
            self.depth_camera_latency_steps + 1,
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )
        self.depth_image_latency_initialized = False
        self.depth_image_delivered_this_step = False
        self.last_depth_camera_update_step = -self.depth_camera_update_interval_steps
        self.depth_camera = None
        self.depth_camera_position = None
        self.depth_camera_rotation = None
        self.depth_camera_axis_rotation = None
        if self.cfg.sensor.enable_depth_camera:
            self._init_depth_camera()
        self.min_action_delay = int(getattr(self.cfg.control, "min_delay", 0))
        self.max_action_delay = int(getattr(self.cfg.control, "max_delay", 0))
        if self.min_action_delay < 0 or self.max_action_delay < self.min_action_delay:
            raise ValueError("control.min_delay and control.max_delay must satisfy 0 <= min_delay <= max_delay")
        self.action_delay_buffer = torch.zeros(
            self.max_action_delay + 1,
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.action_delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        if self.max_action_delay > 0:
            self.action_delay_steps[:] = torch.randint(
                low=self.min_action_delay,
                high=self.max_action_delay + 1,
                size=(self.num_envs,),
                dtype=torch.long,
                device=self.device,
            )
        self.action_delay_env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_scale = torch.ones(self.cfg.commands.num_commands, device=self.device, requires_grad=False)
        self.goal_dist = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_goal_dist = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.min_goal_dist = torch.full((self.num_envs,), float("inf"), dtype=torch.float, device=self.device, requires_grad=False)
        self.reached_goal = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.review_episode_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.difficulty = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self._update_difficulty()
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.terrain_ladder_origins = torch.from_numpy(self.terrain.ladder_origins).to(self.device).to(torch.float)
            self.terrain_platform_centers = torch.from_numpy(self.terrain.platform_centers).to(self.device).to(torch.float)
            self.terrain_ladder_mask = torch.from_numpy(self.terrain.ladder_mask).to(self.device)
            self.terrain_ladder_bar_spacing = torch.from_numpy(self.terrain.ladder_bar_spacing).to(self.device).to(torch.float)
            self.terrain_ladder_bar_counts = torch.from_numpy(self.terrain.ladder_bar_counts).to(self.device)
            self.terrain_ladder_bar_along_distances = torch.from_numpy(
                self.terrain.ladder_bar_along_distances
            ).to(self.device).to(torch.float)
            self.support_height_samples = torch.from_numpy(
                self.terrain.support_heightsamples
            ).view(self.terrain.obs_tot_rows, self.terrain.obs_tot_cols).to(self.device)
            self.terrain_ladder_angles = torch.from_numpy(self.terrain.ladder_angles).to(self.device).to(torch.float)
            self.terrain_ladder_bar_y_scales = torch.from_numpy(self.terrain.ladder_bar_y_scales).to(self.device).to(torch.float)
            self.terrain_ladder_side_rail_half_width = torch.from_numpy(
                self.terrain.ladder_side_rail_half_width
            ).to(self.device).to(torch.float)
        self.goal_targets = self.terrain_platform_centers[self.terrain_levels, self.terrain_types].clone()
        self.goal_targets[:, 2] += self.cfg.rewards.base_height_target
        self._randomize_rough_targets(torch.arange(self.num_envs, device=self.device))
        self._reset_goal_progress(torch.arange(self.num_envs, device=self.device))
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_first_contact = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.bool, device=self.device, requires_grad=False)
        self.feet_last_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.min_feet_last_air_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_ground_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_first_air = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.bool, device=self.device, requires_grad=False)
        self.feet_last_ground_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.min_feet_last_ground_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.foot_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.effector_ladder_plane_distance = torch.zeros(
            self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device, requires_grad=False
        )
        self.effector_nearest_bar_distance = torch.zeros_like(self.effector_ladder_plane_distance)
        self.effector_on_ladder_plane = torch.zeros(
            self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False
        )
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.phase_feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.phase_feet_first_contact = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.bool, device=self.device, requires_grad=False)
        self.phase_feet_last_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.min_phase_feet_last_air_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.phase_feet_ground_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.phase_feet_first_air = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.bool, device=self.device, requires_grad=False)
        self.phase_feet_last_ground_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.min_phase_feet_last_ground_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.phase_foot_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.last_phase_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_force_tensors = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.base_torque_tensors = torch.zeros_like(self.base_force_tensors)
        self.base_force_local = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.base_torque_local = torch.zeros_like(self.base_force_local)
        self.base_force_torque_steps_remaining = 0
        self.foot_push_force = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.foot_push_body_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        self.foot_push_steps_remaining = 0
        num_symmetry_pairs = int(self.symmetry_joint_pair_indices.shape[0]) if hasattr(self, "symmetry_joint_pair_indices") else 0
        self.filtered_symmetry_torque_abs_diff = torch.zeros(
            self.num_envs, num_symmetry_pairs, dtype=torch.float, device=self.device, requires_grad=False
        )
        symmetry_torque_lpf_tau = float(self.cfg.rewards.symmetry_torque_lpf_tau)
        self.symmetry_torque_lpf_alpha = self.dt / (symmetry_torque_lpf_tau + self.dt) if symmetry_torque_lpf_tau > 0.0 else 1.0
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.base_p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.base_d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.base_p_gains[i] = 0.
                self.base_d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self._resample_pd_gains(torch.arange(self.num_envs, device=self.device))

    def _update_feet_air_time_state(self):
        self.foot_contacts = self._get_foot_contacts()
        self.phase_foot_contacts = self._get_phase_foot_contacts()

        current_feet_air_time = self.feet_air_time + self.dt
        current_feet_ground_time = self.feet_ground_time + self.dt
        current_phase_feet_air_time = self.phase_feet_air_time + self.dt
        current_phase_feet_ground_time = self.phase_feet_ground_time + self.dt

        self.feet_first_contact = (self.feet_air_time > 0.) & self.foot_contacts & (~self.last_contacts)
        self.feet_last_air_time = torch.where(
            self.feet_first_contact,
            self.feet_air_time,
            self.feet_last_air_time,
        )
        self.feet_first_air = (self.feet_ground_time > 0.) & (~self.foot_contacts) & self.last_contacts
        self.feet_last_ground_time = torch.where(
            self.feet_first_air,
            self.feet_ground_time,
            self.feet_last_ground_time,
        )
        self.last_contacts = self.foot_contacts
        self.feet_air_time = current_feet_air_time * (~self.foot_contacts).float()
        self.feet_ground_time = current_feet_ground_time * self.foot_contacts.float()
        self.min_feet_last_air_time = self._update_min_valid_last_time(
            self.feet_last_air_time,
            self.min_feet_last_air_time,
        )
        self.min_feet_last_ground_time = self._update_min_valid_last_time(
            self.feet_last_ground_time,
            self.min_feet_last_ground_time,
        )

        self.phase_feet_first_contact = (self.phase_feet_air_time > 0.) & self.phase_foot_contacts & (~self.last_phase_contacts)
        self.phase_feet_last_air_time = torch.where(
            self.phase_feet_first_contact,
            self.phase_feet_air_time,
            self.phase_feet_last_air_time,
        )
        self.phase_feet_first_air = (self.phase_feet_ground_time > 0.) & (~self.phase_foot_contacts) & self.last_phase_contacts
        self.phase_feet_last_ground_time = torch.where(
            self.phase_feet_first_air,
            self.phase_feet_ground_time,
            self.phase_feet_last_ground_time,
        )
        self.last_phase_contacts = self.phase_foot_contacts
        self.phase_feet_air_time = current_phase_feet_air_time * (~self.phase_foot_contacts).float()
        self.phase_feet_ground_time = current_phase_feet_ground_time * self.phase_foot_contacts.float()
        self.min_phase_feet_last_air_time = self._update_min_valid_last_time(
            self.phase_feet_last_air_time,
            self.min_phase_feet_last_air_time,
        )
        self.min_phase_feet_last_ground_time = self._update_min_valid_last_time(
            self.phase_feet_last_ground_time,
            self.min_phase_feet_last_ground_time,
        )

    def _update_min_valid_last_time(self, last_time_tensor, current_min_tensor):
        valid_mask = last_time_tensor > 0.
        masked = torch.where(valid_mask, last_time_tensor, torch.full_like(last_time_tensor, float("inf")))
        min_vals = torch.min(masked, dim=1).values
        return torch.where(torch.isinf(min_vals), current_min_tensor, min_vals)

    def _update_symmetry_torque_state(self):
        if not hasattr(self, "symmetry_joint_pair_indices"):
            return
        if self.symmetry_joint_pair_indices.numel() == 0:
            return
        left_indices = self.symmetry_joint_pair_indices[:, 0]
        right_indices = self.symmetry_joint_pair_indices[:, 1]
        torque_abs_diff = torch.abs(
            torch.abs(self.torques[:, left_indices]) - torch.abs(self.torques[:, right_indices])
        )
        alpha = self.symmetry_torque_lpf_alpha
        self.filtered_symmetry_torque_abs_diff.mul_(1.0 - alpha).add_(alpha * torque_abs_diff)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # This is a reward-scaling meta parameter, not a reward term.
        self.slow_reward_coeff = self.reward_scales.pop("slow_reward_coeff", [1.0, 1.0])
        if len(self.slow_reward_coeff) != 2:
            raise ValueError("rewards.scales.slow_reward_coeff must contain [low, high].")
        self.slow_reward_coeff_lower_reward_limit = float(
            self.reward_scales.pop("slow_reward_coeff_lower_reward_limit", 0.0)
        )
        self.slow_reward_coeff_upper_reward_limit = float(
            self.reward_scales.pop("slow_reward_coeff_upper_reward_limit", 1.0)
        )
        self.slow_reward_coeff_lpf_k = float(self.reward_scales.pop("slow_reward_coeff_lpf_k", 1.0))
        if self.slow_reward_coeff_upper_reward_limit <= self.slow_reward_coeff_lower_reward_limit:
            raise ValueError("slow_reward_coeff_upper_reward_limit must exceed the lower limit.")
        if not 0.0 <= self.slow_reward_coeff_lpf_k <= 1.0:
            raise ValueError("slow_reward_coeff_lpf_k must be within [0, 1].")
        self.slow_reward_coeff_buf = torch.tensor(
            float(self.slow_reward_coeff[0]), dtype=torch.float, device=self.device
        )
        self.episode_return_sums = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.slow_reward_episode_returns = torch.zeros(
            100, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.slow_reward_episode_return_count = 0
        self.slow_reward_episode_return_write_index = 0
        self.slow_reward_names = {
            "lin_vel_z",
            "ang_vel_xy",
            "action_rate",
            "action_smoothness",
            "torques",
            "dof_vel",
            "dof_acc",
        }

        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.cfg.border_size 
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        height_samples = np.asarray(
            getattr(self.terrain, "physics_heightsamples", self.terrain.heightsamples),
            dtype=np.int16,
        ).flatten(order='F')
        self.gym.add_heightfield(self.sim, height_samples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.obs_tot_rows, self.terrain.obs_tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        if self.terrain.vertices.shape[0] == 0 or self.terrain.triangles.shape[0] == 0:
            return

        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.obs_tot_rows, self.terrain.obs_tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        self.dof_name_to_index = {name: i for i, name in enumerate(self.dof_names)}
        symmetry_joint_pairs = [
            ("FL_hip_joint", "FR_hip_joint"),
            ("FL_thigh_joint", "FR_thigh_joint"),
            ("FL_calf_joint", "FR_calf_joint"),
            ("RL_hip_joint", "RR_hip_joint"),
            ("RL_thigh_joint", "RR_thigh_joint"),
            ("RL_calf_joint", "RR_calf_joint"),
        ]
        matched_symmetry_joint_pairs = [
            [self.dof_name_to_index[left_name], self.dof_name_to_index[right_name]]
            for left_name, right_name in symmetry_joint_pairs
            if left_name in self.dof_name_to_index and right_name in self.dof_name_to_index
        ]
        if len(matched_symmetry_joint_pairs) > 0:
            self.symmetry_joint_pair_indices = torch.tensor(
                matched_symmetry_joint_pairs, dtype=torch.long, device=self.device
            )
        else:
            self.symmetry_joint_pair_indices = torch.empty((0, 2), dtype=torch.long, device=self.device)
        # Match the configured terminal body name, not any intermediate frame
        # that merely contains it (for example ``*_effector_center``).
        feet_names = [s for s in body_names if s.endswith(self.cfg.asset.foot_name)]
        distance_link_name = getattr(self.cfg.asset, "distance_link_name", self.cfg.asset.foot_name)
        distance_effector_names = [
            f"{foot_name[:-len(self.cfg.asset.foot_name)]}{distance_link_name}"
            for foot_name in feet_names
        ]
        missing_distance_links = [name for name in distance_effector_names if name not in body_names]
        if missing_distance_links:
            raise ValueError(
                f"Missing distance links for force feet {feet_names}: {missing_distance_links}"
            )
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        init_pos_lower = []
        init_pos_upper = []
        for coordinate in self.cfg.init_state.pos:
            if isinstance(coordinate, (list, tuple)):
                if len(coordinate) != 2:
                    raise ValueError("init_state.pos ranges must contain [min, max].")
                lower, upper = float(coordinate[0]), float(coordinate[1])
            else:
                lower = upper = float(coordinate)
            init_pos_lower.append(min(lower, upper))
            init_pos_upper.append(max(lower, upper))
        self.init_state_pos_lower = torch.tensor(init_pos_lower, device=self.device)
        self.init_state_pos_upper = torch.tensor(init_pos_upper, device=self.device)
        init_pos_center = [
            0.5 * (lower + upper)
            for lower, upper in zip(init_pos_lower, init_pos_upper)
        ]
        base_init_state_list = (
            init_pos_center + self.cfg.init_state.rot
            + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        )
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        self.base_added_mass = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.review_episode_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])
        self.distance_effector_indices = torch.zeros(
            len(distance_effector_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(distance_effector_names)):
            self.distance_effector_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], distance_effector_names[i]
            )
        body_props = self.gym.get_actor_rigid_body_properties(self.envs[0], self.actor_handles[0])
        self.rigid_body_masses = torch.tensor(
            [body_prop.mass for body_prop in body_props],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.foot_name_to_obs_index = {name: i for i, name in enumerate(feet_names)}
        self.distance_effector_names = distance_effector_names

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

        self.base_body_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], self.cfg.asset.base_name)
        if self.base_body_index < 0:
            raise ValueError(f"Could not find base rigid body '{self.cfg.asset.base_name}' in asset body names: {body_names}")

    def _init_depth_camera(self):
        if self.cfg.terrain.mesh_type != "trimesh":
            raise ValueError("Warp depth camera currently requires terrain.mesh_type='trimesh'")

        from legged_gym.utils.warp_depth_camera import WarpDepthCamera

        terrain_offset = [-self.terrain.cfg.border_size, -self.terrain.cfg.border_size, 0.0]
        self.depth_camera = WarpDepthCamera(
            terrain_vertices=self.terrain.vertices,
            terrain_triangles=self.terrain.triangles,
            terrain_offset=terrain_offset,
            num_envs=self.num_envs,
            width=self.cfg.sensor.depth_width,
            height=self.cfg.sensor.depth_height,
            horizontal_fov_deg=self.cfg.sensor.depth_horizontal_fov,
            far_plane=self.cfg.sensor.depth_max,
            device=self.device,
            ladder_triangle_mask=self.terrain.ladder_triangle_mask,
        )
        self.depth_camera_nominal_position = torch.tensor(
            self.cfg.sensor.depth_position,
            dtype=torch.float32,
            device=self.device,
        ).repeat(self.num_envs, 1)
        self.depth_camera_position = torch.empty_like(self.depth_camera_nominal_position)
        self.depth_camera_rotation = torch.empty(
            self.num_envs, 4, dtype=torch.float32, device=self.device
        )
        # Converts Warp's z-forward pinhole frame to Isaac Gym's x-forward camera frame.
        self.depth_camera_axis_rotation = torch.tensor(
            [-0.5, 0.5, -0.5, 0.5],
            dtype=torch.float32,
            device=self.device,
        ).repeat(self.num_envs, 1)
        self._randomize_depth_camera_extrinsics(
            torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        )

    def _randomize_depth_camera_extrinsics(self, env_ids):
        if len(env_ids) == 0:
            return

        position_ranges = torch.as_tensor(
            self.cfg.sensor.depth_position_noise_range,
            dtype=torch.float32,
            device=self.device,
        )
        rotation_ranges = torch.as_tensor(
            self.cfg.sensor.depth_rotation_noise_deg_range,
            dtype=torch.float32,
            device=self.device,
        )
        if position_ranges.shape != (3, 2) or rotation_ranges.shape != (3, 2):
            raise ValueError("Depth camera position and rotation noise ranges must have shape [3, 2]")
        if torch.any(position_ranges[:, 0] > position_ranges[:, 1]):
            raise ValueError("Depth camera position noise lower bounds must not exceed upper bounds")
        if torch.any(rotation_ranges[:, 0] > rotation_ranges[:, 1]):
            raise ValueError("Depth camera rotation noise lower bounds must not exceed upper bounds")

        position_noise = position_ranges[:, 0] + torch.rand(
            len(env_ids), 3, device=self.device
        ) * (position_ranges[:, 1] - position_ranges[:, 0])
        rotation_noise_deg = rotation_ranges[:, 0] + torch.rand(
            len(env_ids), 3, device=self.device
        ) * (rotation_ranges[:, 1] - rotation_ranges[:, 0])

        self.depth_camera_position[env_ids] = (
            self.depth_camera_nominal_position[env_ids] + position_noise
        )
        roll = torch.deg2rad(rotation_noise_deg[:, 0])
        pitch = torch.deg2rad(
            rotation_noise_deg[:, 1] + float(getattr(self.cfg.sensor, "depth_pitch_deg", 0.0))
        )
        yaw = torch.deg2rad(
            rotation_noise_deg[:, 2] + float(getattr(self.cfg.sensor, "depth_yaw_deg", 0.0))
        )
        self.depth_camera_rotation[env_ids] = self._quat_from_euler_xyz(roll, pitch, yaw)

    def _quat_from_euler_xyz(self, roll, pitch, yaw):
        half_roll = 0.5 * roll
        half_pitch = 0.5 * pitch
        half_yaw = 0.5 * yaw
        roll_rotation = torch.stack(
            (
                torch.sin(half_roll),
                torch.zeros_like(half_roll),
                torch.zeros_like(half_roll),
                torch.cos(half_roll),
            ),
            dim=-1,
        )
        pitch_rotation = torch.stack(
            (
                torch.zeros_like(half_pitch),
                torch.sin(half_pitch),
                torch.zeros_like(half_pitch),
                torch.cos(half_pitch),
            ),
            dim=-1,
        )
        yaw_rotation = torch.stack(
            (
                torch.zeros_like(half_yaw),
                torch.zeros_like(half_yaw),
                torch.sin(half_yaw),
                torch.cos(half_yaw),
            ),
            dim=-1,
        )
        return quat_multiply_xyzw(
            quat_multiply_xyzw(yaw_rotation, pitch_rotation),
            roll_rotation,
        )

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_ladder_level
            if not self.cfg.terrain.curriculum:
                max_init_level = self.cfg.terrain.num_rows - 1
            max_init_level = max(1, max_init_level)
            self.ladder_levels = torch.randint(1, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_levels = self.ladder_levels.clone()
            self.terrain_types = torch.randint(0, self.cfg.terrain.num_cols, (self.num_envs,), device=self.device)
            self.max_terrain_level = self.cfg.terrain.num_rows - 1
            self._sample_terrain_levels_and_types(torch.arange(self.num_envs, device=self.device))
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _sample_terrain_levels_and_types(self, env_ids):
        if self.cfg.terrain.mesh_type not in ["heightfield", "trimesh"]:
            return
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if len(env_ids) == 0:
            return

        terrain_num_cols = getattr(self.terrain, "num_cols", self.cfg.terrain.num_cols)
        terrain_kwargs = getattr(self.cfg.terrain, "terrain_kwargs", {})
        rough_prob = float(getattr(self.terrain, "rough_probability", terrain_kwargs.get("rough_probability", 0.25)))
        low_difficulty_prob = float(terrain_kwargs.get("low_difficulty_probability", 0.25))
        sample_draws = torch.rand(len(env_ids), device=self.device)
        use_rough = sample_draws < rough_prob
        use_low_difficulty = (sample_draws >= rough_prob) & (sample_draws < rough_prob + low_difficulty_prob)
        self.review_episode_mask[env_ids] = use_low_difficulty
        self.terrain_types[env_ids] = torch.randint(0, terrain_num_cols, (len(env_ids),), device=self.device)
        saved_ladder_levels = self.ladder_levels[env_ids].clamp(min=1, max=self.max_terrain_level)
        max_lower_levels = torch.clamp(saved_ladder_levels - 1, min=1)
        random_lower_levels = (
            torch.floor(torch.rand(len(env_ids), device=self.device) * max_lower_levels.float()).long() + 1
        )
        sampled_ladder_levels = torch.where(use_low_difficulty, random_lower_levels, saved_ladder_levels)
        self.terrain_levels[env_ids] = torch.where(
            use_rough,
            torch.zeros(len(env_ids), dtype=torch.long, device=self.device),
            sampled_ladder_levels)
        if hasattr(self, "difficulty"):
            self._update_difficulty(env_ids)
        sample_modes = [
            "rough" if is_rough else ("low" if is_low else "saved")
            for is_rough, is_low in zip(use_rough.tolist(), use_low_difficulty.tolist())
        ]
        self._debug_log_sampling_event(
            env_ids,
            "resample",
            rough_prob=float(rough_prob),
            low_difficulty_prob=float(low_difficulty_prob),
            sample_draws=sample_draws.tolist(),
            sample_modes=sample_modes,
        )

    def _randomize_rough_targets(self, env_ids):
        if not hasattr(self, "goal_targets") or not hasattr(self, "terrain_ladder_mask"):
            return
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if len(env_ids) == 0:
            return

        is_ladder = self.terrain_ladder_mask[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        rough_env_ids = env_ids[~is_ladder]
        if len(rough_env_ids) == 0:
            return

        direction = torch.zeros(len(rough_env_ids), 2, device=self.device)
        direction[:, 0] = 1.0
        max_distance = torch.full((len(rough_env_ids),), 0.5 * self.terrain.env_length, device=self.device)
        min_distance = torch.minimum(torch.full_like(max_distance, 0.5), 0.5 * max_distance)
        target_distance = min_distance + torch.rand(len(rough_env_ids), device=self.device) * (max_distance - min_distance)
        target_offsets = direction * target_distance.unsqueeze(1)
        self.goal_targets[rough_env_ids, :2] = self.env_origins[rough_env_ids, :2] + target_offsets
        self.goal_targets[rough_env_ids, 2] = self.cfg.rewards.base_height_target

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.add_noise = bool(self.cfg.noise.add_noise)
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
        self.cfg.domain_rand.push_foot_interval = np.ceil(self.cfg.domain_rand.push_foot_interval_s / self.dt)
        self.base_force_interval_steps = int(np.ceil(self.cfg.domain_rand.base_force_interval_s / self.sim_params.dt))
        self.base_force_duration_steps = int(np.ceil(self.cfg.domain_rand.base_force_duration_s / self.sim_params.dt))
        self.depth_camera_update_interval_steps = max(
            1,
            int(np.ceil(float(self.cfg.sensor.depth_update_interval_s) / self.dt)),
        )
        self.depth_camera_latency_steps = max(
            0,
            int(np.ceil(
                float(getattr(self.cfg.sensor, "depth_latency_s", 0.0))
                / self.dt
            )),
        )

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        target_unreached_geom = gymutil.WireframeSphereGeometry(0.08, 8, 8, None, color=(0, 0, 1))
        target_reached_geom = gymutil.WireframeSphereGeometry(0.08, 8, 8, None, color=(0, 1, 0))
        for i in range(self.num_envs):
            if hasattr(self, "goal_targets"):
                target = self.goal_targets[i].cpu().numpy()
                target_pose = gymapi.Transform(gymapi.Vec3(target[0], target[1], target[2]), r=None)
                target_geom = target_reached_geom if self.reached_goal[i, 0] > 0.5 else target_unreached_geom
                gymutil.draw_lines(target_geom, self.gym, self.viewer, self.envs[i], target_pose)

            if not getattr(self, "draw_height_measurements", True):
                continue
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None, height_samples=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.obs_horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        if height_samples is None:
            height_samples = self.height_samples
        px = torch.clip(px, 0, height_samples.shape[0]-2)
        py = torch.clip(py, 0, height_samples.shape[1]-2)

        heights1 = height_samples[px, py]
        heights2 = height_samples[px+1, py]
        heights3 = height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _sample_height_map_at_world_points(self, points, height_samples):
        """Sample a height map at world-frame points of shape [env, point, 3]."""
        grid_points = ((points[..., :2] + self.terrain.cfg.border_size)
                       / self.terrain.obs_horizontal_scale).long()
        px = torch.clip(grid_points[..., 0].reshape(-1), 0, height_samples.shape[0] - 2)
        py = torch.clip(grid_points[..., 1].reshape(-1), 0, height_samples.shape[1] - 2)
        heights = torch.min(height_samples[px, py], height_samples[px + 1, py])
        heights = torch.min(heights, height_samples[px, py + 1])
        return heights.view(*grid_points.shape[:-1]) * self.terrain.cfg.vertical_scale

    #------------ reward functions----------------
    def _reward_alive(self):
        # Reward for being alive
        return torch.ones(self.num_envs, device=self.device)
    
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        """Penalize height error only where the local scan is flat."""
        if not isinstance(self.measured_heights, torch.Tensor):
            return torch.zeros(self.num_envs, device=self.device)
        local_terrain_height = torch.mean(
            self.measured_heights[:, self._get_flat_height_sample_mask()], dim=1
        )
        base_height = self.root_states[:, 2] - local_terrain_height
        height_error = torch.square(base_height - self.cfg.rewards.base_height_target)
        return height_error * self._get_flat_terrain_mask()
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _get_goal_delta(self):
        return self.reached_goal[:, 0], self.goal_dist

    def _get_flat_terrain_mask(self):
        if not self.cfg.terrain.measure_heights:
            return torch.ones(self.num_envs, device=self.device)
        if not isinstance(self.measured_heights, torch.Tensor):
            return torch.ones(self.num_envs, device=self.device)
        selected_heights = self.measured_heights[:, self._get_flat_height_sample_mask()]
        local_height_std = torch.std(selected_heights, dim=1, unbiased=False)
        return (local_height_std < self.cfg.rewards.flat_height_std_threshold).float()

    def _get_flat_height_sample_mask(self):
        """Return the scan points shared by flat detection and base-height reward."""
        x_coords = self.height_points[0, :, 0]
        y_coords = self.height_points[0, :, 1]
        return (x_coords >= -0.2) & (x_coords <= 0.4) & (torch.abs(y_coords) <= 0.2)

    def _get_local_flat_height(self):
        if not self.cfg.terrain.measure_heights:
            return torch.zeros(self.num_envs, device=self.device)
        if not isinstance(self.measured_heights, torch.Tensor):
            return torch.zeros(self.num_envs, device=self.device)
        x_coords = self.height_points[0, :, 0]
        y_coords = self.height_points[0, :, 1]
        local_mask = (x_coords >= -0.1) & (x_coords <= 0.5) & (torch.abs(y_coords) <= 0.2)
        return torch.mean(self.measured_heights[:, local_mask], dim=1)

    def _update_difficulty(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if len(env_ids) == 0:
            return

        if not hasattr(self, "terrain_levels") or not hasattr(self, "max_terrain_level"):
            self.difficulty[env_ids] = 0.
            return

        max_level = max(int(self.max_terrain_level), 1)
        if max_level <= 1:
            self.difficulty[env_ids] = 0.
            return

        difficulty = (self.terrain_levels[env_ids].float() - 1.) / float(max_level - 1)
        self.difficulty[env_ids] = torch.clamp(difficulty, 0., 1.)

    def _get_foot_contacts(self):
        return self.contact_forces[:, self.feet_indices, 2] > self.cfg.rewards.contact_force_threshold

    def _get_phase_foot_contacts(self):
        return self.contact_forces[:, self.feet_indices, 2] > self.cfg.rewards.phase_contact_force_threshold

    def _update_depth_camera_observations(self):
        self.depth_image_delivered_this_step = False
        if not self.cfg.sensor.enable_depth_camera:
            return
        delivery_slot = self.common_step_counter % self.depth_image_latency_buf.shape[0]
        if self.depth_image_latency_valid[delivery_slot]:
            self.depth_image_noisy_buf[:] = self.depth_image_latency_buf[delivery_slot]
            self.depth_image_latency_valid[delivery_slot] = False
            self.depth_image_delivered_this_step = True
        if (self.common_step_counter - self.last_depth_camera_update_step
                < self.depth_camera_update_interval_steps):
            return
        self.last_depth_camera_update_step = self.common_step_counter

        min_depth = float(self.cfg.sensor.depth_min)
        max_depth = float(self.cfg.sensor.depth_max)
        camera_base_positions = self.rigid_body_states[:, self.base_body_index, :3]
        camera_base_orientations = self.rigid_body_states[:, self.base_body_index, 3:7]
        camera_positions = camera_base_positions + quat_apply(
            camera_base_orientations,
            self.depth_camera_position,
        )
        camera_orientations = quat_multiply_xyzw(
            camera_base_orientations,
            quat_multiply_xyzw(
                self.depth_camera_rotation,
                self.depth_camera_axis_rotation,
            ),
        )
        depth, ladder_hits = self.depth_camera.render(camera_positions, camera_orientations)
        depth = torch.nan_to_num(depth, nan=max_depth, posinf=max_depth, neginf=max_depth)
        depth = torch.clamp(depth, min=min_depth, max=max_depth)
        self.depth_image_buf[:] = depth.reshape(self.num_envs, -1)
        self.depth_camera_ladder_hits = ladder_hits.reshape(self.num_envs, -1)
        depth_edge_mask = self._get_depth_edge_mask(self.depth_image_buf)
        noisy_depth = torch.clamp(
            self._add_uniform_noise(
                self.depth_image_buf,
                self.cfg.noise.noise_scales.depth_image * self.cfg.noise.noise_level,
            ),
            min=min_depth,
            max=max_depth,
        )
        noisy_depth = self._add_depth_speckle_noise(noisy_depth, depth_edge_mask)
        if not self.depth_image_latency_initialized:
            # Warm up the first frame immediately so the policy never receives zeros.
            self.depth_image_noisy_buf[:] = noisy_depth
            self.depth_image_latency_initialized = True
            self.depth_image_delivered_this_step = True
        elif self.depth_camera_latency_steps == 0:
            self.depth_image_noisy_buf[:] = noisy_depth
            self.depth_image_delivered_this_step = True
        else:
            delivery_slot = (
                self.common_step_counter + self.depth_camera_latency_steps
            ) % self.depth_image_latency_buf.shape[0]
            self.depth_image_latency_buf[delivery_slot] = noisy_depth
            self.depth_image_latency_valid[delivery_slot] = True

    def _reward_position_tracking(self):
        goal_reached, goal_dist = self._get_goal_delta()
        goal_dir = self.commands[:, :2] / goal_dist.unsqueeze(1).clamp(min=1e-6)
        velocity_xy = self.base_lin_vel[:, :2]
        velocity_norm = torch.norm(velocity_xy, dim=1).clamp(min=1e-6)
        flat_mask = self._get_flat_terrain_mask()
        speed_limit = torch.where(
            flat_mask > 0.5,
            torch.full_like(velocity_norm, float(self.cfg.rewards.goal_speed_limit)),
            torch.full_like(velocity_norm, float(self.cfg.rewards.nonflat_goal_speed_limit)),
        )
        speed_over = torch.clamp(velocity_norm - speed_limit, min=0.)
        min_dist_decrease_speed = torch.clamp(self.min_goal_dist - goal_dist, min=0.) / self.dt
        dist_increase_speed = torch.clamp(goal_dist - self.last_goal_dist, min=0.) / self.dt
        self.min_goal_dist[:] = torch.minimum(self.min_goal_dist, goal_dist)
        self.last_goal_dist[:] = goal_dist
        heading_error = torch.atan2(goal_dir[:, 1], goal_dir[:, 0])
        heading_gate = torch.clamp(torch.cos(3. * heading_error), min=0.)
        progress_reward_multiplier = 1. + self.difficulty * (
            float(self.cfg.rewards.progress_reward_max_difficulty_multiplier) - 1.
        )
        progress_reward = min_dist_decrease_speed * heading_gate * progress_reward_multiplier
        return (1. - goal_reached) * (progress_reward - 1. * dist_increase_speed - 5. * speed_over) + 1.5 * goal_reached

    def _reward_heading_tracking(self):
        _, goal_dist = self._get_goal_delta()
        heading_error = torch.atan2(self.commands[:, 1], self.commands[:, 0])
        return torch.exp(-10. * torch.square(heading_error)) * torch.exp(-4. * torch.square(goal_dist))

    def _reward_foot_slippage(self):
        foot_contacts = self._get_foot_contacts().float()
        foot_lin_vel_world = self.rigid_body_states[:, self.feet_indices, 7:10]
        foot_lin_vel_base = quat_rotate_inverse(
            self.base_quat.unsqueeze(1).repeat(1, len(self.feet_indices), 1).reshape(-1, 4),
            foot_lin_vel_world.reshape(-1, 3)
        ).view(self.num_envs, len(self.feet_indices), 3)
        return torch.sum(foot_contacts * torch.norm(foot_lin_vel_base[:, :, :2], dim=2), dim=1)

    def _reward_foot_clearance(self):
        # Matches the IsaacLab Go2 term: penalize swing feet away from a target
        # clearance, weighted by horizontal foot speed.
        foot_pos = self.rigid_body_states[:, self.feet_indices, :3]
        foot_vel = self.rigid_body_states[:, self.feet_indices, 7:10]
        ground_height = self._get_local_flat_height().unsqueeze(1)
        foot_height = foot_pos[:, :, 2] - ground_height
        foot_xy_speed = torch.norm(foot_vel[:, :, :2], dim=-1)
        clearance_error = torch.square(self.cfg.rewards.foot_clearance_target - foot_height)
        return torch.sum(clearance_error * foot_xy_speed, dim=1) * self._get_flat_terrain_mask()

    def _reward_ladder_side_clearance(self):
        """Keep feet inside the ladder rails while they are over the ladder span."""
        if not hasattr(self, "terrain_ladder_side_rail_half_width"):
            return torch.zeros(self.num_envs, device=self.device)

        effector_positions = self.rigid_body_states[:, self.distance_effector_indices, :3]
        ladder_origins = self.terrain_ladder_origins[self.terrain_levels, self.terrain_types]
        rail_half_width = self.terrain_ladder_side_rail_half_width[
            self.terrain_levels, self.terrain_types
        ]
        lateral_offset = torch.abs(effector_positions[..., 1] - ladder_origins[:, None, 1])
        inside_clearance = rail_half_width[:, None] - lateral_offset
        threshold = self.cfg.rewards.ladder_side_clearance_threshold
        normalized_deficit = torch.clamp(threshold - inside_clearance, min=0.0) / threshold
        return torch.sum(
            torch.square(normalized_deficit) * self.effector_on_ladder_plane.float(), dim=1
        )

    def _reward_feet_air_time(self):
        # Reward moderate air time and penalize overly long swing time.
        goal_reached, _ = self._get_goal_delta()
        lower = self.cfg.rewards.half_phase_lower
        upper = torch.clamp(self.min_feet_last_air_time, max=self.cfg.rewards.half_phase_upper).unsqueeze(1)
        air_time_reward = torch.clamp(self.feet_last_air_time, max=upper) - lower
        rew_airTime = torch.sum(air_time_reward * self.feet_first_contact.float(), dim=1) # reward only on first contact with the ground
        rew_airTime *= (1. - goal_reached)
        return rew_airTime

    def _reward_feet_ground_time(self):
        goal_reached, _ = self._get_goal_delta()
        lower = self.cfg.rewards.half_phase_lower
        upper = torch.clamp(self.min_phase_feet_last_ground_time, max=self.cfg.rewards.half_phase_upper).unsqueeze(1)
        ground_time_reward = torch.clamp(self.phase_feet_last_ground_time, max=upper) - lower
        rew_ground_time = torch.sum(ground_time_reward * self.phase_feet_first_air.float(), dim=1)
        rew_ground_time *= (1. - goal_reached)
        return rew_ground_time

    def _reward_ladder_contact_precision(self):
        """Reward contacts within a circular tolerance around a rung center."""
        distance_threshold = self.cfg.rewards.ladder_contact_precision_threshold
        min_bar_x_scale_level = int(
            self.cfg.terrain.terrain_kwargs.get("bar_x_scale_min_level", 1)
        )
        squared_alignment_distance = (
            torch.square(self.effector_ladder_plane_distance)
            + torch.square(self.effector_nearest_bar_distance)
        )
        aligned_contact = (
            self._get_foot_contacts()
            & self.effector_on_ladder_plane
            & (squared_alignment_distance < torch.square(
                torch.as_tensor(distance_threshold, device=self.device)
            ))
        )
        is_eligible_ladder = (
            self.terrain_ladder_mask[self.terrain_levels, self.terrain_types]
            & (self.terrain_levels >= min_bar_x_scale_level)
        )
        return torch.sum(aligned_contact.float(), dim=1) * is_eligible_ladder.float()

    def _reward_excess_feet_air_time(self):
        phase_contact = self._get_phase_foot_contacts()
        effective_phase_air_time = self.phase_feet_air_time + self.dt * (~phase_contact).float()
        exceeds_limit = (effective_phase_air_time > self.cfg.rewards.half_phase_upper) & (~phase_contact)
        rew_excess_air_time = torch.sum(exceeds_limit.float(), dim=1)
        return rew_excess_air_time

    def _reward_contact_symmetry(self):
        required_feet = ("FL", "FR", "RL", "RR")
        if not hasattr(self, "foot_name_to_obs_index"):
            return torch.zeros(self.num_envs, device=self.device)
        matched_indices = {}
        for leg_prefix in required_feet:
            matched_name = next((name for name in self.foot_name_to_obs_index if name.startswith(leg_prefix)), None)
            if matched_name is None:
                return torch.zeros(self.num_envs, device=self.device)
            matched_indices[leg_prefix] = self.foot_name_to_obs_index[matched_name]
        contact = self._get_foot_contacts()
        phase_contact = self._get_phase_foot_contacts()
        fl_rr_match = (
            (contact[:, matched_indices["FL"]] == contact[:, matched_indices["RR"]]) &
            (phase_contact[:, matched_indices["FL"]] == phase_contact[:, matched_indices["RR"]])
        ).float()
        fr_rl_match = (
            (contact[:, matched_indices["FR"]] == contact[:, matched_indices["RL"]]) &
            (phase_contact[:, matched_indices["FR"]] == phase_contact[:, matched_indices["RL"]])
        ).float()
        pair_match = torch.stack((fl_rr_match, fr_rl_match), dim=1)
        return (2.0 * torch.mean(pair_match, dim=1) - 1.0) * self._get_flat_terrain_mask()

    def _reward_symmetry_torque(self):
        if not hasattr(self, "filtered_symmetry_torque_abs_diff"):
            return torch.zeros(self.num_envs, device=self.device)
        if self.filtered_symmetry_torque_abs_diff.shape[1] == 0:
            return torch.zeros(self.num_envs, device=self.device)
        return torch.sum(self.filtered_symmetry_torque_abs_diff, dim=1)
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still_when_reached_goal(self):
        goal_reached, _ = self._get_goal_delta()
        flat_mask = self._get_flat_terrain_mask()
        # Penalize the realized pose so gravity/contact compliance cannot evade it.
        return flat_mask * goal_reached * torch.sum(
            torch.abs(self.dof_pos - self.default_dof_pos), dim=1
        )

    def _reward_action_smoothness(self):
        return torch.sum(torch.square(self.last_last_actions - 2. * self.last_actions + self.actions), dim=1)

    def _reward_flat_orientation_when_flat(self):
        goal_reached, _ = self._get_goal_delta()
        flat_mask = self._get_flat_terrain_mask()
        orientation_mask = torch.maximum(flat_mask, goal_reached)
        return orientation_mask * (1. + goal_reached) * torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_stand_still_contact_when_reached_goal(self):
        goal_reached, _ = self._get_goal_delta()
        missing_contacts = (~self._get_foot_contacts()).float()
        return goal_reached * torch.sum(missing_contacts, dim=1)

    def _reward_base_collision(self):
        return (torch.norm(self.contact_forces[:, 0, :], dim=-1) > 0.1).float()

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return (torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)) \
            * self._get_flat_terrain_mask()
