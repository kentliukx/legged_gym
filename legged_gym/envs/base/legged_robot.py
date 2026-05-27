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
from legged_gym.utils.math import quat_apply, quat_apply_yaw, quat_rotate_inverse, wrap_to_pi, torch_rand_float, torch_rand_sqrt_float
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
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
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

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
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
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
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
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def compute_observations(self):
        """ Computes observations
        """
        ladder_obs = self._get_ladder_observations()
        has_ladder_obs = self._get_has_ladder_observations()
        foot_contact = self._get_foot_contacts().float()
        height_scan = torch.clip(
                self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1) * self.obs_scales.height_measurements

        obs_parts = [
            # prioproception
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            # goal
            self.commands * self.commands_scale,
            self.reached_goal,

            # privileged information
            foot_contact,
            has_ladder_obs,
            ladder_obs,

            # height scan
            height_scan
        ]

        self.obs_buf = torch.cat(obs_parts, dim=-1)

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

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
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

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
            props[0].mass += np.random.uniform(rng[0], rng[1])
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
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

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
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

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


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:9 + self.num_actions] = (
            noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        )
        dof_vel_start = 9 + self.num_actions
        noise_vec[dof_vel_start:dof_vel_start + self.num_actions] = (
            noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        )
        action_start = dof_vel_start + self.num_actions
        contact_obs_start = action_start + self.num_actions
        noise_vec[contact_obs_start:contact_obs_start + len(self.feet_indices)] = 0. # foot contact flags
        command_start = contact_obs_start + len(self.feet_indices)
        noise_vec[command_start:command_start + self.cfg.commands.num_commands] = 0. # commands
        reached_goal_start = command_start + self.cfg.commands.num_commands
        noise_vec[reached_goal_start:reached_goal_start + 1] = 0. # reached goal flag
        has_ladder_start = reached_goal_start + 1
        noise_vec[has_ladder_start:has_ladder_start + 1] = 0. # has ladder flag
        ladder_obs_start = has_ladder_start + 1
        noise_vec[ladder_obs_start:ladder_obs_start + 4] = 0. # ladder geometry
        num_height_obs = len(self.cfg.terrain.measured_points_x) * len(self.cfg.terrain.measured_points_y) if self.cfg.terrain.measure_heights else 0
        if self.cfg.terrain.measure_heights:
            height_obs_start = ladder_obs_start + 4
            noise_vec[height_obs_start:height_obs_start + num_height_obs] = (
                noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements
            )
        return noise_vec

    def _get_has_ladder_observations(self):
        has_ladder = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.float)
        if hasattr(self, "terrain_ladder_mask"):
            has_ladder[:, 0] = self.terrain_ladder_mask[self.terrain_levels, self.terrain_types].float()
        return has_ladder

    def _get_ladder_observations(self):
        ladder_obs = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.float)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            return ladder_obs
        if not hasattr(self, "terrain_ladder_bar_spacing") or not hasattr(self, "terrain_ladder_angles"):
            return ladder_obs

        bar_spacing = self.terrain_ladder_bar_spacing[self.terrain_levels, self.terrain_types]
        ladder_angle_deg = self.terrain_ladder_angles[self.terrain_levels, self.terrain_types]
        is_ladder = self.terrain_ladder_mask[self.terrain_levels, self.terrain_types]
        ladder_angle_rad = torch.deg2rad(ladder_angle_deg)

        forward = quat_apply(self.base_quat, self.forward_vec)
        base_heading = torch.atan2(forward[:, 1], forward[:, 0])
        ladder_up_yaw_rel = wrap_to_pi(-base_heading)
        ladder_up_yaw_rel = torch.where(is_ladder, ladder_up_yaw_rel, torch.zeros_like(ladder_up_yaw_rel))

        ladder_obs[:, 0] = bar_spacing
        ladder_obs[:, 1] = bar_spacing * torch.sin(ladder_angle_rad)
        ladder_obs[:, 2] = bar_spacing * torch.cos(ladder_angle_rad)
        ladder_obs[:, 3] = ladder_up_yaw_rel
        return ladder_obs

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
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_scale = torch.ones(self.cfg.commands.num_commands, device=self.device, requires_grad=False)
        self.goal_dist = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.min_goal_dist = torch.full((self.num_envs,), float("inf"), dtype=torch.float, device=self.device, requires_grad=False)
        self.reached_goal = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.difficulty = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self._update_difficulty()
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.terrain_ladder_origins = torch.from_numpy(self.terrain.ladder_origins).to(self.device).to(torch.float)
            self.terrain_platform_centers = torch.from_numpy(self.terrain.platform_centers).to(self.device).to(torch.float)
            self.terrain_ladder_mask = torch.from_numpy(self.terrain.ladder_mask).to(self.device)
            self.terrain_ladder_bar_spacing = torch.from_numpy(self.terrain.ladder_bar_spacing).to(self.device).to(torch.float)
            self.terrain_ladder_angles = torch.from_numpy(self.terrain.ladder_angles).to(self.device).to(torch.float)
            self.goal_targets = self.terrain_platform_centers[self.terrain_levels, self.terrain_types].clone()
            self.goal_targets[:, 2] += self.cfg.rewards.base_height_target
            self._randomize_rough_targets(torch.arange(self.num_envs, device=self.device))
            self._reset_goal_progress(torch.arange(self.num_envs, device=self.device))
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
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
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
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
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
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

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

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
            self.max_terrain_level = self.cfg.terrain.num_rows
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

        target_angle = (torch.rand(len(rough_env_ids), device=self.device) - 0.5) * np.deg2rad(60.0)
        direction = torch.stack((torch.cos(target_angle), torch.sin(target_angle)), dim=1)
        max_distance_x = (0.5 * self.terrain.env_length) / direction[:, 0].clamp(min=1e-6)
        max_distance_y = torch.full_like(max_distance_x, 0.5 * self.terrain.env_width)
        side_distance = max_distance_y / torch.abs(direction[:, 1]).clamp(min=1e-6)
        max_distance = torch.minimum(max_distance_x, side_distance)
        min_distance = torch.minimum(torch.full_like(max_distance, 0.5), 0.5 * max_distance)
        target_distance = min_distance + torch.rand(len(rough_env_ids), device=self.device) * (max_distance - min_distance)
        target_offsets = direction * target_distance.unsqueeze(1)
        self.goal_targets[rough_env_ids, :2] = self.env_origins[rough_env_ids, :2] + target_offsets
        self.goal_targets[rough_env_ids, 2] = self.cfg.rewards.base_height_target

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

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
                target_pose = gymapi.Transform(gymapi.Vec3(target[0], target[1], target[2] + 0.1), r=None)
                target_geom = target_reached_geom if self.reached_goal[i, 0] > 0.5 else target_unreached_geom
                gymutil.draw_lines(target_geom, self.gym, self.viewer, self.envs[i], target_pose)

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

    def _get_heights(self, env_ids=None):
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
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

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
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
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
        local_height_std = torch.std(self.measured_heights, dim=1, unbiased=False)
        return (local_height_std < self.cfg.rewards.flat_height_std_threshold).float()

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

    def _reward_position_tracking(self):
        goal_reached, goal_dist = self._get_goal_delta()
        goal_dir = self.commands[:, :2] / goal_dist.unsqueeze(1).clamp(min=1e-6)
        velocity_xy = self.base_lin_vel[:, :2]
        velocity_norm = torch.norm(velocity_xy, dim=1).clamp(min=1e-6)
        speed_over = torch.clamp(velocity_norm - self.cfg.rewards.goal_speed_limit, min=0.)
        min_dist_decrease_speed = torch.clamp(self.min_goal_dist - goal_dist, min=0.) / self.dt
        self.min_goal_dist[:] = torch.minimum(self.min_goal_dist, goal_dist)
        heading_error = torch.atan2(goal_dir[:, 1], goal_dir[:, 0])
        heading_gate = torch.clamp(torch.cos(3. * heading_error), min=0.)
        progress_reward_multiplier = 1. + self.difficulty * (
            float(self.cfg.rewards.progress_reward_max_difficulty_multiplier) - 1.
        )
        progress_reward = min_dist_decrease_speed * heading_gate * progress_reward_multiplier
        return (1. - goal_reached) * (progress_reward - 5. * speed_over) + 1.5 * goal_reached

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

    def _reward_feet_air_time(self):
        # Reward moderate air time and penalize overly long swing time.
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        goal_reached, _ = self._get_goal_delta()
        contact = self._get_foot_contacts()
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        air_time = self.feet_air_time
        lower = self.cfg.rewards.feet_air_time_threshold
        upper = self.cfg.rewards.feet_air_time_upper_limit
        air_time_reward = torch.where(
            air_time <= upper,
            air_time - lower,
            upper - air_time
        )
        rew_airTime = torch.sum(air_time_reward * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= (1. - goal_reached)
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still_when_reached_goal(self):
        goal_reached, _ = self._get_goal_delta()
        flat_mask = self._get_flat_terrain_mask()
        desired_dof_pos = self.actions * self.cfg.control.action_scale + self.default_dof_pos
        return flat_mask * goal_reached * torch.sum(torch.abs(desired_dof_pos - self.default_dof_pos), dim=1)

    def _reward_action_smoothness(self):
        return torch.sum(torch.square(self.last_last_actions - 2. * self.last_actions + self.actions), dim=1)

    def _reward_flat_orientation_when_flat(self):
        goal_reached, _ = self._get_goal_delta()
        flat_mask = self._get_flat_terrain_mask()
        return flat_mask * (1. + 1. * goal_reached) * torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_stand_still_contact_when_reached_goal(self):
        goal_reached, _ = self._get_goal_delta()
        missing_contacts = (~self._get_foot_contacts()).float()
        return goal_reached * torch.sum(missing_contacts, dim=1)

    def _reward_base_collision(self):
        return (torch.norm(self.contact_forces[:, 0, :], dim=-1) > 0.1).float()

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
