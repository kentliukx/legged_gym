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

from legged_gym import LEGGED_GYM_ROOT_DIR
import multiprocessing as mp
import os
import queue
import time

import isaacgym
from isaacgym import gymapi, gymutil
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from legged_gym.utils.math import quat_apply_yaw

import numpy as np
import torch


def show_depth_camera(frame_queue):
    import matplotlib.pyplot as plt

    plt.ion()
    figure, axis = plt.subplots(num="Robot depth camera")
    image = axis.imshow(np.zeros((36, 64)), cmap="turbo", vmin=0.0, vmax=1.0)
    axis.set_title("Normalized forward depth")
    axis.axis("off")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.show()

    while plt.fignum_exists(figure.number):
        try:
            frame = frame_queue.get(timeout=0.05)
        except queue.Empty:
            plt.pause(0.001)
            continue
        if frame is None:
            break
        image.set_data(frame)
        figure.canvas.draw_idle()
        figure.canvas.flush_events()
    plt.close(figure)


def get_student_diagnostics(actor_critic, observations, robot_index):
    if not hasattr(actor_critic, "estimator") or not hasattr(actor_critic, "reconstructed_terrain_obs"):
        return None
    if actor_critic.reconstructed_terrain_obs is None:
        return None

    split_obs = actor_critic._split_observations(observations)
    estimator_raw = actor_critic.estimator(actor_critic._encode_history(split_obs["proprio_history"]))
    estimated_state = torch.cat(
        (
            estimator_raw[..., :3],
            torch.sigmoid(estimator_raw[..., 3:7]),
            estimator_raw[..., 7:14],
        ),
        dim=-1,
    )
    estimated_target = torch.cat(
        (
            split_obs["base_lin_vel"],
            split_obs["foot_contacts"],
            split_obs["applied_force"],
            split_obs["applied_torque"],
            split_obs["friction"],
        ),
        dim=-1,
    )
    reconstructed_terrain = actor_critic.reconstructed_terrain_obs
    height_dim = actor_critic.height_dim
    return {
        "estimated": estimated_state[robot_index].detach(),
        "estimated_target": estimated_target[robot_index].detach(),
        "reconstructed_height": reconstructed_terrain[robot_index, :height_dim].detach(),
        "reconstructed_height_target": split_obs["height_scan_simplified"][robot_index].detach(),
        "reconstructed_ladder": reconstructed_terrain[robot_index, height_dim:].detach(),
        "ladder_target": split_obs["ladder_info"][robot_index].detach(),
    }


def print_student_diagnostics(diagnostics, step):
    def values(tensor):
        return np.array2string(
            tensor.cpu().numpy(),
            precision=3,
            suppress_small=True,
            floatmode="fixed",
        )

    estimated = diagnostics["estimated"]
    target = diagnostics["estimated_target"]
    print(f"\n[student diagnostics] step={step}")
    print(f"  base_lin_vel   estimated={values(estimated[0:3])} target={values(target[0:3])}")
    print(f"  foot_contacts  estimated={values(estimated[3:7])} target={values(target[3:7])}")
    print(f"  applied_force  estimated={values(estimated[7:10])} target={values(target[7:10])}")
    print(f"  applied_torque estimated={values(estimated[10:13])} target={values(target[10:13])}")
    print(f"  friction       estimated={values(estimated[13:14])} target={values(target[13:14])}")
    print(
        f"  ladder_obs     reconstructed={values(diagnostics['reconstructed_ladder'])} "
        f"target={values(diagnostics['ladder_target'])}"
    )


def draw_reconstructed_heightmap(env, robot_index, reconstructed_height, sphere_geometry):
    height_scale = float(env.obs_scales.height_measurements)
    reconstructed_height = torch.clamp(reconstructed_height, -1.0, 1.0) / max(height_scale, 1e-6)
    world_heights = env.root_states[robot_index, 2] - 0.5 - reconstructed_height
    local_points = env.height_points[robot_index]
    world_points = quat_apply_yaw(
        env.base_quat[robot_index].repeat(local_points.shape[0]),
        local_points,
    )
    world_points = world_points + env.root_states[robot_index, :3]
    world_points[:, 2] = world_heights

    for point in world_points.cpu().numpy():
        pose = gymapi.Transform(gymapi.Vec3(point[0], point[1], point[2]), r=None)
        gymutil.draw_lines(sphere_geometry, env.gym, env.viewer, env.envs[robot_index], pose)


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing

    env_cfg.terrain.num_cols = 1
    env_cfg.env.num_envs = 1
    env_cfg.env.debug_terrain_sampling = False
    env_cfg.terrain.border_size = 5
    env_cfg.terrain.curriculum = True
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.sim.physx.max_gpu_contact_pairs = 2**20

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    if args.sample:
        ppo_runner.alg.actor_critic.eval()
        if env.device != "cpu":
            ppo_runner.alg.actor_critic.to(env.device)
        policy = ppo_runner.alg.actor_critic.act
    else:
        policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    frame_wall_time = time.perf_counter()
    diagnostics = None
    diagnostic_print_interval = max(1, int(round(DIAGNOSTIC_PRINT_INTERVAL_S / env.dt)))
    reconstructed_height_geometry = gymutil.WireframeSphereGeometry(
        0.025,
        4,
        4,
        None,
        color=(0.65, 0.0, 1.0),
    )
    reconstructed_height_target_geometry = gymutil.WireframeSphereGeometry(
        0.018,
        4,
        4,
        None,
        color=(0.55, 1.0, 0.55),
    )
    depth_process = None
    depth_frame_queue = None
    if VISUALIZE_DEPTH_CAMERA and not args.headless and env.cfg.sensor.enable_depth_camera:
        mp_context = mp.get_context("spawn")
        depth_frame_queue = mp_context.Queue(maxsize=1)
        depth_process = mp_context.Process(
            target=show_depth_camera,
            args=(depth_frame_queue,),
            daemon=True,
        )
        depth_process.start()

    try:
        for i in range(100*int(env.max_episode_length)):
            actions = policy(obs.detach())
            if PRINT_STUDENT_DIAGNOSTICS or VISUALIZE_RECONSTRUCTED_HEIGHTMAP:
                with torch.inference_mode():
                    diagnostics = get_student_diagnostics(
                        ppo_runner.alg.actor_critic,
                        obs.detach(),
                        robot_index,
                    )
            if PRINT_STUDENT_DIAGNOSTICS and diagnostics is not None and i % diagnostic_print_interval == 0:
                print_student_diagnostics(diagnostics, i)
            obs, _, rews, dones, infos = env.step(actions.detach())
            ppo_runner.alg.actor_critic.reset(dones)
            depth_camera_updated = (
                env.common_step_counter <= 1
                or env.common_step_counter % env.depth_camera_update_interval_steps == 0
            )
            if (VISUALIZE_RECONSTRUCTED_HEIGHTMAP
                    and diagnostics is not None
                    and env.viewer is not None):
                draw_reconstructed_heightmap(
                    env,
                    robot_index,
                    diagnostics["reconstructed_height"],
                    reconstructed_height_geometry,
                )
                draw_reconstructed_heightmap(
                    env,
                    robot_index,
                    diagnostics["reconstructed_height_target"],
                    reconstructed_height_target_geometry,
                )
            if (depth_frame_queue is not None
                    and depth_process.is_alive()
                    and depth_camera_updated):
                depth_frame = env.depth_image_buf[robot_index].reshape(
                    env.cfg.sensor.depth_height,
                    env.cfg.sensor.depth_width,
                ).cpu().numpy()
                try:
                    depth_frame_queue.put_nowait(depth_frame)
                except queue.Full:
                    pass
            if not args.headless:
                frame_wall_time += env.dt
                sleep_time = frame_wall_time - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    frame_wall_time = time.perf_counter()
            if RECORD_FRAMES:
                if i % 2:
                    filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                    env.gym.write_viewer_image_to_file(env.viewer, filename)
                    img_idx += 1 
            if MOVE_CAMERA:
                camera_position += camera_vel * env.dt
                env.set_camera(camera_position, camera_position + camera_direction)

            if i < stop_state_log:
                command_x = env.commands[robot_index, 0].item() if env.commands.shape[1] > 0 else 0.0
                command_y = env.commands[robot_index, 1].item() if env.commands.shape[1] > 1 else 0.0
                command_yaw = env.commands[robot_index, 2].item() if env.commands.shape[1] > 2 else 0.0
                logger.log_states(
                    {
                        'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                        'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                        'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                        'dof_torque': env.torques[robot_index, joint_index].item(),
                        'command_x': command_x,
                        'command_y': command_y,
                        'command_yaw': command_yaw,
                        'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                        'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                        'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                        'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                        'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                    }
                )
            elif i==stop_state_log:
                logger.plot_states()
            if  0 < i < stop_rew_log:
                if infos["episode"]:
                    num_episodes = torch.sum(env.reset_buf).item()
                    if num_episodes>0:
                        logger.log_rewards(infos["episode"], num_episodes)
            elif i==stop_rew_log:
                logger.print_rewards()
    finally:
        if depth_frame_queue is not None and depth_process.is_alive():
            try:
                depth_frame_queue.put_nowait(None)
            except queue.Full:
                pass
            depth_process.join(timeout=1.0)

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    VISUALIZE_DEPTH_CAMERA = True
    PRINT_STUDENT_DIAGNOSTICS = True
    DIAGNOSTIC_PRINT_INTERVAL_S = 1.0
    VISUALIZE_RECONSTRUCTED_HEIGHTMAP = True
    args = get_args()
    play(args)
