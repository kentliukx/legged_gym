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


def show_depth_camera(frame_queue, min_depth, max_depth, depth_height, depth_width):
    import matplotlib.pyplot as plt

    plt.ion()
    figure, axis = plt.subplots(num="Robot depth camera")
    image = axis.imshow(np.zeros((depth_height, depth_width)), cmap="turbo", vmin=min_depth, vmax=max_depth)
    axis.set_title("Forward depth [m]")
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


def show_height_comparison(frame_queue, height_shape):
    import matplotlib.pyplot as plt

    plt.ion()
    figure = plt.figure(num="Scanned vs reconstructed height", figsize=(7, 8))
    grid = figure.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 0.045), wspace=0.28)
    axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])]
    colorbar_axis = figure.add_subplot(grid[0, 2])
    images = [
        axes[0].imshow(np.zeros(height_shape), cmap="viridis", origin="lower", vmin=-1.0, vmax=1.0),
        axes[1].imshow(np.zeros(height_shape), cmap="viridis", origin="lower", vmin=-1.0, vmax=1.0),
    ]
    axes[0].set_title("Scanned height")
    axes[1].set_title("Reconstructed height")
    for axis in axes:
        axis.set_xlabel("y sample")
        axis.set_ylabel("x sample")
    colorbar = figure.colorbar(images[0], cax=colorbar_axis)
    colorbar.set_label("Terrain height relative to reference (high = positive)")
    figure.subplots_adjust(left=0.10, right=0.91, bottom=0.08, top=0.90)
    figure.show()

    while plt.fignum_exists(figure.number):
        try:
            scanned, reconstructed = frame_queue.get(timeout=0.05)
        except queue.Empty:
            plt.pause(0.001)
            continue
        if scanned is None:
            break
        for image, data in zip(images, (scanned, reconstructed)):
            image.set_data(data)
        mse = np.mean((scanned - reconstructed) ** 2)
        figure.suptitle(f"Height scan comparison | MSE={mse:.6f}")
        figure.canvas.draw_idle()
        figure.canvas.flush_events()
    plt.close(figure)


def get_student_diagnostics(actor_critic, observations, robot_index):
    if (not hasattr(actor_critic, "estimator")
            or not hasattr(actor_critic, "reconstructed_height_obs")
            or not hasattr(actor_critic, "reconstructed_ladder_obs")):
        return None
    if (actor_critic.reconstructed_height_obs is None
            or actor_critic.reconstructed_ladder_obs is None):
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
            split_obs["friction"],
            split_obs["added_mass"],
            split_obs["applied_force"],
            split_obs["applied_torque"],
        ),
        dim=-1,
    )
    return {
        "estimated": estimated_state[robot_index].detach(),
        "estimated_target": estimated_target[robot_index].detach(),
        "reconstructed_height": actor_critic.reconstructed_height_obs[robot_index].detach(),
        "reconstructed_height_target": split_obs["height_scan"][robot_index].detach(),
        "reconstructed_ladder": actor_critic.reconstructed_ladder_obs[robot_index].detach(),
        "ladder_target": torch.cat(
            (
                split_obs["effector_ladder_plane_distance"][robot_index],
                split_obs["effector_nearest_bar_distance"][robot_index],
                split_obs["ladder_info"][robot_index],
            ),
            dim=-1,
        ).detach(),
    }


def print_student_diagnostics(diagnostics, env, robot_index, step):
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
    print(f"  friction       estimated={values(estimated[7:8])} target={values(target[7:8])}")
    print(f"  added_mass     estimated={values(estimated[8:9])} target={values(target[8:9])}")
    print(f"  applied_force  estimated={values(estimated[9:12])} target={values(target[9:12])}")
    print(f"  applied_torque estimated={values(estimated[12:15])} target={values(target[12:15])}")
    print(
        f"  ladder_obs     reconstructed={values(diagnostics['reconstructed_ladder'])} "
        f"target={values(diagnostics['ladder_target'])}"
    )
    print(f"  distance link order={env.distance_effector_names}")
    print(
        "  ladder_plane_distance[m] "
        f"signed={values(env.effector_ladder_plane_distance[robot_index])}"
    )
    print(
        "  nearest_bar_distance[m] "
        f"absolute={values(env.effector_nearest_bar_distance[robot_index])}"
    )
    height_mse = torch.mean(
        (diagnostics["reconstructed_height"] - diagnostics["reconstructed_height_target"]) ** 2
    )
    ladder_mse = torch.mean(
        (diagnostics["reconstructed_ladder"] - diagnostics["ladder_target"]) ** 2
    )
    print(f"  height_scan MSE={height_mse.item():.6f}  ladder MSE={ladder_mse.item():.6f}")


def print_base_height_diagnostic(env, robot_index, step):
    curr_climbing_ladder = bool(env.curr_climbing_ladder[robot_index].item())
    print(
        f"[base height] step={step} height={env.base_height_buf[robot_index].item():.4f}m "
        f"target={env.cfg.rewards.base_height_target:.4f}m "
        f"curr_climbing_ladder={curr_climbing_ladder}"
    )

def get_height_visualization_frame(env, robot_index):
    local_points = env.height_points[robot_index]
    world_points = quat_apply_yaw(
        env.base_quat[robot_index].repeat(local_points.shape[0]),
        local_points,
    )
    world_points = world_points + env.root_states[robot_index, :3]
    return {
        "base_height": env.root_states[robot_index, 2].detach().clone(),
        "env_index": robot_index,
        "world_points": world_points.detach().clone(),
    }


def draw_reconstructed_heightmap(env, reconstructed_height, visualization_frame, sphere_geometry):
    height_scale = float(env.obs_scales.height_measurements)
    reconstructed_height = torch.clamp(reconstructed_height, -1.0, 1.0) / max(height_scale, 1e-6)
    world_heights = visualization_frame["base_height"] - 0.5 - reconstructed_height
    world_points = visualization_frame["world_points"].clone()
    world_points[:, 2] = world_heights

    for point in world_points.cpu().numpy():
        pose = gymapi.Transform(gymapi.Vec3(point[0], point[1], point[2]), r=None)
        gymutil.draw_lines(
            sphere_geometry,
            env.gym,
            env.viewer,
            env.envs[visualization_frame["env_index"]],
            pose,
        )


def draw_effector_ladder_distance_lines(env, robot_index):
    """Draw each effector's distance projection used by the environment buffers."""
    if env.viewer is None or not hasattr(env, "effector_on_ladder_plane"):
        return

    effector_positions = env.rigid_body_states[robot_index, env.distance_effector_indices, :3]
    on_ladder_plane = env.effector_on_ladder_plane[robot_index]
    distances = env.effector_ladder_plane_distance[robot_index]
    ladder_angle = torch.deg2rad(
        env.terrain_ladder_angles[env.terrain_levels[robot_index], env.terrain_types[robot_index]]
    )
    inward_normal = torch.stack((torch.sin(ladder_angle), torch.zeros_like(ladder_angle), -torch.cos(ladder_angle)))
    ladder_tangent = torch.stack((torch.cos(ladder_angle), torch.zeros_like(ladder_angle), torch.sin(ladder_angle)))
    ladder_origin = env.terrain_ladder_origins[
        env.terrain_levels[robot_index], env.terrain_types[robot_index]
    ]
    bar_along_distances = env.terrain_ladder_bar_along_distances[
        env.terrain_levels[robot_index], env.terrain_types[robot_index]
    ]
    bar_count = env.terrain_ladder_bar_counts[
        env.terrain_levels[robot_index], env.terrain_types[robot_index]
    ]
    valid_bars = torch.arange(bar_along_distances.shape[0], device=env.device) < bar_count

    for foot_index in range(effector_positions.shape[0]):
        position = effector_positions[foot_index]
        if on_ladder_plane[foot_index].item():
            projection = position - distances[foot_index] * inward_normal
            color = gymapi.Vec3(1.0, 0.35, 0.0)  # Orange: perpendicular to ladder plane.
            along_ladder = torch.dot(position - ladder_origin, ladder_tangent)
            nearest_bar_index = torch.argmin(
                torch.abs(along_ladder - bar_along_distances).masked_fill(~valid_bars, float("inf"))
            )
            nearest_bar = ladder_origin + ladder_tangent * bar_along_distances[nearest_bar_index]
            # A rung runs along y, so its closest point shares the projection's y.
            nearest_bar[1] = projection[1]
            gymutil.draw_line(
                gymapi.Vec3(*projection.cpu().tolist()),
                gymapi.Vec3(*nearest_bar.cpu().tolist()),
                gymapi.Vec3(0.3, 1.0, 0.1),  # Green: projection-to-nearest-rung distance.
                env.gym,
                env.viewer,
                env.envs[robot_index],
            )
        else:
            projection = position.clone()
            projection[2] -= distances[foot_index]
            color = gymapi.Vec3(0.0, 0.9, 1.0)  # Cyan: world-z height above ground/platform.
        start = gymapi.Vec3(*position.cpu().tolist())
        end = gymapi.Vec3(*projection.cpu().tolist())
        gymutil.draw_line(start, end, color, env.gym, env.viewer, env.envs[robot_index])


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    teacher_mode = args.mode == "teacher"
    default_teacher_checkpoint = train_cfg.runner.teacher_checkpoint
    # override some parameters for testing

    env_cfg.terrain.num_cols = 1
    env_cfg.env.num_envs = 1
    env_cfg.env.debug_terrain_sampling = True
    env_cfg.terrain.border_size = 5
    env_cfg.terrain.terrain_kwargs["rough_probability"] = 0.2
    env_cfg.terrain.terrain_kwargs["low_difficulty_probability"] = 0.2
    env_cfg.sim.physx.max_gpu_contact_pairs = 2**20
    env_cfg.noise.min_noise_level = env_cfg.noise.noise_level
    env_cfg.domain_rand.min_push_vel_xy = env_cfg.domain_rand.max_push_vel_xy
    env_cfg.domain_rand.min_push_foot_vel_xy = env_cfg.domain_rand.max_push_foot_vel_xy

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.draw_height_measurements = VISUALIZE_HEIGHTMAP and not teacher_mode
    obs = env.get_observations()
    # load policy
    use_default_teacher_checkpoint = (
        teacher_mode and not args.resume and args.load_run is None and args.checkpoint is None
    )
    train_cfg.runner.resume = not use_default_teacher_checkpoint
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    if teacher_mode:
        if use_default_teacher_checkpoint:
            teacher_checkpoint = os.path.join(LEGGED_GYM_ROOT_DIR, default_teacher_checkpoint)
            print(f"Loading Teacher model from: {teacher_checkpoint}")
            ppo_runner.load(teacher_checkpoint)
        if args.sample:
            ppo_runner.alg.actor_critic.eval()
            if env.device != "cpu":
                ppo_runner.alg.actor_critic.to(env.device)
            policy = ppo_runner.alg.actor_critic.act
            print("Play mode: Teacher stochastic actions")
        else:
            policy = ppo_runner.get_inference_policy(device=env.device)
            print("Play mode: Teacher deterministic actions")
    elif args.sample:
        student_policy = None
        ppo_runner.alg.actor_critic.eval()
        if env.device != "cpu":
            ppo_runner.alg.actor_critic.to(env.device)
        policy = ppo_runner.alg.actor_critic.act
        print("Play mode: Student stochastic actions")
    else:
        student_policy = None
        policy = ppo_runner.get_inference_policy(device=env.device)
        print("Play mode: Student deterministic actions")
    
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
    height_process = None
    height_frame_queue = None
    mp_context = None
    if VISUALIZE_DEPTH_CAMERA and not args.headless and env.cfg.sensor.enable_depth_camera:
        mp_context = mp.get_context("spawn")
        depth_frame_queue = mp_context.Queue(maxsize=1)
        depth_process = mp_context.Process(
            target=show_depth_camera,
            args=(
                depth_frame_queue,
                float(env.cfg.sensor.depth_min),
                float(env.cfg.sensor.depth_max),
                int(env.cfg.sensor.depth_height),
                int(env.cfg.sensor.depth_width),
            ),
            daemon=True,
        )
        depth_process.start()
    if VISUALIZE_HEIGHT_COMPARISON and not teacher_mode and not args.headless:
        if mp_context is None:
            mp_context = mp.get_context("spawn")
        height_frame_queue = mp_context.Queue(maxsize=1)
        height_process = mp_context.Process(
            target=show_height_comparison,
            args=(
                height_frame_queue,
                (
                    len(env.cfg.terrain.measured_points_x),
                    len(env.cfg.terrain.measured_points_y),
                ),
            ),
            daemon=True,
        )
        height_process.start()

    try:
        for i in range(100*int(env.max_episode_length)):
            with torch.inference_mode():
                actions = policy(obs.detach())
                if (not teacher_mode and (PRINT_STUDENT_DIAGNOSTICS
                        or VISUALIZE_HEIGHTMAP
                        or VISUALIZE_HEIGHT_COMPARISON)):
                    diagnostics = get_student_diagnostics(
                        ppo_runner.alg.actor_critic,
                        obs.detach(),
                        robot_index,
                    )
                    if diagnostics is not None:
                        diagnostics["height_visualization_frame"] = get_height_visualization_frame(
                            env,
                            robot_index,
                        )
            if (not teacher_mode and PRINT_STUDENT_DIAGNOSTICS
                    and diagnostics is not None and i % diagnostic_print_interval == 0):
                print_student_diagnostics(diagnostics, env, robot_index, i)
            if PRINT_BASE_HEIGHT and i % diagnostic_print_interval == 0:
                print_base_height_diagnostic(env, robot_index, i)
            obs, _, rews, dones, infos = env.step(actions.detach())
            with torch.inference_mode():
                ppo_runner.alg.actor_critic.reset(dones)
            depth_camera_updated = env.depth_image_delivered_this_step
            if (not teacher_mode and VISUALIZE_HEIGHTMAP
                    and diagnostics is not None
                    and env.viewer is not None):
                draw_reconstructed_heightmap(
                    env,
                    diagnostics["reconstructed_height"],
                    diagnostics["height_visualization_frame"],
                    reconstructed_height_geometry,
                )
                draw_reconstructed_heightmap(
                    env,
                    diagnostics["reconstructed_height_target"],
                    diagnostics["height_visualization_frame"],
                    reconstructed_height_target_geometry,
                )
            if VISUALIZE_EFFECTOR_LADDER_DISTANCES:
                draw_effector_ladder_distance_lines(env, robot_index)
            if (depth_frame_queue is not None
                    and depth_process.is_alive()
                    and depth_camera_updated):
                depth_frame = env.depth_image_noisy_buf[robot_index].reshape(
                    env.cfg.sensor.depth_height,
                    env.cfg.sensor.depth_width,
                ).cpu().numpy()
                try:
                    depth_frame_queue.put_nowait(depth_frame)
                except queue.Full:
                    pass
            if (height_frame_queue is not None
                    and height_process.is_alive()
                    and diagnostics is not None
                    and depth_camera_updated):
                height_shape = (
                    len(env.cfg.terrain.measured_points_x),
                    len(env.cfg.terrain.measured_points_y),
                )
                # height_scan stores reference_height - terrain_height. Negate
                # it so the visualization follows the intuitive high=positive convention.
                scanned_height = -np.rot90(diagnostics["reconstructed_height_target"].reshape(
                    height_shape
                ).T.cpu().numpy())
                reconstructed_height = -np.rot90(diagnostics["reconstructed_height"].reshape(
                    height_shape
                ).T.cpu().numpy())
                try:
                    height_frame_queue.put_nowait((scanned_height, reconstructed_height))
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

            if PLOT_STATES and i < stop_state_log:
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
            elif PLOT_STATES and i==stop_state_log:
                logger.plot_states()
            if  0 < i < stop_rew_log:
                if infos.get("episode"):
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
        if height_frame_queue is not None and height_process.is_alive():
            try:
                height_frame_queue.put_nowait((None, None))
            except queue.Full:
                pass
            height_process.join(timeout=1.0)

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    VISUALIZE_DEPTH_CAMERA = True
    PLOT_STATES = True
    PRINT_STUDENT_DIAGNOSTICS = True
    PRINT_BASE_HEIGHT = True
    DIAGNOSTIC_PRINT_INTERVAL_S = 1.0
    VISUALIZE_HEIGHTMAP = False
    VISUALIZE_HEIGHT_COMPARISON = False
    VISUALIZE_EFFECTOR_LADDER_DISTANCES = True
    args = get_args()
    play(args)
