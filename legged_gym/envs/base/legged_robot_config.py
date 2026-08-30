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

from .base_config import BaseConfig

class LeggedRobotCfg(BaseConfig):
    @classmethod
    def apply_easy(cls, env_cfg):
        terrain_kwargs = env_cfg.terrain.terrain_kwargs
        terrain_kwargs["bar_x_scale"] = (3.0, 1.0)
        terrain_kwargs["bar_y_scale"] = (1.3, 0.65)
        env_cfg.terrain.max_init_ladder_level = 3
        env_cfg.domain_rand.push_robots = False
        env_cfg.domain_rand.push_robot_foot = False
        env_cfg.rewards.ladder_contact_precision_center_threshold = 0.03
        env_cfg.rewards.ladder_contact_precision_effector_threshold = 1
        env_cfg.env.ignore_nonprecision_for_progress_reward = True
        env_cfg.noise.add_noise = False

    class env:
        num_envs = 2048
        # The final four entries are clean contact precision reserved for the
        # imitation TeacherPolicy; Student network slices end at 2710.
        num_observations = 2714
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 12
        proprioception_history_len = 10
        # Set by task_registry from --mode: Teacher receives clean precision;
        # Student receives its noisy sensor counterpart.
        use_noisy_contact_precision = False
        ignore_nonprecision_for_progress_reward = False
        debug_terrain_sampling = False
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 15 # episode length in seconds

    class sensor:
        enable_depth_camera = True
        depth_width = 54
        depth_height = 36
        depth_horizontal_fov = 73
        depth_update_interval_s = 0.1
        # Frames are delivered to the policy after this fixed camera latency.
        depth_latency_s = 0.04
        depth_min = 0.1
        depth_max = 2.0
        depth_position = [0.35, 0.0, 0.08]
        # Relative to the base frame: yaw is around +z, positive pitch looks downward.
        depth_pitch_deg = 15.0
        depth_yaw_deg = 0.0
        # Per-environment installation error, sampled once when environments are created.
        depth_position_noise_range = [
            [-0.01, 0.01],
            [-0.01, 0.01],
            [-0.01, 0.01],
        ]
        # Roll, pitch, yaw noise ranges in degrees.
        depth_rotation_noise_deg_range = [
            [-2, 2],
            [-2, 2],
            [-2, 2],
        ]

    class terrain:
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        obs_horizontal_scale = 0.01 # [m], height observation map resolution
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = True
        curriculum_move_up_ratio = 0.9
        curriculum_move_down_ratio = 0.5
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # ladder terrain only:
        measure_heights = True
        # 21 x 11 = 231 height points over a 1.6m x 0.8m rectangle.
        measured_points_x = [-0.8, -0.72, -0.64, -0.56, -0.48, -0.4, -0.32,
                             -0.24, -0.16, -0.08, 0., 0.08, 0.16, 0.24,
                             0.32, 0.4, 0.48, 0.56, 0.64, 0.72, 0.8]
        measured_points_y = [-0.4, -0.32, -0.24, -0.16, -0.08, 0.,
                             0.08, 0.16, 0.24, 0.32, 0.4]
        terrain_kwargs = {
            "bar_mesh_file": "{LEGGED_GYM_ROOT_DIR}/resources/terrains/round_bar.STL",
            "side_bar_mesh_file": "{LEGGED_GYM_ROOT_DIR}/resources/terrains/side_bar.STL",
            "bar_spacing": (0.15, 0.25),
            "bar_position_noise_std": 0.01,
            "bar_count": (8, 10),
            "ladder_angle": (10, 80),
            "bar_x_scale": (1.0, 1.0),
            "bar_x_scale_min_level": 5,
            "bar_y_scale": (0.65, 0.65),
            "bar_y_scale_random_multiplier": (0.8, 1.3),
            "bar_y_scale_min_level": 7,
            "bar_y_scale_curve_power": 0.5,
            "platform_length": 2,
            "platform_width": 1.5,
            "platform_gap": 0.1,
            "ladder_x_offset": -1.0,
            "rough_probability": 0.15,
            "low_difficulty_probability": 0.25,
            "rough_height_range": (-0.04, 0),
            "rough_grid_size": 0.1,
            "edge_obstacle_count": (40, 50),
            "edge_obstacle_abs_y_min": 0.7,
            "edge_obstacle_margin": 1,
            "edge_obstacle_height": (0.5, 1.5),
            "edge_obstacle_radius": (0.1, 0.3),
        }
        max_init_ladder_level = 9 # starting ladder curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 50 # number of terrain cols (types)
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 2
        resampling_time = 10. # time before command are changed[s]
        heading_command = True
        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0] # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0] # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state:
        pos = [0.0, 0.0, 1.] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # target angles when action = 0.0
            "joint_a": 0., 
            "joint_b": 0.}

    class control:
        control_type = 'P' # P: position, V: velocity, T: torques, UNITREE: Unitree torque-speed limited position PD
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 1
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        use_actuator_network = False
        min_delay = 0  # [physics steps], actuator command delay lower bound
        max_delay = 0  # [physics steps], actuator command delay upper bound
        # Unitree-style actuator torque-speed envelope. Defaults match Go2HV.
        motor_velocity_x1 = 13.5  # [rad/s], max speed at full torque
        motor_velocity_x2 = 30.0  # [rad/s], no-load speed
        motor_torque_y1 = 20.2    # [N*m], torque limit when velocity and torque have same sign
        motor_torque_y2 = 23.4    # [N*m], torque limit when velocity and torque have opposite signs
        dof_friction = 0.0
        motor_static_friction = 0.0
        motor_dynamic_friction = 0.0
        motor_friction_activation_velocity = 0.01

    class asset:
        file = ""
        name = "legged_robot"  # actor name
        base_name = "base" # rigid body name used for base-directed disturbances
        foot_name = "None" # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up

        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        randomize_friction = True
        friction_range = [-0.5, 1.5]

        randomize_base_mass = True
        added_mass_range = [-1.0, 2.0]

        randomize_pd_gains = True
        stiffness_multiplier_range = [0.8, 1.2]
        damping_multiplier_range = [0.8, 1.2]

        apply_base_force_torque = True
        base_force_interval_s = 4.
        base_force_duration_s = 2
        max_base_force = [10., 10., 10.]
        max_base_torque = [5., 5., 5.]

        # increase over time
        difficulty_scale_level_range = [3, 5]

        push_robots = True
        push_interval_s = 2.
        min_push_vel_xy = 0.
        max_push_vel_xy = 0.5

        push_robot_foot = True
        push_foot_interval_s = 1.
        min_push_foot_vel_xy = 0.
        max_push_foot_vel_xy = 5.

    class rewards:
        class scales:
            # Fixed rewards
            # Primary task rewards
            alive = 2
            position_tracking = 5
            # on ladder
            ladder_side_clearance = -0.2
            ladder_contact_precision = 0.1
            collision = -1
            feet_contact_forces = -0.01
            excess_step_length = -5
            # gait related
            feet_air_time = 1
            feet_ground_time = 1
            excess_feet_air_time = -5
            contact_symmetry = 0.05
            foot_slippage = -0.2
            foot_clearance = -0.2
            # reached goal
            stand_still_when_reached_goal = -0.5
            stand_still_contact_when_reached_goal = -0.2

            # Increasing rewards
            increasing_reward_coeff = [0.4, 1.0]
            increasing_reward_upper_reward_limit = 30
            increasing_reward_lower_reward_limit = 0
            increasing_reward_lpf_k = 0.05
            increasing_reward_names = [
                "lin_vel_z",
                "ang_vel_xy",
                "action_rate",
                "action_smoothness",
                "torques",
                "dof_vel",
                "dof_acc",
                "flat_orientation_when_flat",
                "base_height",
                "dof_pos_limits",
                "effector_velocity",
            ]
            # slow movement
            lin_vel_z = -2
            ang_vel_xy = -0.05
            action_rate = -0.02
            action_smoothness = -0.02
            torques = -2e-4
            dof_vel = -5e-4
            dof_acc = -2e-7
            effector_velocity = -0.02
            # natural movement
            flat_orientation_when_flat = -5.0
            base_height = -20.0
            dof_pos_limits = -10

            # Unused rewards
            heading_tracking = 0
            termination = -0.0
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            orientation = 0
            base_collision = 0
            feet_stumble = 0.0
            symmetry_torque = -0

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.95 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.35
        ladder_base_height_target_lowest = 0.35
        ladder_base_height_target_highest = 0.25
        max_contact_force = 150. # forces above this value are penalized
        foot_slip_threshold = 1.0
        half_phase_lower = 0.3
        half_phase_upper = 0.6
        phase_step_length_threshold = 0.25
        phase_step_length_ladder_margin = 0.05
        symmetry_torque_lpf_tau = 0.2
        goal_radius = 0.15
        goal_speed_limit = 0.4
        nonflat_goal_speed_limit = 0.4
        progress_reward_max_difficulty_multiplier = 2
        contact_force_threshold = 1.0
        phase_contact_force_threshold = 20.0
        foot_clearance_target = 0.15
        ladder_side_clearance_threshold = 0.1
        ladder_contact_precision_center_threshold = 0.02
        ladder_contact_precision_effector_threshold = 0.04

    class normalization:
        class obs_scales:
            lin_vel = 1.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 1.0
            applied_wrench = 0.2
        clip_observations = 100.
        clip_actions = 10.

    class noise:
        add_noise = True
        min_noise_level = 0.0
        noise_level = 1.0 # scales other values
        difficulty_scale_level_range = [3, 5]
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            ang_vel = 0.2
            gravity = 0.05
            depth_image = 0.01  # [m], per-pixel uniform depth noise
            depth_dropout_prob = 0.005
            depth_outlier_prob = 0.002
            # Per-environment probability sampled for every depth-camera frame.
            depth_edge_dropout_prob = [0.1, 0.7]
            depth_edge_threshold = 0.1  # [m], neighboring depth jump treated as an invalid-pixel edge
            # Dilation is inward (nearer/bar side). Nearer edges use more negative values.
            depth_edge_dilation_near = -4
            depth_edge_dilation_far = -1
            depth_edge_dilation_distance_range = [0.1, 2.0]
            # Larger values keep far edges at the far dilation and steepen near-range growth.
            depth_edge_dilation_distance_exponent = 8.0
            # Maximum per-foot flip probability for Student contact precision.
            # It is scaled by the common noise curriculum level.
            contact_precision_flip_prob = 0.1

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 10.0
            max_gpu_contact_pairs = 2**22 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 2
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class LeggedRobotCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        # Split every logical minibatch into this many microbatches and accumulate gradients.
        mini_batch_divide = 4
        learning_rate = 1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.
        estimator_loss_coef = 1.0
        ladder_reconstruction_loss_coef = 1.0
        height_reconstruction_loss_coef = 1.0
    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 48 # per iteration
        max_iterations = 20000 # number of policy updates

        # logging
        save_interval = 50 # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt
