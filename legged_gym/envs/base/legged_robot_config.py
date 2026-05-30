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
    class env:
        num_envs = 4096
        num_observations = 302
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 12
        debug_terrain_sampling = False
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 10 # episode length in seconds

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
            "bar_spacing": (0.15, 0.25),
            "bar_count": (10, 14),
            "ladder_angle": (10, 80),
            "bar_x_scale": (3.0, 1.0),
            "bar_x_scale_min_level": 5,
            "platform_length": 2,
            "platform_width": 2,
            "platform_gap": 0.1,
            "ladder_x_offset": -1.0,
            "rough_probability": 0.3,
            "low_difficulty_probability": 0.3,
            "rough_height_range": (-0.04, 0),
            "rough_grid_size": 0.1,
        }
        max_init_ladder_level = 5 # starting ladder curriculum state
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
        control_type = 'P' # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 1
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        use_actuator_network = False

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
        friction_range = [0.5, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 1.]
        randomize_pd_gains = True
        stiffness_multiplier_range = [1.0, 1.0]
        damping_multiplier_range = [1.0, 1.0]
        push_robots = False
        push_interval_s = 15
        max_push_vel_xy = 1.
        apply_base_force_torque = True
        base_force_interval_s = 15.
        base_force_duration_s = 0.2
        max_base_force = [0., 0., 0.]
        max_base_torque = [0., 0., 0.]

    class rewards:
        class scales:
            # Primary task rewards
            alive = 2
            position_tracking = 3
            heading_tracking = 0.5

            # Slow movement rewards
            lin_vel_z = -2
            ang_vel_xy = -0.02

            # Shaping rewards
            flat_orientation_when_flat = -1.0
            foot_slippage = -0.1
            feet_air_time = 0.1
            excess_feet_air_time = -3
            contact_symmetry = 0.1
            collision = -0.1
            stand_still_when_reached_goal = -0.5
            stand_still_contact_when_reached_goal = -0.5

            # Slow movement rewards
            action_rate = -0.01
            action_smoothness = -0.01
            torques = -1e-4
            dof_vel = -1e-4
            dof_acc = -5e-8
            dof_pos_limits = -5

            # Unused rewards
            termination = -0.0
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            orientation = 0
            base_collision = 0
            feet_stumble = 0.0
            base_height = 0 

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.95 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.3
        max_contact_force = 200. # forces above this value are penalized
        foot_slip_threshold = 1.0
        feet_air_time_threshold = 0.4
        feet_air_time_upper_limit = 0.8
        goal_radius = 0.15
        goal_speed_limit = 0.5
        progress_reward_max_difficulty_multiplier = 1.5
        contact_force_threshold = 1.0
        flat_height_std_threshold = 0.2

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
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

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
            max_gpu_contact_pairs = 2**24 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 10
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class LeggedRobotCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 48 # per iteration
        max_iterations = 5000 # number of policy updates

        # logging
        save_interval = 50 # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt
