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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class GO2LadderCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        pass

    class terrain( LeggedRobotCfg.terrain ):
        pass

    class init_state( LeggedRobotCfg.init_state ):
        pos = [-2.0, 0.0, 0.35] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            "FL_thigh_joint": 0.7, "FL_hip_joint": 0.1, "FL_calf_joint": -1.7,
            "FR_thigh_joint": 0.7, "FR_hip_joint": -0.1, "FR_calf_joint": -1.7,
            "RL_thigh_joint": 0.7, "RL_hip_joint": 0.1, "RL_calf_joint": -1.7,
            "RR_thigh_joint": 0.7, "RR_hip_joint": -0.1, "RR_calf_joint": -1.7
        }

    class control( LeggedRobotCfg.control ):
        control_type = 'UNITREE'
        # PD Drive parameters:
        stiffness = {'joint': 25}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        min_delay = 0
        max_delay = 3
        # Unitree RL Lab Go2HV actuator torque-speed curve.
        motor_velocity_x1 = 13.5
        motor_velocity_x2 = 30.0
        motor_torque_y1 = 20.2
        motor_torque_y2 = 23.4
        dof_friction = 0.01
        motor_static_friction = 0.0
        motor_dynamic_friction = 0.0
        motor_friction_activation_velocity = 0.01

    class asset( LeggedRobotCfg.asset ):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf"
        name = "go2"
        foot_name = "effector"
        rubber_name = "foot_front"
        penalize_contacts_on = ["thigh", "hip"]
        terminate_after_contacts_on = ["base", "Head_upper", "Head_lower"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter

    class domain_rand( LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [-0.5, 1.5]

        randomize_base_mass = True
        added_mass_range = [-1.0, 2.0]

        randomize_pd_gains = True
        stiffness_multiplier_range = [0.8, 1.2]
        damping_multiplier_range = [0.8, 1.2]

        push_robots = False

        apply_base_force_torque = True
        base_force_interval_s = 4.
        base_force_duration_s = 2
        max_base_force = [10., 10., 10.]
        max_base_torque = [5., 5., 5.]

        apply_rubber_foot_damping = False
        rubber_foot_damping_threshold = 0.03
        rubber_foot_xy_damping = 200.0
        rubber_foot_z_up_damping_xy_ratio = 100.0
        rubber_foot_max_damping_force = 50.0

  
    class rewards( LeggedRobotCfg.rewards ):
        pass

    class noise( LeggedRobotCfg.noise ):
        pass

class GO2LadderCfgPPO( LeggedRobotCfgPPO ):
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCriticRecurrent'
        experiment_name = 'go2_ladder'
        teacher_checkpoint = 'logs/go2_ladder/Teacher/model_30000.pt'
