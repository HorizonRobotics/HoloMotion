# Project HoloMotion
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
# This file was originally copied from the [ASAP] repository:
# https://github.com/LeCAR-Lab/ASAP
# Modifications have been made to fit the needs of this project.

import torch


class BaseSimulator:
    """Base class for robotic simulation environments.

    Provides a framework for simulation setup, environment creation, and
    control over robotic assets and simulation properties.
    """

    def __init__(self, config, device):
        """Initializes the base simulator with configuration settings.

        Args:
            config (dict): Configuration dictionary for the simulation.
            device (str): Device type for simulation ('cpu' or 'cuda').

        """
        self.config = config
        self.sim_device = device
        self.headless = False

        self._rigid_body_pos: torch.Tensor
        self._rigid_body_rot: torch.Tensor
        self._rigid_body_vel: torch.Tensor
        self._rigid_body_ang_vel: torch.Tensor

    def set_headless(self, headless):
        """Sets the headless mode for the simulator.

        Args:
            headless (bool): If True, runs the simulation without
                graphical display.

        """
        self.headless = headless

    def setup(self):
        """Initializes the simulator parameters and environment.

        This method should be implemented by subclasses to set specific
        simulator configurations.
        """
        raise NotImplementedError(
            "The 'setup' method must be implemented in subclasses."
        )

    def setup_terrain(self, mesh_type):
        """Configures the terrain based on specified mesh type.

        Args:
            mesh_type (str): Type of terrain mesh ('plane', 'heightfield',
                'trimesh').

        """
        raise NotImplementedError(
            "The 'setup_terrain' method must be implemented in subclasses."
        )

    def load_assets(self, robot_config):
        """Loads the robot assets into the simulation environment.

        save self.num_dofs, self.num_bodies, self.dof_names, self.body_names
        Args:
            robot_config (dict): holomotion Configuration for the robot asset.
        """
        raise NotImplementedError(
            "The 'load_assets' method must be implemented in subclasses."
        )

    def create_envs(self, num_envs, env_origins, base_init_state, env_config):
        """Creates and initializes environments with specified configurations.

        Args:
            num_envs (int): Number of environments to create.
            env_origins (list): List of origin positions for each environment.
            base_init_state (array): Initial state of the base.
            env_config (dict): Configuration for each environment.

        """
        raise NotImplementedError(
            "The 'create_envs' method must be implemented in subclasses."
        )

    def get_dof_limits_properties(self):
        """Retrieves the DOF (degrees of freedom) limits and properties.

        Returns:
            Tuple of tensors representing position limits, velocity limits,
            and torque limits for each DOF.

        """
        raise NotImplementedError(
            "The 'get_dof_limits_properties' method must be implemented "
            "in subclasses."
        )

    def find_rigid_body_indice(self, body_name):
        """Finds the index of a specified rigid body.

        Args:
            body_name (str): Name of the rigid body to locate.

        Returns:
            int: Index of the rigid body.

        """
        raise NotImplementedError(
            "The 'find_rigid_body_indice' method must be implemented "
            "in subclasses."
        )

    def prepare_sim(self):
        """Prepares the simulation environment and refreshes tensors."""
        raise NotImplementedError(
            "The 'prepare_sim' method must be implemented in subclasses."
        )

    def refresh_sim_tensors(self):
        """Refreshes state tensors in the simulation for up-to-date values."""
        raise NotImplementedError(
            "The 'refresh_sim_tensors' method must be implemented "
            "in subclasses."
        )

    def apply_torques_at_dof(self, torques):
        """Applies the specified torques to the robot's DOF.

        Args:
            torques (tensor): Tensor containing torques to apply.

        """
        raise NotImplementedError(
            "The 'apply_torques_at_dof' method must be implemented "
            "in subclasses."
        )

    def set_actor_root_state_tensor(self, set_env_ids, root_states):
        """Sets the root state tensor for specified actors within environments.

        Args:
            set_env_ids (tensor): Tensor of environment IDs where states
                will be set.
            root_states (tensor): New root states to apply.

        """
        raise NotImplementedError(
            "The 'set_actor_root_state_tensor' method must be implemented "
            "in subclasses."
        )

    def set_dof_state_tensor(self, set_env_ids, dof_states):
        """Sets the DOF state tensor for specified actors within environments.

        Args:
            set_env_ids (tensor): Tensor of environment IDs where states
                will be set.
            dof_states (tensor): New DOF states to apply.

        """
        raise NotImplementedError(
            "The 'set_dof_state_tensor' method must be implemented "
            "in subclasses."
        )

    def simulate_at_each_physics_step(self):
        """Advances the simulation by a single physics step."""
        raise NotImplementedError(
            "The 'simulate_at_each_physics_step' method must be implemented "
            "in subclasses."
        )

    def setup_viewer(self):
        """Sets up a viewer for visualizing the simulation."""
        raise NotImplementedError(
            "The 'setup_viewer' method must be implemented in subclasses."
        )

    def render(self, sync_frame_time=True):
        """Renders the simulation frame-by-frame.

        Args:
            sync_frame_time (bool): Whether to synchronize the frame time.

        """
        raise NotImplementedError(
            "The 'render' method must be implemented in subclasses."
        )
