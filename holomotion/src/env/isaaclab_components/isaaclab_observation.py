import isaaclab.envs.mdp as isaaclab_mdp
import isaaclab.sim as sim_utils
from dataclasses import fields as dataclass_fields
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import (
    ActionTermCfg,
    CommandTerm,
    CommandTermCfg,
    EventTermCfg as EventTerm,
    ObservationGroupCfg,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
import torch
from isaaclab.markers import (
    VisualizationMarkers,
    VisualizationMarkersCfg,
)
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass


import isaaclab.utils.math as isaaclab_math
import isaaclab.utils.noise as isaaclab_noise
from omegaconf import DictConfig, ListConfig, OmegaConf

from holomotion.src.env.isaaclab_components.isaaclab_utils import (
    resolve_holo_config,
)


class ObservationFunctions:
    """Atomic observation functions.

    The most foundamental observation functions are defined here, aiming to
    utize the convenient functions from isaaclab apis. For complex observation
    composition patterns, we'll use the custom observation serizliazer.
    """

    @staticmethod
    def _get_body_indices(
        robot: Articulation, keybody_names: list[str] | None
    ) -> list[int]:
        """Convert body names to indices.

        Args:
            robot: Robot articulation asset
            keybody_names: List of body names. If None, returns all body indices.

        Returns:
            List of body indices corresponding to the given names
        """
        if keybody_names is None:
            return list(range(robot.num_bodies))

        body_indices = []
        for name in keybody_names:
            if name not in robot.body_names:
                raise ValueError(
                    f"Body '{name}' not found in robot.body_names: {robot.body_names}"
                )
            body_indices.append(robot.body_names.index(name))

        return body_indices

    # ------- Robot Head / mid360 States -------
    @staticmethod
    def _get_obs_head_pos_quat_vel(
        env: ManagerBasedRLEnv, robot_asset_name: str = "robot"
    ):
        """Head (mid360) features in torso frame with first-frame anchor.

        Returns [B,13]: pos(3), quat_wxyz->xyzw(4), lin_vel(3), ang_vel(3), all in torso frame and anchored.
        """
        robot_ptr = env.scene[robot_asset_name]
        body_names = robot_ptr.body_names
        if body_names is None:
            raise RuntimeError("robot.body_names is empty")
        try:
            torso_idx = body_names.index("torso_link")
        except ValueError:
            raise ValueError(
                f"'torso_link' not found in body_names: {body_names}"
            )

        B = env.num_envs
        device = env.device
        # Mid360 extrinsics relative to torso (rotation about Y by pitch)
        rel_pos_t = torch.tensor(
            [0.0002835, 0.00003, 0.41618], dtype=torch.float, device=device
        )
        pitch = torch.tensor(
            0.04014257279586953, dtype=torch.float, device=device
        )
        half = pitch * 0.5
        # WXYZ
        rel_quat_wxyz = torch.stack(
            [
                torch.cos(half),
                torch.zeros_like(half),
                torch.sin(half),
                torch.zeros_like(half),
            ],
            dim=-1,
        )
        rel_quat_wxyz = rel_quat_wxyz.expand(B, -1)

        # World pose/vel from torso + extrinsics (WXYZ math)
        torso_pos_w = robot_ptr.data.body_pos_w[:, torso_idx, :]
        torso_quat_wxyz = robot_ptr.data.body_quat_w[:, torso_idx, :]
        torso_lin_w = robot_ptr.data.body_lin_vel_w[:, torso_idx, :]
        torso_ang_w = robot_ptr.data.body_ang_vel_w[:, torso_idx, :]

        rel_pos = rel_pos_t.expand(B, -1)
        r_world = isaaclab_math.quat_apply(torso_quat_wxyz, rel_pos)
        pos_w = torso_pos_w + r_world
        quat_wxyz = isaaclab_math.quat_mul(torso_quat_wxyz, rel_quat_wxyz)
        lin_w = torso_lin_w + torch.cross(torso_ang_w, r_world, dim=-1)
        ang_w = torso_ang_w

        # Convert to torso frame (WXYZ math)
        rel_p = pos_w - torso_pos_w
        torso_inv_wxyz = isaaclab_math.quat_inv(torso_quat_wxyz)
        pos_torso = isaaclab_math.quat_apply(torso_inv_wxyz, rel_p)
        lin_torso = isaaclab_math.quat_apply(
            torso_inv_wxyz, lin_w - torch.cross(ang_w, rel_p, dim=-1)
        )
        ang_torso = isaaclab_math.quat_apply(torso_inv_wxyz, ang_w)
        quat_torso_wxyz = isaaclab_math.quat_mul(torso_inv_wxyz, quat_wxyz)
        # export quaternion as XYZW to match common obs format
        quat_torso_xyzw = quat_torso_wxyz[..., [1, 2, 3, 0]]

        # First-frame anchor normalization (in torso frame)
        if not hasattr(env, "head_anchor_set"):
            env.head_anchor_set = torch.zeros(
                B, dtype=torch.bool, device=device
            )
            env.head_anchor_pos = torch.zeros(B, 3, device=device)
            env.head_anchor_quat_wxyz = torch.zeros(B, 4, device=device)
            env.head_anchor_quat_wxyz[:, 0] = 1.0  # identity W
        unset = ~env.head_anchor_set
        if unset.any():
            env.head_anchor_pos[unset] = pos_torso[unset]
            env.head_anchor_quat_wxyz[unset] = quat_torso_wxyz[unset]
            env.head_anchor_set[unset] = True
        q0_inv = isaaclab_math.quat_inv(env.head_anchor_quat_wxyz)
        pos_rel = isaaclab_math.quat_apply(
            q0_inv, pos_torso - env.head_anchor_pos
        )
        lin_rel = isaaclab_math.quat_apply(q0_inv, lin_torso)
        ang_rel = isaaclab_math.quat_apply(q0_inv, ang_torso)
        quat_rel_wxyz = isaaclab_math.quat_mul(q0_inv, quat_torso_wxyz)
        quat_rel_xyzw = quat_rel_wxyz[..., [1, 2, 3, 0]]
        return torch.cat([pos_rel, quat_rel_xyzw, lin_rel, ang_rel], dim=-1)

    @staticmethod
    def _get_obs_rel_headlink_lin_vel(
        env: ManagerBasedRLEnv, robot_asset_name: str = "robot"
    ) -> torch.Tensor:  # [num_envs, 3]
        """Headlink relative linear velocity, expressed in the headlink's frame.

        Definitions:
        - Headlink: a virtual rigid sensor frame fixed to `torso_link` using the
          extrinsics defined below (translation `rel_pos_t` and rotation `rel_quat_wxyz`).
        - Relative linear velocity: v_head - v_torso_origin, both measured in the world
          frame before re-expression. For a rigid mount, this equals ω_torso × r_world.
        - Expression frame: the instantaneous headlink frame (i.e., result is in headlink axes).

        Returns:
            Tensor of shape [num_envs, 3]: headlink relative linear velocity in headlink frame.
        """
        robot_ptr = env.scene[robot_asset_name]
        body_names = robot_ptr.body_names
        if body_names is None:
            raise RuntimeError("robot.body_names is empty")
        torso_idx = body_names.index("torso_link")

        num_envs = env.num_envs
        device = env.device
        # Headlink extrinsics relative to torso: translation + rotation about Y (pitch)
        rel_pos_t = torch.tensor(
            [0.0002835, 0.00003, 0.41618], dtype=torch.float, device=device
        )  # [3]
        pitch = torch.tensor(
            0.04014257279586953, dtype=torch.float, device=device
        )
        half = pitch * 0.5
        # Quaternion (WXYZ) for rotation about Y by 'pitch'
        rel_quat_wxyz = torch.stack(
            [
                torch.cos(half),
                torch.zeros_like(half),
                torch.sin(half),
                torch.zeros_like(half),
            ],
            dim=-1,
        ).expand(num_envs, -1)  # [num_envs, 4]

        # Torso world state
        torso_quat_wxyz = robot_ptr.data.body_quat_w[
            :, torso_idx, :
        ]  # [num_envs, 4]
        torso_lin_w = robot_ptr.data.body_lin_vel_w[
            :, torso_idx, :
        ]  # [num_envs, 3]
        torso_ang_w = robot_ptr.data.body_ang_vel_w[
            :, torso_idx, :
        ]  # [num_envs, 3]

        # Headlink world pose from torso + extrinsics
        rel_pos = rel_pos_t.expand(num_envs, -1)  # [num_envs, 3]
        r_world = isaaclab_math.quat_apply(
            torso_quat_wxyz, rel_pos
        )  # [num_envs, 3]
        head_quat_wxyz = isaaclab_math.quat_mul(
            torso_quat_wxyz, rel_quat_wxyz
        )  # [num_envs, 4]

        # World-frame velocities
        head_lin_w = torso_lin_w + torch.cross(
            torso_ang_w, r_world, dim=-1
        )  # [num_envs, 3]
        # Relative linear velocity in world frame
        rel_lin_w = (
            head_lin_w - torso_lin_w
        )  # [num_envs, 3] == ω_torso × r_world

        # Re-express in headlink frame
        head_inv_wxyz = isaaclab_math.quat_inv(head_quat_wxyz)  # [num_envs, 4]
        rel_lin_head = isaaclab_math.quat_apply(
            head_inv_wxyz, rel_lin_w
        )  # [num_envs, 3]
        return rel_lin_head

    @staticmethod
    def _get_obs_rel_headlink_ang_vel(
        env: ManagerBasedRLEnv, robot_asset_name: str = "robot"
    ) -> torch.Tensor:  # [num_envs, 3]
        """Headlink relative angular velocity, expressed in the headlink's frame.

        Definitions:
        - Headlink: a virtual rigid sensor frame fixed to `torso_link` using the
          extrinsics defined below.
        - Relative angular velocity: ω_head - ω_torso, measured in the world frame,
          then re-expressed in the headlink frame.
        - For a rigid mount (no neck articulation), ω_head == ω_torso, so the result
          is identically zero. If an articulated head exists, replace ω_head with the
          head link's world angular velocity before the subtraction.

        Returns:
            Tensor of shape [num_envs, 3]: headlink relative angular velocity in headlink frame.
        """
        robot_ptr = env.scene[robot_asset_name]
        body_names = robot_ptr.body_names
        if body_names is None:
            raise RuntimeError("robot.body_names is empty")
        torso_idx = body_names.index("torso_link")

        num_envs = env.num_envs
        device = env.device
        # Headlink extrinsics (rotation about Y by pitch)
        pitch = torch.tensor(
            0.04014257279586953, dtype=torch.float, device=device
        )
        half = pitch * 0.5
        rel_quat_wxyz = torch.stack(
            [
                torch.cos(half),
                torch.zeros_like(half),
                torch.sin(half),
                torch.zeros_like(half),
            ],
            dim=-1,
        ).expand(num_envs, -1)  # [num_envs, 4]

        torso_quat_wxyz = robot_ptr.data.body_quat_w[
            :, torso_idx, :
        ]  # [num_envs, 4]
        torso_ang_w = robot_ptr.data.body_ang_vel_w[
            :, torso_idx, :
        ]  # [num_envs, 3]

        # For the rigid mount, ω_head_w == ω_torso_w
        head_ang_w = torso_ang_w  # [num_envs, 3]
        rel_ang_w = (
            head_ang_w - torso_ang_w
        )  # [num_envs, 3] -> zeros for rigid mount

        # Re-express in headlink frame
        head_quat_wxyz = isaaclab_math.quat_mul(
            torso_quat_wxyz, rel_quat_wxyz
        )  # [num_envs, 4]
        head_inv_wxyz = isaaclab_math.quat_inv(head_quat_wxyz)  # [num_envs, 4]
        rel_ang_head = isaaclab_math.quat_apply(
            head_inv_wxyz, rel_ang_w
        )  # [num_envs, 3]
        return rel_ang_head

    # ------- Robot Root States -------
    @staticmethod
    def _get_obs_global_robot_root_pos(env: ManagerBasedRLEnv):
        """Asset root position in the environment frame."""
        return isaaclab_mdp.root_pos_w(env)

    @staticmethod
    def _get_obs_global_robot_root_rot_wxyz(env: ManagerBasedRLEnv):
        """Asset root orientation (w, x, y, z) in the environment frame."""
        return isaaclab_mdp.root_quat_w(env)

    @staticmethod
    def _get_obs_global_robot_root_rot_xyzw(env: ManagerBasedRLEnv):
        """Asset root orientation (x, y, z, w) in the environment frame."""
        return ObservationFunctions._get_obs_global_robot_root_rot_wxyz(env)[
            ..., [1, 2, 3, 0]
        ]

    @staticmethod
    def _get_obs_global_robot_root_rot_mat(env: ManagerBasedRLEnv):
        """Asset root orientation as a 3x3 matrix, flattened to the first two rows (6D)."""
        return isaaclab_math.matrix_from_quat(
            ObservationFunctions._get_obs_global_robot_root_rot_wxyz(env)
        )[..., :2]  # [num_envs, 6]

    @staticmethod
    def _get_obs_global_robot_root_lin_vel(env: ManagerBasedRLEnv):
        """Asset root linear velocity in the environment frame."""
        return isaaclab_mdp.root_lin_vel_w(env)  # [num_envs, 3]

    @staticmethod
    def _get_obs_global_robot_root_ang_vel(env: ManagerBasedRLEnv):
        """Asset root angular velocity in the environment frame."""
        return isaaclab_mdp.root_ang_vel_w(env)  # [num_envs, 3]

    @staticmethod
    def _get_obs_rel_robot_root_lin_vel(env: ManagerBasedRLEnv):
        """Relative root linear velocity in the root frame."""
        return isaaclab_mdp.base_lin_vel(env)  # [num_envs, 3]

    @staticmethod
    def _get_obs_rel_robot_root_ang_vel(env: ManagerBasedRLEnv):
        """Relative root angular velocity in the root frame."""
        return isaaclab_mdp.base_ang_vel(env)  # [num_envs, 3]

    @staticmethod
    def _get_obs_rel_anchor_lin_vel(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        anchor_bodylink_name: str = "torso_link",
    ):
        """Relative anchor linear velocity in the anchor frame."""
        torso_global_rot_quat_wxyz = (
            ObservationFunctions._get_obs_global_robot_bodylink_rot_wxyz(
                env, robot_asset_name, [anchor_bodylink_name]
            )
        )  # [num_envs, 1, 4]
        torso_global_lin_vel = (
            ObservationFunctions._get_obs_global_robot_bodylink_lin_vel(
                env, robot_asset_name, [anchor_bodylink_name]
            )
        )  # [num_envs, 1, 3]
        torso_rel_lin_vel = isaaclab_math.quat_apply(
            isaaclab_math.quat_inv(torso_global_rot_quat_wxyz),
            torso_global_lin_vel,
        )  # [num_envs, 1, 3]
        return torso_rel_lin_vel.squeeze(1)  # [num_envs, 3]

    @staticmethod
    def _get_obs_projected_gravity(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
    ) -> torch.Tensor:  # [num_envs, 3]
        """Gravity vector projected into the robot's root frame.

        Projects the world-frame gravity vector into the robot's base frame
        using the inverse root orientation quaternion.
        """
        robot_ptr = env.scene[robot_asset_name]
        g_w: torch.Tensor = robot_ptr.data.GRAVITY_VEC_W  # [num_envs, 3]
        root_quat_wxyz: torch.Tensor = (
            ObservationFunctions._get_obs_global_robot_root_rot_wxyz(env)
        )  # [num_envs, 4]

        # Project gravity into root frame using inverse quaternion
        projected_gravity: torch.Tensor = isaaclab_math.quat_apply_inverse(
            root_quat_wxyz, g_w
        )  # [num_envs, 3]

        # projected_gravity: torch.Tensor = isaaclab_math.quat_rotate_inverse(
        #     root_quat_wxyz, g_w
        # )  # [num_envs, 3]

        return projected_gravity

    @staticmethod
    def _get_obs_global_robot_root_yaw(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
    ):
        """Robot's yaw heading in the environment frame (in radians)."""
        robot_ptr = env.scene[robot_asset_name]
        return robot_ptr.data.heading_w  # [num_envs, ]

    # @torch.compile
    @staticmethod
    def _get_obs_robot_root_heading_aligned_quat(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
    ):
        """A quaternion representing only the robot's yaw heading."""
        global_yaw = ObservationFunctions._get_obs_global_robot_root_yaw(
            env,
            robot_asset_name,
        )  # [num_envs, ]
        zero_roll = torch.zeros_like(global_yaw, device=env.device)
        zero_pitch = torch.zeros_like(global_yaw, device=env.device)
        heading_aligned_quat = isaaclab_math.quat_from_angle_axis(
            roll=zero_roll,
            pitch=zero_pitch,
            yaw=global_yaw,
        )  # [num_envs, 4]
        return heading_aligned_quat  # [num_envs, 4]

    # @torch.compile
    @staticmethod
    def _get_obs_rel_robot_root_roll_pitch(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
    ):
        """Robot's roll and pitch relative to its heading-aligned frame."""
        heading_aligned_quat = (
            ObservationFunctions._get_obs_robot_root_heading_aligned_quat(
                env,
                robot_asset_name,
            )
        )  # [num_envs, 4]
        robot_quat_in_heading_aligned_frame = isaaclab_math.quat_mul(
            isaaclab_math.quat_inv(heading_aligned_quat),
            ObservationFunctions._get_obs_global_robot_root_rot_wxyz(env),
        )  # [num_envs, 4]
        rel_roll, rel_pitch, _ = isaaclab_math.get_euler_xyz(
            robot_quat_in_heading_aligned_frame
        )  # [num_envs, 3]
        return torch.stack([rel_roll, rel_pitch], dim=-1)  # [num_envs, 2]

    # ------- Robot Bodylink States -------
    @staticmethod
    def _get_obs_global_robot_bodylink_pos(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_names: list[str] | None = None,
    ):
        """Positions of specified bodylinks in the environment frame."""
        robot_ptr = env.scene[robot_asset_name]
        keybody_idxs = ObservationFunctions._get_body_indices(
            robot_ptr, keybody_names
        )
        keybody_global_pos = robot_ptr.data.body_pos_w[:, keybody_idxs]
        return keybody_global_pos  # [num_envs, num_keybodies, 3]

    @staticmethod
    def _get_obs_global_robot_bodylink_rot_wxyz(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_names: list[str] | None = None,
    ):
        """Orientations (w, x, y, z) of specified bodylinks in the environment frame."""
        robot_ptr = env.scene[robot_asset_name]
        keybody_idxs = ObservationFunctions._get_body_indices(
            robot_ptr, keybody_names
        )
        keybody_global_rot = robot_ptr.data.body_quat_w[:, keybody_idxs]
        return keybody_global_rot  # [num_envs, num_keybodies, 4]

    @staticmethod
    def _get_obs_global_robot_bodylink_rot_xyzw(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_names: list[str] | None = None,
    ):
        """Orientations (x, y, z, w) of specified bodylinks in the environment frame."""
        return ObservationFunctions._get_obs_global_robot_bodylink_rot_wxyz(
            env,
            robot_asset_name,
            keybody_names,
        )[..., [1, 2, 3, 0]]  # [num_envs, num_keybodies, 4]

    @staticmethod
    def _get_obs_global_robot_bodylink_rot_mat(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_names: list[str] | None = None,
    ):
        """Orientations of specified bodylinks as a 3x3 matrix, flattened to the first two rows (6D)."""
        keybody_global_rot_wxyz = (
            ObservationFunctions._get_obs_global_robot_bodylink_rot_wxyz(
                env,
                robot_asset_name,
                keybody_names,
            )
        )
        return isaaclab_math.matrix_from_quat(keybody_global_rot_wxyz)[
            ..., :2
        ]  # [num_envs, num_keybodies, 6]

    @staticmethod
    def _get_obs_global_robot_bodylink_lin_vel(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_names: list[str] | None = None,
    ):
        """Linear velocities of specified bodylinks in the environment frame."""
        robot_ptr = env.scene[robot_asset_name]
        keybody_idxs = ObservationFunctions._get_body_indices(
            robot_ptr, keybody_names
        )
        keybody_global_lin_vel = robot_ptr.data.body_lin_vel_w[:, keybody_idxs]
        return keybody_global_lin_vel  # [num_envs, num_keybodies, 3]

    @staticmethod
    def _get_obs_global_robot_bodylink_ang_vel(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_names: list[str] | None = None,
    ):
        """Angular velocities of specified bodylinks in the environment frame."""
        robot_ptr = env.scene[robot_asset_name]
        keybody_idxs = ObservationFunctions._get_body_indices(
            robot_ptr, keybody_names
        )
        keybody_global_ang_vel = robot_ptr.data.body_ang_vel_w[:, keybody_idxs]
        return keybody_global_ang_vel  # [num_envs, num_keybodies, 3]

    @staticmethod
    def _get_obs_root_rel_robot_bodylink_pos(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_names: list[str] | None = None,
    ) -> torch.Tensor:  # [num_envs, num_keybodies, 3]
        """Positions of specified bodylinks relative to the robot's root frame."""
        # Get global states
        keybody_global_pos: torch.Tensor = (
            ObservationFunctions._get_obs_global_robot_bodylink_pos(
                env, robot_asset_name, keybody_names
            )
        )  # [num_envs, num_keybodies, 3]

        global_root_pos: torch.Tensor = (
            ObservationFunctions._get_obs_global_robot_root_pos(env)
        )  # [num_envs, 3]
        root_global_rot_wxyz: torch.Tensor = (
            ObservationFunctions._get_obs_global_robot_root_rot_wxyz(env)
        )  # [num_envs, 4]

        # Transform to root frame
        # Position relative to root
        rel_pos_global: torch.Tensor = (
            keybody_global_pos - global_root_pos[..., None, :]
        )  # [num_envs, num_keybodies, 3]

        # Rotate to root frame using inverse root rotation
        root_inv_rot: torch.Tensor = isaaclab_math.quat_inv(
            root_global_rot_wxyz
        )  # [num_envs, 4]
        num_bodies = keybody_global_pos.shape[1]
        rel_pos_root: torch.Tensor = isaaclab_math.quat_apply(
            root_inv_rot[..., None, :].expand(-1, num_bodies, -1),
            rel_pos_global,
        )  # [num_envs, num_keybodies, 3]

        return rel_pos_root

    @staticmethod
    def _get_obs_root_rel_robot_bodylink_rot_wxyz(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_names: list[str] | None = None,
    ) -> torch.Tensor:  # [num_envs, num_keybodies, 4]
        """Orientations (w, x, y, z) of specified bodylinks relative to the robot's root frame."""
        # Get global states
        keybody_global_rot: torch.Tensor = (
            ObservationFunctions._get_obs_global_robot_bodylink_rot_wxyz(
                env, robot_asset_name, keybody_names
            )
        )  # [num_envs, num_keybodies, 4]

        root_global_rot_wxyz: torch.Tensor = (
            ObservationFunctions._get_obs_global_robot_root_rot_wxyz(env)
        )  # [num_envs, 4]

        # Transform to root frame by multiplying with inverse root rotation
        root_inv_rot: torch.Tensor = isaaclab_math.quat_inv(
            root_global_rot_wxyz
        )  # [num_envs, 4]
        num_bodies = keybody_global_rot.shape[1]
        rel_rot_root: torch.Tensor = isaaclab_math.quat_mul(
            root_inv_rot[..., None, :].expand(-1, num_bodies, -1),
            keybody_global_rot,
        )  # [num_envs, num_keybodies, 4]

        return rel_rot_root

    @staticmethod
    def _get_obs_root_rel_robot_bodylink_rot_xyzw(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_names: list[str] | None = None,
    ) -> torch.Tensor:  # [num_envs, num_keybodies, 4]
        """Orientations (x, y, z, w) of specified bodylinks relative to the robot's root frame."""
        return ObservationFunctions._get_obs_root_rel_robot_bodylink_rot_wxyz(
            env, robot_asset_name, keybody_names
        )[
            ..., [1, 2, 3, 0]
        ]  # [num_envs, num_keybodies, 4] - convert WXYZ to XYZW

    @staticmethod
    def _get_obs_root_rel_robot_bodylink_rot_mat(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_names: list[str] | None = None,
    ) -> torch.Tensor:  # [num_envs, num_keybodies, 6]
        """Orientations of specified bodylinks relative to the robot's root frame, as a 3x3 matrix, flattened to the first two rows (6D)."""
        keybody_rel_rot_wxyz: torch.Tensor = (
            ObservationFunctions._get_obs_root_rel_robot_bodylink_rot_wxyz(
                env, robot_asset_name, keybody_names
            )
        )  # [num_envs, num_keybodies, 4]

        return isaaclab_math.matrix_from_quat(keybody_rel_rot_wxyz)[
            ..., :2
        ]  # [num_envs, num_keybodies, 6]

    @staticmethod
    def _get_obs_root_rel_robot_bodylink_lin_vel(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_names: list[str] | None = None,
    ) -> torch.Tensor:  # [num_envs, num_keybodies, 3]
        """Linear velocities of specified bodylinks relative to the robot's root frame."""
        # Get global states
        keybody_global_lin_vel: torch.Tensor = (
            ObservationFunctions._get_obs_global_robot_bodylink_lin_vel(
                env, robot_asset_name, keybody_names
            )
        )  # [num_envs, num_keybodies, 3]
        root_global_lin_vel: torch.Tensor = (
            ObservationFunctions._get_obs_global_robot_root_lin_vel(env)
        )  # [num_envs, 3]
        root_global_rot_wxyz: torch.Tensor = (
            ObservationFunctions._get_obs_global_robot_root_rot_wxyz(env)
        )  # [num_envs, 4]

        # Compute relative velocity in world frame
        rel_lin_vel_w = keybody_global_lin_vel - root_global_lin_vel.unsqueeze(
            1
        )

        # Transform to root frame by rotating with inverse root rotation
        root_inv_rot: torch.Tensor = isaaclab_math.quat_inv(
            root_global_rot_wxyz
        )  # [num_envs, 4]
        rel_lin_vel_root: torch.Tensor = isaaclab_math.quat_apply(
            root_inv_rot.unsqueeze(1), rel_lin_vel_w
        )  # [num_envs, num_keybodies, 3]

        return rel_lin_vel_root

    @staticmethod
    def _get_obs_root_rel_robot_bodylink_ang_vel(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_names: list[str] | None = None,
    ) -> torch.Tensor:  # [num_envs, num_keybodies, 3]
        """Angular velocities of specified bodylinks relative to the robot's root frame."""
        # Get global states
        keybody_global_ang_vel: torch.Tensor = (
            ObservationFunctions._get_obs_global_robot_bodylink_ang_vel(
                env, robot_asset_name, keybody_names
            )
        )  # [num_envs, num_keybodies, 3]
        root_global_ang_vel: torch.Tensor = (
            ObservationFunctions._get_obs_global_robot_root_ang_vel(env)
        )  # [num_envs, 3]
        root_global_rot_wxyz: torch.Tensor = (
            ObservationFunctions._get_obs_global_robot_root_rot_wxyz(env)
        )  # [num_envs, 4]

        # Compute relative angular velocity in world frame
        rel_ang_vel_w = keybody_global_ang_vel - root_global_ang_vel.unsqueeze(
            1
        )

        # Transform to root frame by rotating with inverse root rotation
        root_inv_rot: torch.Tensor = isaaclab_math.quat_inv(
            root_global_rot_wxyz
        )  # [num_envs, 4]
        rel_ang_vel_root: torch.Tensor = isaaclab_math.quat_apply(
            root_inv_rot.unsqueeze(1), rel_ang_vel_w
        )  # [num_envs, num_keybodies, 3]

        return rel_ang_vel_root

    # ------- Flat Bodylink Observations -------
    @staticmethod
    def _get_obs_global_robot_bodylink_pos_flat(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_names: list[str] | None = None,
    ) -> torch.Tensor:  # [num_envs, num_keybodies * 3]
        """Flattened positions of specified bodylinks in the environment frame."""
        bodylink_pos = ObservationFunctions._get_obs_global_robot_bodylink_pos(
            env, robot_asset_name, keybody_names
        )  # [num_envs, num_keybodies, 3]
        return bodylink_pos.reshape(
            bodylink_pos.shape[0], -1
        )  # [num_envs, num_keybodies * 3]

    @staticmethod
    def _get_obs_global_robot_bodylink_rot_wxyz_flat(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_names: list[str] | None = None,
    ) -> torch.Tensor:  # [num_envs, num_keybodies * 4]
        """Flattened orientations (w, x, y, z) of specified bodylinks in the environment frame."""
        bodylink_rot = (
            ObservationFunctions._get_obs_global_robot_bodylink_rot_wxyz(
                env, robot_asset_name, keybody_names
            )
        )  # [num_envs, num_keybodies, 4]
        return bodylink_rot.reshape(
            bodylink_rot.shape[0], -1
        )  # [num_envs, num_keybodies * 4]

    @staticmethod
    def _get_obs_global_robot_bodylink_rot_xyzw_flat(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_names: list[str] | None = None,
    ) -> torch.Tensor:  # [num_envs, num_keybodies * 4]
        """Flattened orientations (x, y, z, w) of specified bodylinks in the environment frame."""
        bodylink_rot = (
            ObservationFunctions._get_obs_global_robot_bodylink_rot_xyzw(
                env, robot_asset_name, keybody_names
            )
        )  # [num_envs, num_keybodies, 4]
        return bodylink_rot.reshape(
            bodylink_rot.shape[0], -1
        )  # [num_envs, num_keybodies * 4]

    @staticmethod
    def _get_obs_global_robot_bodylink_rot_mat_flat(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_names: list[str] | None = None,
    ) -> torch.Tensor:  # [num_envs, num_keybodies * 6]
        """Flattened orientation matrices (6D) of specified bodylinks in the environment frame."""
        bodylink_rot_mat = (
            ObservationFunctions._get_obs_global_robot_bodylink_rot_mat(
                env, robot_asset_name, keybody_names
            )
        )  # [num_envs, num_keybodies, 6]
        return bodylink_rot_mat.reshape(
            bodylink_rot_mat.shape[0], -1
        )  # [num_envs, num_keybodies * 6]

    @staticmethod
    def _get_obs_global_robot_bodylink_lin_vel_flat(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_names: list[str] | None = None,
    ) -> torch.Tensor:  # [num_envs, num_keybodies * 3]
        """Flattened linear velocities of specified bodylinks in the environment frame."""
        bodylink_lin_vel = (
            ObservationFunctions._get_obs_global_robot_bodylink_lin_vel(
                env, robot_asset_name, keybody_names
            )
        )  # [num_envs, num_keybodies, 3]
        return bodylink_lin_vel.reshape(
            bodylink_lin_vel.shape[0], -1
        )  # [num_envs, num_keybodies * 3]

    @staticmethod
    def _get_obs_global_robot_bodylink_ang_vel_flat(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_names: list[str] | None = None,
    ) -> torch.Tensor:  # [num_envs, num_keybodies * 3]
        """Flattened angular velocities of specified bodylinks in the environment frame."""
        bodylink_ang_vel = (
            ObservationFunctions._get_obs_global_robot_bodylink_ang_vel(
                env, robot_asset_name, keybody_names
            )
        )  # [num_envs, num_keybodies, 3]
        return bodylink_ang_vel.reshape(
            bodylink_ang_vel.shape[0], -1
        )  # [num_envs, num_keybodies * 3]

    @staticmethod
    def _get_obs_root_rel_robot_bodylink_pos_flat(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_names: list[str] | None = None,
    ) -> torch.Tensor:  # [num_envs, num_keybodies * 3]
        """Flattened positions of specified bodylinks relative to the robot's root frame."""
        bodylink_pos = (
            ObservationFunctions._get_obs_root_rel_robot_bodylink_pos(
                env, robot_asset_name, keybody_names
            )
        )  # [num_envs, num_keybodies, 3]
        return bodylink_pos.reshape(
            bodylink_pos.shape[0], -1
        )  # [num_envs, num_keybodies * 3]

    @staticmethod
    def _get_obs_root_rel_robot_bodylink_rot_wxyz_flat(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_names: list[str] | None = None,
    ) -> torch.Tensor:  # [num_envs, num_keybodies * 4]
        """Flattened orientations (w, x, y, z) of specified bodylinks relative to the robot's root frame."""
        bodylink_rot = (
            ObservationFunctions._get_obs_root_rel_robot_bodylink_rot_wxyz(
                env, robot_asset_name, keybody_names
            )
        )  # [num_envs, num_keybodies, 4]
        return bodylink_rot.reshape(
            bodylink_rot.shape[0], -1
        )  # [num_envs, num_keybodies * 4]

    @staticmethod
    def _get_obs_root_rel_robot_bodylink_rot_xyzw_flat(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_names: list[str] | None = None,
    ) -> torch.Tensor:  # [num_envs, num_keybodies * 4]
        """Flattened orientations (x, y, z, w) of specified bodylinks relative to the robot's root frame."""
        bodylink_rot = (
            ObservationFunctions._get_obs_root_rel_robot_bodylink_rot_xyzw(
                env, robot_asset_name, keybody_names
            )
        )  # [num_envs, num_keybodies, 4]
        return bodylink_rot.reshape(
            bodylink_rot.shape[0], -1
        )  # [num_envs, num_keybodies * 4]

    @staticmethod
    def _get_obs_root_rel_robot_bodylink_rot_mat_flat(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_names: list[str] | None = None,
    ) -> torch.Tensor:  # [num_envs, num_keybodies * 6]
        """Flattened orientation matrices (6D) of specified bodylinks relative to the robot's root frame."""
        bodylink_rot_mat = (
            ObservationFunctions._get_obs_root_rel_robot_bodylink_rot_mat(
                env, robot_asset_name, keybody_names
            )
        )  # [num_envs, num_keybodies, 6]
        return bodylink_rot_mat.reshape(
            bodylink_rot_mat.shape[0], -1
        )  # [num_envs, num_keybodies * 6]

    @staticmethod
    def _get_obs_root_rel_robot_bodylink_lin_vel_flat(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_names: list[str] | None = None,
    ) -> torch.Tensor:  # [num_envs, num_keybodies * 3]
        """Flattened linear velocities of specified bodylinks relative to the robot's root frame."""
        bodylink_lin_vel = (
            ObservationFunctions._get_obs_root_rel_robot_bodylink_lin_vel(
                env, robot_asset_name, keybody_names
            )
        )  # [num_envs, num_keybodies, 3]
        return bodylink_lin_vel.reshape(
            bodylink_lin_vel.shape[0], -1
        )  # [num_envs, num_keybodies * 3]

    @staticmethod
    def _get_obs_root_rel_robot_bodylink_ang_vel_flat(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_names: list[str] | None = None,
    ) -> torch.Tensor:  # [num_envs, num_keybodies * 3]
        """Flattened angular velocities of specified bodylinks relative to the robot's root frame."""
        bodylink_ang_vel = (
            ObservationFunctions._get_obs_root_rel_robot_bodylink_ang_vel(
                env, robot_asset_name, keybody_names
            )
        )  # [num_envs, num_keybodies, 3]
        return bodylink_ang_vel.reshape(
            bodylink_ang_vel.shape[0], -1
        )  # [num_envs, num_keybodies * 3]

    # ------- Robot DoF States -------
    @staticmethod
    def _get_obs_dof_pos(env: ManagerBasedRLEnv):
        """Joint positions relative to the default joint angles."""
        return isaaclab_mdp.joint_pos_rel(env)  # [num_envs, num_dofs]

    @staticmethod
    def _get_obs_dof_vel(env: ManagerBasedRLEnv):
        """Joint velocities."""
        return isaaclab_mdp.joint_vel_rel(env)  # [num_envs, num_dofs]

    @staticmethod
    def _get_obs_last_actions(env: ManagerBasedRLEnv):
        """Last action output by the policy."""
        return isaaclab_mdp.last_action(env)  # [num_envs, num_actions]

    # ------- Reference Motion States -------
    @staticmethod
    def _get_obs_ref_motion_states(
        env: ManagerBasedRLEnv,
        ref_motion_command_name: str = "ref_motion",
        ref_prefix: str = "ref_",
    ):
        """Reference motion states (flattened) via RefMotionCommand schema."""
        command = env.command_manager.get_term(ref_motion_command_name)
        obs_fn_name = f"_get_obs_{command.cfg.command_obs_name}"
        obs_fn = getattr(command, obs_fn_name)
        return obs_fn(obs_prefix=ref_prefix)

    @staticmethod
    def _get_obs_ref_motion_states_fut(
        env: ManagerBasedRLEnv,
        ref_motion_command_name: str = "ref_motion",
        ref_prefix: str = "ref_",
    ):
        """Future reference motion states (flattened)."""
        command = env.command_manager.get_term(ref_motion_command_name)
        obs_fn_name = f"_get_obs_{command.cfg.command_obs_name}_fut"
        obs_fn = getattr(command, obs_fn_name)
        return obs_fn(obs_prefix=ref_prefix)

    @staticmethod
    def _get_obs_vr_ref_motion_states(
        env: ManagerBasedRLEnv,
        ref_motion_command_name: str = "ref_motion",
        ref_prefix: str = "ref_",
    ):
        command = env.command_manager.get_term(ref_motion_command_name)
        return command._get_obs_vr_ref_motion_states(obs_prefix=ref_prefix)

    @staticmethod
    def _get_obs_vr_ref_motion_states_fut(
        env: ManagerBasedRLEnv,
        ref_motion_command_name: str = "ref_motion",
        ref_prefix: str = "ref_",
    ):
        """Future reference motion states (flattened)."""
        command = env.command_manager.get_term(ref_motion_command_name)
        return command._get_obs_vr_ref_motion_fut(obs_prefix=ref_prefix)

    @staticmethod
    def _get_obs_ref_dof_pos_cur(
        env: ManagerBasedRLEnv,
        ref_motion_command_name: str = "ref_motion",
        ref_prefix: str = "ref_",
    ) -> torch.Tensor:  # [num_envs, num_dofs]
        """Reference current DoF positions in simulator DoF order."""
        command = env.command_manager.get_term(ref_motion_command_name)
        return command.get_ref_motion_dof_pos_cur(prefix=ref_prefix)

    @staticmethod
    def _get_obs_ref_dof_vel_cur(
        env: ManagerBasedRLEnv,
        ref_motion_command_name: str = "ref_motion",
        ref_prefix: str = "ref_",
    ) -> torch.Tensor:  # [num_envs, num_dofs]
        """Reference current DoF velocities in simulator DoF order."""
        command = env.command_manager.get_term(ref_motion_command_name)
        return command.get_ref_motion_dof_vel_cur(prefix=ref_prefix)

    @staticmethod
    def _get_obs_ref_root_height_cur(
        env: ManagerBasedRLEnv,
        ref_motion_command_name: str = "ref_motion",
        ref_prefix: str = "ref_",
    ) -> torch.Tensor:  # [num_envs, 1]
        """Reference current root height: world z minus env-origin z."""
        command = env.command_manager.get_term(ref_motion_command_name)
        world_pos = command.get_ref_motion_root_global_pos_cur(
            prefix=ref_prefix
        )  # [B, 3]
        height = (world_pos[..., 2] - env.scene.env_origins[..., 2]).unsqueeze(
            -1
        )  # [B,1]
        return height

    @staticmethod
    def _get_obs_ref_dof_pos_fut(
        env: ManagerBasedRLEnv,
        ref_motion_command_name: str = "ref_motion",
        ref_prefix: str = "ref_",
    ) -> torch.Tensor:  # [num_envs, n_fut_frames * num_dofs]
        """Future reference DoF positions (flattened over time) in simulator DoF order."""
        command = env.command_manager.get_term(ref_motion_command_name)
        dof_pos_fut = command.get_ref_motion_dof_pos_fut(
            prefix=ref_prefix
        )  # [B, T, D(sim)]
        B, T, D = dof_pos_fut.shape
        return dof_pos_fut.reshape(B, T * D)

    @staticmethod
    def _get_obs_ref_dof_pos_fut_1_4(
        env: ManagerBasedRLEnv,
        ref_motion_command_name: str = "ref_motion",
        ref_prefix: str = "ref_",
    ) -> torch.Tensor:  # [num_envs, n_fut_frames * num_dofs]
        """Future reference DoF positions (flattened over time) in simulator DoF order."""
        command = env.command_manager.get_term(ref_motion_command_name)
        dof_pos_fut = command.get_ref_motion_dof_pos_fut(
            prefix=ref_prefix
        )  # [B, T, D(sim)]
        dof_pos_fut = dof_pos_fut[:, :4, ...]
        B, T, D = dof_pos_fut.shape
        return dof_pos_fut.reshape(B, T * D)

    @staticmethod
    def _get_obs_ref_dof_pos_fut_5_8(
        env: ManagerBasedRLEnv,
        ref_motion_command_name: str = "ref_motion",
        ref_prefix: str = "ref_",
    ) -> torch.Tensor:  # [num_envs, n_fut_frames * num_dofs]
        """Future reference DoF positions (flattened over time) in simulator DoF order."""
        command = env.command_manager.get_term(ref_motion_command_name)
        dof_pos_fut = command.get_ref_motion_dof_pos_fut(
            prefix=ref_prefix
        )  # [B, T, D(sim)]
        dof_pos_fut = dof_pos_fut[:, 4:8, ...]
        B, T, D = dof_pos_fut.shape
        return dof_pos_fut.reshape(B, T * D)

    @staticmethod
    def _get_obs_ref_dof_vel_fut(
        env: ManagerBasedRLEnv,
        ref_motion_command_name: str = "ref_motion",
        ref_prefix: str = "ref_",
    ) -> torch.Tensor:  # [num_envs, n_fut_frames * num_dofs]
        """Future reference DoF velocities (flattened over time) in simulator DoF order."""
        command = env.command_manager.get_term(ref_motion_command_name)
        dof_vel_fut = command.get_ref_motion_dof_vel_fut(
            prefix=ref_prefix
        )  # [B, T, D(sim)]
        B, T, D = dof_vel_fut.shape
        return dof_vel_fut.reshape(B, T * D)

    @staticmethod
    def _get_obs_ref_dof_vel_fut_1_4(
        env: ManagerBasedRLEnv,
        ref_motion_command_name: str = "ref_motion",
        ref_prefix: str = "ref_",
    ) -> torch.Tensor:  # [num_envs, n_fut_frames * num_dofs]
        """Future reference DoF velocities (flattened over time) in simulator DoF order."""
        command = env.command_manager.get_term(ref_motion_command_name)
        dof_vel_fut = command.get_ref_motion_dof_vel_fut(
            prefix=ref_prefix
        )  # [B, T, D(sim)]
        dof_vel_fut = dof_vel_fut[:, :4, ...]
        B, T, D = dof_vel_fut.shape
        return dof_vel_fut.reshape(B, T * D)

    @staticmethod
    def _get_obs_ref_root_height_fut(
        env: ManagerBasedRLEnv,
        ref_motion_command_name: str = "ref_motion",
        ref_prefix: str = "ref_",
    ) -> torch.Tensor:  # [num_envs, n_fut_frames]
        """Future reference root heights per frame: world z minus env-origin z."""
        command = env.command_manager.get_term(ref_motion_command_name)
        world_pos = command.get_ref_motion_root_global_pos_fut(
            prefix=ref_prefix
        )  # [B, T, 3]
        heights = (
            world_pos[..., 2] - env.scene.env_origins[:, None, 2]
        )  # [B, T]
        return heights

    @staticmethod
    def _get_obs_ref_root_height_fut_1_4(
        env: ManagerBasedRLEnv,
        ref_motion_command_name: str = "ref_motion",
        ref_prefix: str = "ref_",
    ) -> torch.Tensor:  # [num_envs, n_fut_frames]
        """Future reference root heights per frame: world z minus env-origin z."""
        command = env.command_manager.get_term(ref_motion_command_name)
        world_pos = command.get_ref_motion_root_global_pos_fut(
            prefix=ref_prefix
        )  # [B, T, 3]
        heights = (
            world_pos[..., 2] - env.scene.env_origins[:, None, 2]
        )  # [B, T]
        heights = heights[:, :4]
        return heights
    
    @staticmethod
    def _get_obs_ref_root_height_fut_5_8(
        env: ManagerBasedRLEnv,
        ref_motion_command_name: str = "ref_motion",
        ref_prefix: str = "ref_",
    ) -> torch.Tensor:  # [num_envs, n_fut_frames]
        """Future reference root heights per frame: world z minus env-origin z."""
        command = env.command_manager.get_term(ref_motion_command_name)
        world_pos = command.get_ref_motion_root_global_pos_fut(
            prefix=ref_prefix
        )  # [B, T, 3]
        heights = (
            world_pos[..., 2] - env.scene.env_origins[:, None, 2]
        )  # [B, T]
        heights = heights[:, 4:8]
        return heights

    # @torch.compile
    @staticmethod
    def _get_obs_global_anchor_diff(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        ref_motion_command_name: str = "ref_motion",
        ref_prefix: str = "ref_",
    ):
        command = env.command_manager.get_term(ref_motion_command_name)
        global_ref_motion_anchor_pos = (
            command.get_ref_motion_anchor_bodylink_global_pos_cur(
                prefix=ref_prefix
            )
        )
        global_ref_motino_anchor_rot_wxyz = (
            command.get_ref_motion_anchor_bodylink_global_rot_wxyz_cur(
                prefix=ref_prefix
            )
        )
        global_robot_anchor_pos = (
            ObservationFunctions._get_obs_global_robot_bodylink_pos(
                env, robot_asset_name, [command.anchor_bodylink_name]
            ).squeeze(1)
        )
        global_robot_anchor_rot_wxyz = (
            ObservationFunctions._get_obs_global_robot_bodylink_rot_wxyz(
                env, robot_asset_name, [command.anchor_bodylink_name]
            ).squeeze(1)
        )
        pos_diff, rot_diff = isaaclab_math.subtract_frame_transforms(
            t01=global_robot_anchor_pos,
            q01=global_robot_anchor_rot_wxyz,
            t02=global_ref_motion_anchor_pos,
            q02=global_ref_motino_anchor_rot_wxyz,
        )
        rot_diff_mat = isaaclab_math.matrix_from_quat(rot_diff)
        return torch.cat(
            [
                pos_diff,
                rot_diff_mat[..., :2].reshape(env.num_envs, -1),
            ],
            dim=-1,
        )  # [num_envs, 9]

    @staticmethod
    def _get_obs_global_anchor_pos_diff(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        ref_motion_command_name: str = "ref_motion",
        ref_prefix: str = "ref_",
    ):
        command = env.command_manager.get_term(ref_motion_command_name)
        global_ref_motion_anchor_pos = (
            command.get_ref_motion_anchor_bodylink_global_pos_cur(
                prefix=ref_prefix
            )
        )
        global_ref_motino_anchor_rot_wxyz = (
            command.get_ref_motion_anchor_bodylink_global_rot_wxyz_cur(
                prefix=ref_prefix
            )
        )
        global_robot_anchor_pos = (
            ObservationFunctions._get_obs_global_robot_bodylink_pos(
                env, robot_asset_name, [command.anchor_bodylink_name]
            ).squeeze(1)
        )
        global_robot_anchor_rot_wxyz = (
            ObservationFunctions._get_obs_global_robot_bodylink_rot_wxyz(
                env, robot_asset_name, [command.anchor_bodylink_name]
            ).squeeze(1)
        )
        pos_diff, _ = isaaclab_math.subtract_frame_transforms(
            t01=global_robot_anchor_pos,
            q01=global_robot_anchor_rot_wxyz,
            t02=global_ref_motion_anchor_pos,
            q02=global_ref_motino_anchor_rot_wxyz,
        )
        return pos_diff

    @staticmethod
    def _get_obs_global_anchor_rot_diff(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        ref_motion_command_name: str = "ref_motion",
        ref_prefix: str = "ref_",
    ):
        command = env.command_manager.get_term(ref_motion_command_name)
        global_ref_motion_anchor_pos = (
            command.get_ref_motion_anchor_bodylink_global_pos_cur(
                prefix=ref_prefix
            )
        )
        global_ref_motino_anchor_rot_wxyz = (
            command.get_ref_motion_anchor_bodylink_global_rot_wxyz_cur(
                prefix=ref_prefix
            )
        )
        global_robot_anchor_pos = (
            ObservationFunctions._get_obs_global_robot_bodylink_pos(
                env, robot_asset_name, [command.anchor_bodylink_name]
            ).squeeze(1)
        )
        global_robot_anchor_rot_wxyz = (
            ObservationFunctions._get_obs_global_robot_bodylink_rot_wxyz(
                env, robot_asset_name, [command.anchor_bodylink_name]
            ).squeeze(1)
        )
        _, rot_diff = isaaclab_math.subtract_frame_transforms(
            t01=global_robot_anchor_pos,
            q01=global_robot_anchor_rot_wxyz,
            t02=global_ref_motion_anchor_pos,
            q02=global_ref_motino_anchor_rot_wxyz,
        )
        rot_diff_mat = isaaclab_math.matrix_from_quat(rot_diff)
        return rot_diff_mat[..., :2].reshape(env.num_envs, -1)

    @staticmethod
    def _get_obs_velocity_command(
        env: ManagerBasedRLEnv,
    ):
        """Velocity command.

        This function should return the velocity command which
        has already been serialized into flattened vectors. Note that we also
        add a model switch mask dimension, when commands are small, the mode
        is set to 0, otherwise it is set to 1.
        """
        velocity_command = isaaclab_mdp.generated_commands(
            env,
            command_name="base_velocity",
        )  # [num_envs, 3]
        move_mask = velocity_command.norm(dim=-1) > 0.1
        return torch.cat(
            [
                move_mask.unsqueeze(-1),
                velocity_command,
            ],
            dim=-1,
        )  # [num_envs, 4]

    @staticmethod
    def _get_obs_place_holder(env: ManagerBasedRLEnv, n_dim: int):
        return torch.zeros(env.num_envs, n_dim, device=env.device)


@configclass
class ObservationsCfg:
    pass


def build_observations_config(obs_config_dict: dict):
    """Build isaaclab-compatible ObservationsCfg from a config dictionary."""

    if isinstance(obs_config_dict, (DictConfig, ListConfig)):
        obs_config_dict = OmegaConf.to_container(obs_config_dict, resolve=True)

    obs_cfg = ObservationsCfg()
    obs_term_field_names = {
        field.name for field in dataclass_fields(ObservationTermCfg)
    }

    # Create observation groups dynamically
    for group_name, group_cfg in obs_config_dict.items():
        group_cfg = resolve_holo_config(group_cfg)

        isaaclab_obs_group_cfg = ObsGroup()

        for key, value in group_cfg.items():
            if key == "atomic_obs_list":
                continue
            if hasattr(isaaclab_obs_group_cfg, key):
                setattr(isaaclab_obs_group_cfg, key, value)

        # Add observation terms to the group
        for obs_term_dict in group_cfg["atomic_obs_list"]:
            for obs_name, obs_params in obs_term_dict.items():
                # Look for observation function in ObservationFunctions class
                method_name = f"_get_obs_{obs_name}"

                if hasattr(ObservationFunctions, method_name):
                    # Use custom observation function
                    func = getattr(ObservationFunctions, method_name)
                elif hasattr(isaaclab_mdp, obs_name):
                    # Use isaaclab isaaclab_mdp function directly
                    func = getattr(isaaclab_mdp, obs_name)
                else:
                    raise ValueError(
                        f"Unknown observation function: {obs_name}"
                    )

                obs_params = resolve_holo_config(obs_params)

                obs_term_kwargs = {"func": func}
                try:
                    params_cfg = obs_params.get("params", {})
                except AttributeError:
                    print(f"No params found for {obs_name}")

                obs_term_kwargs["params"] = resolve_holo_config(params_cfg)

                noise_cfg = obs_params.get("noise")
                if noise_cfg is not None:
                    noise_cfg = resolve_holo_config(noise_cfg)
                    if isinstance(noise_cfg, dict) and "type" in noise_cfg:
                        noise_cls = getattr(isaaclab_noise, noise_cfg["type"])
                        noise_params = resolve_holo_config(
                            noise_cfg.get("params", {})
                        )
                        obs_term_kwargs["noise"] = noise_cls(**noise_params)
                    else:
                        obs_term_kwargs["noise"] = noise_cfg

                for field_name in obs_term_field_names:
                    if field_name in {"func", "params", "noise"}:
                        continue
                    if field_name in obs_params:
                        obs_term_kwargs[field_name] = obs_params[field_name]

                obs_term = ObsTerm(**obs_term_kwargs)

                # Add observation term to group
                setattr(isaaclab_obs_group_cfg, obs_name, obs_term)

        # Add group to main observations config
        setattr(obs_cfg, group_name, isaaclab_obs_group_cfg)

    return obs_cfg
