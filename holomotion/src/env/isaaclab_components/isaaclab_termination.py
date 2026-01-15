from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import TerminationTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
import torch
from isaaclab.assets import Articulation
from holomotion.src.env.isaaclab_components.isaaclab_motion_tracking_command import (
    RefMotionCommand,
)
import isaaclab.utils.math as isaaclab_math
import isaaclab.envs.mdp as isaaclab_mdp

from holomotion.src.env.isaaclab_components.isaaclab_utils import (
    _get_body_indices,
    resolve_holo_config,
)


def time_out(env: ManagerBasedRLEnv):
    return isaaclab_mdp.terminations.time_out(env)


def bad_orientation(env: ManagerBasedRLEnv, limit_angle: float = 0.8):
    return isaaclab_mdp.terminations.bad_orientation(
        env, limit_angle=limit_angle
    )


def root_height_below_minimum(env: ManagerBasedRLEnv, minimum_height: float):
    return isaaclab_mdp.terminations.root_height_below_minimum(
        env, minimum_height=minimum_height
    )


def global_bodylink_pos_far(
    env: ManagerBasedRLEnv,
    threshold: float,
    command_name: str = "ref_motion",
    keybody_names: list[str] | None = None,
    ref_prefix: str = "ref_",
) -> torch.Tensor:
    """Any body link position deviates more than threshold (world frame)."""
    command: RefMotionCommand = env.command_manager.get_term(command_name)
    ref_pos_w = command.get_ref_motion_bodylink_global_pos_cur(
        prefix=ref_prefix
    )  # [B, Nb, 3]
    robot_pos_w = command.robot.data.body_pos_w  # [B, Nb, 3]

    keybody_idxs = _get_body_indices(command.robot, keybody_names)

    if keybody_idxs is not None and len(keybody_idxs) > 0:
        idxs = torch.as_tensor(
            keybody_idxs,
            device=ref_pos_w.device,
            dtype=torch.long,
        )
        ref_pos_w = ref_pos_w[:, idxs]
        robot_pos_w = robot_pos_w[:, idxs]

    error = torch.norm(ref_pos_w - robot_pos_w, dim=-1)  # [B, Nb]
    return torch.any(error > threshold, dim=-1)  # [B]


def anchor_ref_z_far(
    env: ManagerBasedRLEnv,
    threshold: float,
    command_name: str = "ref_motion",
    ref_prefix: str = "ref_",
) -> torch.Tensor:
    """Anchor link z difference exceeds threshold (world frame)."""
    command: RefMotionCommand = env.command_manager.get_term(command_name)
    ref_z = command.get_ref_motion_anchor_bodylink_global_pos_cur(
        prefix=ref_prefix
    )[:, -1]
    robot_z = command.global_robot_anchor_pos_cur[:, -1]
    return (ref_z - robot_z).abs() > threshold


def ref_gravity_projection_far(
    env: ManagerBasedRLEnv,
    threshold: float,
    asset_name: str = "robot",
    command_name: str = "ref_motion",
    ref_prefix: str = "ref_",
) -> torch.Tensor:
    """Difference in projected gravity z-component between ref and robot exceeds threshold.

    Project world gravity into the anchor body frames using inverse quaternion rotation
    and compare z-components.
    """
    command: RefMotionCommand = env.command_manager.get_term(command_name)
    g_w = env.scene[asset_name].data.GRAVITY_VEC_W  # [B, 3]

    # Reference anchor orientation (xyzw) from motion cache
    ref_anchor_quat_xyzw = command.get_ref_motion_anchor_bodylink_global_rot_wxyz_cur(
        prefix=ref_prefix
    )  # [B, 4]

    motion_projected_gravity_b = isaaclab_math.quat_apply_inverse(
        ref_anchor_quat_xyzw, g_w
    )  # [B, 3]

    # motion_projected_gravity_b = isaaclab_math.quat_rotate_inverse(
    #     ref_anchor_quat_xyzw, g_w
    # )  # [B, 3]

    # Robot anchor orientation (xyzw) from sim
    robot_anchor_quat_wxyz = command.robot.data.body_quat_w[
        :, command.anchor_bodylink_idx
    ]  # [B, 4]

    robot_projected_gravity_b = isaaclab_math.quat_apply_inverse(
        robot_anchor_quat_wxyz, g_w
    )  # [B, 3]

    # robot_projected_gravity_b = isaaclab_math.quat_rotate_inverse(
    #     robot_anchor_quat_wxyz, g_w
    # )  # [B, 3]

    return (
        motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]
    ).abs() > threshold


def keybody_ref_z_far(
    env: ManagerBasedRLEnv,
    threshold: float,
    command_name: str = "ref_motion",
    keybody_names: list[str] | None = None,
    ref_prefix: str = "ref_",
) -> torch.Tensor:
    """Any key body link z difference exceeds threshold (world frame)."""
    command: RefMotionCommand = env.command_manager.get_term(command_name)
    ref_pos_w = command.get_ref_motion_bodylink_global_pos_cur(
        prefix=ref_prefix
    )  # [B, Nb, 3]
    robot_pos_w = command.robot.data.body_pos_w  # [B, Nb, 3]

    keybody_idxs = _get_body_indices(command.robot, keybody_names)

    if keybody_idxs is not None and len(keybody_idxs) > 0:
        idxs = torch.as_tensor(
            keybody_idxs,
            device=ref_pos_w.device,
            dtype=torch.long,
        )
        ref_pos_w = ref_pos_w[:, idxs]
        robot_pos_w = robot_pos_w[:, idxs]

    error_z = (ref_pos_w[..., 2] - robot_pos_w[..., 2]).abs()  # [B, Nb]
    return torch.any(error_z > threshold, dim=-1)  # [B]


def motion_end(
    env: ManagerBasedRLEnv,
    command_name: str = "ref_motion",
) -> torch.Tensor:
    """Terminate when reference motion frames exceed their end frames.

    Returns a boolean mask of shape [num_envs].
    """
    command: RefMotionCommand = env.command_manager.get_term(command_name)
    result = command.motion_end_mask.clone().bool()
    return result


@configclass
class TerminationsCfg:
    pass


def build_terminations_config(
    termination_config_dict: dict,
) -> TerminationsCfg:
    terminations_cfg = TerminationsCfg()

    # Explicit mapping from names to functions (Hydra-friendly signatures)
    fn_map = {
        "time_out": time_out,
        "bad_orientation": bad_orientation,
        "root_height_below_minimum": root_height_below_minimum,
        "global_bodylink_pos_far": global_bodylink_pos_far,
        "anchor_ref_z_far": anchor_ref_z_far,
        "ref_gravity_projection_far": ref_gravity_projection_far,
        "keybody_ref_z_far": keybody_ref_z_far,
        "motion_end": motion_end,
    }

    for termination_name, termination_cfg in termination_config_dict.items():
        if termination_name not in fn_map:
            raise ValueError(
                f"Unknown termination function: {termination_name}. "
                f"Supported: {list(fn_map.keys())}"
            )

        func = fn_map[termination_name]
        params = termination_cfg.get("params", {})

        # Verbosely construct the TerminationTermCfg; for standard terms like time_out,
        # mark time_out=True for Manager-based auto handling.
        term_cfg = TerminationTermCfg(
            func=func,
            params=params,
            time_out=(termination_name == "time_out")
            or termination_cfg.get("time_out", False),
        )
        setattr(terminations_cfg, termination_name, term_cfg)

    return terminations_cfg
