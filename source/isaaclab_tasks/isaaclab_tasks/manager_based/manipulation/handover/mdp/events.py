import torch

from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms, quat_mul
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv

# Global phase flags (per environment)
phase_flags = {}
GRASP_THRESHOLD = 1.0
RELEASE_THRESHOLD = 0.1
Z_OFFSET = 0.07
REACH_TOL = 0.03
Z_TOL = 0.03
XY_TOL = 0.03
READY_JOINT_TOL = 0.5

def reset_phase_flags(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
):
    """Initialize phase flags for all environments and reset them to False at episode start."""
    global phase_flags
    
    # env_ids의 최대값을 기준으로 필요한 크기 계산
    max_env_id = torch.max(env_ids).item() + 1
    
    # phase_flags가 아직 초기화되지 않았거나 크기가 부족하면 확장
    if (not phase_flags or "phase1_complete" not in phase_flags or 
        phase_flags["phase1_complete"].shape[0] <= max_env_id):
        
        # 더 큰 크기로 초기화
        new_size = int(max(max_env_id, env.num_envs))
        phase_flags["phase1_complete"] = torch.zeros(new_size, dtype=torch.bool, device=env.device)
        phase_flags["phase2_complete"] = torch.zeros(new_size, dtype=torch.bool, device=env.device)
        phase_flags["phase3_complete"] = torch.zeros(new_size, dtype=torch.bool, device=env.device)
        phase_flags["phase4_complete"] = torch.zeros(new_size, dtype=torch.bool, device=env.device)
        phase_flags["phase5_complete"] = torch.zeros(new_size, dtype=torch.bool, device=env.device)
    else:
        current_size = phase_flags["phase1_complete"].shape[0]
        for key in ("phase1_complete", "phase2_complete", "phase3_complete", "phase4_complete", "phase5_complete", "phase6_complete"):
            if key not in phase_flags:
                phase_flags[key] = torch.zeros(current_size, dtype=torch.bool, device=env.device)
    
    # 특정 환경들만 리셋
    phase_flags["phase1_complete"][env_ids] = False
    phase_flags["phase2_complete"][env_ids] = False
    phase_flags["phase3_complete"][env_ids] = False
    phase_flags["phase4_complete"][env_ids] = False
    phase_flags["phase5_complete"][env_ids] = False


def check_and_update_phase_flags(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Check phase conditions and update latch flags for specific environments."""

    # Robot is an articulation (has joints)
    robot: Articulation = env.scene[robot_cfg.name]
    left_finger_sensor = env.scene.sensors["contact_forces_left_finger_pad"]
    right_finger_sensor = env.scene.sensors["contact_forces_right_finger_pad"]
    ee_frame: FrameTransformer = env.scene["ee_frame"]
    object: RigidObject = env.scene[object_cfg.name]
    ascend_command = env.command_manager.get_command("ascend")
    descend_command = env.command_manager.get_command("descend")

    # contact sensor of fingers (env_ids에 해당하는 환경들만)
    left_forces = left_finger_sensor.data.force_matrix_w[env_ids]
    right_forces = right_finger_sensor.data.force_matrix_w[env_ids]
    left_forces_sum = torch.sum(left_forces, dim=(1, 2))
    right_forces_sum = torch.sum(right_forces, dim=(1, 2))
    left_force_magnitudes = torch.norm(left_forces_sum, dim=-1)
    right_force_magnitudes = torch.norm(right_forces_sum, dim=-1)

    left_grasp_contact = (left_force_magnitudes >= GRASP_THRESHOLD)
    right_grasp_contact = (right_force_magnitudes >= GRASP_THRESHOLD)
    grasping = left_grasp_contact & right_grasp_contact

    left_release_contact = (left_force_magnitudes <= RELEASE_THRESHOLD)
    right_release_contact = (right_force_magnitudes <= RELEASE_THRESHOLD)
    releasing = left_release_contact & right_release_contact

    # Phase 1 goal: Reach & Lift
    ee_pos_w = ee_frame.data.target_pos_w[env_ids, 0, :]
    object_pos = object.data.root_pos_w[env_ids]
    ee_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w[env_ids], robot.data.root_quat_w[env_ids], ee_pos_w)
    obj_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w[env_ids], robot.data.root_quat_w[env_ids], object_pos)
    lift_target = env._object_initial_pos_b[env_ids].clone() if hasattr(env, "_object_initial_pos_b") else obj_pos_b.clone()
    lift_target[:, 2] += Z_OFFSET
    obj_z = obj_pos_b[:, 2]
    lift_z = lift_target[:, 2]
    object_reached = torch.norm(ee_pos_b - obj_pos_b, dim=1) < REACH_TOL
    object_lifted = torch.abs(obj_z - lift_z) < Z_TOL
    phase1_condition = object_reached & object_lifted & grasping
    phase_flags["phase1_complete"][env_ids] = phase_flags["phase1_complete"][env_ids] | phase1_condition

    # Phase 2 goal: Ascend
    ascend_pos_b = ascend_command[env_ids, :3]
    ascend_z_err = torch.abs(ascend_pos_b[:, 2] - obj_pos_b[:, 2])
    ascend_xy_err = torch.norm(ascend_pos_b[:, :2] - obj_pos_b[:, :2], dim=1)
    phase2_condition = (ascend_z_err < Z_TOL) & (ascend_xy_err < XY_TOL) & grasping
    phase_flags["phase2_complete"][env_ids] = phase_flags["phase2_complete"][env_ids] | (
        phase2_condition & phase_flags["phase1_complete"][env_ids]
    )

    # Phase 3 goal: Descend
    descend_pos_b = descend_command[env_ids, :3]
    descend_z_err = torch.abs(descend_pos_b[:, 2] - obj_pos_b[:, 2])
    descend_xy_err = torch.norm(descend_pos_b[:, :2] - obj_pos_b[:, :2], dim=1)
    phase3_condition = (descend_z_err < Z_TOL) & (descend_xy_err < XY_TOL) & grasping
    phase_flags["phase3_complete"][env_ids] = phase_flags["phase3_complete"][env_ids] | (
        phase3_condition & phase_flags["phase2_complete"][env_ids]
    )

    # Phase 4 goal: Drop & Leave
    # Leave target: use descend command with a Z offset
    leave_pos_b = descend_pos_b.clone()
    leave_pos_b[:, 2] += Z_OFFSET
    leave_z_err = torch.abs(leave_pos_b[:, 2] - ee_pos_b[:, 2])
    leave_xy_err = torch.norm(leave_pos_b[:, :2] - ee_pos_b[:, :2], dim=1)
    phase4_condition = (leave_z_err < Z_TOL) & (leave_xy_err < XY_TOL) & releasing
    phase_flags["phase4_complete"][env_ids] = phase_flags["phase4_complete"][env_ids] | (
        phase4_condition & phase_flags["phase3_complete"][env_ids]
    )

    # Phase 5 goal: Go Back to Initial pose
    joint_ids, _ = robot.find_joints(["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "left_outer_knuckle_joint"])
    if len(joint_ids) > 0:
        joint_pos = robot.data.joint_pos[env_ids][:, joint_ids]
        default_joint_pos = robot.data.default_joint_pos[env_ids][:, joint_ids]
        joint_deviation = torch.linalg.norm(joint_pos - default_joint_pos, dim=1)
        phase5_condition = (joint_deviation < READY_JOINT_TOL) & releasing
    else:
        phase5_condition = releasing

    phase_flags["phase5_complete"][env_ids] = phase_flags["phase5_complete"][env_ids] | (
        phase5_condition & phase_flags["phase4_complete"][env_ids]
    )


def cache_object_initial_pose(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Cache the object's initial 6D pose expressed in the robot's root frame.

    This stores per-env tensors on the env instance so observation terms can
    read a fixed value during the entire episode.
    """
    robot: Articulation | RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    # Object position in robot root frame
    obj_pos_w = obj.data.root_pos_w[env_ids, :3]
    robot_pos_w = robot.data.root_pos_w[env_ids]
    robot_quat_w = robot.data.root_quat_w[env_ids]
    obj_pos_b, _ = subtract_frame_transforms(robot_pos_w, robot_quat_w, obj_pos_w)

    # Object orientation in robot root frame (quaternion)
    obj_quat_w = obj.data.root_quat_w[env_ids]
    robot_quat_w_inv = torch.cat([robot_quat_w[:, 0:1], -robot_quat_w[:, 1:]], dim=1)
    obj_quat_b = quat_mul(quat_mul(robot_quat_w_inv, obj_quat_w), robot_quat_w)

    # Allocate buffers if first use
    if not hasattr(env, "_object_initial_pos_b"):
        env._object_initial_pos_b = torch.zeros(env.num_envs, 3, device=env.device)
    if not hasattr(env, "_object_initial_quat_b"):
        env._object_initial_quat_b = torch.zeros(env.num_envs, 4, device=env.device)

    # Write for just the resetting envs
    env._object_initial_pos_b[env_ids] = obj_pos_b
    env._object_initial_quat_b[env_ids] = obj_quat_b


def cache_ee_initial_pose(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
):
    """Cache the end-effector's initial position expressed in the robot's root frame."""

    robot: Articulation | RigidObject = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    ee_pos_w = ee_frame.data.target_pos_w[env_ids, 0, :]
    robot_pos_w = robot.data.root_pos_w[env_ids]
    robot_quat_w = robot.data.root_quat_w[env_ids]
    ee_pos_b, _ = subtract_frame_transforms(robot_pos_w, robot_quat_w, ee_pos_w)

    if not hasattr(env, "_ee_initial_pos_b"):
        env._ee_initial_pos_b = torch.zeros(env.num_envs, 3, device=env.device)
    if not hasattr(env, "_prev_ee_pos_b"):
        env._prev_ee_pos_b = torch.zeros(env.num_envs, 3, device=env.device)

    env._ee_initial_pos_b[env_ids] = ee_pos_b
    env._prev_ee_pos_b[env_ids] = ee_pos_b
