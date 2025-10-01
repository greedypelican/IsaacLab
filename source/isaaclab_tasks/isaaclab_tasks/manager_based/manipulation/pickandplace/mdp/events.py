import torch

from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms, quat_mul
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv

# Global phase flags (per environment)
phase_flags = {}
GRASP_THRESHOLD = 1.0
RELEASE_THRESHOLD = 0.05
ASCEND_Z_POS = 0.15
ASCEND_Z_TOL = 0.03
PLANAR_TOL = 0.03
READY_JOINT_TOL = 0.2

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
        phase_flags["phase6_complete"] = torch.zeros(new_size, dtype=torch.bool, device=env.device)
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
    phase_flags["phase6_complete"][env_ids] = False


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
    move_command = env.command_manager.get_command("move")
    target_command = env.command_manager.get_command("target")

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

    # Phase 2: ee_frame과 object가 가깝고 grasping 상태 (ROBOT ROOT FRAME)
    ee_pos_w = ee_frame.data.target_pos_w[env_ids, 0, :]  # (N,3)
    object_pos = object.data.root_pos_w[env_ids]          # (N,3)
    ee_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w[env_ids], robot.data.root_quat_w[env_ids], ee_pos_w
    )
    obj_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w[env_ids], robot.data.root_quat_w[env_ids], object_pos
    )
    distance_ee_object = torch.norm(ee_pos_b - obj_pos_b, dim=1)
    phase1_condition = (distance_ee_object < 0.03) & grasping
    phase_flags["phase1_complete"][env_ids] = phase_flags["phase1_complete"][env_ids] | phase1_condition

    # Phase 2 target: initial object position lifted by offset (robot frame)
    if hasattr(env, "_object_initial_pos_b"):
        ascend_target = env._object_initial_pos_b[env_ids].clone()
    else:
        ascend_target = obj_pos_b.clone()
    ascend_target[:, 2] = ASCEND_Z_POS

    obj_z = obj_pos_b[:, 2]
    ascend_z = ascend_target[:, 2]
    ascend_z_reached = torch.abs(obj_z - ascend_z) < ASCEND_Z_TOL

    phase2_condition = ascend_z_reached & grasping
    phase_flags["phase2_complete"][env_ids] = phase_flags["phase2_complete"][env_ids] | (
        phase2_condition & phase_flags["phase1_complete"][env_ids]
    )

    # Phase 3: object가 move command와 가깝고 grasping 상태 (ROBOT ROOT FRAME)
    move_pos_b = move_command[env_ids, :3]
    move_xy_err = torch.norm(move_pos_b[:, :2] - obj_pos_b[:, :2], dim=1)
    phase3_condition = ascend_z_reached & (move_xy_err < PLANAR_TOL) & grasping
    phase_flags["phase3_complete"][env_ids] = phase_flags["phase3_complete"][env_ids] | (
        phase3_condition & phase_flags["phase2_complete"][env_ids]
    )

    # Phase 4: object가 target command 상부와 가깝고 grasping 상태 (ROBOT ROOT FRAME)
    target_pos_b = target_command[env_ids, :3]
    target_pre_descend = target_pos_b.clone()
    target_pre_descend[:, 2] = ASCEND_Z_POS
    pre_target_xy_err = torch.norm(target_pre_descend[:, :2] - obj_pos_b[:, :2], dim=1)
    pre_target_z = target_pre_descend[:, 2]
    pre_target_z_reached = torch.abs(obj_z - pre_target_z) < ASCEND_Z_TOL
    phase4_condition = pre_target_z_reached & (pre_target_xy_err < PLANAR_TOL) & grasping
    phase_flags["phase4_complete"][env_ids] = phase_flags["phase4_complete"][env_ids] | (
        phase4_condition & phase_flags["phase3_complete"][env_ids]
    )

    # Phase 5: object가 target command와 가깝고 grasping 상태 (ROBOT ROOT FRAME)
    target_xy_err = torch.norm(target_pos_b[:, :2] - obj_pos_b[:, :2], dim=1)
    target_z_err = torch.abs(obj_z - target_pos_b[:, 2])
    phase5_condition = (target_xy_err < PLANAR_TOL) & (target_z_err < ASCEND_Z_TOL) & grasping
    prev_phase5 = phase_flags["phase5_complete"][env_ids].clone()
    update_mask = (phase5_condition & phase_flags["phase4_complete"][env_ids]) & ~prev_phase5
    phase_flags["phase5_complete"][env_ids] = prev_phase5 | (
        phase5_condition & phase_flags["phase4_complete"][env_ids]
    )

    # Cache object position in ROBOT ROOT FRAME at the first time phase5 is achieved
    if not hasattr(env, "_object_pos_b_at_phase5"):
        env._object_pos_b_at_phase5 = torch.zeros(env.num_envs, 3, device=env.device)
    if torch.any(update_mask):
        new_env_ids = env_ids[update_mask]
        obj_pos_b_cache, _ = subtract_frame_transforms(
            robot.data.root_pos_w[new_env_ids], robot.data.root_quat_w[new_env_ids], object_pos[update_mask]
        )
        env._object_pos_b_at_phase5[new_env_ids] = obj_pos_b_cache

    # Phase 6: releasing 상태이면서 기본 관절 자세 근처에 도달하면 ready 단계로 전환
    joint_ids, _ = robot.find_joints(["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "left_outer_knuckle_joint"])
    if len(joint_ids) > 0:
        joint_pos = robot.data.joint_pos[env_ids][:, joint_ids]
        default_joint_pos = robot.data.default_joint_pos[env_ids][:, joint_ids]
        joint_deviation = torch.linalg.norm(joint_pos - default_joint_pos, dim=1)
        phase6_condition = (joint_deviation < READY_JOINT_TOL) & releasing
    else:
        phase6_condition = releasing

    phase_flags["phase6_complete"][env_ids] = phase_flags["phase6_complete"][env_ids] | (
        phase6_condition & phase_flags["phase5_complete"][env_ids]
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
