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
    
    # 특정 환경들만 리셋
    phase_flags["phase1_complete"][env_ids] = False
    phase_flags["phase2_complete"][env_ids] = False
    phase_flags["phase3_complete"][env_ids] = False
    phase_flags["phase4_complete"][env_ids] = False


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

    # Phase 3: object가 ascend command와 가깝고 grasping 상태 (ROBOT ROOT FRAME)
    des_pos_b_ascend = ascend_command[env_ids, :3]                    # already in robot frame
    distance_ascend = torch.norm(des_pos_b_ascend - obj_pos_b, dim=1)
    phase2_condition = (distance_ascend < 0.03) & grasping
    phase_flags["phase2_complete"][env_ids] = phase_flags["phase2_complete"][env_ids] | (phase2_condition & phase_flags["phase1_complete"][env_ids])

    # Phase 4: object가 descend command와 가깝고 grasping 상태 (ROBOT ROOT FRAME)
    des_pos_b_descend = descend_command[env_ids, :3]                  # already in robot frame
    distance_descend = torch.norm(des_pos_b_descend - obj_pos_b, dim=1)
    phase3_condition = (distance_descend < 0.03) & grasping
    prev_phase3 = phase_flags["phase3_complete"][env_ids].clone()
    update_mask = (phase3_condition & phase_flags["phase2_complete"][env_ids]) & ~prev_phase3
    phase_flags["phase3_complete"][env_ids] = prev_phase3 | (phase3_condition & phase_flags["phase2_complete"][env_ids])

    # Cache object position in ROBOT ROOT FRAME at the first time phase3 is achieved
    if not hasattr(env, "_object_pos_b_at_phase3"):
        env._object_pos_b_at_phase3 = torch.zeros(env.num_envs, 3, device=env.device)
    if torch.any(update_mask):
        new_env_ids = env_ids[update_mask]
        # compute object position in robot frame for those envs
        obj_pos_b, _ = subtract_frame_transforms(
            robot.data.root_pos_w[new_env_ids], robot.data.root_quat_w[new_env_ids], object_pos[update_mask]
        )
        env._object_pos_b_at_phase3[new_env_ids] = obj_pos_b

    # Phase 5: initial joint state 와 가깝고 releasing 상태
    joint_ids, _ = robot.find_joints(["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "left_outer_knuckle_joint"])
    # Calculate deviation using L2 norm
    joint_deviation = torch.linalg.norm(robot.data.joint_pos[:, joint_ids] - robot.data.default_joint_pos[:, joint_ids], dim=1)
    phase4_condition = (joint_deviation < 0.1) & releasing
    # phase3가 완료된 환경들에서만 phase4 완료 업데이트
    phase_flags["phase4_complete"][env_ids] = phase_flags["phase4_complete"][env_ids] | (phase4_condition & phase_flags["phase3_complete"][env_ids])


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