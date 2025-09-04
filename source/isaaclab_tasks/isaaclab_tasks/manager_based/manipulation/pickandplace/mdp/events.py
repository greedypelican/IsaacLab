import torch

from isaaclab.utils.math import combine_frame_transforms
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv

# Global phase flags (per environment)
phase_flags = {}

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

    robot: RigidObject = env.scene[robot_cfg.name]
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
    left_contact = (left_force_magnitudes > 1.0)
    right_contact = (right_force_magnitudes > 1.0)
    grasping = left_contact & right_contact

    left_grasp_contact = (left_force_magnitudes > 1.0)
    right_grasp_contact = (right_force_magnitudes > 1.0)
    grasping = left_grasp_contact & right_grasp_contact
    left_release_contact = (left_force_magnitudes > 1.0) #0.01
    right_release_contact = (right_force_magnitudes > 1.0) #0.01
    releasing = ~(left_release_contact & right_release_contact)

    # Phase 1: ee_frame과 object가 가깝고 grasping 상태
    ee_pos = ee_frame.data.target_pos_w[env_ids, 0, :]  # ee position
    object_pos = object.data.root_pos_w[env_ids]
    distance_ee_object = torch.norm(ee_pos - object_pos, dim=1)
    phase1_condition = (distance_ee_object < 0.03) & grasping
    phase_flags["phase1_complete"][env_ids] = phase_flags["phase1_complete"][env_ids] | phase1_condition

    # Phase 2: object가 ascend command와 가깝고 grasping 상태
    des_pos_b_ascend = ascend_command[env_ids, :3]
    des_pos_w_ascend, _ = combine_frame_transforms(robot.data.root_pos_w[env_ids], robot.data.root_quat_w[env_ids], des_pos_b_ascend)
    distance_ascend = torch.norm(des_pos_w_ascend - object_pos, dim=1)
    phase2_condition = (distance_ascend < 0.03) & grasping
    # phase1이 완료된 환경들에서만 phase2 업데이트
    phase_flags["phase2_complete"][env_ids] = phase_flags["phase2_complete"][env_ids] | (phase2_condition & phase_flags["phase1_complete"][env_ids])

    # Phase 3: object가 descend command와 가깝고 grasping 상태
    des_pos_b_descend = descend_command[env_ids, :3]
    des_pos_w_descend, _ = combine_frame_transforms(robot.data.root_pos_w[env_ids], robot.data.root_quat_w[env_ids], des_pos_b_descend)
    distance_descend = torch.norm(des_pos_w_descend - object_pos, dim=1)
    phase3_condition = (distance_descend < 0.03) & grasping
    # phase2가 완료된 환경들에서만 phase3 업데이트
    phase_flags["phase3_complete"][env_ids] = phase_flags["phase3_complete"][env_ids] | (phase3_condition & phase_flags["phase2_complete"][env_ids])

    # Phase 4: object가 descend command와 가깝고 release 상태
    phase4_condition = (distance_descend < 0.03) & releasing  # grasping 해제
    # phase3가 완료된 환경들에서만 phase4 업데이트
    phase_flags["phase4_complete"][env_ids] = phase_flags["phase4_complete"][env_ids] | (phase4_condition & phase_flags["phase3_complete"][env_ids])