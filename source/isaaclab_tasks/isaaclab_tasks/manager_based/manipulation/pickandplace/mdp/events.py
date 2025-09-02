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
    
    # 특정 환경들만 리셋
    phase_flags["phase1_complete"][env_ids] = False
    phase_flags["phase2_complete"][env_ids] = False
    phase_flags["phase3_complete"][env_ids] = False
    
    # Reset object dropping tracking for reset environments
    from .terminations import _prev_grasped_above_threshold
    if env.num_envs in _prev_grasped_above_threshold:
        _prev_grasped_above_threshold[env.num_envs][env_ids] = False


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
    object: RigidObject = env.scene[object_cfg.name]
    command_1 = env.command_manager.get_command("ascend")
    command_2 = env.command_manager.get_command("descend")

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

    # distance between object and pick command (env_ids에 해당하는 환경들만)
    des_pos_b_1 = command_1[env_ids, :3]
    des_pos_w_1, _ = combine_frame_transforms(robot.data.root_pos_w[env_ids], robot.data.root_quat_w[env_ids], des_pos_b_1)
    distance_1 = torch.norm(des_pos_w_1 - object.data.root_pos_w[env_ids], dim=1)

    des_pos_b_2 = command_2[env_ids, :3]
    des_pos_w_2, _ = combine_frame_transforms(robot.data.root_pos_w[env_ids], robot.data.root_quat_w[env_ids], des_pos_b_2)
    distance_2 = torch.norm(des_pos_w_2 - object.data.root_pos_w[env_ids], dim=1)

    # phase1_condition과 phase2_condition은 env_ids 크기와 동일
    phase1_condition = (distance_1 < 0.05) & grasping
    phase_flags["phase1_complete"][env_ids] = phase_flags["phase1_complete"][env_ids] | phase1_condition

    # phase2_condition: phase1이 완료되고, distance2가 가깝고, grasping 상태
    phase2_condition = (distance_2 < 0.05) & grasping
    # phase1이 완료된 환경들에서만 phase2 업데이트
    phase_flags["phase2_complete"][env_ids] = phase_flags["phase2_complete"][env_ids] | (phase2_condition & phase_flags["phase1_complete"][env_ids])


def log_phase_metrics(env: ManagerBasedRLEnv, env_ids: torch.Tensor):
    """Log phase metrics to environment extras."""
    if not phase_flags:
        return
    
    env.extras["Metrics/phase_0_count"] = torch.sum(~phase_flags["phase1_complete"]).item()
    env.extras["Metrics/phase_1_count"] = torch.sum(phase_flags["phase1_complete"] & ~phase_flags["phase2_complete"]).item()
    env.extras["Metrics/phase_2_count"] = torch.sum(phase_flags["phase2_complete"]).item()