from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg


# local tf -> world tf
def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    w = quat[..., 0]
    q_xyz = quat[..., 1:4]  # (x, y, z)
    t = 2.0 * torch.cross(q_xyz, vec, dim=-1)
    v_prime = vec + w.unsqueeze(-1) * t + torch.cross(q_xyz, t, dim=-1)
    return v_prime

# world tf -> local tf
def quat_conjugate_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """
    Rotate vec by the conjugate of q.
    For quaternion in format (w, x, y, z), the conjugate is (w, -x, -y, -z).
    """
    q_conj = torch.cat([quat[..., 0:1], -quat[..., 1:4]], dim=-1)
    v_prime = quat_apply(q_conj, vec)
    return v_prime


@configclass
class EventCfg:
    robot_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.7, 1.3),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.5, 2.0),
            "damping_distribution_params": (0.5, 2.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    robot_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.2, 2.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    robot_joint_pos_limits = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint_2", "joint_3", "joint_5"]),
            "lower_limit_distribution_params": (0.0, 0.01),
            "upper_limit_distribution_params": (0.0, 0.01),
            "operation": "add",
            "distribution": "gaussian",
        },
    )


@configclass
class KinovaReachEnvCfg(DirectRLEnvCfg):
    """
    Configuration for the Kinova reaching environment.
    """
    # Environment
    episode_length_s = 8
    decimation = 6
    action_space = 14
    observation_space = 23
    state_space = 0

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.0, replicate_physics=True)

    # Robot
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"/home/haryun/workspace/IsaacSim/Robot/Collected_kinova_robot2/kinova_robot4.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,  # Enable self-collision
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "joint_1": 0.0,
                "joint_2": 0.262,
                "joint_3": -2.269,
                "joint_4": 0.0,
                "joint_5": 0.960,
                "joint_6": 1.571,
                "robotiq_85_left_knuckle_joint": 0.0,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "kinova_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["joint_1", "joint_2", "joint_3"],
                effort_limit_sim=39.0,
                velocity_limit_sim=0.7,
                stiffness=40.0,
                damping=1.0,
            ),
            "kinova_forearm": ImplicitActuatorCfg(
                joint_names_expr=["joint_4", "joint_5", "joint_6"],
                effort_limit_sim=9.0,
                velocity_limit_sim=0.7,
                stiffness=15.0,
                damping=0.5,
            ),
            "kinova_hand": ImplicitActuatorCfg(
                joint_names_expr=["robotiq_85_left_knuckle_joint"],
                effort_limit_sim=10.0,
                velocity_limit_sim=0.2,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )

    # Markers
    markers = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Markers")
    markers.markers["frame"].scale = (0.07, 0.07, 0.07)

    # scales for rewards and penalties
    DIST_REWARD_SCALE = 1.0 
    ORI_REWARD_SCALE = 1.0
    ACTION_PENALTY_SCALE = 0.015
    DELTA_PENALTY_SCALE = 0.1
    ACCEL_PENALTY_SCALE = 0.03
    TORQUE_PENALTY_SCALE = 0.0003

    # Parameters for stuck condition:
    stuck_distance_threshold: float = 0.2  # If tip is farther than this from target, it is considered far.
    stuck_velocity_threshold: float = 0.3  # If joint velocity norm is below this, tip is considered not moving.

    # Stable termination
    stable_frames_required = 10

    events: EventCfg = EventCfg()


class KinovaReachEnv(DirectRLEnv):
    """
    Reaching environment with incremental (delta) actions and stable termination.
    """
    cfg: KinovaReachEnvCfg

    def __init__(self, cfg: KinovaReachEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(self.device)
        self.effort_penalty = torch.zeros(self.num_envs, device=self.device)

        # Target position (per environment)
        self.target_pos = torch.tensor([0.5, 0.0, 0.3], device=self.device).repeat(self.num_envs, 1)

        # Action history for delta penalty and acceleration penalty
        self.actions_last = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.actions_prev = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)

        # Counter for stable frames at goal
        self.time_at_goal = torch.zeros(self.num_envs, device=self.device)

        # Initialize self.out_of_bounds as a tensor of zeros.
        self.out_of_bounds = torch.zeros(self.num_envs, device=self.device)

        self.stuck_condition = torch.zeros(self.num_envs, device=self.device)
        
        # Flag to indicate if the first reset has been done (training start)
        self.initial_reset_done = False

        self.last_episode_distance = torch.zeros(self.num_envs, device=self.device)

    def _setup_scene(self) -> None:
        """Add the robot, ground plane, table, and markers to the scene, then clone envs."""
        self._robot = Articulation(self.cfg.robot)
        self.scene.target_markers = VisualizationMarkers(self.cfg.markers)
        self.scene.tip_markers = VisualizationMarkers(self.cfg.markers)
        spawn_ground_plane("/World/ground", GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))

        # Table
        table_cfg = sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        )
        table_cfg.func(
            "/World/envs/env_.*/Table",
            table_cfg,
            translation=(0.55, 0.0, 0.0),
            orientation=(0.70711, 0.0, 0.0, 0.70711),
        )
        # Replicate the environment
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self._robot

        # Light
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _get_target(self) -> torch.Tensor:
        root_pos = self._robot.data.root_state_w[:, :3]
        target = self.target_pos + root_pos
        return target

    def _get_gripper_tip_pos(self) -> torch.Tensor:
        # Get gripper's orientation and translation using the correct attribute.
        gripper_pos = self._robot.data.body_pos_w[:, self._robot.find_bodies("robotiq_85_base_link")[0][0]]
        gripper_quat = self._robot.data.body_quat_w[:, self._robot.find_bodies("robotiq_85_base_link")[0][0]]
        tip_pos = gripper_pos
        return gripper_quat, tip_pos

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        target_pos = torch.clamp(actions, self.robot_dof_lower_limits, self.robot_dof_upper_limits)     
        self.actions = target_pos
        target = self._get_target()
        self.scene.target_markers.visualize(target)
        _, tip_pos = self._get_gripper_tip_pos()
        self.scene.tip_markers.visualize(tip_pos)

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self.actions)

    def _get_observations(self) -> dict:
        indices = [0, 1, 2, 3, 4, 5, 7]
        joint_pos_7 = self._robot.data.joint_pos[:, indices]
        joint_vel_7 = self._robot.data.joint_vel[:, indices]
        _, tip_pos = self._get_gripper_tip_pos()
        target = self._get_target()
        to_target = target - tip_pos

        obs = torch.cat((joint_pos_7, joint_vel_7, tip_pos, target, to_target), dim=-1)
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    def _get_rewards(self) -> torch.Tensor:
        target = self._get_target()
        gripper_quat, tip_pos = self._get_gripper_tip_pos()

        # ---------- reward ---------- #
        # distance
        dist = torch.norm(abs(tip_pos - target), dim=-1)
        self.last_episode_distance = dist.detach()
        raw_dist_reward = torch.exp(-2.3 * (dist - 0.176)) - 0.15
        dist_reward = self.cfg.DIST_REWARD_SCALE * raw_dist_reward

        # orientation
        gripper_up = torch.tensor([0.0, 0.0, 1.0], device=self.device)\
            .unsqueeze(0).repeat(self.num_envs, 1)
        gripper_front = torch.tensor([1.0, 0.0, 0.0], device=self.device)\
            .unsqueeze(0).repeat(self.num_envs, 1)
        gripper_up_world = quat_apply(gripper_quat, gripper_up)
        gripper_front_world = quat_apply(gripper_quat, gripper_front)
        global_up = torch.tensor([0.0, 0.0, 1.0], device=self.device)\
            .unsqueeze(0).repeat(self.num_envs, 1)
        global_front = torch.tensor([1.0, 0.0, 0.0], device=self.device)\
            .unsqueeze(0).repeat(self.num_envs, 1)
        
        down_alignment = torch.sum(gripper_front_world * global_up, dim=-1)
        front_alignment = torch.sum(gripper_up_world * global_front, dim=-1)
        alignment = (1.5 * (1 - (down_alignment + 1) / 2) + ((front_alignment + 1) / 2)) / 2.5

        raw_orientation_reward = torch.exp(1.5 * (alignment -1.4)) -0.12
        orientation_reward = self.cfg.ORI_REWARD_SCALE * raw_orientation_reward
        # ---------- reward ---------- #


        # ---------- penalty ---------- #
        # torque
        torques = self._robot.data.applied_torque
        indices = [self._robot.find_joints(jn)[0][0] for jn in ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]]
        used_torques = torques[:, indices]
        torque_penalty = self.cfg.TORQUE_PENALTY_SCALE * torch.sum(used_torques**2, dim=-1)
        
        # action
        action_penalty = torch.sum(self.actions**2, dim=-1) * self.cfg.ACTION_PENALTY_SCALE

        # delta(stepwise change)
        delta_actions = self.actions - self.actions_last
        delta_penalty = self.cfg.DELTA_PENALTY_SCALE * torch.sum(delta_actions**2, dim=-1)

        # acceleration
        acceleration = self.actions - 2 * self.actions_last + self.actions_prev
        accel_penalty = self.cfg.ACCEL_PENALTY_SCALE * torch.sum(acceleration**2, dim=-1)
        
        # out of bounds
        self.out_of_bounds = (tip_pos[:, 2] < 0.15).float()
        out_of_bounds_penalty = 10.0 * self.out_of_bounds

        # stuck
        joint_vel_norm = torch.norm(self._robot.data.joint_vel, dim=-1)
        self.stuck_condition = ((dist > self.cfg.stuck_distance_threshold) & (joint_vel_norm < self.cfg.stuck_velocity_threshold)).float()
        stuck_penalty = 10.0 * self.stuck_condition
        # ---------- penalty ---------- #


        reward = dist_reward + orientation_reward \
                - torque_penalty -action_penalty - delta_penalty - accel_penalty - out_of_bounds_penalty - stuck_penalty
        self.extras["log"] = {
            "distance": dist.mean(),
            "down_alignment": down_alignment,
            "front_alignment": front_alignment,
            "dist_reward": dist_reward.mean(),
            "orientation_reward": orientation_reward.mean(),
            "torque_penalty": torque_penalty.mean(),
            "action_penalty": action_penalty.mean(),
            "delta_penalty": delta_penalty.mean(),
            "accel_penalty": accel_penalty.mean(),
            "out_of_bounds_penalty": out_of_bounds_penalty.mean(),
            "stuck_penalty": stuck_penalty.mean(),
        }

        self.actions_prev = self.actions_last.clone()
        self.actions_last = self.actions.clone()

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        End episode if:
          - The end-effector is stable (i.e. close to target with low joint velocities),
          - or if the time limit is reached,
          - or if the gripper's tip goes below z = 0,
          - or if the tip is far from the target and not moving (stuck).
        """
        target = self._get_target()
        _, tip_pos = self._get_gripper_tip_pos()
        distance = torch.norm(abs(tip_pos - target), dim=-1)
        joint_vel_norm = torch.norm(self._robot.data.joint_vel, dim=-1)

        dist_thresh = 0.01
        vel_thresh = 0.03
        is_stable = (distance < dist_thresh) & (joint_vel_norm < vel_thresh)
        self.time_at_goal[is_stable] += 1
        self.time_at_goal[~is_stable] = 0

        # Stuck condition for termination.
        stuck_condition = (distance > self.cfg.stuck_distance_threshold) & (joint_vel_norm < self.cfg.stuck_velocity_threshold)
        terminated = (self.time_at_goal >= self.cfg.stable_frames_required) | (tip_pos[:, 2] < 0.15) | stuck_condition
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        """
        Reset states for given environments:
          - Randomize initial joint positions and target positions,
          - Reset counters and action history.
        """
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        super()._reset_idx(env_ids)

        default_joint_pos = self._robot.data.default_joint_pos[env_ids].clone()

        # On the first reset (training starting), force joint_pos to default_joint_pos.
        if not self.initial_reset_done:
            joint_pos = default_joint_pos
            self.initial_reset_done = True
        else:
            random_joints = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
            random_joint_indices = [self._robot.find_joints(jn)[0][0] for jn in random_joints]
            random_offsets = sample_uniform(-0.2, 0.2, (len(env_ids), len(random_joint_indices)), self.device)

            for i, j in enumerate(random_joint_indices):
                default_joint_pos[:, j] += random_offsets[:, i]

            # Use vectorized selection based on self.out_of_bounds for env_ids.
            mask = (
                (self.out_of_bounds[env_ids].unsqueeze(-1) == 1.0) 
                | (self.stuck_condition[env_ids].unsqueeze(-1) == 1.0)
                | (self.last_episode_distance[env_ids] >= 0.03).unsqueeze(-1)
                )
            joint_pos = torch.where(mask, default_joint_pos,
                                    self._robot.data.joint_pos[env_ids].clone())
            
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = self._robot.data.joint_vel[env_ids]

        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        x_min, x_max = 0.1, 0.5
        y_min, y_max = -0.3, 0.3
        z_min, z_max = 0.2, 0.5

        self.target_pos[env_ids, 0] = sample_uniform(x_min, x_max, (len(env_ids), ), self.device)
        self.target_pos[env_ids, 1] = sample_uniform(y_min, y_max, (len(env_ids), ), self.device)
        self.target_pos[env_ids, 2] = sample_uniform(z_min, z_max, (len(env_ids), ), self.device)
        
        self.time_at_goal[env_ids] = 0
        self.actions_last[env_ids] = 0.0