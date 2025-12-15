from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


RIGHT_LEROBOT_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Right_Robot",
    articulation_root_prim_path="/right_base",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/jhr/Isaac/Assets/Collected_bimanual_lerobot/so100_right_flat.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "right_shoulder_pan": 0.0,
            "right_shoulder_lift": 0.0,
            "right_elbow_flex": 0.0,
            "right_wrist_flex": 0.0,
            "right_wrist_roll": 0.0,
            "right_gripper": 0.0,
        },
        pos=(-0.15, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    actuators={
        "right_gripper": ImplicitActuatorCfg(
            joint_names_expr=["right_gripper"],
            effort_limit_sim=10,
            velocity_limit_sim=10,
            stiffness=17.8,
            damping=0.60,
        ),
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=["right_shoulder_pan", "right_shoulder_lift", "right_elbow_flex", "right_wrist_flex", "right_wrist_roll"],
            effort_limit_sim=10,
            velocity_limit_sim=10,
            stiffness=17.8,
            damping=0.60,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)


LEFT_LEROBOT_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Left_Robot",
    articulation_root_prim_path="/left_base",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/jhr/Isaac/Assets/Collected_bimanual_lerobot/so100_left_flat.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "left_shoulder_pan": 0.0,
            "left_shoulder_lift": 0.0,
            "left_elbow_flex": 0.0,
            "left_wrist_flex": 0.0,
            "left_wrist_roll": 0.0,
            "left_gripper": 0.0,
        },
        pos=(0.15, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    actuators={
        "left_gripper": ImplicitActuatorCfg(
            joint_names_expr=["left_gripper"],
            effort_limit_sim=10,
            velocity_limit_sim=10,
            stiffness=17.8,
            damping=0.60,
        ),
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=["left_shoulder_pan", "left_shoulder_lift", "left_elbow_flex", "left_wrist_flex", "left_wrist_roll"],
            effort_limit_sim=10,
            velocity_limit_sim=10,
            stiffness=17.8,
            damping=0.60,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)


# joint limit written in USD (degree)
LEROBOT_JOINT_LIMLITS = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 90.0),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-160.0, 160.0),
    "gripper": (-10, 100.0),
}

# motor limit written in real device (normalized to related range)
LEROBOT_MOTOR_LIMITS = {
    'shoulder_pan': (-100.0, 100.0),
    'shoulder_lift': (-100.0, 100.0),
    'elbow_flex': (-100.0, 100.0),
    'wrist_flex': (-100.0, 100.0),
    'wrist_roll': (-100.0, 100.0),
    'gripper': (0.0, 100.0),
}

LEROBOT_REST_POSE_RANGE = {
    "shoulder_pan": (0 - 30.0, 0 + 30.0),  # 0 degree
    "shoulder_lift": (-100.0 - 30.0, -100.0 + 30.0),  # -100 degree
    "elbow_flex": (90.0 - 30.0, 90.0 + 30.0),  # 90 degree
    "wrist_flex": (50.0 - 30.0, 50.0 + 30.0),  # 50 degree
    "wrist_roll": (0.0 - 30.0, 0.0 + 30.0),  # 0 degree
    "gripper": (-10.0 - 30.0, -10.0 + 30.0),  # -10 degree
}
