import math
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

KINOVAGEN3N6_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot", 
    spawn=sim_utils.UsdFileCfg(
        usd_path="../Assets/Collected_kinovagen3n6/kinovagen3n6_v2.usd", 
        activate_contact_sensors=True, 
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False, 
            max_depenetration_velocity=5.0, 
        ), 
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, 
            solver_position_iteration_count=12, 
            solver_velocity_iteration_count=1, 
        ), 
    ), 
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint_1": 0.0, 
            "joint_2": -math.pi/12, # -15
            "joint_3": -2*math.pi/3, # -120
            "joint_4": 0.0, 
            "joint_5": -math.pi/4, # -45
            "joint_6": math.pi/2, # 90
            "left_inner_finger_joint": 0.0, 
            "right_inner_finger_joint": 0.0, 
            "left_outer_finger_joint": 0.0 , 
            "right_outer_finger_joint": 0.0, 
            "left_inner_finger_knuckle_joint": 0.0, 
            "right_inner_finger_knuckle_joint": 0.0, 
            "left_outer_knuckle_joint": 0.0, 
            "right_outer_knuckle_joint": 0.0, 
        }, 
        pos=(0.0, 0.0, 0.0), 
        rot=(1.0, 0.0, 0.0, 0.0), 
    ), 
    actuators={
        "kinova_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["joint_1", "joint_2", "joint_3"], 
            effort_limit_sim=39.0, 
            velocity_limit_sim=4*math.pi/9, 
            stiffness=40.0, 
            damping=1.0, 
        ), 
        "kinova_forearm": ImplicitActuatorCfg(
            joint_names_expr=["joint_4", "joint_5", "joint_6"], 
            effort_limit_sim=9.0, 
            velocity_limit_sim=7*math.pi/18, 
            stiffness=15.0, 
            damping=0.5, 
        ), 
        
        "gripper_drive": ImplicitActuatorCfg( # finger control joint
            joint_names_expr=["left_outer_knuckle_joint"],  # "right_outer_knuckle_joint" is its mimic joint
            effort_limit_sim=330.0, #16.5 * 20
            velocity_limit_sim=29*math.pi/9, 
            stiffness=3.4, #0.17 * 20
            damping=0.004, #0.0002 * 20
        ),
        "gripper_finger": ImplicitActuatorCfg( # enable the gripper to grasp in a parallel manner
            joint_names_expr=[".*_inner_finger_joint"],
            effort_limit_sim=15.0, # 0.5 * 20
            velocity_limit_sim=29*math.pi/9, 
            stiffness=0.04, # 0.002 * 20
            damping=0.0002, # 0.00001 * 20
        ),
        "gripper_passive": ImplicitActuatorCfg(  # set PD to zero for passive joints in close-loop gripper
            joint_names_expr=[".*_inner_finger_knuckle_joint", ".*_outer_finger_joint", "right_outer_knuckle_joint"],
            effort_limit_sim=0.2, # 0.01 * 20
            velocity_limit_sim=29*math.pi/9,
            stiffness=0.0,
            damping=0.0,
        ),
    }, 
)