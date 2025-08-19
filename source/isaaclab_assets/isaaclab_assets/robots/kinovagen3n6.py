import math
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

KINOVAGEN3N6_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot", 
    spawn=sim_utils.UsdFileCfg(
        usd_path="../Assets/Collected_kinovagen3n6/kinovagen3n6.usd", 
        activate_contact_sensors=False, 
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
            "joint_2": math.pi/12, # 15
            "joint_3": -2*math.pi/3, # -120
            "joint_4": 0.0, 
            "joint_5": -math.pi/4, # -45
            "joint_6": math.pi/2, # 90
            # "joint_1": 0.0, 
            # "joint_2": 0.262, 
            # "joint_3": -2.269, 
            # "joint_4": 0.0, 
            # "joint_5": 0.960, 
            # "joint_6": 1.571, 
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
            velocity_limit_sim=1.39, 
            stiffness=40.0, 
            damping=1.0, 
        ), 
        "kinova_forearm": ImplicitActuatorCfg(
            joint_names_expr=["joint_4", "joint_5", "joint_6"], 
            effort_limit_sim=9.0, 
            velocity_limit_sim=1.22, 
            stiffness=15.0, 
            damping=0.5, 
        ), 
        "robotiq_inner_finger": ImplicitActuatorCfg(
            joint_names_expr=["left_inner_finger_joint", "right_inner_finger_joint"], 
            effort_limit_sim=0.5, 
            velocity_limit_sim=10.0, 
            stiffness=0.17, 
            damping=0.0002, 
        ), 
        "robotiq_outer_finger": ImplicitActuatorCfg(
            joint_names_expr=["left_outer_finger_joint", "right_outer_finger_joint"], 
            velocity_limit_sim=10.0, 
            stiffness=1.7, 
            damping=0.0002, 
        ), 
        "robotiq_inner_finger_knuckle": ImplicitActuatorCfg(
            joint_names_expr=["left_inner_finger_knuckle_joint", "right_inner_finger_knuckle_joint"], 
            velocity_limit_sim=10.0, 
            stiffness=0.17, 
            damping=0.0002, 
        ), 
        "robotiq_outer_knuckle": ImplicitActuatorCfg(
            joint_names_expr=["left_outer_knuckle_joint", "right_outer_knuckle_joint"], 
            velocity_limit_sim=10.0, 
            effort_limit_sim=17.0, 
            stiffness=17.0, 
            damping=0.0002, 
        ), 
        # "robotiq_gripper": ImplicitActuatorCfg(
        #     joint_names_expr=["left_inner_finger_joint", "right_inner_finger_joint", 
        #                       "left_outer_finger_joint", "right_outer_finger_joint", 
        #                       "left_inner_finger_knuckle_joint", "right_inner_finger_knuckle_joint", 
        #                       "left_outer_knuckle_joint", "right_outer_knuckle_joint"], 
        #     effort_limit_sim=20.0, 
        #     velocity_limit_sim=10.0, 
        #     stiffness=5.0, 
        #     damping=0.05, 
        # ), 
    }, 
)