from isaaclab.sensors.camera import TiledCameraCfg
import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


GO2_CUSTOM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="../Assets/Collected_go2/go2.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.46),
        joint_pos={
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            "F[L,R]_calf_joint": -1.5,
            "R[L,R]_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=23.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)

FRONT_DEPTHCAMERA_CFG = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/camera/Camera_front",
    update_period=1/50,
    width=80,    # 1280
    height=45,    # 720
    data_types=["distance_to_image_plane"],  # RGB와 depth 데이터
    spawn=None,
)

UNDER_FRONT_DEPTHCAMERA_CFG = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/camera/Camera_under_front",
    update_period=1/50,
    width=80,    # 1280
    height=45,    # 720
    data_types=["distance_to_image_plane"],  # RGB와 depth 데이터
    spawn=None,
)

UNDER_REAR_DEPTHCAMERA_CFG = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/camera/Camera_under_rear",
    update_period=1/50,
    width=80,    # 1280
    height=45,    # 720
    data_types=["distance_to_image_plane"],  # RGB와 depth 데이터
    spawn=None,
)