from __future__ import annotations

import os

import numpy as np
import torch

from pxr import Gf
from omni.isaac.core.utils.stage import get_current_stage

import omni.isaac.lab.sim as sim_utils

from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensor, ContactSensorCfg
from omni.isaac.lab.sim import SimulationCfg, SimulationContext
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import subtract_frame_transforms
from omni.physx import acquire_physx_interface
import carb.settings

from .fluid_object import FluidObject, FluidObjectCfg
from .suction.suction_controller import SuctionControllerNoTimer


@configclass
class PsmBloodAbsorptionEnvCfg(DirectRLEnvCfg):
    # 环境配置
    episode_length_s = 10
    decimation = 2
    action_space = 3
    state_space = 0
    observation_space = 18

    # 仿真配置
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=2,
        disable_contact_processing=False,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=sim_utils.PhysxCfg(
            gpu_max_particle_contacts=2**22,
        ),
    )

    # 场景配置
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=16,
        env_spacing=3.0,
        replicate_physics=False,
    )

    CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))

    spawn_pos_tissue = Gf.Vec3f(0.0, 0.30, 0.0)
    spawn_pos_fluid = spawn_pos_tissue + Gf.Vec3f(0.0, 0.0, 0.05)
    spawn_pos_glass2 = Gf.Vec3f(0.0, 0.70, 0.01)
    glass2_particle_height = 0.03

    tissue_setup = UsdFileCfg(
        usd_path=f"{CURRENT_PATH}/usd_models/whole_sence.usd",
        scale=(1.0, 1.0, 1.0),
        rigid_props=RigidBodyPropertiesCfg(
            disable_gravity=True,
            kinematic_enabled=True,
        ),
    )

    table_setup = UsdFileCfg(
        usd_path=f"{CURRENT_PATH}/usd_models/table.usd",
        scale=(1.0, 1.0, 1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )
    table_pos = Gf.Vec3f(0.0, 0.0, 0.457)
    table_height_offset = 0.914

    glass2 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Glass2",
        init_state=RigidObjectCfg.InitialStateCfg(pos=spawn_pos_glass2, rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{CURRENT_PATH}/usd_models/Tall_Glass.usd",
            semantic_tags=[("class", "Glass2")],
            scale=(0.01, 0.01, 0.01),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    liquidCfg = FluidObjectCfg()
    liquidCfg.numParticlesX = 10
    liquidCfg.numParticlesY = 10
    liquidCfg.numParticlesZ = 3
    liquidCfg.density = 1060.0
    liquidCfg.particle_mass = 0.001
    liquidCfg.particleSpacing = 0.004
    liquidCfg.viscosity = 3.5

    # PSM机器人配置
    psm_robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/PSM",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{CURRENT_PATH}/usd_models/psm_all.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "psm_rev_joint": 0.0,
                "psm_yaw_joint": 0.0,
                "psm_pitch_back_joint": 0.0,
                "psm_pitch_bottom_joint": 0.0,
                "psm_pitch_end_joint": 0.0,
                "psm_main_insertion_joint": 0.07,
                "suction_tool_pitch_joint": 0.0,
                "suction_tool_end_joint": 0.0,
            },
            pos=(0.0, -0.20, 0.0),
        ),
        actuators={
            "psm": ImplicitActuatorCfg(
                joint_names_expr=[
                    "psm_rev_joint",
                    "psm_yaw_joint",
                    "psm_pitch_back_joint",
                    "psm_pitch_bottom_joint",
                    "psm_pitch_end_joint",
                    "psm_main_insertion_joint",
                    "suction_tool_pitch_joint",
                    "suction_tool_end_joint",
                ],
                effort_limit=12000,
                stiffness=800.0,
                damping=40.0,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )

    # Differential IK 动作配置
    ik_joint_names = (
        # "psm_rev_joint",
        "psm_yaw_joint",
        # "psm_pitch_back_joint",
        # "psm_pitch_bottom_joint",
        "psm_pitch_end_joint",
        "psm_main_insertion_joint",
    )
    tool_joint_names = (
        "suction_tool_pitch_joint",
        "suction_tool_end_joint",
    )
    action_scale_lin = 0.003
    workspace_low_offset = (-0.10, -0.10, -0.02)
    workspace_high_offset = (0.10, 0.10, 0.30)

    # 吸附参数
    psm_tip_body_name = "suction_tool_end_link"
    psm_tip_local_offset = (0.0, -0.011957148076033514, 0.0)
    psm_tip_local_axis = (0.0, -1.0, 0.0)
    tip_contact_force_threshold = 0.5
    height_axis = 2
    height_limit = 0.92
    suction_cone_half_angle_deg = 60.0
    suction_cone_range = 0.07
    suction_force_scale = 0.02
    suction_epsilon = 1e-6
    inlet_radius = 0.008
    inlet_depth = 0.012
    use_body_quat_for_tip_dir = True
    outflow_speed = 0.02
    max_particle_speed = 0.4

    # 奖励参数
    reward_absorb_weight = 6.0
    centroid_progress_weight = 100.0
    centroid_progress_clip = 0.02
    reward_cone_coverage_weight = 0.4
    reward_inlet_coverage_weight = 0.8
    reward_action_weight = 0.02
    reward_time_penalty = 0.005
    reward_task_complete = 10.0
    reward_collision_force_weight = 0.03
    absorbed_delta_ema_alpha = 0.2
    severe_contact_force_threshold = 2.0
    severe_contact_patience = 2

    # 成功条件
    success_absorption_ratio = 0.75
    joint_limit_termination_tolerance = 1e-3

    debug_print_ee_pos = False
    debug_print_ee_pos_interval = 1


class PsmBloodAbsorptionEnv(DirectRLEnv):
    """低维观测的 PSM 粒子吸取环境."""

    cfg: PsmBloodAbsorptionEnvCfg

    def __init__(self, cfg: PsmBloodAbsorptionEnvCfg, render_mode: str | None = None, **kwargs):
        self._tip_contact_sensor = None
        self._ee_jacobi_idx: int | None = None
        super().__init__(cfg, render_mode, **kwargs)

        self.stage = get_current_stage()
        self._suction_controller = SuctionControllerNoTimer(cfg=cfg, num_envs=self.num_envs)

        self._initial_particles_pos = None
        self._initial_particles_vel = None
        self._particles_initialized = False

        self._step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._absorbed_count = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._absorbed_delta = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._absorbed_delta_ema = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._blood_centroid = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self._prev_blood_centroid = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self._blood_centroid_distance = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._prev_blood_centroid_distance = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._valid_in_cone_ratio = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._valid_in_inlet_ratio = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._task_state_dirty = True
        self._task_state_apply_suction = False
        self._severe_contact_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._episode_reward_sums = {
            "absorb_reward": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "centroid_progress_reward": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "cone_coverage_reward": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "inlet_coverage_reward": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "action_penalty": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "collision_force_penalty": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "time_penalty": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "task_complete": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
        }
        self._episode_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._episode_joint_limit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._episode_severe_collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._episode_time_out = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._raw_actions = torch.zeros((self.num_envs, self.cfg.action_space), dtype=torch.float32, device=self.device)

        self._expected_particle_count = float(
            self.cfg.liquidCfg.numParticlesX * self.cfg.liquidCfg.numParticlesY * self.cfg.liquidCfg.numParticlesZ
        )
        self._success_threshold = self._expected_particle_count * float(self.cfg.success_absorption_ratio)

        self._joint_lower_limits = self._psm.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self._joint_upper_limits = self._psm.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        joint_names = list(self._psm.data.joint_names)
        self._joint_name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
        self._ik_joint_ids = [self._joint_name_to_idx[name] for name in self.cfg.ik_joint_names]
        self._tool_joint_ids = [self._joint_name_to_idx[name] for name in self.cfg.tool_joint_names]
        self._ik_joint_lower_limits = self._joint_lower_limits[self._ik_joint_ids]
        self._ik_joint_upper_limits = self._joint_upper_limits[self._ik_joint_ids]
        self._tool_joint_default_pos = self._psm.data.default_joint_pos[0, self._tool_joint_ids].clone()
        self._joint_pos_des = self._psm.data.default_joint_pos[:, self._ik_joint_ids].clone()

        self._psm_body_name_to_idx, self._psm_body_name_to_path = self._build_psm_body_lookup()
        self._register_tip_contact_sensor()
        self._tip_body_idx = self._resolve_required_body_idx(self.cfg.psm_tip_body_name)
        self._tip_local_offset = torch.tensor(self.cfg.psm_tip_local_offset, dtype=torch.float32, device=self.device)
        self._tip_local_axis = torch.tensor(self.cfg.psm_tip_local_axis, dtype=torch.float32, device=self.device)
        self._tip_local_axis = self._tip_local_axis / torch.linalg.vector_norm(self._tip_local_axis).clamp_min(1.0e-9)

        self._ik_controller_cfg = DifferentialIKControllerCfg(
            command_type="position",
            use_relative_mode=False,
            ik_method="dls",
        )
        self._ik_controller = DifferentialIKController(self._ik_controller_cfg, num_envs=self.num_envs, device=self.device)
        self._psm_entity_cfg = SceneEntityCfg("psm", joint_names=list(self.cfg.ik_joint_names), body_names=[self.cfg.psm_tip_body_name])
        self._ik_commands = torch.zeros((self.num_envs, self._ik_controller.action_dim), dtype=torch.float32, device=self.device)
        self._ee_goal_pos_w = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

        self._workspace_local_low = self._build_workspace_local_low()
        self._workspace_local_high = self._build_workspace_local_high()
        self._workspace_low_w = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self._workspace_high_w = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self._update_workspace_bounds()

        self._obs_state = torch.zeros(
            (self.num_envs, int(self.cfg.observation_space)),
            dtype=torch.float32,
            device=self.device,
        )

    @staticmethod
    def _gf_vec3_to_tensor(vec: Gf.Vec3f, device: torch.device | str) -> torch.Tensor:
        return torch.tensor((float(vec[0]), float(vec[1]), float(vec[2])), dtype=torch.float32, device=device)

    def _build_workspace_local_low(self) -> torch.Tensor:
        lift = torch.tensor((0.0, 0.0, float(self.cfg.table_height_offset)), dtype=torch.float32, device=self.device)
        spawn_pos_tissue = self._gf_vec3_to_tensor(self.cfg.spawn_pos_tissue, self.device)
        workspace_low_offset = torch.tensor(self.cfg.workspace_low_offset, dtype=torch.float32, device=self.device)
        return spawn_pos_tissue + lift + workspace_low_offset

    def _build_workspace_local_high(self) -> torch.Tensor:
        lift = torch.tensor((0.0, 0.0, float(self.cfg.table_height_offset)), dtype=torch.float32, device=self.device)
        spawn_pos_tissue = self._gf_vec3_to_tensor(self.cfg.spawn_pos_tissue, self.device)
        workspace_high_offset = torch.tensor(self.cfg.workspace_high_offset, dtype=torch.float32, device=self.device)
        return spawn_pos_tissue + lift + workspace_high_offset

    def _update_workspace_bounds(self) -> None:
        self._workspace_low_w[:] = self.scene.env_origins + self._workspace_local_low.unsqueeze(0)
        self._workspace_high_w[:] = self.scene.env_origins + self._workspace_local_high.unsqueeze(0)

    def _resolve_ik_handles(self) -> None:
        self._psm_entity_cfg.resolve(self.scene)
        self._ee_jacobi_idx = self._psm_entity_cfg.body_ids[0] - 1

    def _setup_scene(self):
        physx_interface = acquire_physx_interface()
        physx_interface.overwrite_gpu_setting(1)

        sim_context = SimulationContext()
        sim_context.set_render_mode(sim_context.RenderMode.FULL_RENDERING)
        settings = carb.settings.get_settings()
        settings.set_bool("/physics/disableContactProcessing", False)
        settings.set("/rtx/translucency/enabled", True)

        self.cfg.table_setup.func(
            prim_path="/World/envs/env_0/Table",
            cfg=self.cfg.table_setup,
            translation=self.cfg.table_pos,
        )

        lift = Gf.Vec3f(0.0, 0.0, self.cfg.table_height_offset)
        spawn_pos_tissue = self.cfg.spawn_pos_tissue + lift
        spawn_pos_fluid = self.cfg.spawn_pos_fluid + lift
        spawn_pos_glass2 = self.cfg.spawn_pos_glass2

        self.liquid = FluidObject(cfg=self.cfg.liquidCfg, lower_pos=spawn_pos_fluid)
        self.liquid.spawn_fluid_direct()

        self.cfg.tissue_setup.func(
            prim_path="/World/envs/env_0/TissueSetup",
            cfg=self.cfg.tissue_setup,
            translation=spawn_pos_tissue,
        )

        self.cfg.glass2.init_state.pos = spawn_pos_glass2
        self._glass2 = RigidObject(self.cfg.glass2)
        self.scene.rigid_objects["glass2"] = self._glass2

        psm_pos = list(self.cfg.psm_robot.init_state.pos)
        psm_pos[2] += self.cfg.table_height_offset
        self.cfg.psm_robot.init_state.pos = tuple(psm_pos)
        self._psm = Articulation(self.cfg.psm_robot)
        self.scene.articulations["psm"] = self._psm

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        if self._ee_jacobi_idx is None:
            self._resolve_ik_handles()

        self._raw_actions[:] = torch.clamp(actions, -1.0, 1.0)
        delta_pos = self._raw_actions * float(self.cfg.action_scale_lin)

        tip_pos_w, _ = self._compute_tip_pose_and_direction_w()
        tip_quat_w = self._compute_tip_quat_w()
        uninitialized_goal = torch.linalg.vector_norm(self._ee_goal_pos_w, dim=-1) < 1.0e-6
        self._ee_goal_pos_w[uninitialized_goal] = tip_pos_w[uninitialized_goal]
        self._ee_goal_pos_w += delta_pos
        self._ee_goal_pos_w = torch.clamp(self._ee_goal_pos_w, self._workspace_low_w, self._workspace_high_w)

        root_pose_w = self._psm.data.root_state_w[:, 0:7]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], tip_pos_w, tip_quat_w
        )
        ee_goal_pos_b, _ = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], self._ee_goal_pos_w, tip_quat_w
        )

        self._ik_commands[:] = ee_goal_pos_b
        self._ik_controller.set_command(self._ik_commands, ee_quat=ee_quat_b)

        jacobian = self._psm.root_physx_view.get_jacobians()[:, self._ee_jacobi_idx, :, self._ik_joint_ids]
        joint_pos = self._psm.data.joint_pos[:, self._ik_joint_ids]
        self._joint_pos_des[:] = self._ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        self._joint_pos_des[:] = torch.clamp(self._joint_pos_des, self._ik_joint_lower_limits, self._ik_joint_upper_limits)

        self._task_state_dirty = True
        self._task_state_apply_suction = True

    def _apply_action(self):
        tool_targets = self._tool_joint_default_pos.unsqueeze(0).expand(self.num_envs, -1)
        self._psm.set_joint_position_target(self._joint_pos_des, joint_ids=self._ik_joint_ids)
        self._psm.set_joint_position_target(tool_targets, joint_ids=self._tool_joint_ids)

    def _register_tip_contact_sensor(self) -> None:
        tip_body_path = self._psm_body_name_to_path.get(self.cfg.psm_tip_body_name)

        tip_contact_cfg = ContactSensorCfg(
            prim_path=tip_body_path,
            update_period=0.0,
            history_length=1,
            track_air_time=True,
            force_threshold=float(self.cfg.tip_contact_force_threshold),
            debug_vis=False,
        )
        self._tip_contact_sensor = ContactSensor(cfg=tip_contact_cfg)
        self.scene.sensors["tip_contact"] = self._tip_contact_sensor
        if self.sim.is_playing():
            self._tip_contact_sensor._initialize_callback(None)

    def _get_tip_contact_force(self) -> torch.Tensor:
        if self._tip_contact_sensor is None:
            return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        net_forces_w = self.scene["tip_contact"].data.net_forces_w
        contact_force = torch.linalg.vector_norm(net_forces_w, dim=-1)
        if contact_force.ndim > 1:
            contact_force = torch.amax(contact_force, dim=1)
        return contact_force.to(dtype=torch.float32)

    def _build_psm_body_lookup(self) -> tuple[dict[str, int], dict[str, str]]:
        env_zero_ns = self.scene.env_regex_ns.replace(".*", "0")
        body_name_to_idx: dict[str, int] = {}
        body_name_to_path: dict[str, str] = {}

        for body_idx, (body_name, body_path) in enumerate(zip(self._psm.body_names, self._psm.root_physx_view.link_paths[0])):
            body_name_to_idx[body_name] = body_idx
            if body_path.startswith(env_zero_ns):
                body_name_to_path[body_name] = body_path.replace(env_zero_ns, self.scene.env_regex_ns, 1)
            else:
                body_name_to_path[body_name] = body_path

        return body_name_to_idx, body_name_to_path

    def _resolve_required_body_idx(self, body_name: str) -> int:
        if body_name not in self._psm_body_name_to_idx:
            available = ", ".join(self._psm.body_names)
            raise RuntimeError(f"Required PSM body '{body_name}' not found. Available bodies: {available}")
        return self._psm_body_name_to_idx[body_name]

    @staticmethod
    def _quat_rotate_torch(quat_wxyz: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        quat_vec = quat_wxyz[..., 1:]
        uv = torch.linalg.cross(quat_vec, vec, dim=-1)
        uuv = torch.linalg.cross(quat_vec, uv, dim=-1)
        return vec + 2.0 * (quat_wxyz[..., :1] * uv + uuv)

    def _compute_tip_quat_w(self) -> torch.Tensor:
        body_quat_w = getattr(self._psm.data, "body_quat_w", None)
        if body_quat_w is None:
            return self._psm.data.root_state_w[:, 3:7]
        return body_quat_w[:, self._tip_body_idx]

    def _compute_tip_pose_and_direction_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        tip_body_pos_w = self._psm.data.body_pos_w[:, self._tip_body_idx]
        local_offset = self._tip_local_offset.unsqueeze(0).expand(self.num_envs, -1)
        local_axis = self._tip_local_axis.unsqueeze(0).expand(self.num_envs, -1)

        body_quat_w = getattr(self._psm.data, "body_quat_w", None)
        if body_quat_w is None:
            tip_pos_w = tip_body_pos_w + local_offset
            tip_dir_w = local_axis
        else:
            tip_body_quat_w = body_quat_w[:, self._tip_body_idx]
            tip_pos_w = tip_body_pos_w + self._quat_rotate_torch(tip_body_quat_w, local_offset)
            if self.cfg.use_body_quat_for_tip_dir:
                tip_dir_w = self._quat_rotate_torch(tip_body_quat_w, local_axis)
            else:
                tip_dir_w = local_axis

        tip_dir_w = tip_dir_w / torch.linalg.vector_norm(tip_dir_w, dim=-1, keepdim=True).clamp_min(1.0e-9)
        return tip_pos_w, tip_dir_w

    def _refresh_post_step_task_state(self) -> None:
        if not self._task_state_dirty:
            return

        tip_pos_w, tip_dir_w = self._compute_tip_pose_and_direction_w()

        if self._task_state_apply_suction:
            suction_stats = self._suction_controller.step(
                step_count=self._step_count.detach().cpu().numpy(),
                num_envs=self.num_envs,
                psm=self._psm,
                liquid=self.liquid,
                glass2=self._glass2,
                env_origins=self.scene.env_origins,
            )
            self._absorbed_delta[:] = torch.from_numpy(suction_stats["absorbed_delta"]).to(self.device)
            self._absorbed_count += self._absorbed_delta
            alpha = float(self.cfg.absorbed_delta_ema_alpha)
            self._absorbed_delta_ema.mul_(1.0 - alpha).add_(alpha * self._absorbed_delta)

        self._update_particle_task_state(tip_pos_w=tip_pos_w, tip_dir_w=tip_dir_w)
        self._task_state_apply_suction = False
        self._task_state_dirty = False

    def _update_particle_task_state(self, tip_pos_w: torch.Tensor, tip_dir_w: torch.Tensor) -> None:
        tip_pos_local = tip_pos_w - self.scene.env_origins
        prev_centroid = self._blood_centroid.clone()
        prev_distance = self._blood_centroid_distance.clone()
        self._prev_blood_centroid.copy_(prev_centroid)
        self._prev_blood_centroid_distance.copy_(prev_distance)

        suction_radius = float(self.cfg.suction_cone_range)
        inlet_depth = float(self.cfg.inlet_depth)
        inlet_radius = float(self.cfg.inlet_radius)
        cos_theta = float(np.cos(np.deg2rad(float(self.cfg.suction_cone_half_angle_deg))))
        epsilon = max(float(getattr(self.cfg, "suction_epsilon", 1.0e-6)), 1.0e-12)

        for env_idx in range(self.num_envs):
            particles_pos, _ = self.liquid.get_particles_position(env_idx)
            if len(particles_pos) == 0:
                if int(self._step_count[env_idx].item()) <= 0:
                    self._blood_centroid[env_idx] = tip_pos_w[env_idx]
                    self._prev_blood_centroid[env_idx] = tip_pos_w[env_idx]
                    self._blood_centroid_distance[env_idx] = 0.0
                    self._prev_blood_centroid_distance[env_idx] = 0.0
                else:
                    self._blood_centroid[env_idx] = prev_centroid[env_idx]
                    self._blood_centroid_distance[env_idx] = prev_distance[env_idx]
                self._valid_in_cone_ratio[env_idx] = 0.0
                self._valid_in_inlet_ratio[env_idx] = 0.0
                continue

            particles_pos_tensor = torch.as_tensor(particles_pos, dtype=torch.float32, device=self.device)
            valid_mask = particles_pos_tensor[:, self.cfg.height_axis] >= float(self.cfg.height_limit)
            if not torch.any(valid_mask):
                if int(self._step_count[env_idx].item()) <= 0:
                    self._blood_centroid[env_idx] = tip_pos_w[env_idx]
                    self._prev_blood_centroid[env_idx] = tip_pos_w[env_idx]
                    self._blood_centroid_distance[env_idx] = 0.0
                    self._prev_blood_centroid_distance[env_idx] = 0.0
                else:
                    self._blood_centroid[env_idx] = prev_centroid[env_idx]
                    self._blood_centroid_distance[env_idx] = prev_distance[env_idx]
                self._valid_in_cone_ratio[env_idx] = 0.0
                self._valid_in_inlet_ratio[env_idx] = 0.0
                continue

            valid_positions = particles_pos_tensor[valid_mask]
            centroid_local = valid_positions.mean(dim=0)
            centroid_w = centroid_local + self.scene.env_origins[env_idx]
            relative_positions = valid_positions - tip_pos_local[env_idx]
            distances = torch.linalg.vector_norm(relative_positions, dim=1)
            axial_depth = torch.sum(relative_positions * tip_dir_w[env_idx], dim=1)
            radial_offset = relative_positions - axial_depth.unsqueeze(1) * tip_dir_w[env_idx]
            radial_distance = torch.linalg.vector_norm(radial_offset, dim=1)
            cos_alpha = axial_depth / distances.clamp_min(epsilon)
            in_cone = (distances < suction_radius) & (cos_alpha >= cos_theta)
            in_inlet = (axial_depth > 0.0) & (axial_depth < inlet_depth) & (radial_distance < inlet_radius)
            valid_count = max(valid_positions.shape[0], 1)
            current_distance = torch.linalg.vector_norm(centroid_local - tip_pos_local[env_idx])

            self._blood_centroid[env_idx] = centroid_w
            self._blood_centroid_distance[env_idx] = current_distance
            self._valid_in_cone_ratio[env_idx] = in_cone.to(dtype=torch.float32).sum() / float(valid_count)
            self._valid_in_inlet_ratio[env_idx] = in_inlet.to(dtype=torch.float32).sum() / float(valid_count)

            if int(self._step_count[env_idx].item()) <= 0:
                self._prev_blood_centroid[env_idx] = centroid_w
                self._prev_blood_centroid_distance[env_idx] = current_distance

    def _normalize_workspace_positions(self, pos_w: torch.Tensor) -> torch.Tensor:
        workspace_range = (self._workspace_high_w - self._workspace_low_w).clamp_min(1.0e-6)
        normalized = 2.0 * (pos_w - self._workspace_low_w) / workspace_range - 1.0
        return torch.clamp(normalized, -1.0, 1.0)

    def _maybe_print_tip_position(self, tip_pos_w: torch.Tensor) -> None:
        if not bool(getattr(self.cfg, "debug_print_ee_pos", False)) or self.num_envs <= 0:
            return

        interval = max(int(getattr(self.cfg, "debug_print_ee_pos_interval", 50)), 1)
        step = int(self._step_count[0].item())
        if step % interval != 0:
            return

        tip_pos = tip_pos_w[0].detach().cpu().tolist()
        print(f"[EE] env=0 step={step} tip_pos_w=({tip_pos[0]:.4f}, {tip_pos[1]:.4f}, {tip_pos[2]:.4f})")

    def _update_low_dim_observation(self) -> None:
        self._refresh_post_step_task_state()

        tip_pos_w, tip_dir_w = self._compute_tip_pose_and_direction_w()
        reset_goal_mask = self._step_count <= 0
        self._ee_goal_pos_w[reset_goal_mask] = tip_pos_w[reset_goal_mask]
        self._maybe_print_tip_position(tip_pos_w)
        tip_pos_normalized = self._normalize_workspace_positions(tip_pos_w)
        workspace_range = (self._workspace_high_w - self._workspace_low_w).clamp_min(1.0e-6)
        goal_error_normalized = torch.clamp(2.0 * (self._ee_goal_pos_w - tip_pos_w) / workspace_range, -1.0, 1.0)
        blood_centroid_rel_normalized = torch.clamp((self._blood_centroid - tip_pos_w) / workspace_range, -1.0, 1.0)

        absorbed_ratio = torch.clamp(self._absorbed_count / self._expected_particle_count, min=0.0, max=1.0).unsqueeze(1)
        absorbed_delta_ema = torch.clamp(self._absorbed_delta_ema, min=0.0, max=1.0).unsqueeze(1)
        valid_in_cone_ratio = torch.clamp(self._valid_in_cone_ratio, min=0.0, max=1.0).unsqueeze(1)
        valid_in_inlet_ratio = torch.clamp(self._valid_in_inlet_ratio, min=0.0, max=1.0).unsqueeze(1)
        contact_ratio = torch.clamp(
            self._get_tip_contact_force() / max(float(self.cfg.tip_contact_force_threshold), 1.0e-6),
            min=0.0,
            max=1.0,
        ).unsqueeze(1)
        step_ratio = torch.clamp(
            self._step_count.to(dtype=torch.float32) / max(float(self.max_episode_length), 1.0),
            min=0.0,
            max=1.0,
        ).unsqueeze(1)

        self._obs_state[:] = torch.cat(
            (
                tip_pos_normalized,
                goal_error_normalized,
                tip_dir_w,
                blood_centroid_rel_normalized,
                valid_in_cone_ratio,
                valid_in_inlet_ratio,
                absorbed_ratio,
                absorbed_delta_ema,
                contact_ratio,
                step_ratio,
            ),
            dim=1,
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._refresh_post_step_task_state()

        ik_joint_pos = self._psm.data.joint_pos[:, self._ik_joint_ids]
        tolerance = float(self.cfg.joint_limit_termination_tolerance)
        joint_limit_reached = torch.any(
            (ik_joint_pos <= self._ik_joint_lower_limits + tolerance)
            | (ik_joint_pos >= self._ik_joint_upper_limits - tolerance),
            dim=1,
        )

        contact_force = self._get_tip_contact_force()
        severe_contact = contact_force > float(self.cfg.severe_contact_force_threshold)
        self._severe_contact_counter[:] = torch.where(
            severe_contact,
            self._severe_contact_counter + 1,
            torch.zeros_like(self._severe_contact_counter),
        )
        severe_collision = self._severe_contact_counter >= int(self.cfg.severe_contact_patience)
        success = self._absorbed_count >= self._success_threshold

        terminated = success | joint_limit_reached | severe_collision
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        self._terminated[:] = terminated
        self._episode_success[:] = success
        self._episode_joint_limit[:] = joint_limit_reached
        self._episode_severe_collision[:] = severe_collision
        self._episode_time_out[:] = truncated
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        self._refresh_post_step_task_state()

        absorb_reward = self.cfg.reward_absorb_weight * self._absorbed_delta
        centroid_progress = torch.clamp(
            self._prev_blood_centroid_distance - self._blood_centroid_distance,
            min=-float(self.cfg.centroid_progress_clip),
            max=float(self.cfg.centroid_progress_clip),
        )
        centroid_progress_reward = self.cfg.centroid_progress_weight * centroid_progress
        cone_coverage_reward = self.cfg.reward_cone_coverage_weight * self._valid_in_cone_ratio
        inlet_coverage_reward = self.cfg.reward_inlet_coverage_weight * self._valid_in_inlet_ratio
        action_penalty = self.cfg.reward_action_weight * torch.sum(self._raw_actions**2, dim=1)
        contact_force = self._get_tip_contact_force()
        collision_force_penalty = self.cfg.reward_collision_force_weight * torch.clamp(
            contact_force - float(self.cfg.tip_contact_force_threshold), min=0.0
        )
        time_penalty = torch.full(
            (self.num_envs,),
            float(self.cfg.reward_time_penalty),
            dtype=torch.float32,
            device=self.device,
        )
        task_complete = self.cfg.reward_task_complete * (self._absorbed_count >= self._success_threshold).float()

        reward = (
            task_complete
            + absorb_reward
            + centroid_progress_reward
            + cone_coverage_reward
            + inlet_coverage_reward
            - action_penalty
            - collision_force_penalty
            - time_penalty
        ).float()

        self._episode_reward_sums["absorb_reward"] += absorb_reward
        self._episode_reward_sums["centroid_progress_reward"] += centroid_progress_reward
        self._episode_reward_sums["cone_coverage_reward"] += cone_coverage_reward
        self._episode_reward_sums["inlet_coverage_reward"] += inlet_coverage_reward
        self._episode_reward_sums["action_penalty"] -= action_penalty
        self._episode_reward_sums["collision_force_penalty"] -= collision_force_penalty
        self._episode_reward_sums["time_penalty"] -= time_penalty
        self._episode_reward_sums["task_complete"] += task_complete

        self.extras["log"] = {
            "Metrics/absorbed_count": self._absorbed_count.mean(),
            "Metrics/absorbed_delta": self._absorbed_delta.mean(),
            "Metrics/blood_centroid_distance": self._blood_centroid_distance.mean(),
            "Metrics/valid_in_cone_ratio": self._valid_in_cone_ratio.mean(),
            "Metrics/valid_in_inlet_ratio": self._valid_in_inlet_ratio.mean(),
            "Metrics/absorbed_delta_ema": self._absorbed_delta_ema.mean(),
            "Metrics/success_rate": (self._absorbed_count >= self._success_threshold).float().mean(),
        }

        return reward

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        if self._ee_jacobi_idx is None:
            self._resolve_ik_handles()
        self._ik_controller.reset(env_ids=env_ids)
        self._update_workspace_bounds()

        ids = env_ids.tolist()
        finished_mask = (
            self._episode_success[env_ids]
            | self._episode_joint_limit[env_ids]
            | self._episode_severe_collision[env_ids]
            | self._episode_time_out[env_ids]
        )
        finished_env_ids = env_ids[finished_mask]

        if finished_env_ids.numel() > 0:
            if "log" not in self.extras:
                self.extras["log"] = dict()

            reward_logs = {}
            for key, values in self._episode_reward_sums.items():
                reward_logs[f"Episode_Reward/{key}"] = values[finished_env_ids].mean() / self.max_episode_length_s

            success_mask = self._episode_success[finished_env_ids]
            joint_limit_mask = (~success_mask) & self._episode_joint_limit[finished_env_ids]
            severe_collision_mask = (~success_mask) & (~joint_limit_mask) & self._episode_severe_collision[finished_env_ids]
            time_out_mask = (
                (~success_mask)
                & (~joint_limit_mask)
                & (~severe_collision_mask)
                & self._episode_time_out[finished_env_ids]
            )

            termination_logs = {
                "Episode_Termination/success": success_mask.sum().to(dtype=torch.float32),
                "Episode_Termination/joint_limit": joint_limit_mask.sum().to(dtype=torch.float32),
                "Episode_Termination/severe_collision": severe_collision_mask.sum().to(dtype=torch.float32),
                "Episode_Termination/time_out": time_out_mask.sum().to(dtype=torch.float32),
            }
            self.extras["log"].update(reward_logs)
            self.extras["log"].update(termination_logs)

        joint_pos = self._psm.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(joint_pos)
        self._psm.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._psm.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._joint_pos_des[env_ids] = joint_pos[:, self._ik_joint_ids]

        if self._initial_particles_pos is not None:
            for env_id in ids:
                self.liquid.set_particles_position(
                    self._initial_particles_pos.copy(),
                    self._initial_particles_vel.copy(),
                    int(env_id),
                )

        self._suction_controller.reset(ids)
        self._step_count[env_ids] = 0
        self._absorbed_count[env_ids] = 0.0
        self._absorbed_delta[env_ids] = 0.0
        self._absorbed_delta_ema[env_ids] = 0.0
        self._blood_centroid_distance[env_ids] = 0.0
        self._prev_blood_centroid_distance[env_ids] = 0.0
        self._valid_in_cone_ratio[env_ids] = 0.0
        self._valid_in_inlet_ratio[env_ids] = 0.0
        self._severe_contact_counter[env_ids] = 0
        self._terminated[env_ids] = False
        for values in self._episode_reward_sums.values():
            values[env_ids] = 0.0
        self._episode_success[env_ids] = False
        self._episode_joint_limit[env_ids] = False
        self._episode_severe_collision[env_ids] = False
        self._episode_time_out[env_ids] = False
        self._raw_actions[env_ids] = 0.0
        self._ik_commands[env_ids] = 0.0
        self._obs_state[env_ids] = 0.0

        tip_pos_w, _ = self._compute_tip_pose_and_direction_w()
        self._ee_goal_pos_w[env_ids] = tip_pos_w[env_ids]
        self._blood_centroid[env_ids] = tip_pos_w[env_ids]
        self._prev_blood_centroid[env_ids] = tip_pos_w[env_ids]
        self._task_state_dirty = True
        self._task_state_apply_suction = False

    def _get_observations(self) -> dict:
        self._update_low_dim_observation()

        if not self._particles_initialized:
            pos, vel = self.liquid.get_particles_position(0)
            if len(pos) > 0:
                self._initial_particles_pos = pos.copy()
                self._initial_particles_vel = vel.copy()
                self._particles_initialized = True

        self._step_count += 1

        return {"policy": self._obs_state.clone()}
