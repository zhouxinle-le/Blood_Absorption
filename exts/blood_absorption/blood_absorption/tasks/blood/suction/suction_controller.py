from __future__ import annotations

import math
import numpy as np

class SuctionControllerNoTimer:
    """Particle suction/transfer logic without capture timer state machine."""
    # 这是一个无状态的控制器，意味着它只根据当前帧的物理状态进行计算，
    # 没有“吸取进度条”或“捕获计时器”等中间状态。

    def __init__(self, cfg, num_envs: int):
        self.cfg = cfg
        self._num_envs = int(num_envs)
        self._tip_body_idx: int | None = None
        self._tip_local_offset = np.asarray(getattr(self.cfg, "psm_tip_local_offset", (0.0, 0.0, 0.0)), dtype=np.float32)
        self._tip_local_axis = np.asarray(getattr(self.cfg, "psm_tip_local_axis", (0.0, 0.0, -1.0)), dtype=np.float32)
        axis_norm = np.linalg.norm(self._tip_local_axis)
        self._tip_local_axis = self._tip_local_axis / axis_norm

    def reset(self, env_ids) -> None:
        # 因为是无状态控制器，reset不需要做任何事情，保留此方法是为了与其他控制器的接口兼容
        return None

    def step(self, step_count, num_envs: int, psm, liquid, glass2, env_origins) -> dict[str, np.ndarray]:
        if hasattr(env_origins, "detach"):
            env_origins = env_origins.detach().cpu().numpy()
        env_origins = np.asarray(env_origins, dtype=np.float32)

        # 初始化记录各项指标的数组
        absorbed_delta = np.zeros((num_envs,), dtype=np.float32) # 本步吸收的粒子数
        min_dist = np.full((num_envs,), float(self.cfg.suction_cone_range), dtype=np.float32)
        inlet_count = np.zeros((num_envs,), dtype=np.float32)    # 进入吸入口的粒子数
        cone_count = np.zeros((num_envs,), dtype=np.float32)     # 受到吸力影响的粒子数

        # 预先提取一些物理参数以加快循环内的计算速度
        # dt = float(self.cfg.sim.dt * self.cfg.decimation)
        dt = self.cfg.sim.dt
        suction_radius = float(self.cfg.suction_cone_range)
        cos_theta = math.cos(math.radians(float(self.cfg.suction_cone_half_angle_deg)))
        epsilon = max(float(getattr(self.cfg, "suction_epsilon", 1e-6)), 1e-12) # 防止除以0
        force_scale = float(getattr(self.cfg, "suction_force_scale", 1.0))
        particle_mass = max(float(self.cfg.liquidCfg.particle_mass), epsilon)

        # 遍历所有并行的仿真环境
        for env_idx in range(num_envs):
            # 1. 获取当前环境中吸头的位置和方向
            tip_pos, tip_dir = self._get_tip_pose(env_idx, psm, env_origins)
            # 2. 获取当前环境中所有液体粒子的位置和速度
            particles_pos, particles_vel = liquid.get_particles_position(env_idx)
            if len(particles_pos) == 0:
                continue

            # 过滤掉高度超出限制的异常粒子
            valid_mask = particles_pos[:, self.cfg.height_axis] >= float(self.cfg.height_limit)
            if not valid_mask.any():
                continue

            # 3. 核心空间计算：计算粒子相对于吸头的位置向量 r
            r = particles_pos - tip_pos
            dist = np.linalg.norm(r, axis=1) # 粒子到吸头的直线距离
            min_dist[env_idx] = float(dist.min()) # 记录最近的粒子距离
            
            # 投影：计算粒子在吸头轴线(tip_dir)上的深度 (ax) 和垂直于轴线的径向距离 (rad)
            ax = np.dot(r, tip_dir) 
            r_perp = r - np.outer(ax, tip_dir)
            rad = np.linalg.norm(r_perp, axis=1)

            in_cone = np.zeros(len(particles_pos), dtype=bool)
            in_inlet = np.zeros(len(particles_pos), dtype=bool)

            # 5. 判定吸入口内粒子：深度 ax > 0 且小于吸管深度，且径向距离小于吸管半径
            in_inlet = (
                (ax > 0.0)
                & (ax < float(self.cfg.inlet_depth))
                & (rad < float(self.cfg.inlet_radius))
                & valid_mask
            )

            # 6. 如果开启了手动吸力，则对圆锥范围内的粒子施加物理速度改变
            in_cone = self._apply_manual_suction(
                particles_vel=particles_vel,
                relative_positions=r,
                distances=dist,
                cone_axis=tip_dir,
                valid_mask=valid_mask,
                cos_theta=cos_theta,
                suction_radius=suction_radius,
                force_scale=force_scale,
                particle_mass=particle_mass,
                epsilon=epsilon,
                dt=dt,
            )

            # 进入吸入口的粒子将被标记为待移除/转移
            remove_mask = in_inlet

            inlet_count[env_idx] = float(in_inlet.sum())
            cone_count[env_idx] = float(in_cone.sum())

            # 7. 粒子转移：如果粒子进入了吸入口，且允许传送，则把它们传送到 glass2
            if remove_mask.any():
                absorbed_delta[env_idx] = float(
                    self._transfer_particles(
                        env_idx=env_idx,
                        ready_to_transfer=remove_mask,
                        particles_pos=particles_pos,
                        particles_vel=particles_vel,
                        glass2=glass2,
                        env_origins=env_origins,
                    )
                )

            # # 8. 安全机制：限制粒子的最大速度，防止由于受力过大导致仿真崩溃穿模
            self._limit_particle_speed(particles_vel)

            # 将更新后的位置和速度写回物理仿真引擎
            liquid.set_particles_position(particles_pos, particles_vel, env_idx)

        # 返回各项统计数据，供外部调用（如 RL 的 Reward 计算或日志记录）使用
        return {
            "absorbed_delta": absorbed_delta,
            "min_dist": min_dist,
            "inlet_count": inlet_count,
            "cone_count": cone_count,
        }

    @staticmethod
    def _rotate_vector_by_quat(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
        quat_vec = quat_wxyz[1:]
        uv = np.cross(quat_vec, vec)
        uuv = np.cross(quat_vec, uv)
        return vec + 2.0 * (quat_wxyz[0] * uv + uuv)

    def _resolve_tip_body_idx(self, psm) -> int:
        if self._tip_body_idx is None:
            body_names = list(psm.body_names)
            body_name = str(getattr(self.cfg, "psm_tip_body_name", ""))
            if body_name not in body_names:
                available = ", ".join(body_names)
                raise RuntimeError(f"Required PSM tip body '{body_name}' not found. Available bodies: {available}")
            self._tip_body_idx = body_names.index(body_name)
        return self._tip_body_idx

    def _get_tip_pose(self, env_idx: int, psm, env_origins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tip_body_idx = self._resolve_tip_body_idx(psm)

        tip_body_pos_w = psm.data.body_pos_w[env_idx, tip_body_idx].cpu().numpy()
        tip_pos = tip_body_pos_w + self._tip_local_offset
        tip_dir = self._tip_local_axis.copy()

        if hasattr(psm.data, "body_quat_w"):
            quat_wxyz = psm.data.body_quat_w[env_idx, tip_body_idx].cpu().numpy()
            tip_pos = tip_body_pos_w + self._rotate_vector_by_quat(quat_wxyz, self._tip_local_offset)
            if self.cfg.use_body_quat_for_tip_dir:
                tip_dir = self._rotate_vector_by_quat(quat_wxyz, self._tip_local_axis)

        tip_dir = tip_dir / (np.linalg.norm(tip_dir) + 1e-9)
        return tip_pos - env_origins[env_idx], tip_dir

    def _apply_manual_suction(self, particles_vel, relative_positions, distances, cone_axis, 
                              valid_mask, cos_theta, suction_radius, force_scale, particle_mass, epsilon, dt) -> np.ndarray:
        if suction_radius <= 0.0 or force_scale <= 0.0:
            return np.zeros(len(particles_vel), dtype=bool)

        # 判断粒子是否在吸力圆锥内：1. 距离小于吸取半径; 2. 相对方向与吸头轴向的夹角小于半锥角 (cos值更大)
        cos_alpha = np.dot(relative_positions, cone_axis) / (distances + epsilon)
        in_cone = (distances < suction_radius) & (cos_alpha >= cos_theta) & valid_mask
        if not in_cone.any():
            return in_cone

        idx = np.where(in_cone)[0]
        dist = distances[idx]
        
        # 吸力方向：指向吸头 (-relative_positions)
        direction = -relative_positions[idx] / (dist[:, None] + epsilon)
        
        # 吸力大小计算公式：距离越近力越大，在 suction_radius 处力降为 0
        # accel_mag = (force_scale / particle_mass) * (1.0 / (dist + epsilon) - 1.0 / suction_radius)
        accel_mag = force_scale/particle_mass *(suction_radius - dist)**2
        accel_mag = np.maximum(accel_mag, 0.0) # 确保加速度不为负
        
        # 更新粒子的速度：v = v0 + a * dt
        particles_vel[idx] += direction * accel_mag[:, None] * dt
        return in_cone

    def _transfer_particles(
        self,
        env_idx: int,
        ready_to_transfer: np.ndarray,
        particles_pos: np.ndarray,
        particles_vel: np.ndarray,
        glass2,
        env_origins: np.ndarray,
    ) -> int:
        # 获取目标容器 (glass2) 的位置
        glass2_pos = glass2.data.root_pos_w[env_idx].cpu().numpy() - env_origins[env_idx]
        indices = np.where(ready_to_transfer)[0]
        
        # 生成随机偏移量，避免所有粒子传送到同一个确切的点导致物理引擎碰撞计算爆炸
        random_offsets = np.random.uniform(-0.015, 0.015, (len(indices), 3))
        random_offsets[:, 2] = np.abs(random_offsets[:, 2]) + 0.01 # 确保在容器底部上方

        # 修改被吞噬粒子的位置和速度，实现瞬间转移
        particles_pos[indices] = (
            glass2_pos + np.array([0.0, 0.0, float(self.cfg.glass2_particle_height)]) + random_offsets
        )
        particles_vel[indices] = np.array([0.0, 0.0, float(self.cfg.outflow_speed)], dtype=np.float32)
        return len(indices)

    def _limit_particle_speed(self, particles_vel: np.ndarray) -> None:
        # 计算所有粒子的速度大小（L2范数）
        vel_norm = np.linalg.norm(particles_vel, axis=1, keepdims=True) + 1e-9
        # 找出超速的粒子
        too_fast = vel_norm[:, 0] > float(self.cfg.max_particle_speed)
        # 对超速粒子进行等比例缩放（保持方向不变，长度截断为最大限制速度）
        if too_fast.any():
            particles_vel[too_fast] = (
                particles_vel[too_fast] / vel_norm[too_fast]
            ) * float(self.cfg.max_particle_speed)
