from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .suction.geometry import compute_cone_and_inlet_masks, compute_particle_relation


@dataclass
class ParticleTaskState:
    absorbed_delta: torch.Tensor
    absorbed_count: torch.Tensor
    absorbed_delta_ema: torch.Tensor
    blood_centroid: torch.Tensor
    prev_blood_centroid: torch.Tensor
    blood_centroid_distance: torch.Tensor
    prev_blood_centroid_distance: torch.Tensor
    valid_in_cone_ratio: torch.Tensor
    valid_in_inlet_ratio: torch.Tensor


@dataclass(frozen=True)
class ParticleRewardInputs:
    raw_actions: torch.Tensor
    contact_force: torch.Tensor


class ParticleTaskTracker:
    def __init__(self, cfg, num_envs: int, device: torch.device | str):
        self.cfg = cfg
        self._num_envs = int(num_envs)
        self.device = torch.device(device)

        self._cos_theta = float(np.cos(np.deg2rad(float(self.cfg.suction_cone_half_angle_deg))))
        self._suction_radius = float(self.cfg.suction_cone_range)
        self._inlet_depth = float(self.cfg.inlet_depth)
        self._inlet_radius = float(self.cfg.inlet_radius)
        self._epsilon = max(float(getattr(self.cfg, "suction_epsilon", 1.0e-6)), 1.0e-12)
        self._height_axis = int(self.cfg.height_axis)
        self._height_limit = float(self.cfg.height_limit)
        self._ema_alpha = float(self.cfg.absorbed_delta_ema_alpha)

        self.state = ParticleTaskState(
            absorbed_delta=self._zeros(),
            absorbed_count=self._zeros(),
            absorbed_delta_ema=self._zeros(),
            blood_centroid=self._zeros((self._num_envs, 3)),
            prev_blood_centroid=self._zeros((self._num_envs, 3)),
            blood_centroid_distance=self._zeros(),
            prev_blood_centroid_distance=self._zeros(),
            valid_in_cone_ratio=self._zeros(),
            valid_in_inlet_ratio=self._zeros(),
        )

    def _zeros(self, shape: tuple[int, ...] | None = None) -> torch.Tensor:
        if shape is None:
            shape = (self._num_envs,)
        return torch.zeros(shape, dtype=torch.float32, device=self.device)

    @staticmethod
    def _to_numpy(value) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    def refresh(
        self,
        liquid,
        tip_pos_w: torch.Tensor,
        tip_dir_w: torch.Tensor,
        env_origins,
        step_count,
        suction_stats: dict[str, np.ndarray] | None = None,
    ) -> None:
        if suction_stats is not None:
            self._apply_suction_stats(suction_stats)

        self._update_particle_metrics(
            liquid=liquid,
            tip_pos_w=tip_pos_w,
            tip_dir_w=tip_dir_w,
            env_origins=env_origins,
            step_count=step_count,
        )

    def _apply_suction_stats(self, suction_stats: dict[str, np.ndarray]) -> None:
        absorbed_delta = torch.from_numpy(suction_stats["absorbed_delta"]).to(device=self.device, dtype=torch.float32)
        self.state.absorbed_delta[:] = absorbed_delta
        self.state.absorbed_count += absorbed_delta
        self.state.absorbed_delta_ema.mul_(1.0 - self._ema_alpha).add_(self._ema_alpha * absorbed_delta)

    def _update_particle_metrics(self, liquid, tip_pos_w: torch.Tensor, tip_dir_w: torch.Tensor, env_origins, step_count) -> None:
        prev_centroid = self.state.blood_centroid.clone()
        prev_distance = self.state.blood_centroid_distance.clone()
        self.state.prev_blood_centroid.copy_(prev_centroid)
        self.state.prev_blood_centroid_distance.copy_(prev_distance)

        tip_pos_w_np = self._to_numpy(tip_pos_w).astype(np.float32, copy=False)
        tip_dir_w_np = self._to_numpy(tip_dir_w).astype(np.float32, copy=False)
        env_origins_np = self._to_numpy(env_origins).astype(np.float32, copy=False)
        tip_pos_local_np = tip_pos_w_np - env_origins_np
        step_count_np = self._to_numpy(step_count).astype(np.int64, copy=False)

        for env_idx in range(self._num_envs):
            particles_pos, _ = liquid.read_particles(env_idx)
            if len(particles_pos) == 0:
                self._set_empty_metrics(env_idx, tip_pos_w_np[env_idx], prev_centroid, prev_distance, step_count_np)
                continue

            valid_mask = particles_pos[:, self._height_axis] >= self._height_limit
            if not np.any(valid_mask):
                self._set_empty_metrics(env_idx, tip_pos_w_np[env_idx], prev_centroid, prev_distance, step_count_np)
                continue

            valid_positions = particles_pos[valid_mask]
            centroid_local = valid_positions.mean(axis=0)
            centroid_w = centroid_local + env_origins_np[env_idx]
            _, distances, axial_depth, radial_distance = compute_particle_relation(
                valid_positions,
                tip_pos_local_np[env_idx],
                tip_dir_w_np[env_idx],
                self._epsilon,
            )
            full_mask = np.ones((valid_positions.shape[0],), dtype=bool)
            in_cone, in_inlet = compute_cone_and_inlet_masks(
                distances=distances,
                axial_depth=axial_depth,
                radial_distance=radial_distance,
                valid_mask=full_mask,
                suction_radius=self._suction_radius,
                cos_theta=self._cos_theta,
                inlet_depth=self._inlet_depth,
                inlet_radius=self._inlet_radius,
                epsilon=self._epsilon,
            )
            valid_count = max(valid_positions.shape[0], 1)
            current_distance = float(np.linalg.norm(centroid_local - tip_pos_local_np[env_idx]))

            self.state.blood_centroid[env_idx] = torch.as_tensor(centroid_w, dtype=torch.float32, device=self.device)
            self.state.blood_centroid_distance[env_idx] = current_distance
            self.state.valid_in_cone_ratio[env_idx] = float(in_cone.sum()) / float(valid_count)
            self.state.valid_in_inlet_ratio[env_idx] = float(in_inlet.sum()) / float(valid_count)

            if int(step_count_np[env_idx]) <= 0:
                self.state.prev_blood_centroid[env_idx] = torch.as_tensor(centroid_w, dtype=torch.float32, device=self.device)
                self.state.prev_blood_centroid_distance[env_idx] = current_distance

    def _set_empty_metrics(
        self,
        env_idx: int,
        tip_pos_w: np.ndarray,
        prev_centroid: torch.Tensor,
        prev_distance: torch.Tensor,
        step_count_np: np.ndarray,
    ) -> None:
        if int(step_count_np[env_idx]) <= 0:
            tip_pos_tensor = torch.as_tensor(tip_pos_w, dtype=torch.float32, device=self.device)
            self.state.blood_centroid[env_idx] = tip_pos_tensor
            self.state.prev_blood_centroid[env_idx] = tip_pos_tensor
            self.state.blood_centroid_distance[env_idx] = 0.0
            self.state.prev_blood_centroid_distance[env_idx] = 0.0
        else:
            self.state.blood_centroid[env_idx] = prev_centroid[env_idx]
            self.state.blood_centroid_distance[env_idx] = prev_distance[env_idx]

        self.state.valid_in_cone_ratio[env_idx] = 0.0
        self.state.valid_in_inlet_ratio[env_idx] = 0.0

    def reset(self, env_ids: torch.Tensor, tip_pos_w: torch.Tensor) -> None:
        self.state.absorbed_delta[env_ids] = 0.0
        self.state.absorbed_count[env_ids] = 0.0
        self.state.absorbed_delta_ema[env_ids] = 0.0
        self.state.blood_centroid_distance[env_ids] = 0.0
        self.state.prev_blood_centroid_distance[env_ids] = 0.0
        self.state.valid_in_cone_ratio[env_ids] = 0.0
        self.state.valid_in_inlet_ratio[env_ids] = 0.0
        self.state.blood_centroid[env_ids] = tip_pos_w[env_ids]
        self.state.prev_blood_centroid[env_ids] = tip_pos_w[env_ids]
