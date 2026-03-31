from __future__ import annotations

from typing import Any

import torch


class BloodVisionObservationManager:
    """Builds camera + proprio observations for the blood vision task."""

    def __init__(self, cfg, num_envs: int, device: torch.device | str):
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.device = torch.device(device)

        self._camera: Any | None = None
        self._scene: Any | None = None

        self._obs_camera = torch.zeros(
            (
                self.num_envs,
                int(self.cfg.num_channels),
                int(self.cfg.obs_camera_height),
                int(self.cfg.obs_camera_width),
            ),
            dtype=torch.float32,
            device=self.device,
        )
        self._obs_position = torch.zeros(
            (self.num_envs, int(self.cfg.position_observation_dim)),
            dtype=torch.float32,
            device=self.device,
        )

    def bind_runtime(self, camera: Any, scene: Any) -> None:
        self._camera = camera
        self._scene = scene

    def reset(self, env_ids: torch.Tensor) -> None:
        self._obs_camera[env_ids] = 0.0
        self._obs_position[env_ids] = 0.0

    @staticmethod
    def _normalize_workspace_positions(
        pos_w: torch.Tensor,
        workspace_low_w: torch.Tensor,
        workspace_high_w: torch.Tensor,
    ) -> torch.Tensor:
        workspace_range = (workspace_high_w - workspace_low_w).clamp_min(1.0e-6)
        normalized = 2.0 * (pos_w - workspace_low_w) / workspace_range - 1.0
        return torch.clamp(normalized, -1.0, 1.0)

    def set_fixed_camera_pose(self) -> None:
        if self._camera is None or self._scene is None:
            return

        eyes_tensor = torch.tensor(self.cfg.camera_pos, dtype=torch.float32, device=self.device).unsqueeze(0)
        targets_tensor = torch.tensor(self.cfg.camera_target, dtype=torch.float32, device=self.device).unsqueeze(0)
        eyes = self._scene.env_origins + eyes_tensor
        targets = self._scene.env_origins + targets_tensor
        self._camera.set_world_poses_from_view(eyes=eyes, targets=targets)

    def _build_camera_observation(self) -> None:
        if self._camera is None:
            self._obs_camera.zero_()
            return

        camera_data = self._camera.data.output["rgb"]
        if camera_data is None or camera_data.numel() == 0:
            self._obs_camera.zero_()
            return

        rgb = camera_data[..., :3]
        if rgb.dtype != torch.float32:
            rgb = rgb.float()
        if rgb.max() > 1.0:
            rgb = rgb / 255.0

        rgb_nchw = rgb.permute(0, 3, 1, 2).contiguous()
        target_size = (int(self.cfg.obs_camera_height), int(self.cfg.obs_camera_width))
        if tuple(rgb_nchw.shape[-2:]) == target_size:
            self._obs_camera[:] = rgb_nchw
        else:
            self._obs_camera[:] = torch.nn.functional.interpolate(
                rgb_nchw,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )

    def _build_position_observation(
        self,
        tip_pos_w: torch.Tensor,
        tip_dir_w: torch.Tensor,
        ee_goal_pos_w: torch.Tensor,
        workspace_low_w: torch.Tensor,
        workspace_high_w: torch.Tensor,
        raw_actions: torch.Tensor,
        contact_force: torch.Tensor,
        step_count: torch.Tensor,
        max_episode_length: int,
    ) -> torch.Tensor:
        tip_pos_normalized = self._normalize_workspace_positions(
            tip_pos_w,
            workspace_low_w,
            workspace_high_w,
        )
        workspace_range = (workspace_high_w - workspace_low_w).clamp_min(1.0e-6)
        goal_error_normalized = torch.clamp(2.0 * (ee_goal_pos_w - tip_pos_w) / workspace_range, -1.0, 1.0)
        contact_ratio = torch.clamp(
            contact_force / max(float(self.cfg.severe_contact_force_threshold), 1.0e-6),
            min=0.0,
            max=1.0,
        ).unsqueeze(1)
        step_ratio = torch.clamp(
            step_count.to(dtype=torch.float32) / max(float(max_episode_length), 1.0),
            min=0.0,
            max=1.0,
        ).unsqueeze(1)

        return torch.cat(
            (
                tip_pos_normalized,
                goal_error_normalized,
                tip_dir_w,
                raw_actions,
                contact_ratio,
                step_ratio,
            ),
            dim=1,
        )

    def update(
        self,
        tip_pos_w: torch.Tensor,
        tip_dir_w: torch.Tensor,
        ee_goal_pos_w: torch.Tensor,
        workspace_low_w: torch.Tensor,
        workspace_high_w: torch.Tensor,
        raw_actions: torch.Tensor,
        contact_force: torch.Tensor,
        step_count: torch.Tensor,
        max_episode_length: int,
    ) -> None:
        self.set_fixed_camera_pose()
        self._build_camera_observation()
        self._obs_position[:] = self._build_position_observation(
            tip_pos_w=tip_pos_w,
            tip_dir_w=tip_dir_w,
            ee_goal_pos_w=ee_goal_pos_w,
            workspace_low_w=workspace_low_w,
            workspace_high_w=workspace_high_w,
            raw_actions=raw_actions,
            contact_force=contact_force,
            step_count=step_count,
            max_episode_length=max_episode_length,
        )

    def get_observations(self) -> dict[str, torch.Tensor]:
        return {
            "camera": self._obs_camera.clone(),
            "position": self._obs_position.clone(),
        }
