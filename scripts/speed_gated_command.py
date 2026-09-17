"""Isolated training-only velocity sampler; no installed task changes."""
import json
from pathlib import Path
import sys

import torch
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/go2_recovery"))
from speed_gate_math import GateCurriculum


class GatedVelocityCommand(UniformVelocityCommand):
    """Preserve reverse/stand/y/yaw draws; scale only moving positive vx.

    Curriculum observations use raw exp(-error^2 / 0.5^2), not weighted
    training returns. Only completed, unreset command windows may advance.
    This is a bounded engineering adaptation, not the author's bin algorithm.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if (cfg.ranges.lin_vel_x != (-0.5, 1.0) or cfg.heading_command
                or cfg.rel_standing_envs != 0.3 or env.step_dt != 0.02):
            raise ValueError("Unexpected parent command contract")
        self.course = GateCurriculum()
        self._resetting = False
        self.age = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.samples = torch.zeros_like(self.age)
        self.lin_sum = torch.zeros(self.num_envs, dtype=torch.float64, device=self.device)
        self.ang_sum = torch.zeros_like(self.lin_sum)
        self.sample_cap = torch.full_like(self.lin_sum, 0.8)
        self.windows = []
        self.discarded_reset_windows = 0
        self.resampled = 0
        self.standing_draws = 0
        self.reverse_draws = 0
        self.positive_draws = 0
        self.log_dir = Path(env.cfg.log_dir)

    def reset(self, env_ids=None):
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        self.discarded_reset_windows += int((self.samples[ids] > 0).sum().item())
        self._resetting = True
        try:
            return super().reset(ids)
        finally:
            self._resetting = False

    def _update_metrics(self):
        super()._update_metrics()
        self.age += 1
        linear = torch.exp(-torch.sum((self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2]) ** 2, dim=1) / 0.25)
        angular = torch.exp(-(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2]) ** 2 / 0.25)
        if not bool(torch.isfinite(linear).all() and torch.isfinite(angular).all()):
            raise ValueError("Nonfinite measured curriculum reward")
        # Warm-up excludes reset-state observations and first 0.5s of transitions.
        mature = self.age > 25
        self.lin_sum += torch.where(mature, linear, 0).double()
        self.ang_sum += torch.where(mature, angular, 0).double()
        self.samples += mature.long()

    def _resample_command(self, env_ids):
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if not self._resetting:
            records = []
            for i in ids.tolist():
                count = int(self.samples[i])
                if count:
                    records.append({"cap_at_sample": round(float(self.sample_cap[i]), 8),
                        "vx": float(self.vel_command_b[i, 0]),
                        "mean_linear_reward": float(self.lin_sum[i]) / count,
                        "mean_angular_reward": float(self.ang_sum[i]) / count,
                        "duration_s": round(count * self._env.step_dt, 8),
                        "terminated": False, "standing": bool(self.is_standing_env[i])})
            before = self.course.snapshot()
            self.course.observe(records)
            self.windows.extend(records)
            if self.course.snapshot() != before:
                self.save_course()
        super()._resample_command(env_ids)
        original = self.vel_command_b[ids].clone()
        standing = self.is_standing_env[ids].clone()
        positive = (original[:, 0] > 0) & ~standing
        cap = self.course.snapshot()["cap"]
        self.vel_command_b[ids[positive], 0] = original[positive, 0] * cap
        current = self.vel_command_b[ids]
        if (not torch.equal(current[:, 1:], original[:, 1:])
                or not torch.equal(current[~positive], original[~positive])
                or not torch.equal(self.is_standing_env[ids], standing)
                or not bool((current[positive, 0] <= cap).all())):
            raise AssertionError("Sampler changed preserved command channels")
        self.resampled += len(ids)
        self.standing_draws += int(standing.sum())
        self.reverse_draws += int(((original[:, 0] < 0) & ~standing).sum())
        self.positive_draws += int(positive.sum())
        self.sample_cap[ids] = cap
        self.age[ids] = self.samples[ids] = 0
        self.lin_sum[ids] = self.ang_sum[ids] = 0

    def save_course(self):
        document = {"protocol": "bounded_speed_gate_training_v1", "course": self.course.snapshot(),
            "common_control_steps": self._env.common_step_counter,
            "resampled": self.resampled, "standing_draws": self.standing_draws,
            "reverse_draws": self.reverse_draws, "positive_draws": self.positive_draws,
            "discarded_reset_windows": self.discarded_reset_windows,
            "complete_command_windows": self.windows,
            "not_acceptance": True, "warmup_s": 0.5,
            "note": "Unweighted linear and angular exponential scores, std0.5; actual endpoint retention must be independently evaluated. Unfinished and reset windows never promote."}
        (self.log_dir / "speed_curriculum.json").write_text(json.dumps(document, indent=2, allow_nan=False), encoding="utf-8")
