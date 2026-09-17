"""One command-gated completed-duration penalty; no sensor/history writes.

Raw semantics match Isaac Lab Spot ``air_time_variance_penalty``: sum of
sample variances (correction=1) of last air/contact times, each clipped at .5s.
These sensor timers use the sensor's force-norm threshold, not the evaluation
CSV's >5N vertical-contact duty definition. Last durations can be stale after
command changes, and all-zero timers after reset are not completed cycles.

The caller alone supplies reward weight -10 and RewardManager supplies dt.
This module imports no simulator and never changes commands or observations.
"""
from __future__ import annotations

from types import SimpleNamespace
from weakref import WeakKeyDictionary

import torch


ORDERED_FOOT_NAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
FOOT_NAMES = frozenset(ORDERED_FOOT_NAMES)
OFFICIAL_SOURCE_SHA256 = "14e2d61f0061f27191e23228469e590c27ffef001646812993b9d4e0cce929a6"
_SENSOR_CONFIG_CACHE = WeakKeyDictionary()


def duration_variance_components(
    last_air_time: torch.Tensor,
    last_contact_time: torch.Tensor,
    commands: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(raw_variance, eligible_bool, gated_positive_penalty)``.

    Timers have shape [N,4], commands [N,3]=(vx,vy,wz), on one device/dtype.
    Strict command bounds: vx>.1, abs(vy)<.1, abs(wz)<.3. No weighting,
    time-step multiplication, reset, warmup mask, desired gait period or state.
    Reject invalid measurements even on command-ineligible rows.
    """
    inputs = (("last_air_time", last_air_time, 4),
              ("last_contact_time", last_contact_time, 4), ("commands", commands, 3))
    for name, value, width in inputs:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.dtype not in (torch.float32, torch.float64):
            raise TypeError(f"{name} must be float32 or float64")
        if value.layout != torch.strided or value.device.type not in ("cpu", "cuda"):
            raise ValueError(f"{name} must be a dense CPU/CUDA tensor")
        if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] != width:
            raise ValueError(f"{name} must have nonempty shape [N,{width}]")
    if any(v.shape[0] != last_air_time.shape[0] or v.device != last_air_time.device
           or v.dtype != last_air_time.dtype for _, v, _ in inputs):
        raise ValueError("all inputs must have the same batch, device and dtype")
    invalid = (~torch.isfinite(last_air_time)).any() | (~torch.isfinite(last_contact_time)).any()
    invalid = invalid | (~torch.isfinite(commands)).any()
    invalid = invalid | (last_air_time < 0).any() | (last_contact_time < 0).any()
    if bool(invalid):
        raise ValueError("measurements must be finite and durations nonnegative")
    raw = torch.var(torch.clamp(last_air_time, max=0.5), dim=1, correction=1)
    raw = raw + torch.var(torch.clamp(last_contact_time, max=0.5), dim=1, correction=1)
    eligible = (commands[:, 0] > 0.1) & (commands[:, 1].abs() < 0.1) & (commands[:, 2].abs() < 0.3)
    return raw, eligible, torch.where(eligible, raw, torch.zeros_like(raw))


def runtime_duration_variance_components(env, sensor_cfg, command_name="base_velocity"):
    """Read exactly the resolved four feet and expose the three diagnostics.

    ``sensor_cfg`` must be an Isaac Lab SceneEntityCfg resolved by its manager.
    Any permutation of the four actual foot body IDs is valid; all-body slices,
    duplicates and calf/base substitutions are rejected. No Isaac Lab import is
    needed here, so the same wrapper can be CPU-tested using a fake environment.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    if sensor.cfg.track_air_time is not True:
        raise ValueError("contact sensor must track air/contact time")
    ids = sensor_cfg.body_ids
    if not isinstance(ids, (list, tuple)) or len(ids) != 4:
        raise ValueError("sensor_cfg.body_ids must explicitly resolve four feet")
    if any(type(i) is not int or not 0 <= i < len(sensor.body_names) for i in ids) or len(set(ids)) != 4:
        raise ValueError("foot IDs must be four unique in-range integer sensor-body IDs")
    if frozenset(sensor.body_names[i] for i in ids) != FOOT_NAMES:
        raise ValueError("duration penalty requires FL/FR/RL/RR foot bodies, not joint/robot indices")
    data = sensor.data
    return duration_variance_components(
        data.last_air_time[:, ids], data.last_contact_time[:, ids],
        env.command_manager.get_command(command_name),
    )


def command_gated_air_time_variance_penalty(env, sensor_cfg, command_name="base_velocity"):
    """RewardTermCfg callable returning a positive unweighted [N] penalty."""
    return runtime_duration_variance_components(env, sensor_cfg, command_name)[2]


def balanced_gait_duration_reward(env):
    """No-parameter trainer entry; cache only resolved sensor-body metadata.

    Use RewardTermCfg(func=balanced_gait_duration_reward, weight=-10, params={}).
    Cached IDs belong to the sensor object, not the articulation or its joints.
    Weak keys avoid retaining closed environments. No runtime tensor is cached,
    and no field in the environment, sensor or their history is written.
    """
    sensor = env.scene.sensors["contact_forces"]
    cfg = _SENSOR_CONFIG_CACHE.get(sensor)
    if cfg is None:
        names = sensor.body_names
        if any(names.count(name) != 1 for name in ORDERED_FOOT_NAMES):
            raise ValueError("contact sensor must have exactly one of each Go2 foot body")
        cfg = SimpleNamespace(name="contact_forces", body_ids=[names.index(name) for name in ORDERED_FOOT_NAMES])
        _SENSOR_CONFIG_CACHE[sensor] = cfg
    return command_gated_air_time_variance_penalty(env, cfg)
