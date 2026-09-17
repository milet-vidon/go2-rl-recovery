"""Read-only training exposure measurement; never alters command/contact history.

Completed windows are exact unchanged nonzero command segments, discarded on
episode resets. Sensor timing contacts use its configured norm threshold, NOT
the evaluation's current vertical-force >5N definition. Switch counts are raw
50Hz observations, not a debounced gait-frequency claim. Finite-error flags are
sticky device values, checked at each24-step telemetry flush and at finish.
"""
import json
from pathlib import Path

import torch


def eligible(command):
    return (command[:, 0] > .1) & (command[:, 1].abs() < .1) & (command[:, 2].abs() < .3)


class ExposureWindows:
    def __init__(self, n, device):
        self.steps = torch.zeros(n, device=device, dtype=torch.long)
        self.touch = torch.zeros((n, 4), device=device, dtype=torch.long)
        self.lift = self.touch.clone()
        self.whole_eligible = torch.ones(n, device=device, dtype=torch.bool)
        self.known_start = torch.zeros(n, device=device, dtype=torch.bool)
        # initial_partial, complete, eligible_complete, cycle_complete,
        # reset_discarded, eligible_intervals. Copy as one batch only at summary.
        self.counts = torch.zeros(6, device=device, dtype=torch.long)
        self.invalid = torch.zeros((), device=device, dtype=torch.bool)
        self.total_intervals = 0

    def update(self, command, next_command, before_contact, after_contact, done):
        n = self.steps.shape[0]
        if command.shape != (n, 3) or next_command.shape != (n, 3) or before_contact.shape != (n, 4) or after_contact.shape != (n, 4) or done.shape != (n,):
            raise ValueError("Exposure shape mismatch")
        if before_contact.dtype != torch.bool or after_contact.dtype != torch.bool or done.dtype != torch.bool:
            raise ValueError("Contact/done masks must be bool")
        self.invalid |= ~torch.isfinite(command).all() | ~torch.isfinite(next_command).all()
        gate = eligible(command)
        self.steps += 1
        self.whole_eligible &= gate
        self.touch += (~before_contact & after_contact).long()
        self.lift += (before_contact & ~after_contact).long()
        changed = (next_command != command).any(-1)
        initial_partial = changed & ~done & ~self.known_start
        complete = changed & ~done & self.known_start
        qualifies = complete & self.whole_eligible
        cycles = qualifies & (self.touch >= 2).all(-1) & (self.lift >= 2).all(-1)
        self.counts += torch.stack((initial_partial.sum(), complete.sum(), qualifies.sum(),
                                    cycles.sum(), done.sum(), (gate & ~done).sum()))
        self.total_intervals += n
        cleared = changed | done
        self.steps.masked_fill_(cleared, 0)
        self.touch.masked_fill_(cleared[:, None], 0)
        self.lift.masked_fill_(cleared[:, None], 0)
        self.whole_eligible |= cleared
        self.known_start |= cleared

    def summary(self):
        values = torch.cat((self.counts, (self.steps > 0).sum().reshape(1),
                            self.invalid.long().reshape(1))).detach().cpu().tolist()
        if values[7]:
            raise ValueError("Nonfinite command observed since exposure start")
        return {"environment_intervals": self.total_intervals,
                "eligible_nonreset_environment_intervals": values[5],
                "completed_command_windows": values[1],
                "completed_eligible_command_windows": values[2],
                "completed_eligible_windows_two_raw_touch_and_lift_per_foot": values[3],
                "reset_discarded_windows": values[4],
                "initial_partial_windows_discarded": values[0],
                "unfinished_windows_at_end": values[6],
                "minimum32_cycle_exposed_windows_met": values[3] >= 32,
                "contact_semantics": "Sensor current_contact_time>0 at50Hz; sensor norm threshold, not evaluator vertical5N; counts not debounced",
                "window_semantics": "Exact same commanded vector; reset and initial partial windows discarded. Identical-valued resampling is not a boundary."}


class DurationExposure:
    def __init__(self, env, path):
        self.env, self.base = env, env.unwrapped
        self.sensor = self.base.scene.sensors["contact_forces"]
        cfg = self.sensor.cfg
        if cfg.history_length != 3 or cfg.update_period != .005 or cfg.track_air_time is not True or cfg.force_threshold != 1.0:
            raise ValueError("Exposure requires the pinned per-substep contact sensor configuration")
        # .data is a lazy-updating property and can refresh reset rows. Read only
        # cached buffers already updated by the normal simulation/reward path.
        self.data = self.sensor._data
        names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        self.ids, actual = self.sensor.find_bodies(names, preserve_order=True)
        if actual != names or len(self.ids) != 4:
            raise ValueError("Wrong ordered feet in exposure logger")
        self.windows = ExposureWindows(self.base.num_envs, self.base.device)
        self.original_step = env.step
        self.stream = Path(path).open("x", encoding="utf-8", newline="\n")
        self.control_steps = 0
        # Preserve double accumulation of per-step float reductions, without
        # synchronizing each scalar to Python after every control interval.
        self.totals = torch.zeros(3, device=self.base.device, dtype=torch.float64)
        self.invalid = torch.zeros((), device=self.base.device, dtype=torch.bool)

        def measured_step(action):
            command = self.base.command_manager.get_command("base_velocity").clone()
            before = (self.data.current_contact_time[:, self.ids] > 0).clone()
            result = self.original_step(action)
            done = result[2].bool()
            after = self.data.current_contact_time[:, self.ids] > 0
            next_command = self.base.command_manager.get_command("base_velocity")
            self.windows.update(command, next_command, before, after, done)
            air = self.data.last_air_time[:, self.ids].clamp(max=.5)
            contact = self.data.last_contact_time[:, self.ids].clamp(max=.5)
            raw = air.var(-1, correction=1) + contact.var(-1, correction=1)
            self.invalid |= ~torch.isfinite(raw).all()
            gate = eligible(command) & ~done
            # Masked fixed-shape reductions avoid CUDA nonzero synchronization
            # from raw[~done]. Done samples are still excluded from these means.
            raw_sum = torch.where(~done, raw, torch.zeros_like(raw)).sum()
            raw_count = (~done).sum()
            self.totals += torch.stack((raw_sum.double(), (raw * gate).sum().double(), raw_count.double()))
            self.control_steps += 1
            if self.control_steps % 24 == 0:
                windows = self.windows.summary()
                values = torch.cat((self.invalid.double().reshape(1), gate.float().mean().double().reshape(1),
                                    raw_sum.double().reshape(1), raw_count.double().reshape(1),
                                    air.mean(0).double(), contact.mean(0).double())).detach().cpu().tolist()
                if values[0]:
                    raise ValueError("Nonfinite measured completed-duration variance since exposure start")
                self.stream.write(json.dumps({"control_step": self.control_steps,
                    "eligible_fraction": values[1], "raw_mean_nonreset": values[2] / values[3] if values[3] else None,
                    "last_air_mean_by_foot": values[4:8], "last_contact_mean_by_foot": values[8:12],
                    **windows}, allow_nan=False) + "\n")
                self.stream.flush()
            return result

        self.env.step = measured_step

    def finish(self):
        self.env.step = self.original_step
        self.stream.close()
        windows = self.windows.summary()
        values = torch.cat((self.totals, self.invalid.double().reshape(1))).detach().cpu().tolist()
        if values[3]:
            raise ValueError("Nonfinite measured completed-duration variance since exposure start")
        return {**windows, "control_steps": self.control_steps,
                "raw_variance_mean_nonreset": values[0] / max(1, values[2]),
                "gated_variance_sum": values[1],
                "training_acceptance": False, "observational_only": True,
                "initial_partial_windows_excluded": True,
                "lazy_sensor_property_never_read": True}
