"""Pure bookkeeping for an experimental, jointly gated speed curriculum.

This is neither a simulator controller nor a standing/recovery acceptance test.
Callers must summarize complete command windows; this module cannot establish
from an aggregate record whether the underlying command stayed unchanged.

Only 32 eligible windows make a decision, with no overlapping reuse. Both raw
mean rewards must be strictly greater than 0.8 in a successful window. A
decision with at least 80% successes advances 0.8 -> 0.9 -> 1.0 m/s. At the
1.0 cap, decisions continue to be recorded but the cap cannot advance further.
After an advance, remaining records bearing the old cap are rejected, not
carried into the new stage. The current cap is compared exactly, not fuzzily.

Each observe call validates its entire input batch before changing state.
Splitting valid records into different batches does not change decisions.
All returned data are copies, and caller input records are never modified.
"""

from collections.abc import Mapping
from copy import deepcopy
import math


class SpeedGateInputError(ValueError):
    """An input batch is malformed; the curriculum was left unchanged."""


class GateCurriculum:
    """Deterministic finite curriculum, beginning at the fixed 0.8 m/s cap."""

    CAPS = (0.8, 0.9, 1.0)
    # Literal decimal edges avoid 0.8 - 0.2 rounding above the intended 0.6.
    LOWER_EDGES = (0.6, 0.7, 0.8)
    WINDOW_COUNT = 32
    MIN_DURATION_S = 3.5
    REWARD_THRESHOLD = 0.8
    NUMERIC_FIELDS = (
        "cap_at_sample", "vx", "mean_linear_reward",
        "mean_angular_reward", "duration_s",
    )
    BOOL_FIELDS = ("terminated", "standing")

    def __init__(self):
        self._stage = 0
        self._pending_count = 0
        self._pending_successes = 0
        self._eligible_count = 0
        self._rejected_count = 0
        self._rejected_by_reason = {
            "standing": 0,
            "terminated": 0,
            "short_duration": 0,
            "different_cap": 0,
            "outside_command_band": 0,
        }
        self._decisions = []

    @property
    def cap(self):
        """Current upper command cap, in m/s."""
        return self.CAPS[self._stage]

    @classmethod
    def _validated_records(cls, records):
        if isinstance(records, (str, bytes, Mapping)):
            raise SpeedGateInputError("records must be an iterable of mappings")
        try:
            batch = list(records)
        except TypeError as exc:
            raise SpeedGateInputError("records must be an iterable of mappings") from exc
        validated = []
        for index, record in enumerate(batch):
            if not isinstance(record, Mapping):
                raise SpeedGateInputError(f"record {index} must be a mapping")
            value = {}
            for field in cls.NUMERIC_FIELDS:
                raw = record.get(field)
                if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                    raise SpeedGateInputError(f"record {index}: {field} must be a finite number")
                try:
                    number = float(raw)
                except (ValueError, OverflowError) as exc:
                    raise SpeedGateInputError(f"record {index}: {field} must be finite") from exc
                if not math.isfinite(number):
                    raise SpeedGateInputError(f"record {index}: {field} must be finite")
                value[field] = number
            for field in cls.BOOL_FIELDS:
                raw = record.get(field)
                if type(raw) is not bool:
                    raise SpeedGateInputError(f"record {index}: {field} must be a bool")
                value[field] = raw
            validated.append(value)
        return validated

    def _rejection_reason(self, record):
        if record["standing"]:
            return "standing"
        if record["terminated"]:
            return "terminated"
        if record["duration_s"] < self.MIN_DURATION_S:
            return "short_duration"
        if record["cap_at_sample"] != self.cap:
            return "different_cap"
        if not self.LOWER_EDGES[self._stage] <= record["vx"] <= self.cap:
            return "outside_command_band"
        return None

    def observe(self, records):
        """Consume a valid batch in order and return an independent snapshot.

        Required fields are cap_at_sample, vx, mean_linear_reward,
        mean_angular_reward, duration_s, terminated, and standing. Numeric
        fields accept finite Python int/float values (not bool); the last two
        fields require actual bool values. Extra fields are ignored. Finite
        durations below 3.5 s, including negative values, are ineligible rather
        than malformed. Invalid records abort the whole batch before mutation.

        Rejected-reason counts are mutually exclusive using the documented
        order: standing, terminated, duration, cap, then command band.
        """
        batch = self._validated_records(records)
        for record in batch:
            reason = self._rejection_reason(record)
            if reason is not None:
                self._rejected_count += 1
                self._rejected_by_reason[reason] += 1
                continue

            self._eligible_count += 1
            self._pending_count += 1
            self._pending_successes += int(
                record["mean_linear_reward"] > self.REWARD_THRESHOLD
                and record["mean_angular_reward"] > self.REWARD_THRESHOLD
            )
            if self._pending_count != self.WINDOW_COUNT:
                continue

            cap_before = self.cap
            # Integer form of success_rate >= 0.8; 26/32 passes, 25/32 fails.
            threshold_met = self._pending_successes * 5 >= self.WINDOW_COUNT * 4
            if threshold_met and self._stage < len(self.CAPS) - 1:
                self._stage += 1
            self._decisions.append({
                "count": self._pending_count,
                "successes": self._pending_successes,
                "success_rate": self._pending_successes / self._pending_count,
                "cap_before": cap_before,
                "cap_after": self.cap,
                "threshold_met": threshold_met,
                "promoted": self.cap != cap_before,
            })
            self._pending_count = 0
            self._pending_successes = 0
        return self.snapshot()

    def snapshot(self):
        """Return serializable counters and decisions without sharing state."""
        return {
            "cap": self.cap,
            "pending_count": self._pending_count,
            "pending_successes": self._pending_successes,
            "eligible_count": self._eligible_count,
            "rejected_count": self._rejected_count,
            "rejected_by_reason": dict(self._rejected_by_reason),
            "decisions": deepcopy(self._decisions),
        }
