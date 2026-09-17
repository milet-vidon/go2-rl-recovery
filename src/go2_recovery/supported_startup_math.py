"""Fixed, measurement-only startup routing; no simulator or tensor dependency.

The caller must invoke ``supported_startup_mask`` ONCE at the real policy-start
boundary and latch its result for that episode. This pure function cannot
enforce an episode lifecycle. It does not assess standing acceptance, reset
state/history, choose a later handoff, or read requested poses, IDs or outcomes.

Input is the JSON-native mapping emitted by the frozen ``_start_snapshot``.
Its geometry boolean must come from the original normal_stance_geometry, and
its four forces from the original four named feet's CURRENT sample. This
module checks internal consistency, not measurement provenance or geometry
from unavailable body coordinates. Upstream source/contract checks remain
necessary. Extra snapshot fields are neither read nor copied.
"""

from collections.abc import Mapping
import math


PROTOCOL = "supported_nonfallen_startup_predicate_v1"
CONDITIONS = (
    "fresh_contacts", "four_current_feet_gt_5N", "normal_geometry",
    "no_base_contact", "tilt_lt_20deg", "height_in_closed_0p20_0p55m",
    "linear_speed_lt_0p5m_s", "angular_speed_lt_1rad_s",
    "not_eligible_settled_fallen",
)
BOOLEAN_FIELDS = (
    "contacts_fresh_since_pose_write", "geometry_ok", "base_contact",
    "any_body_contact", "settled", "standing_at_policy_start",
    "fallen_at_policy_start", "eligible_settled_fallen_recovery",
)
SCALAR_FIELDS = (
    "tilt_from_upright_deg", "height_m", "quiet_supported_window_s", "gravity_error",
)
VECTOR_WIDTHS = {
    "foot_vertical_forces_N": 4,
    "root_linear_velocity_w_m_s": 3,
    "root_angular_velocity_w_rad_s": 3,
}
REQUIRED_FIELDS = (*BOOLEAN_FIELDS, *SCALAR_FIELDS, *VECTOR_WIDTHS, "vertical_foot_contacts")


class SupportedStartupInputError(ValueError):
    """Invalid measurements: records are diagnostics, NOT a fallback mask."""

    def __init__(self, records, message="Invalid policy-start snapshot; no routing mask returned"):
        self.records = records
        self.invalid_indices = tuple(i for i, row in enumerate(records) if not row["valid"])
        super().__init__(f"{message}; invalid row indices={self.invalid_indices}")


def _finite_number(value):
    # Exact types deliberately reject bool, numeric strings and implicit casts.
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def evaluate_supported_startup_snapshot(snapshot):
    """Return a detached diagnostic record; invalid input has no actor choice.

    Validation of original count/fallen/eligibility/settled labels is not an
    extra selection threshold. In particular settled=False is fully allowed.
    The standing label is checked only for necessary conditions: its original
    contact history is absent here, so standing=False is not reverse-inferred.
    """
    values, errors = {}, []
    conditions = dict.fromkeys(CONDITIONS)
    record = {
        "protocol": PROTOCOL, "valid": False, "conditions": conditions,
        "inputs": values, "invalid_reasons": errors,
        "selected_standing": None, "selected_actor": None,
        "standing_acceptance_assessed": False,
    }
    if not isinstance(snapshot, Mapping):
        errors.append("snapshot must be a mapping")
        return record
    for key in REQUIRED_FIELDS:
        if key not in snapshot:
            errors.append(f"missing {key}")
            continue
        value = snapshot[key]
        if key in BOOLEAN_FIELDS:
            valid = type(value) is bool
        elif key in SCALAR_FIELDS:
            valid = _finite_number(value)
        elif key in VECTOR_WIDTHS:
            valid = (type(value) in (list, tuple) and len(value) == VECTOR_WIDTHS[key]
                     and all(_finite_number(item) for item in value))
        else:
            valid = type(value) is int and 0 <= value <= 4
        if not valid:
            errors.append(f"invalid type, dimensions or finite value: {key}")
        else:
            values[key] = list(value) if key in VECTOR_WIDTHS else value

    # Do not invent defaults for missing data or evaluate partial predicates.
    if errors:
        return record
    forces = values["foot_vertical_forces_N"]
    linear = math.hypot(*values["root_linear_velocity_w_m_s"])
    angular = math.hypot(*values["root_angular_velocity_w_rad_s"])
    values["root_linear_speed_m_s"] = linear
    values["root_angular_speed_rad_s"] = angular
    tilt, height = values["tilt_from_upright_deg"], values["height_m"]
    fresh = values["contacts_fresh_since_pose_write"]
    geometry, base = values["geometry_ok"], values["base_contact"]
    grounded, settled = values["any_body_contact"], values["settled"]
    standing, fallen = values["standing_at_policy_start"], values["fallen_at_policy_start"]
    eligible = values["eligible_settled_fallen_recovery"]
    count = sum(force > 5.0 for force in forces)
    conditions.update({
        "fresh_contacts": fresh,
        "four_current_feet_gt_5N": count == 4,
        "normal_geometry": geometry,
        "no_base_contact": not base,
        "tilt_lt_20deg": tilt < 20.0,
        "height_in_closed_0p20_0p55m": 0.20 <= height <= 0.55,
        "linear_speed_lt_0p5m_s": linear < 0.5,
        "angular_speed_lt_1rad_s": angular < 1.0,
        "not_eligible_settled_fallen": not eligible,
    })
    if not fresh:
        errors.append("contacts are not fresh at policy start")
    if not math.isfinite(linear) or not math.isfinite(angular):
        errors.append("derived root speed is not finite")
    if not 0.0 <= tilt <= 180.0 or values["gravity_error"] < 0.0:
        errors.append("tilt or gravity-error domain is invalid")
    quiet = values["quiet_supported_window_s"]
    if quiet < 0.0:
        errors.append("quiet supported window cannot be negative")
    if values["vertical_foot_contacts"] != count:
        errors.append("vertical-foot count contradicts current forces >5N")
    if (base or any(abs(force) > 1.0 for force in forces)) and not grounded:
        errors.append("any-body-contact flag contradicts measured foot/base contact")
    if settled != (fresh and quiet + 1e-9 >= 0.25):
        errors.append("settled flag contradicts original fresh/quiet-window definition")
    if settled and not grounded:
        errors.append("settled flag contradicts lack of current body support")
    expected_fallen = grounded and (tilt >= 60.0 or (base and height < 0.26))
    if fallen != expected_fallen:
        errors.append("fallen flag contradicts original measured-state definition")
    if eligible != (fallen and settled and not standing):
        errors.append("eligible flag contradicts original fallen/settled/standing definition")
    necessary_standing = (
        geometry and not base and count == 4 and 0.30 < height < 0.55
        and linear < 0.50 and angular < 1.0 and values["gravity_error"] < 0.35
    )
    if standing and (fallen or not necessary_standing):
        errors.append("standing flag contradicts necessary original instantaneous conditions")
    if errors:
        return record
    record["valid"] = True
    record["selected_standing"] = all(conditions.values())
    record["selected_actor"] = "stand" if record["selected_standing"] else "roll"
    return record


def supported_startup_mask(snapshots):
    """Return ``(tuple[bool], list[record])`` once all rows are valid.

    A single invalid row raises SupportedStartupInputError with all diagnostic
    records and returns NO mask. Never treat invalid as an ordinary roll row.
    This helper has no mode, timer, gate, RNG, file access or mutable state.
    """
    if type(snapshots) not in (list, tuple) or not snapshots:
        raise SupportedStartupInputError([], "Expected a nonempty list/tuple of real start snapshots")
    records = [evaluate_supported_startup_snapshot(snapshot) for snapshot in snapshots]
    if any(not record["valid"] for record in records):
        raise SupportedStartupInputError(records)
    return tuple(record["selected_standing"] for record in records), records
