"""CPU stdlib tests only; actual report counts are NOT new simulation results."""

import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "go2_recovery"))
from supported_startup_math import (BOOLEAN_FIELDS, REQUIRED_FIELDS, SCALAR_FIELDS,
    VECTOR_WIDTHS, SupportedStartupInputError, evaluate_supported_startup_snapshot,
    supported_startup_mask)


def ordinary():
    return {
        "contacts_fresh_since_pose_write": True, "geometry_ok": True,
        "base_contact": False, "any_body_contact": True, "settled": False,
        "standing_at_policy_start": False, "fallen_at_policy_start": False,
        "eligible_settled_fallen_recovery": False, "tilt_from_upright_deg": 5.0,
        "height_m": 0.28, "quiet_supported_window_s": 0.0, "gravity_error": 0.087,
        "foot_vertical_forces_N": [20.0] * 4, "vertical_foot_contacts": 4,
        "root_linear_velocity_w_m_s": [0.10, 0.0, 0.0],
        "root_angular_velocity_w_rad_s": [0.0, 0.20, 0.0],
    }


def with_changes(**changes):
    state = ordinary()
    state.update(changes)
    return state


class SupportedStartupTests(unittest.TestCase):
    def assert_invalid(self, state, reason=None):
        with self.assertRaises(SupportedStartupInputError) as context:
            supported_startup_mask([state])
        record = context.exception.records[0]
        self.assertFalse(record["valid"])
        self.assertIsNone(record["selected_standing"])
        self.assertIsNone(record["selected_actor"])
        self.assertFalse(record["standing_acceptance_assessed"])
        if reason:
            self.assertTrue(any(reason in item for item in record["invalid_reasons"]))

    def test_ordinary_selection_is_not_settled_or_accepted(self):
        mask, records = supported_startup_mask([ordinary()])
        self.assertEqual(mask, (True,))
        self.assertTrue(all(records[0]["conditions"].values()))
        self.assertFalse(records[0]["inputs"]["settled"])
        self.assertFalse(records[0]["inputs"]["standing_at_policy_start"])
        self.assertFalse(records[0]["standing_acceptance_assessed"])

    def test_current_force_strict_boundary_each_foot(self):
        for index in range(4):
            for value, selected in [(5.0, False), (math.nextafter(5.0, math.inf), True),
                                    (math.nextafter(5.0, -math.inf), False)]:
                state = ordinary()
                state["foot_vertical_forces_N"][index] = value
                state["vertical_foot_contacts"] = 4 if selected else 3
                with self.subTest(index=index, value=value):
                    self.assertEqual(supported_startup_mask([state])[0], (selected,))

    def test_tilt_strict_boundary(self):
        for value, selected in [(20.0, False), (math.nextafter(20.0, 0.0), True),
                                (math.nextafter(20.0, math.inf), False)]:
            with self.subTest(value=value):
                self.assertEqual(supported_startup_mask([with_changes(tilt_from_upright_deg=value)])[0], (selected,))

    def test_height_closed_endpoints(self):
        for value, selected in [(.20, True), (.55, True),
                                (math.nextafter(.20, 0.0), False),
                                (math.nextafter(.55, math.inf), False)]:
            with self.subTest(value=value):
                self.assertEqual(supported_startup_mask([with_changes(height_m=value)])[0], (selected,))

    def test_speed_strict_boundaries_and_world_vector_norm(self):
        for field, limit, diagonal in [("root_linear_velocity_w_m_s", .5, [.3, .4, 0.0]),
                                      ("root_angular_velocity_w_rad_s", 1.0, [.6, .8, 0.0])]:
            for vector, selected in [([limit, 0.0, 0.0], False),
                                     ([math.nextafter(limit, 0.0), 0.0, 0.0], True),
                                     ([math.nextafter(limit, math.inf), 0.0, 0.0], False),
                                     (diagonal, False), ([-limit, 0.0, 0.0], False)]:
                with self.subTest(field=field, vector=vector):
                    self.assertEqual(supported_startup_mask([with_changes(**{field: vector})])[0], (selected,))

    def test_geometry_and_base_fail_are_valid_nonselection(self):
        for state in [with_changes(geometry_ok=False), with_changes(base_contact=True)]:
            mask, records = supported_startup_mask([state])
            self.assertEqual(mask, (False,))
            self.assertTrue(records[0]["valid"])
            self.assertEqual(records[0]["selected_actor"], "roll")

    def test_fallen_eligibility_and_unsettled_ambiguity_are_not_shortcuts(self):
        fallen = with_changes(tilt_from_upright_deg=90.0, fallen_at_policy_start=True,
                              settled=True, quiet_supported_window_s=1.0,
                              eligible_settled_fallen_recovery=True)
        ambiguous = with_changes(tilt_from_upright_deg=90.0, fallen_at_policy_start=True)
        self.assertEqual(supported_startup_mask([fallen, ambiguous])[0], (False, False))

    def test_joint_speed_and_settled_are_not_new_selection_gates(self):
        state = with_changes(max_joint_speed_rad_s=1e6)
        mask, records = supported_startup_mask([state])
        self.assertEqual(mask, (True,))
        self.assertNotIn("max_joint_speed_rad_s", records[0]["inputs"])
        instant_standing = with_changes(height_m=.32, standing_at_policy_start=True)
        self.assertEqual(supported_startup_mask([instant_standing])[0], (True,))

    def test_missing_every_required_field_is_invalid(self):
        for key in REQUIRED_FIELDS:
            state = ordinary()
            del state[key]
            with self.subTest(key=key):
                self.assert_invalid(state, "missing")

    def test_pseudo_booleans_and_numeric_bool_are_rejected(self):
        for key in BOOLEAN_FIELDS:
            for value in [0, 1, "false", None, []]:
                with self.subTest(key=key, value=value):
                    self.assert_invalid(with_changes(**{key: value}), "invalid type")
        for key in SCALAR_FIELDS:
            self.assert_invalid(with_changes(**{key: True}), "invalid type")
        self.assert_invalid(with_changes(vertical_foot_contacts=True), "invalid type")
        self.assert_invalid(with_changes(vertical_foot_contacts=4.0), "invalid type")

    def test_nonfinite_scalars_and_vectors_are_invalid(self):
        for value in [float("nan"), float("inf"), -float("inf")]:
            for key in SCALAR_FIELDS:
                with self.subTest(value=value, key=key):
                    self.assert_invalid(with_changes(**{key: value}), "finite")
            for key, width in VECTOR_WIDTHS.items():
                vector = [0.0] * width
                vector[-1] = value
                with self.subTest(value=value, key=key):
                    self.assert_invalid(with_changes(**{key: vector}), "finite")
        self.assert_invalid(with_changes(height_m=10 ** 1000), "finite")
        self.assert_invalid(with_changes(root_linear_velocity_w_m_s=[1.7e308] * 3), "derived root speed")

    def test_wrong_dimensions_nested_values_and_string_casts_are_invalid(self):
        for key, width in VECTOR_WIDTHS.items():
            for vector in [None, "0,0,0", [0.0] * (width-1), [0.0] * (width+1),
                           [[0.0]] * width, [True] * width, ["0"] * width]:
                with self.subTest(key=key, vector=vector):
                    self.assert_invalid(with_changes(**{key: vector}), "dimensions")
        self.assert_invalid(with_changes(height_m="0.28"), "invalid type")

    def test_stale_contacts_are_invalid_not_roll_fallback(self):
        self.assert_invalid(with_changes(contacts_fresh_since_pose_write=False), "not fresh")

    def test_contradictory_original_flags_are_invalid(self):
        cases = [
            ({"vertical_foot_contacts": 3}, "count contradicts"),
            ({"any_body_contact": False}, "any-body-contact"),
            ({"eligible_settled_fallen_recovery": True}, "eligible flag"),
            ({"fallen_at_policy_start": True}, "fallen flag"),
            ({"settled": True}, "settled flag"),
            ({"quiet_supported_window_s": .26}, "settled flag"),
            ({"standing_at_policy_start": True}, "standing flag"),
            ({"quiet_supported_window_s": -1.0}, "cannot be negative"),
            ({"tilt_from_upright_deg": -1.0}, "domain"),
            ({"tilt_from_upright_deg": 181.0}, "domain"),
            ({"gravity_error": -1.0}, "domain"),
        ]
        for changes, reason in cases:
            with self.subTest(changes=changes):
                self.assert_invalid(with_changes(**changes), reason)

    def test_one_invalid_row_blocks_entire_batch_but_preserves_records(self):
        with self.assertRaises(SupportedStartupInputError) as context:
            supported_startup_mask([ordinary(), with_changes(height_m=None), ordinary()])
        self.assertEqual(context.exception.invalid_indices, (1,))
        self.assertEqual(len(context.exception.records), 3)
        self.assertEqual([row["valid"] for row in context.exception.records], [True, False, True])
        self.assertIsNone(context.exception.records[1]["selected_standing"])
        self.assertFalse(hasattr(context.exception, "mask"))

    def test_invalid_batch_and_snapshot_types(self):
        for value in [[], (), {}, None, "states", iter([ordinary()])]:
            with self.subTest(value=type(value)), self.assertRaises(SupportedStartupInputError):
                supported_startup_mask(value)
        self.assert_invalid(None, "mapping")

    def test_no_mutation_aliasing_pose_id_outcome_or_rng_inputs(self):
        state = ordinary()
        original = copy.deepcopy(state)
        first = supported_startup_mask([state])
        self.assertEqual(state, original)
        first[1][0]["inputs"]["foot_vertical_forces_N"][0] = -999
        self.assertEqual(state, original)
        state.update({"trial": object(), "pose": object(), "bank_state_id": object(),
                      "classification": object(), "results": object(), "successes": object()})
        second = supported_startup_mask([state])
        self.assertEqual(second[0], (True,))
        self.assertEqual(second[1][0]["inputs"]["foot_vertical_forces_N"], [20.0] * 4)
        self.assertEqual(second, supported_startup_mask([state]))
        tree = ast.parse((ROOT / "src/go2_recovery/supported_startup_math.py").read_text(encoding="utf-8"))
        modules = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        modules.update(alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names)
        self.assertEqual(modules, {"collections.abc", "math"})

    def test_actual_frozen_snapshots_offline_mapping_20_0_0(self):
        cases = [
            ("upright", "20260917-handoff1999-to3547-upright-control", 20,
             "2f3e3f6f5f2fae44883bf9a03a740404425f5bbfffb4710b60802801a125eeae"),
            ("side", "20260917-handoff1999-to3547-side", 0,
             "e77c40de96649e454ff60d6f8b681f08c7dac3d61730c81c292605025d942715"),
            ("upside_down", "20260917-handoff1999-to3547-upside_down", 0,
             "a6e853374d433e375047cac3c40b865736b54535054564ccdaed9cbc6f5f3605"),
        ]
        for pose, directory, expected, sha in cases:
            with self.subTest(pose=pose):
                path = ROOT / "evaluations" / directory / "model_1999.pt_recovery_metrics.json"
                raw = path.read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), sha)
                snapshots = json.loads(raw)["results"][pose]["policy_start_state"]
                self.assertEqual(len(snapshots), 20)
                mask, records = supported_startup_mask(snapshots)
                self.assertEqual(sum(mask), expected)
                self.assertTrue(all(row["valid"] for row in records))
                self.assertTrue(all(not row["standing_acceptance_assessed"] for row in records))
                # Release snapshots explicitly have stale/null contact data.
                releases = json.loads(raw)["results"][pose]["release_state_before_settling"]
                with self.assertRaises(SupportedStartupInputError):
                    supported_startup_mask(releases)


if __name__ == "__main__":
    unittest.main(verbosity=2)
