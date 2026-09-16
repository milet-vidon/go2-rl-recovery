"""CPU-only state-bank validation/replay tests; never launches Isaac Sim."""

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch


_ROOT = Path(__file__).parents[1]
_SPEC = importlib.util.spec_from_file_location("recovery_state_bank", _ROOT / "src/go2_recovery/recovery_state_bank.py")
bank_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bank_module)


def fixture():
    angles = np.array([-math.pi / 2, math.pi / 2, math.pi] * 2, dtype=np.float32)
    poses = np.zeros((6, 7), dtype=np.float32)
    poses[:, 2] = .2
    poses[:, 3] = np.cos(angles / 2)
    poses[:, 4] = np.sin(angles / 2)
    states = {
        "root_pose_local": poses, "root_velocity_w": np.zeros((6, 6), dtype=np.float32),
        "joint_pos": np.zeros((6, 12), dtype=np.float32), "joint_vel": np.zeros((6, 12), dtype=np.float32),
        "pose_class": np.array(["left", "right", "back"] * 2),
        "requested_pose_class": np.array(["left", "right", "back", "random", "side", "upside_down"]),
        "state_id": np.arange(6, dtype=np.int64), "split": np.array([0, 1] * 3, dtype=np.int8),
    }
    metadata = {
        "schema_version": "nominal_pd_fallen_v1", "controller": "nominal_pose_PD",
        "self_collisions_enabled": True, "joint_names": [f"joint{i}" for i in range(12)],
        "body_names": ["base", "FL_foot", "FR_foot", "RL_foot", "RR_foot"],
        "num_states": 6, "num_joints": 12, "split_counts": {"train": 3, "heldout": 3},
        "replay_validation": {"thresholds": {"root_speed": .10}, "statistics": {"accepted_count": 6},
                              "states": [{"state_id": i, "height_drift_m": .001} for i in range(6)]},
    }
    return states, metadata


def quaternion_matrix(q):
    """Independent unit-quaternion rotation-matrix oracle for transform tests."""
    w, x, y, z = q.unbind(-1)
    return torch.stack((1 - 2 * (y*y + z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), dim=-1).reshape(-1, 3, 3)


class BankValidationTests(unittest.TestCase):
    def setUp(self):
        self.states, self.metadata = fixture()

    def validate(self):
        bank_module.validate_recovery_state_bank(self.states, self.metadata, self.metadata["joint_names"])

    def test_valid_fixture_and_negative_equivalent_quaternions(self):
        self.validate()
        self.states["root_pose_local"][:, 3:] *= -1
        self.validate()

    def test_all_matrix_fields_reject_nonfinite_values(self):
        for key in ("root_pose_local", "root_velocity_w", "joint_pos", "joint_vel"):
            for bad in (float("nan"), float("inf")):
                with self.subTest(key=key, bad=bad):
                    data = copy.deepcopy(self.states)
                    data[key][0, 0] = bad
                    with self.assertRaisesRegex(ValueError, "finite"):
                        bank_module.validate_recovery_state_bank(data, self.metadata)

    def test_large_finite_values_cannot_overflow_float32_load(self):
        self.states["root_velocity_w"] = self.states["root_velocity_w"].astype(np.float64)
        self.states["root_velocity_w"][0, 0] = 1e100
        with self.assertRaisesRegex(ValueError, "float32"):
            self.validate()

    def test_shapes_and_float_dtypes(self):
        for key, value in (("joint_vel", np.zeros((6, 11), dtype=np.float32)),
                           ("root_pose_local", np.zeros((6, 8), dtype=np.float32)),
                           ("root_velocity_w", np.zeros((6, 6), dtype=np.int32))):
            with self.subTest(key=key):
                data = dict(self.states, **{key: value})
                with self.assertRaises(ValueError):
                    bank_module.validate_recovery_state_bank(data, self.metadata)

    def test_quaternion_norm_and_actual_pose_label_consistency(self):
        self.states["root_pose_local"][0, 3:] = 0
        with self.assertRaisesRegex(ValueError, "quaternion"):
            self.validate()
        self.states, self.metadata = fixture()
        self.states["pose_class"][0] = "right"
        with self.assertRaisesRegex(ValueError, "disagrees"):
            self.validate()
        self.states, self.metadata = fixture()
        self.states["root_pose_local"][0, 3:] = [1, 0, 0, 0]
        with self.assertRaisesRegex(ValueError, "disagrees"):
            self.validate()

    def test_both_splits_required_and_other_split_values_rejected(self):
        for split in (np.zeros(6, dtype=np.int8), np.ones(6, dtype=np.int8),
                      np.array([0, 1, 2, 0, 1, 0], dtype=np.int8), np.zeros(6, dtype=bool)):
            with self.subTest(split=split):
                data = dict(self.states, split=split)
                with self.assertRaisesRegex(ValueError, "split"):
                    bank_module.validate_recovery_state_bank(data, self.metadata)

    def test_unique_ids_and_string_arrays(self):
        self.states["state_id"][0] = self.states["state_id"][1]
        with self.assertRaisesRegex(ValueError, "unique"):
            self.validate()
        self.states, self.metadata = fixture()
        self.states["pose_class"] = self.states["pose_class"].astype(object)
        with self.assertRaisesRegex(ValueError, "string array"):
            self.validate()

    def test_state_ids_require_int64_representability(self):
        for ids in (np.arange(6).astype(str), np.arange(6, dtype=float), np.arange(6).astype(bool)):
            with self.subTest(dtype=ids.dtype):
                with self.assertRaisesRegex(ValueError, "integer array"):
                    bank_module.validate_recovery_state_bank(dict(self.states, state_id=ids), self.metadata)
        self.states["state_id"] = np.arange(6, dtype=np.uint64)
        self.states["state_id"][0] = 2**63
        with self.assertRaisesRegex(ValueError, "int64"):
            self.validate()

    def test_manifest_schema_controller_self_collision_and_names(self):
        for key, value in (("schema_version", "old"), ("controller", "zero_torque"),
                           ("self_collisions_enabled", False), ("body_names", [])):
            with self.subTest(key=key):
                metadata = dict(self.metadata, **{key: value})
                with self.assertRaises(ValueError):
                    bank_module.validate_recovery_state_bank(self.states, metadata)
        with self.assertRaisesRegex(ValueError, "order"):
            bank_module.validate_recovery_state_bank(self.states, self.metadata, self.metadata["joint_names"][::-1])

    def test_manifest_counts_replay_records_and_thresholds(self):
        for key, value in (("num_states", 7), ("num_joints", 11), ("split_counts", {"train": 2, "heldout": 4})):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "disagrees"):
                    bank_module.validate_recovery_state_bank(self.states, dict(self.metadata, **{key: value}))
        self.metadata["replay_validation"]["states"][0]["state_id"] = 999999
        with self.assertRaisesRegex(ValueError, "IDs"):
            self.validate()
        self.states, self.metadata = fixture()
        self.metadata["replay_validation"]["thresholds"]["root_speed"] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.validate()

    def test_missing_arrays_and_replay_statistics(self):
        del self.states["joint_vel"]
        with self.assertRaisesRegex(ValueError, "missing"):
            self.validate()
        self.states, self.metadata = fixture()
        self.metadata["replay_validation"]["statistics"] = {}
        with self.assertRaisesRegex(ValueError, "statistics"):
            self.validate()

    def test_joint_limits_preserve_data_and_reject_single_bad_joint(self):
        data = {"joint_pos": torch.zeros(6, 12)}
        original = data["joint_pos"].clone()
        bank_module.validate_bank_joint_limits(data, torch.full((12,), -.5), torch.full((1, 12), .5))
        torch.testing.assert_close(data["joint_pos"], original)
        data["joint_pos"][4, 7] = .6
        with self.assertRaisesRegex(ValueError, "state 4, joint 7"):
            bank_module.validate_bank_joint_limits(data, torch.full((12,), -.5), torch.full((12,), .5))
        with self.assertRaisesRegex(ValueError, "shape"):
            bank_module.validate_bank_joint_limits(data, torch.zeros(2), torch.ones(2))


class BankLoadTests(unittest.TestCase):
    def setUp(self):
        # Test fixtures and their automatic cleanup stay within the E: workspace.
        self.temporary = tempfile.TemporaryDirectory(prefix="state-bank-test-", dir=_ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.states, self.metadata = fixture()
        self.write()

    def write(self):
        np.savez(self.directory / "states.npz", **self.states)
        self.metadata["sha256"] = hashlib.sha256((self.directory / "states.npz").read_bytes()).hexdigest()
        (self.directory / "manifest.json").write_text(json.dumps(self.metadata), encoding="utf-8")

    def test_directory_and_npz_paths_select_disjoint_splits(self):
        train = bank_module.load_recovery_state_bank(self.directory, self.metadata["joint_names"], "cpu")
        heldout = bank_module.load_recovery_state_bank(self.directory / "states.npz", self.metadata["joint_names"], "cpu", "heldout")
        self.assertEqual(train["root_pose_local"].shape, (3, 7))
        self.assertEqual(train["root_pose_local"].dtype, torch.float32)
        self.assertEqual(train["split"].dtype, torch.int64)
        self.assertEqual(train["state_id"].dtype, torch.int64)
        self.assertEqual(train["source_indices"].tolist(), [0, 2, 4])
        self.assertFalse(set(train["state_id"].tolist()) & set(heldout["state_id"].tolist()))
        self.assertEqual(train["archive_sha256"], self.metadata["sha256"])
        complete = bank_module.load_recovery_state_bank(self.directory, self.metadata["joint_names"], "cpu", "all")
        self.assertEqual(complete["joint_pos"].shape, (6, 12))

    def test_hash_mismatch_is_rejected(self):
        self.metadata["sha256"] = "0" * 64
        (self.directory / "manifest.json").write_text(json.dumps(self.metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            bank_module.load_recovery_state_bank(self.directory, self.metadata["joint_names"], "cpu")

    def test_object_arrays_never_load_with_pickle(self):
        self.states["state_id"] = self.states["state_id"].astype(object)
        self.write()
        with self.assertRaisesRegex(ValueError, "allow_pickle=False"):
            bank_module.load_recovery_state_bank(self.directory, self.metadata["joint_names"], "cpu")

    def test_bad_heldout_data_is_not_hidden_by_loading_train_only(self):
        self.states["joint_vel"][1, 0] = float("nan")
        self.write()
        with self.assertRaisesRegex(ValueError, "finite"):
            bank_module.load_recovery_state_bank(self.directory, self.metadata["joint_names"], "cpu", "train")


class BankTransformTests(unittest.TestCase):
    def test_yaw_translation_pose_velocity_gravity_and_no_mutation(self):
        states, _ = fixture()
        pose = torch.from_numpy(states["root_pose_local"]).double()
        pose[:, :2] = torch.tensor([1., 2.])
        velocity = torch.tensor([[1., 2., 3., 4., 5., 6.]], dtype=torch.float64).repeat(6, 1)
        original_pose, original_velocity = pose.clone(), velocity.clone()
        origins = torch.arange(18, dtype=torch.float64).reshape(6, 3)
        yaw = torch.linspace(-math.pi, math.pi, 6, dtype=torch.float64)
        result_pose, result_velocity = bank_module.transform_bank_root(pose, velocity, origins, yaw)
        c, s = yaw.cos(), yaw.sin()
        rz = torch.zeros(6, 3, 3, dtype=torch.float64)
        rz[:, 0, 0] = c; rz[:, 0, 1] = -s; rz[:, 1, 0] = s; rz[:, 1, 1] = c; rz[:, 2, 2] = 1
        torch.testing.assert_close(result_pose[:, :3], torch.bmm(rz, pose[:, :3, None]).squeeze(-1) + origins)
        torch.testing.assert_close(quaternion_matrix(result_pose[:, 3:]), torch.bmm(rz, quaternion_matrix(pose[:, 3:])), atol=1e-6, rtol=1e-6)
        for offset in (0, 3):
            expected = torch.bmm(rz, velocity[:, offset:offset + 3, None]).squeeze(-1)
            torch.testing.assert_close(result_velocity[:, offset:offset + 3], expected)
        torch.testing.assert_close(result_pose[:, 2], pose[:, 2] + origins[:, 2])
        torch.testing.assert_close(-quaternion_matrix(result_pose[:, 3:])[:, 2], -quaternion_matrix(pose[:, 3:])[:, 2], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(pose, original_pose)
        torch.testing.assert_close(velocity, original_velocity)

    def test_scalar_yaw_shared_origin_and_inverse(self):
        states, _ = fixture()
        pose, velocity = torch.from_numpy(states["root_pose_local"]), torch.randn(6, 6)
        world, rotated = bank_module.transform_bank_root(pose, velocity, [2., 3., 0.5], math.pi / 2)
        world[:, :3] -= torch.tensor([2., 3., 0.5])
        restored, original_vel = bank_module.transform_bank_root(world, rotated, [0., 0., 0.], -math.pi / 2)
        torch.testing.assert_close(restored, pose)
        torch.testing.assert_close(original_vel, velocity)

    def test_invalid_origin_yaw_and_quaternion_rejected(self):
        states, _ = fixture()
        pose, velocity = torch.from_numpy(states["root_pose_local"]), torch.zeros(6, 6)
        for origins, yaw in (([0., 0.], 0), ([0., 0., 0.], [1., 2.]), ([0., 0., 0.], float("nan"))):
            with self.subTest(origins=origins, yaw=yaw):
                with self.assertRaises(ValueError):
                    bank_module.transform_bank_root(pose, velocity, origins, yaw)
        pose[0, 3:] = 0
        with self.assertRaisesRegex(ValueError, "quaternion"):
            bank_module.transform_bank_root(pose, velocity, [0., 0., 0.], 0)


if __name__ == "__main__":
    unittest.main()
