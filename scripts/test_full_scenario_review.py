"""Pure CPU checks for fail-closed review assembly; never imports a simulator."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import build_full_scenario_review as review


def report(view):
    return {"checkpoint_sha256": "a" * 64, "task": review.LOCOMOTION_TASK, "seed": 20260918,
            "video_view": view, "protocol_version": "stand_walk_stop_stance_geometry_v2",
            "rest_geometry_protocol": "stance_geometry_v1", "protocol": {"stand_s": 4},
            "criteria": "test", "rest_stance_criterion": {"geometry_required_fraction": 1.0},
            "acceptance": {key: True for key in ("no_reset", "normal_stance_geometry_at_rest",
                "four_vertical_contacts_at_rest", "no_current_base_contact_at_rest", "geometry_and_support_at_rest")},
            "passed": True}


def recovery_report(bank=True):
    """Only fields actually emitted by evaluate_go2_recovery.py are used."""
    start = {"trial": 0, "settled": bank, "standing_at_policy_start": False if bank else None,
             "fallen_at_policy_start": True if bank else None, "eligible_settled_fallen_recovery": bank,
             "contacts_fresh_since_pose_write": bank, "any_body_contact": True if bank else None,
             "quiet_supported_window_s": 1.0 if bank else 0.0}
    final = {"trial": 0, "geometry_ok": True, "stable_hold_s": 3.2, "height_m": 0.32,
             "gravity_error": 0.05, "vertical_foot_contacts": 4,
             "feet_y_b": [0.15, -0.15, 0.15, -0.15], "knees_y_b": [0.10, -0.10, 0.10, -0.10],
             "max_joint_offset_rad": 0.2}
    if bank:
        start['bank_state_id'] = final['bank_state_id'] = 132
    result = {"trials": 1, "successes": 1, "final_valid_stands": 1, "final_geometry_passes": 1,
              "standing_starts_not_fallen_recovery": 0, "settled_fallen_trials": int(bank),
              "settled_fallen_recovery_successes": int(bank), "settled_fallen_final_valid_stands": int(bank),
              "stable_hold_s": 3.0, "horizon_s": 8.0, "policy_start_state": [start],
              "final_diagnostics": [final]}
    return {"protocol_version": "stance_geometry_v1_state_bank_PD_v1" if bank else "stance_geometry_v1",
            "acceptance_eligible": True, "results": {"side" if bank else "upright": result}}


class ReviewTests(unittest.TestCase):
    def test_checkpoint_identity_is_pinned_to_completed_experiment(self):
        for group, sha in review.EXPECTED_CHECKPOINTS.items():
            review.validate_checkpoint_identity({'checkpoint_sha256': sha}, group)
            with self.assertRaises(ValueError):
                review.validate_checkpoint_identity({'checkpoint_sha256': 'a' * 64,
                                                     'checkpoint': 'other_run/model_1999.pt'}, group)
        with self.assertRaises(ValueError):
            review.validate_checkpoint_identity({'checkpoint_sha256': review.EXPECTED_CHECKPOINTS['current']},
                                                'nominal')

    def test_valid_single_trial_and_recorded_failure_are_accepted(self):
        for bank, pose in ((True, 'side'), (False, 'upright')):
            item = recovery_report(bank)
            review.validate_recovery_result(item, pose)
            result = item['results'][pose]
            for key in ('successes', 'final_valid_stands', 'final_geometry_passes',
                        'settled_fallen_recovery_successes', 'settled_fallen_final_valid_stands'):
                result[key] = 0
            result['final_diagnostics'][0].update(geometry_ok=False, stable_hold_s=0.,
                                                 height_m=.15, gravity_error=1.0, vertical_foot_contacts=1)
            review.validate_recovery_result(item, pose)

    def test_aggregate_success_cannot_hide_ineligible_start(self):
        for changes in ({'settled': False, 'eligible_settled_fallen_recovery': False},
                        {'settled': False}, {'standing_at_policy_start': True},
                        {'fallen_at_policy_start': False}, {'contacts_fresh_since_pose_write': False}):
            with self.subTest(changes=changes):
                item = recovery_report()
                item['results']['side']['policy_start_state'][0].update(changes)
                with self.assertRaises(ValueError):
                    review.validate_recovery_result(item, 'side')

    def test_final_success_cannot_hide_missing_hold_or_bad_final_pose(self):
        for changes in ({'stable_hold_s': 0.}, {'geometry_ok': False}, {'height_m': .15},
                        {'gravity_error': 1.0}, {'vertical_foot_contacts': 3},
                        {'max_joint_offset_rad': .9}, {'feet_y_b': [-.15, .15, .15, -.15]},
                        {'knees_y_b': [-.10, -.10, .10, -.10]}, {'bank_state_id': 999}):
            with self.subTest(changes=changes):
                item = recovery_report()
                item['results']['side']['final_diagnostics'][0].update(changes)
                with self.assertRaises(ValueError):
                    review.validate_recovery_result(item, 'side')

    def test_inconsistent_fallen_subcounts_and_missing_diagnostics_rejected(self):
        for key in ('settled_fallen_trials', 'settled_fallen_recovery_successes',
                    'settled_fallen_final_valid_stands', 'final_geometry_passes'):
            with self.subTest(key=key):
                item = recovery_report()
                item['results']['side'][key] = 0
                with self.assertRaises(ValueError):
                    review.validate_recovery_result(item, 'side')
        item = recovery_report()
        del item['results']['side']['final_diagnostics'][0]['vertical_foot_contacts']
        with self.assertRaises(ValueError):
            review.validate_recovery_result(item, 'side')

    def test_any_time_success_may_legitimately_end_in_failure(self):
        item = recovery_report()
        result = item['results']['side']
        result.update(final_valid_stands=0, final_geometry_passes=0, settled_fallen_final_valid_stands=0)
        result['final_diagnostics'][0].update(stable_hold_s=0., geometry_ok=False, height_m=.15)
        review.validate_recovery_result(item, 'side')

    def test_short_final_hold_is_not_full_final_success(self):
        item = recovery_report()
        result = item['results']['side']
        result.update(successes=0, final_valid_stands=0,
                      settled_fallen_recovery_successes=0, settled_fallen_final_valid_stands=0)
        result['final_diagnostics'][0]['stable_hold_s'] = 2.98
        review.validate_recovery_result(item, 'side')

    @staticmethod
    def fake_capture(declared_frames, decoded_frames):
        cap = MagicMock()
        cap.isOpened.return_value = True
        values = {review.cv2.CAP_PROP_FRAME_COUNT: declared_frames, review.cv2.CAP_PROP_FPS: 25.,
                  review.cv2.CAP_PROP_FRAME_WIDTH: 960., review.cv2.CAP_PROP_FRAME_HEIGHT: 540.}
        cap.get.side_effect = values.__getitem__
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        cap.read.side_effect = [(True, frame)] * decoded_frames + [(False, None)]
        return cap

    def test_recovery_prepolicy_initial_frame_is_retained_and_timed(self):
        for count in (275, 276):
            with self.subTest(count=count):
                with patch.object(review.cv2, 'VideoCapture', return_value=self.fake_capture(count, count)):
                    metadata = review.probe_complete_video('test_only.mp4', 275, allow_initial_frame=True)
                self.assertEqual(metadata['frames'], count)
                self.assertEqual(metadata['decoded_frames'], count)
                self.assertEqual(metadata['duration_s'], count / 25)
                self.assertEqual(metadata['recorded_initial_frames'], count - 275)

    def test_extra_or_truncated_recovery_frames_are_rejected(self):
        for declared, decoded, allow_initial in ((277, 277, True), (276, 275, True), (276, 276, False)):
            with self.subTest(declared=declared, decoded=decoded, allow_initial=allow_initial):
                with patch.object(review.cv2, 'VideoCapture', return_value=self.fake_capture(declared, decoded)):
                    with self.assertRaises(ValueError):
                        review.probe_complete_video('test_only.mp4', 275, allow_initial_frame=allow_initial)

    def test_eight_complete_chapters(self):
        chapters = review.chapter_specs(Path("E:/sources"))
        self.assertEqual(len(chapters), 8)
        self.assertEqual(sum(c["protocol_duration_s"] for c in chapters), 102)
        self.assertEqual(len({c["id"] for c in chapters}), 8)
        self.assertEqual(chapters[-1]["directories"][1].name, "current-heldout-oblique")

    def test_matching_success_and_matching_failure_are_allowed(self):
        left, right = report("front"), report("oblique")
        review.validate_locomotion_reports(left, right)
        for item in (left, right):
            item["acceptance"]["no_reset"] = False
            item["passed"] = False
        review.validate_locomotion_reports(left, right)

    def test_protocol_outcome_and_view_mismatches_rejected(self):
        for key, value in (("checkpoint_sha256", "b" * 64), ("task", "other"), ("seed", 1),
                           ("protocol_version", "old"), ("protocol", {}), ("passed", False),
                           ("acceptance", {}), ("video_view", "front")):
            with self.subTest(key=key):
                right = report("oblique")
                right[key] = value
                with self.assertRaises(ValueError):
                    review.validate_locomotion_reports(report("front"), right)

    def test_missing_fields_and_false_green_flag_rejected(self):
        left, right = report("front"), report("oblique")
        del left["seed"]
        with self.assertRaises(ValueError):
            review.validate_locomotion_reports(left, right)
        left, right = report("front"), report("oblique")
        for item in (left, right):
            item["acceptance"]["no_reset"] = False
        with self.assertRaises(ValueError):
            review.validate_locomotion_reports(left, right)

    def test_caption_surface_does_not_change_original_pixels(self):
        left = np.full((540, 960, 3), 55, dtype=np.uint8)
        right = np.full((540, 960, 3), 133, dtype=np.uint8)
        bar = np.zeros((review.BAR_HEIGHT, 1920, 3), dtype=np.uint8)
        combined = review.compose_frame(left, right, bar)
        np.testing.assert_array_equal(combined[review.BAR_HEIGHT:, :960], left)
        np.testing.assert_array_equal(combined[review.BAR_HEIGHT:, 960:], right)

    def test_failure_label_does_not_claim_recovery(self):
        chapter = {"title": "New nominal model", "kind": "recovery", "display_passed": False,
                   "checkpoint_sha256": "a" * 64, "seed": 20260918, "bank_state_id": "side-0001",
                   "handover": {}, "outcome": {"successes": 0, "final_valid_stands": 0, "trials": 1}}
        lines = review.label_lines(chapter, 2)
        self.assertIn("FAIL", lines[0])
        self.assertIn("successes=0/1", lines[1])
        self.assertIn("side-0001", lines[2])
        self.assertIn("NOT synchronized", lines[3])
        self.assertIn("NOT continuous", lines[4])
        self.assertIn("NOT in", lines[5])

    def test_missing_sources_never_open_encoder_or_create_output(self):
        # The caller sets TEMP to E: when running this test.
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            output = root / "new" / "review.mp4"
            with patch.object(review, "encode_review") as encoder:
                with self.assertRaises(ValueError):
                    review.build(root, output)
                encoder.assert_not_called()
            self.assertFalse(output.parent.exists())

    def test_existing_manifest_prevents_any_preflight_or_write(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            output = root / "review.mp4"
            output.with_suffix(".json").write_text("retain me", encoding="utf-8")
            with patch.object(review, "prepare_chapters") as prepare:
                with self.assertRaises(FileExistsError):
                    review.build(root, output)
                prepare.assert_not_called()
            self.assertEqual(output.with_suffix(".json").read_text(), "retain me")

    def test_encoder_smoke_preserves_complete_frames_and_card_timeline(self):
        # Synthetic test-only colors; never mixed into real review footage.
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            paths = [root / f'{view}.mp4' for view in review.VIEWS]
            for value, path in zip((55, 133), paths):
                writer = review.cv2.VideoWriter(str(path), review.cv2.VideoWriter_fourcc(*'mp4v'),
                                                 25, (960, 540))
                self.assertTrue(writer.isOpened())
                for _ in range(26):
                    writer.write(np.full((540, 960, 3), value, dtype=np.uint8))
                writer.release()
            chapter = {'id': 'synthetic_test_only', 'title': 'Synthetic CPU encoding test',
                       'frames': 26, 'video_paths': paths, 'kind': 'locomotion',
                       'display_passed': False, 'outcome': {'passed': False},
                       'checkpoint_sha256': '0' * 64, 'seed': 0}
            output = root / 'synthetic_review.mp4'
            self.assertEqual(review.encode_review([chapter], output), 76)
            self.assertEqual(chapter['timeline']['footage_start_frame'], 50)
            self.assertEqual(chapter['timeline']['end_frame_exclusive'], 76)
            self.assertEqual(chapter['timeline']['end_s'], 3.04)
            cap = review.cv2.VideoCapture(str(output))
            try:
                decoded = 0
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    decoded += 1
                    self.assertEqual(frame.shape, (review.HEIGHT, review.WIDTH, 3))
                    if decoded > 50:
                        self.assertLess(abs(float(frame[review.BAR_HEIGHT:, :960].mean()) - 55), 8)
                        self.assertLess(abs(float(frame[review.BAR_HEIGHT:, 960:].mean()) - 133), 8)
                self.assertEqual(decoded, 76)
            finally:
                cap.release()


if __name__ == "__main__":
    unittest.main()
