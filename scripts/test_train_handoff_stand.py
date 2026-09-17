"""CPU-only gates/negative tests. No simulator, runner, training or installation."""

import ast
import copy
from contextlib import redirect_stderr
import io
from pathlib import Path
import sys
import unittest

import torch

import train_handoff_stand as train


class TrainingPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args = train.build_parser().parse_args([
            "--mode", "smoke", "--tag", "cpu-preflight-only-test-unused", "--preflight-only"])
        cls.arrays, cls.manifest, cls.train_ids, cls.parent, cls.report = train.preflight(cls.args)

    def test_actual_receipts_parent_and_frozen_models_pass_without_simulator(self):
        self.assertTrue(self.report["preflight_pass"])
        self.assertEqual((self.report["num_envs"], self.report["updates"]), (16, 2))
        self.assertEqual((self.report["expected_final_iteration"], self.report["expected_final_adam_step"]), (3449, 69160))
        self.assertEqual(len(self.report["frozen_models"]), 16)
        self.assertFalse(Path(self.report["run_dir"]).exists())
        self.assertFalse(any(name == "isaaclab" or name.startswith("isaaclab.") for name in sys.modules))

    def test_missing_false_partial_or_changed_input_receipts_rejected(self):
        receipt = train.load_report(train.DEFAULT_INPUT_RECEIPT)
        mutations = [lambda r: r.pop("input_replay_pass"),
                     lambda r: r.update(contact_probe_finite_pass=False),
                     lambda r: r.update(script_sha256="0" * 64),
                     lambda r: r["dataset"].update(archive_sha256="0" * 64),
                     lambda r: r["result"].update(sample_ids=[0, 1]),
                     lambda r: r["result"]["contact_probe"].update(finite=False)]
        for mutation in mutations:
            changed = copy.deepcopy(receipt)
            mutation(changed)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                train.validate_input_receipt(changed, self.arrays, self.report["dataset"])

    def test_missing_false_or_changed_subset_receipts_rejected(self):
        receipt = train.load_report(train.DEFAULT_SUBSET_RECEIPT)
        mutations = [lambda r: r.pop("subset_pass"), lambda r: r.update(validation_pass=False),
                     lambda r: r["coverage"].update(ordinary_rows=0),
                     lambda r: r["source_snapshot"][0].update(sha256="0" * 64),
                     lambda r: r["rounds"][0].update(selected_history_exact=False)]
        for mutation in mutations:
            changed = copy.deepcopy(receipt)
            mutation(changed)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                train.validate_subset_receipt(changed, self.report["dataset"])
        with self.assertRaises(ValueError):
            train.e_file(train.ROOT / "nonexistent-subset-receipt.json", "receipt")

    def test_iteration_adam_std_and_finite_fail_closed(self):
        mutations = [lambda c: c.update(iter=3449),
                     lambda c: c["optimizer_state_dict"]["state"][0].update(step=torch.tensor(69121.)),
                     lambda c: c["model_state_dict"]["std"].fill_(-1),
                     lambda c: c["model_state_dict"]["std"].fill_(float("nan")),
                     lambda c: c["optimizer_state_dict"]["state"][0]["exp_avg"].fill_(float("inf"))]
        for mutation in mutations:
            changed = copy.deepcopy(self.parent)
            mutation(changed)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                train.checkpoint_summary(changed, 3448, 69120)

    def test_exact_full_resume_tree_detects_optimizer_or_model_loss(self):
        train.equal_tree(self.parent["model_state_dict"], self.parent["model_state_dict"])
        changed = copy.deepcopy(self.parent["optimizer_state_dict"])
        changed["state"][0]["exp_avg"].add_(1)
        with self.assertRaises(ValueError):
            train.equal_tree(changed, self.parent["optimizer_state_dict"])

    def test_only_two_fixed_budgets_and_no_overwrite_or_escape(self):
        self.assertEqual(train.BUDGETS, {"smoke": (16, 2), "formal": (128, 100)})
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            train.build_parser().parse_args(["--mode", "formal", "--tag", "bad", "--updates", "300"])
        for value in ("../escape", "E:/escape", "", "a/b"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                train.run_destination(value)
        with self.assertRaises(ValueError):
            train.e_file("C:/not-on-e.json", "receipt")

    def test_no_registration_or_legacy_install_and_sim_first(self):
        tree = ast.parse(Path(train.__file__).read_text(encoding="utf-8"))
        run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_training")
        launch = next(node for node in ast.walk(run) if isinstance(node, ast.Call) and ast.unparse(node.func) == "AppLauncher")
        for node in ast.walk(run):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                text = ast.unparse(node)
                if "isaaclab_tasks" in text or "rsl_rl" in text or text == "import torch":
                    self.assertGreater(node.lineno, launch.lineno)
        calls = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertNotIn("gym.register", calls)
        self.assertNotIn("gym.make", calls)
        self.assertNotIn("os.system", calls)
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        writes = [node.lineno for node in ast.walk(main) if isinstance(node, ast.Call) and ast.unparse(node.func) == "write_json_exclusive"]
        closes = [node.lineno for node in ast.walk(main) if isinstance(node, ast.Call) and ast.unparse(node.func) == "close_resources"]
        self.assertLess(max(writes), min(closes))


if __name__ == "__main__":
    unittest.main()
