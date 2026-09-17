"""Isolated, exact-budget continuation of the interrupted Smith w30 checkpoint.

Starts Isaac Sim before importing RSL-RL, using the official training entry.
Only after full model/optimizer load: restore the saved adaptive LR scalar and
advance the already-completed iteration label. Environment/RNG are restarted.
"""
import hashlib
import json
import runpy
import sys
from pathlib import Path

CHECKPOINT = Path("E:/IsaacLab/repo/logs/rsl_rl/unitree_go2_recovery/2026-09-16_20-41-29_20260916-smithstandpair128x1000_w30/model_2900.pt")
SHA256 = "1f704da68a0ab1a95e81b9362c7b8f9e47352a354c1ed2cf4eaf0e1ec1d21a45"
TRAIN = Path("E:/IsaacLab/repo/scripts/reinforcement_learning/rsl_rl/train.py")

def restore_resume_state(runner):
    if runner.current_learning_iteration != 2900:
        raise RuntimeError("Expected completed iteration2900")
    groups = runner.alg.optimizer.param_groups
    states = runner.alg.optimizer.state
    if len(groups) != 1 or float(groups[0]["lr"]) != 1e-5:
        raise RuntimeError("Unexpected saved optimizer learning rate")
    if len(states) != 17 or any(float(s["step"]) != 58040 for s in states.values()):
        raise RuntimeError("Unexpected Adam state/update budget")
    runner.alg.learning_rate = float(groups[0]["lr"])
    runner.current_learning_iteration = 2901
    return {"checkpoint_iter_completed": 2900, "next_iteration": 2901,
            "saved_optimizer_lr": runner.alg.learning_rate, "adam_step": 58040,
            "remaining_updates": 98, "expected_final": 2998,
            "environment_and_rng_restarted": True}

def main():
    if hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() != SHA256:
        raise RuntimeError("Checkpoint hash mismatch")
    # Match official entry's sim-first import sequence.
    from isaaclab.app import AppLauncher
    original_init = AppLauncher.__init__
    active_launchers = []
    patched_loads = []
    def initialize_then_patch(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        active_launchers.append(self)
        AppLauncher.__init__ = original_init
        from rsl_rl.runners import OnPolicyRunner
        original_load = OnPolicyRunner.load
        patched_loads.append((OnPolicyRunner, original_load))
        def restore_load(runner, path, load_optimizer=True, map_location=None):
            if Path(path).resolve() != CHECKPOINT.resolve() or not load_optimizer:
                raise RuntimeError("Unexpected resume source or missing optimizer restore")
            result = original_load(runner, path, load_optimizer=load_optimizer, map_location=map_location)
            print("EXACT_RESUME_STATE " + json.dumps(restore_resume_state(runner)), flush=True)
            return result
        OnPolicyRunner.load = restore_load
    AppLauncher.__init__ = initialize_then_patch
    sys.path.insert(0, str(TRAIN.parent))
    sys.argv[0] = str(TRAIN)
    try:
        runpy.run_path(str(TRAIN), run_name="__main__")
    except BaseException:
        # Official train.py closes the app only on normal completion.
        for launcher in active_launchers:
            try:
                launcher.app.close()
            except Exception as error:
                print(f"Resume cleanup failed: {error}", file=sys.stderr)
        raise
    finally:
        AppLauncher.__init__ = original_init
        for runner_class, original in patched_loads:
            runner_class.load = original

if __name__ == "__main__":
    main()
