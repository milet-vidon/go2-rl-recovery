"""Bounded, real single-episode diagnostic; never deployment or model promotion.

Frozen roll1999/stand3547 with explicitly selected control3947 or experimental
balanced4246 at cmd=.5. The latter failed its broader screen and gait selection;
neither selection authorizes promotion, a fast-running claim or deployment.
Import and preflight are simulator-free. The separate runtime creates ONE env,
records the initial nominal-PD preparation, and forbids all subsequent resets.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "continuous_recovery_flow_diagnostic_v3_sensor_identity"
TASK = "Isaac-Recovery-Bank-SmithNominal-Flat-Unitree-Go2-Play-v0"
CONTROL = Path("E:/IsaacLab/repo/logs/rsl_rl/unitree_go2_flat/2026-09-17_03-10-21_20260917-speedretention128x300/model_3947.pt")
MODEL_SHA = {"roll": "71e4c7a39ff81836d2ead1c4533988c696e5bd4a0fc939b8cef6b4f6daec1b1c",
             "stand": "5527c4053cb6ef8e6616da65b7dc32a9464eb8d1366f4966a4eb4b65bc913fdb",
             "locomotion": "3dfbe03e0ac332fee688c970b420167affcdb6e8cacee16e40fc12120751df3f"}
MIRROR = ROOT / "scripts/evaluate_handoff_mirror.py"
FROZEN = {
    "scripts/evaluate_handoff_mirror.py": "cf2fbbdceae0e460727a969bf080e1e230ed716f00f914f47381a167df50ba95",
    "scripts/evaluate_handoff_combined.py": "0788beb2e8d4457f6c286d9aff80638b269e080a9b75167f9f7977456380bed0",
    "scripts/evaluate_locomotion_recovery_physics.py": "cc2437ffb0d84da97eda5906333e54070f2d065ea475a148b6be7f1c6cfea4c8",
    "scripts/evaluate_go2_stand_walk_stop.py": "bbba369177d95bc24163bdd2b5ba96506f5db7d4b1ce38875f8a8885c49e22e4",
    "src/go2_recovery/recovery_handoff_math.py": "df8a45a5e64ddc05a3062b73f769134fd3612aa4149f88eae51a67252699369b",
    "src/go2_recovery/roll_mirror_math.py": "9e3edf7cd402c745242fa41e8dcf3d4ccb3818cb474344cff52c104c00426d91",
    "src/go2_recovery/supported_startup_math.py": "47991846f57272192b10171cf43eed5e772fdbc063853cc0601a82110af6fe93",
    "src/go2_recovery/recovery_env_cfg.py": "45836efa79a4aded4b9a8cb3f46ca5ac0a0d9c9a0f7d4609884ff65059a7865f",
    "src/go2_recovery/recovery_control_targets.py": "5220033d6b0f76bc313527dce017bb37f67f19a192034a15ac2a9eb7abd6ecac",
}
BANK = ROOT / "datasets/recovery_states/nominal_pd_v1_20260916"
SCREEN = ROOT / "evaluations/20260917-locomotion-recovery-retention-v1/recovery-normal-seed20260909"
EVIDENCE = {
    SCREEN / "model_3947_stand_walk_stop.json": "21c4415673267e6c248a404fd80ff3125577728e04b8fe8560ae7b812faf96fb",
    SCREEN / "model_3947_stand_walk_stop.csv": "ba4a0eaf216b8ef822356b89e066bc09b5ce750936b70d67cdc1923b1a4136d4",
    SCREEN / "retention_interface.json": "985c447375683a8ee174af78a0ee230b2b01db0c686ada367b6ab5db35a08343",
    BANK / "states.npz": "71b882e92035b4c248e5980339c980dbb242dcb1ec711c05252c33249db48f5a",
    BANK / "manifest.json": "809d90f8c2618e1ad961e2e8b5bfe33df358d20373ad85e341c26af249a342aa",
}
BALANCED_RUN = Path("E:/IsaacLab/repo/logs/rsl_rl/unitree_go2_flat/2026-09-17_16-33-35_20260917-gait-balanced-128x300-first")
BALANCED_SCREEN = ROOT / "evaluations/20260917-gait-balanced-4246-first"
MATCHED_CONTROL_SCREEN = ROOT / "evaluations/20260917-gait-control-4246-first"
BALANCED_SHA = "882baefd00193c4b71bef2a47b5cb7382c37dd9e87b48f7f5863eaa364baebdc"
BALANCED_CASE = BALANCED_SCREEN / "recovery-normal-seed20260909"
BALANCED_PINS = {
    BALANCED_RUN / "model_4246.pt": BALANCED_SHA,
    BALANCED_RUN / "common_physics_training_result.json": "299b45ea7c89e10107c4273e1d4867e546c3017bb8e18ee48cc9664427c91b9e",
    BALANCED_CASE / "model_4246_stand_walk_stop.json": "7d4fb4ddb21851fd4099b12fee523604b5c7f0e4f988d7a7f2967a6066711573",
    BALANCED_CASE / "model_4246_stand_walk_stop.csv": "8554b025142bce52af11cd39172d01c0ff10110d41c0710c8bd7eff4af8ada97",
    BALANCED_CASE / "retention_interface.json": "ba73b4ea40fcc40856fff4d1feca5dc23266bb964c08bd601eb67fd660836242",
    BALANCED_SCREEN / "summary.json": "213a6351d1f909fa8a800131c0cdd2eb75d02b328b0e6078200fbaad2a83d188",
    BALANCED_SCREEN / "identity_audit_v1.json": "47db123af51fca8689fc6341aa004a4fe7049db8ef2f9c420383e21f492517be",
    BALANCED_SCREEN / "matched_comparison.json": "3bec33ce08de6ead1672b43d866a3afea564ffc8743edac385ae97caf9b4726c",
    MATCHED_CONTROL_SCREEN / "summary.json": "6bf38ccf74f3ed18a8e1f4d5fc6682b65707eeed07bad58ee5ba9afc0aeb0e6d",
    MATCHED_CONTROL_SCREEN / "identity_audit_v1.json": "9da94ad375b133e65ec7b162dcac51ca985b1cc0566cb8bf9568082b0f2fc1b3",
    ROOT / "scripts/evaluate_balanced_gait_candidate.py": "e42f2f34f1968e1bf99f6c1dc13aa1b53463fa22893db485b23dcad75df4c8fe",
    ROOT / "scripts/evaluate_common_physics_candidate.py": "f62f2230742b868c71f68b9241ef2ea9a9099a699c233303a5c4983051857ce1",
    ROOT / "scripts/train_balanced_gait_adaptation.py": "574e9cac8a5413293df898b6802fe5c0d1a3da9d34a00c0bb084b7e396dc79c5",
    ROOT / "scripts/audit_balanced_gait_screen_identity.py": "f21c7872ea72d935a2eb458a4a25a6d44955a1938d222c2dc5f9d08815a1cc9f",
    ROOT / "scripts/compare_balanced_gait_screens.py": "7518b1e1431207d7779778348a48ab825f7855d6d4f9f2b2974d3a01c2ece702",
}
OLD_ACCEPTANCE = {"no_reset", "no_base_contact", "supported_height", "level", "walk_tracking",
                  "quiet_stand_stop", "feet_lift_in_walk", "limited_slip", "four_feet_at_rest",
                  "normal_stance_geometry_at_rest", "four_vertical_contacts_at_rest",
                  "no_current_base_contact_at_rest", "geometry_and_support_at_rest"}
HELPERS = ("_uniform", "_pose_quaternions", "_set_pose_class", "_stable_stand", "_stance_geometry",
           "_start_snapshot", "_select_bank_states", "_validate_bank_environment", "_set_bank_states",
           "_attach_bank_provenance", "_diagnostic_scene", "_camera")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def extract_definitions(source, names=HELPERS):
    """Copy exact AST-selected definitions, never imports/top-level code/old main."""
    tree = ast.parse(source)
    found = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    require(len(found) == len(names) and {n.name for n in found} == set(names), "Missing/duplicate frozen helper")
    require(all(not n.decorator_list for n in found), "Unexpected helper decorator side effects")
    result = "from __future__ import annotations\n\n" + "\n\n".join(ast.get_source_segment(source, n) for n in found) + "\n"
    compile(result, "<continuous-frozen-helpers>", "exec")
    return result


def locomotion_spec(selected_model="control3947"):
    require(selected_model in ("control3947", "balanced4246"), "Unknown explicit locomotion selection")
    balanced = selected_model == "balanced4246"
    case = BALANCED_CASE if balanced else SCREEN
    return {"path": str(BALANCED_RUN / "model_4246.pt" if balanced else CONTROL),
            "sha": BALANCED_SHA if balanced else MODEL_SHA["locomotion"], "selected_model": selected_model,
            "interface_path": str(case / "retention_interface.json"),
            "actual_passed_interface_path": str(case / "retention_interface.json"),
            "screen_report_path": str(case / ("model_4246_stand_walk_stop.json" if balanced else "model_3947_stand_walk_stop.json")),
            "screen_protocol": "trained_balanced_gait_candidate_retention_v1" if balanced else "locomotion_recovery_physics_retention_v1"}


def validate_screen(report, selected_model="control3947"):
    spec = locomotion_spec(selected_model)
    pins = BALANCED_PINS if selected_model == "balanced4246" else EVIDENCE
    require(report["checkpoint_sha256"] == spec["sha"], "Wrong screen actor")
    require(report["protocol_version"] == spec["screen_protocol"], "Wrong screen protocol")
    require(report["protocol"] == {"stand_s": 4., "walk_s": 8., "stop_s": 6., "walk_speed": .5,
                                  "lateral_speed": 0., "yaw_rate": 0., "push_delta_vy": 0.}, "Only measured .5 screen qualifies")
    require(report["retention_passed"] is True and report["passed"] is True, "Failed bounded prerequisite")
    require(report["retention_acceptance"] == {"original_checks": True, "straight_drift": True}, "Drift check required")
    require(set(report["acceptance"]) == OLD_ACCEPTANCE and all(v is True for v in report["acceptance"].values()), "Every old criterion required")
    exp = report["retention_experiment"]
    require(exp["physics_mode"] == "recovery" and exp["reference_checked"] is True and
            exp["read_only_interface_steps"] == 900 and exp["trace_sha256"] == pins[Path(spec["interface_path"])],
            "Not the complete measured full-physics screen")
    require(Path(exp["trace_path"]).resolve() == Path(spec["interface_path"]).resolve() and
            exp["no_added_reset_or_history_write"] is True, "Wrong actual interface or reset semantics")
    if selected_model == "balanced4246":
        require(exp["candidate_checkpoint_sha256"] == spec["sha"] and
                exp["parent_control_sha256"] == MODEL_SHA["locomotion"], "Wrong actual candidate lineage")
    walk = report["settled_phase_stats"]["walk"]
    require(abs(walk["vx_b_mean"] - .5) < .12 and abs(walk["vy_b_mean"]) < .12 and
            abs(walk["yaw_rate_mean"]) < .15 and walk["contact_slip_mean"] < .12, "Actual screen values fail")
    return {"report": spec["screen_report_path"], "cmd_m_s": .5,
            "unique_physical_trajectories": 1, "vx_b": walk["vx_b_mean"], "vy_b": walk["vy_b_mean"],
            "notice": ("Speed-specific diagnostic only: balanced4246 passed13/21; gait selection FAILED. No promotion or fast-run claim."
                       if selected_model == "balanced4246" else
                       "Bounded diagnostic prerequisite only; .8 drift FAILED. No complete compatibility/promotion/fast-run claim.")}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def validate_balanced_limitations(summary, comparison):
    require(summary["status"] == "completed_screening_only" and summary["arm"] == "balanced" and
            summary["checkpoint_sha256"] == BALANCED_SHA and len(summary["rows"]) == 21 and
            sum(row["passed"] is True for row in summary["rows"]) == 13 and
            summary["promotion_performed"] is False, "Not the measured incomplete balanced screen")
    require(comparison["diagnostic_selection_passed"] is False and comparison["promotion_performed"] is False and
            comparison["checks"]["balanced_all21_old_physical_criteria"] is False and
            comparison["checks"]["mean_duty_gap_reduction_at_least25percent"] is False,
            "Cannot relabel failed wider/gait selection as accepted")
    return {"screen_passed": 13, "screen_total": 21, "gait_selection_passed": False,
            "promotion_performed": False, "scope": "Explicit .5-command diagnostic only; not an actual speed limiter, full compatibility, natural gait, trot or fast-run acceptance"}


def verify_locomotion(selected_model="control3947"):
    """Bind the selected actor to its OWN measured interface and full provenance."""
    spec = locomotion_spec(selected_model)
    pins = dict(BALANCED_PINS if selected_model == "balanced4246" else EVIDENCE)
    pins[Path(spec["path"])] = spec["sha"]
    def bind(path, digest):
        path = Path(path).resolve()
        require(path.is_file() and sha(path) == digest, "Selected evidence changed/missing: " + str(path))
        require(path not in pins or pins[path] == digest, "Conflicting selected evidence digest")
        pins[path] = digest
    for path, digest in list(pins.items()):
        bind(path, digest)
    report = read_json(spec["screen_report_path"])
    spec["prerequisite"] = validate_screen(report, selected_model)
    if selected_model == "balanced4246":
        # Imported only after source pins; these pure verifiers start no simulator.
        from evaluate_balanced_gait_candidate import candidate_receipt
        from audit_balanced_gait_screen_identity import audit
        from compare_balanced_gait_screens import compare
        proof = candidate_receipt(BALANCED_RUN / "common_physics_training_result.json", Path(spec["path"]), "balanced")
        require(proof == report["candidate_training"], "Screen/full actual training verification differs")
        spec["training_verification"] = proof
        for arm, directory in (("control", MATCHED_CONTROL_SCREEN), ("balanced", BALANCED_SCREEN)):
            summary_path = directory / "summary.json"
            require(audit(summary_path, arm) == read_json(directory / "identity_audit_v1.json"),
                    "Actual21-case identity audit differs: " + arm)
            summary = read_json(summary_path)
            for item in summary["source_snapshot"]:
                bind(item["path"], item["sha256"])
            for key in ("checkpoint", "training_receipt"):
                bind(summary[key], summary[key + "_sha256"])
            receipt = read_json(summary["training_receipt"])
            smoke_path = Path(receipt["smoke_evidence"]["path"])
            bind(smoke_path, receipt["smoke_evidence"]["sha256"])
            for training_path, ledger in ((Path(summary["training_receipt"]), receipt), (smoke_path, read_json(smoke_path))):
                for item in ledger["source_snapshot"] + ledger["actual_yaml"]:
                    bind(item["path"], item["sha256"])
                bind(ledger["checkpoint"], ledger["checkpoint_metadata"]["sha256"])
                for name in ("common_physics_pretrain_guard.json", "common_physics_invocation.json", "training_source_generated.py"):
                    path = training_path.parent / name
                    bind(path, sha(path))
            for row in summary["rows"]:
                actual = read_json(row["report"])
                for key in ("report", "invocation"):
                    bind(row[key], row[key + "_sha256"])
                bind(actual["artifacts"]["csv"], row["csv_sha256"])
                bind(actual["retention_experiment"]["trace_path"], row["trace_sha256"])
                bind(Path(row["report"]).parent / "retention_generated.py", row["generated_sha256"])
                for item in read_json(row["invocation"])["source_and_inputs"]:
                    bind(item["path"], item["sha256"])
        comparison = compare(MATCHED_CONTROL_SCREEN / "summary.json", BALANCED_SCREEN / "summary.json")
        require(comparison == read_json(BALANCED_SCREEN / "matched_comparison.json"), "Actual matched comparison differs")
        spec["selection_limitations"] = validate_balanced_limitations(read_json(BALANCED_SCREEN / "summary.json"), comparison)
        spec["completed_screen_identity_audits"] = 42
    spec["verified_inputs"] = {str(path.resolve()): digest for path, digest in pins.items()}
    return spec


def source_inventory(args, locomotion=None):
    locomotion = locomotion or verify_locomotion(getattr(args, "locomotion_model", "control3947"))
    paths = {ROOT / name: expected for name, expected in FROZEN.items()}
    paths.update({path: digest for path, digest in EVIDENCE.items() if path.parent == BANK})
    paths.update({Path(path): digest for path, digest in locomotion["verified_inputs"].items()})
    paths.update({args.checkpoint: MODEL_SHA["roll"], args.stand_checkpoint: MODEL_SHA["stand"]})
    for path, expected in paths.items():
        require(path.is_file() and sha(path) == expected, f"Frozen input changed/missing: {path}")
    installed = Path("E:/IsaacLab/repo/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2")
    for name in ("recovery_env_cfg.py", "recovery_control_targets.py"):
        require(sha(installed / name) == FROZEN["src/go2_recovery/" + name], "Installed physical source changed")
    extras = [Path(__file__), ROOT / "scripts/continuous_flow_runtime.py", ROOT / "scripts/continuous_flow_phase.py",
              ROOT / "scripts/continuous_flow_motion_metrics.py",
              Path("E:/IsaacLab/repo/source/isaaclab/isaaclab/envs/manager_based_rl_env.py"),
              Path("E:/IsaacLab/repo/source/isaaclab/isaaclab/scene/interactive_scene.py"),
              Path("E:/IsaacLab/userdata/assets/Robots/Unitree/Go2/go2.usd"),
              Path("E:/IsaacLab/userdata/assets/Environments/Grid/default_environment.usd")]
    extras += list(installed.rglob("*.py"))
    preservation_path = ROOT / "configs/overnight_preservation_20260917.json"
    preservation = json.loads(preservation_path.read_text(encoding="utf-8"))
    require(preservation["schema_version"] == "overnight_preservation_v1", "Unknown preserved-model inventory")
    require(len(preservation["files"]) == 16, "Preserve all16 portfolio models")
    for item in preservation["files"]:
        path = Path(item["path"])
        require(path.is_file() and path.stat().st_size == item["bytes"] and sha(path) == item["sha256"],
                "Preserved portfolio model changed: " + str(path))
        paths[path] = item["sha256"]
    extras.append(preservation_path)
    paths.update({path: sha(path) for path in extras})
    return {str(path.resolve()): expected for path, expected in paths.items()}


def verify_inventory(inventory):
    for path, expected in inventory.items():
        require(sha(path) == expected, f"Input/source changed during diagnostic: {path}")


def preflight(args):
    require(args.pose in ("upright", "side", "upside_down") and type(args.seed) is int, "Invalid actual-start request")
    require(args.speed == .5 and args.refinement == "off", "First integration permits only .5 command, no refinement")
    require(args.output_dir.resolve().is_relative_to((ROOT / "evaluations").resolve()), "Use E:/.../evaluations")
    require(not args.output_dir.exists(), "Never overwrite existing evidence")
    locomotion = verify_locomotion(getattr(args, "locomotion_model", "control3947"))
    inventory = source_inventory(args, locomotion)
    return {"protocol": PROTOCOL, "prerequisite": locomotion["prerequisite"], "locomotion": locomotion, "source_and_inputs": inventory,
            "helpers_source": extract_definitions(MIRROR.read_text(encoding="utf-8")),
            "promotion_performed": False, "hardware_deployment": False}


def assert_no_other_runtime():
    """Match the existing serial launcher boundary, excluding only this process."""
    command = ("@(Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -ne " + str(os.getpid()) +
               " -and $_.Name -match '^(python.*|kit.*|isaac-sim.*)\\.exe$' -and "
               "($_.CommandLine -match 'E:[\\\\/]IsaacLab' -or $_.ExecutablePath -match '^E:[\\\\/]IsaacLab[\\\\/]') } | "
               "Select-Object ProcessId,Name,CommandLine) | ConvertTo-Json -Compress")
    result = subprocess.run(["powershell.exe", "-NoProfile", "-Command", command], capture_output=True, text=True, check=True)
    require(not result.stdout.strip() or json.loads(result.stdout) in (None, []), "Another E-drive Python/Kit is active; serial only")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stand_checkpoint", type=Path, required=True)
    parser.add_argument("--locomotion_model", choices=("control3947", "balanced4246"), default="control3947")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--pose", choices=("upright", "side", "upside_down"), default="side")
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--speed", type=float, choices=(.5,), default=.5)
    parser.add_argument("--refinement", choices=("off",), default="off")
    parser.add_argument("--view", choices=("front", "oblique"), default="front")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Record all 50 PD intervals plus only two policy intervals; never full-flow acceptance")
    parser.add_argument("--preflight_only", action="store_true")
    # Importing AppLauncher is deferred until after pure local preflight.
    args, launcher_args = parser.parse_known_args()
    evidence = preflight(args)
    if args.preflight_only:
        print(json.dumps({k: v for k, v in evidence.items() if k != "helpers_source"}, indent=2))
        return
    assert_no_other_runtime()
    for name, value in {"OMNI_USER_HOME": "E:/IsaacLab/userdata", "OV_USER_HOME": "E:/IsaacLab/userdata",
                        "TEMP": "E:/IsaacLab/tmp", "TMP": "E:/IsaacLab/tmp", "PIP_CACHE_DIR": "E:/IsaacLab/cache/pip",
                        "PYTHONDONTWRITEBYTECODE": "1", "ISAACLAB_RECOVERY_BANK_COLLECTION": "1",
                        "ISAACLAB_GO2_USD": str(Path("E:/IsaacLab/userdata/assets/Robots/Unitree/Go2/go2.usd")),
                        "ISAACLAB_GROUND_USD": str(Path("E:/IsaacLab/userdata/assets/Environments/Grid/default_environment.usd"))}.items():
        os.environ[name] = value
    os.environ["CONDA_PREFIX"] = "E:/IsaacLab/env"
    os.environ["PATH"] = "E:\\IsaacLab\\env;E:\\IsaacLab\\env\\Scripts;" + os.environ.get("PATH", "")
    from isaaclab.app import AppLauncher
    app_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(app_parser)
    app_args = app_parser.parse_args(launcher_args)
    app_args.enable_cameras = args.video
    args.device = app_args.device or "cuda:0"
    args.min_contacts, args.angle_deg = 4, 0. if args.pose == "upright" else 30.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "invocation.json").open("x", encoding="utf-8") as handle:
        json.dump({"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                   "evidence": {k: v for k, v in evidence.items() if k != "helpers_source"}}, handle, indent=2)
    app = AppLauncher(app_args).app
    try:
        from continuous_flow_runtime import run
        run(args, evidence)
    finally:
        app.close()


if __name__ == "__main__":
    main()
