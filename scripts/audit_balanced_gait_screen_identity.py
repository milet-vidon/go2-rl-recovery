"""Read-only binding audit for the actual21-case balanced-gait screen.

No Torch/Isaac imports, simulation, promotion or changes to source reports.
Bind each summary label to its unique invocation, actual command trajectory,
report, training receipt and artifacts. This supplements (does not rewrite)
the active comparator and the wrapper's full checkpoint/physics verification.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "balanced_gait_screen_identity_audit_v1"
PARENT_SHA = "3dfbe03e0ac332fee688c970b420167affcdb6e8cacee16e40fc12120751df3f"
TASK = "Isaac-Natural-Stop-Flat-Unitree-Go2-Play-v0"
REPORT_PROTOCOL = "trained_balanced_gait_candidate_retention_v1"
COMMANDS = {"normal": (.5, 0., 0.), "retained08": (.8, 0., 0.),
            "left": (.5, .5, 0.), "right": (.5, -.5, 0.),
            "push05": (.5, 0., .5), "target10": (1., 0., 0.)}
PINS = {
    "run_balanced_gait_candidate_screen.ps1": "9ed912aa5656fd3e7873b1141fb4c9d73fc8d3b6f3256191e729e1860fe9c9f3",
    "evaluate_balanced_gait_candidate.ps1": "7cb9ba878f6bb7bfb89edd059aac2350453a70e6dbe21ba4090848dbe2e2751c",
    "evaluate_balanced_gait_candidate.py": "e42f2f34f1968e1bf99f6c1dc13aa1b53463fa22893db485b23dcad75df4c8fe",
    "train_balanced_gait_adaptation.py": "574e9cac8a5413293df898b6802fe5c0d1a3da9d34a00c0bb084b7e396dc79c5",
    "evaluate_go2_stand_walk_stop.py": "bbba369177d95bc24163bdd2b5ba96506f5db7d4b1ce38875f8a8885c49e22e4",
    "compare_balanced_gait_screens.py": "7518b1e1431207d7779778348a48ab825f7855d6d4f9f2b2974d3a01c2ece702",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "Duplicate JSON key: " + key)
            out[key] = value
        return out
    def constant(value):
        raise ValueError("Nonfinite JSON: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=pairs, parse_constant=constant)


def number(value, expected, label):
    require(type(value) in (int, float) and math.isfinite(value) and value == expected, "Wrong numeric " + label)


def path_equal(actual, expected, label):
    require(type(actual) is str and Path(actual).resolve() == Path(expected).resolve(), "Wrong path " + label)


def expected_cases():
    return ({("recovery", case, 20260909) for case in COMMANDS} |
            {("original", case, seed) for case in COMMANDS if case != "target10"
             for seed in (20260909, 20260910, 20260911)})


def validate_summary(summary, arm):
    require(arm in ("control", "balanced") and summary["arm"] == arm, "Wrong arm")
    require(summary["protocol"] == "balanced_gait_candidate21_screen_v1" and
            summary["status"] == "completed_screening_only", "Incomplete/wrong screen")
    rows = summary["rows"]
    require(type(rows) is list and len(rows) == 21, "Require all21 rows")
    require(all(type(r["seed"]) is int for r in rows), "Seed must be an integer")
    require({(r["physics"], r["case"], r["seed"]) for r in rows} == expected_cases(), "Wrong21-case identity set")
    require(summary["promotion_performed"] is False and summary["training_performed"] is False
            and summary["reused_cases"] == 0, "Screen cannot train/promote/reuse")
    for r in rows:
        require(r["arm"] == arm and r["reused_existing"] is False and type(r["passed"]) is bool,
                "Wrong row arm/reuse/pass type")
    for key in ("report", "invocation"):
        paths = [Path(r[key]).resolve() for r in rows]
        require(len(set(paths)) == 21, "Duplicate actual " + key + " path")


def argument(argv, key):
    require(type(argv) is list and all(type(x) is str for x in argv), "Malformed invocation argv")
    indices = [i for i, value in enumerate(argv) if value == key]
    require(len(indices) == 1 and indices[0] + 1 < len(argv), "Missing/duplicated invocation argument: " + key)
    return argv[indices[0] + 1]


def validate_case(row, report, invocation, arm, checkpoint, checkpoint_sha, receipt_path, receipt_sha):
    """Pure metadata binding; called before independently hashing row artifacts."""
    physics, case, seed = row["physics"], row["case"], row["seed"]
    speed, yaw, push = COMMANDS[case]
    weight = 0.0 if arm == "control" else -10.0
    require(report["protocol_version"] == REPORT_PROTOCOL and report["task"] == TASK and
            report["baseline_metric_protocol_version"] == "stand_walk_stop_stance_geometry_v2", "Wrong actual report protocol/task")
    number(report["seed"], seed, "report seed")
    path_equal(report["checkpoint"], checkpoint, "report checkpoint")
    require(report["checkpoint_sha256"] == checkpoint_sha, "Wrong actual report checkpoint hash")
    exp = report["retention_experiment"]
    require(exp["physics_mode"] == physics and exp["reference_checked"] is (physics == "recovery")
            and exp["no_added_reset_or_history_write"] is True, "Wrong actual physics")
    for key, value in (("stand_s", 4.), ("walk_s", 8.), ("stop_s", 6.), ("walk_speed", speed),
                       ("lateral_speed", 0.), ("yaw_rate", yaw), ("push_delta_vy", push)):
        number(report["protocol"][key], value, "actual command " + key)
    for value in (report["global"]["steps"], exp["read_only_interface_steps"]):
        number(value, 900, "control intervals")
    proof = report["candidate_training"]
    require(proof["arm"] == arm, "Wrong actual training arm")
    number(proof["balanced_duration_weight"], weight, "training weight")
    path_equal(proof["training_receipt"], receipt_path, "training receipt")
    path_equal(proof["checkpoint"], checkpoint, "proof checkpoint")
    require(proof["training_receipt_sha256"] == receipt_sha and proof["checkpoint_sha256"] == checkpoint_sha
            and proof["parent_control_sha256"] == PARENT_SHA, "Wrong actual lineage hashes")
    for key, value in (("formal_updates_verified", 300), ("actual_environment_steps", 921600),
                       ("actual_control_steps", 7200), ("actual_iteration", 4246)):
        number(proof[key], value, key)
    require(proof["actual_adam_steps"] == [85120] * 17 and proof["model_and_optimizer_all_finite"] is True
            and proof["quality_accepted"] is False and proof["promotion_performed"] is False, "Wrong final full-state proof")
    require(invocation["schema"] == "trained_balanced_gait_candidate_invocation_v1"
            and invocation["status"] == "completed_screening_only" and invocation["arm"] == arm
            and invocation["physics_mode"] == physics and invocation["seed"] == seed, "Wrong actual invocation identity")
    path_equal(invocation["checkpoint"], checkpoint, "invocation checkpoint")
    require(invocation["checkpoint_sha256"] == checkpoint_sha and invocation["promotion_performed"] is False, "Wrong invocation model/promotion")
    argv = invocation["args"]
    for key, value in (("--arm", arm), ("--physics_mode", physics), ("--task", TASK), ("--seed", str(seed))):
        require(argument(argv, key) == value, "Wrong actual invocation " + key)
    for key, value in (("--checkpoint", checkpoint), ("--candidate_training_result", receipt_path),
                       ("--output_dir", Path(row["report"]).parent)):
        path_equal(argument(argv, key), value, key)
    for key, value in (("--stand_s", 4.), ("--walk_s", 8.), ("--stop_s", 6.), ("--walk_speed", speed),
                       ("--lateral_speed", 0.), ("--yaw_rate", yaw), ("--push_speed", push)):
        number(float(argument(argv, key)), value, "invocation " + key)
    require(argv.count("--no_video") == 1 and report["artifacts"]["video"] is None and report["video_view"] is None, "Not the no-video21-case screen")
    acceptance = report["acceptance"]
    expected = {"no_reset", "no_base_contact", "supported_height", "level", "walk_tracking", "quiet_stand_stop",
                "feet_lift_in_walk", "limited_slip", "four_feet_at_rest", "normal_stance_geometry_at_rest",
                "four_vertical_contacts_at_rest", "no_current_base_contact_at_rest", "geometry_and_support_at_rest"}
    if yaw:
        expected |= {"lateral_tracking", "yaw_tracking", "quiet_yaw"}
    require(set(acceptance) == expected and all(type(v) is bool for v in acceptance.values()), "Wrong actual acceptance fields/types")
    passed = all(acceptance.values())
    walk = report["settled_phase_stats"]["walk"]
    for name in ("vx_b_mean", "vy_b_mean", "yaw_rate_mean", "contact_slip_mean"):
        require(type(walk[name]) in (int, float) and math.isfinite(walk[name]), "Nonfinite actual walk metric")
    drift = abs(walk["vy_b_mean"]) < .12 and abs(walk["yaw_rate_mean"]) < .15 if yaw == push == 0 else None
    retained = passed and drift is not False
    require(report["passed"] is passed and report["retention_passed"] is retained
            and row["passed"] is retained and invocation["retention_passed"] is retained, "False/nonboolean passed label")
    require(report["retention_acceptance"] == {"original_checks": passed, "straight_drift": drift}
            and row["original_checks"] is passed and row["straight_drift"] is drift, "Changed actual retention gate")
    failed = sorted([k for k,v in acceptance.items() if not v] + (["straight_drift"] if drift is False else []))
    require(sorted(row["failed_criteria"]) == failed, "Hidden/changed failure list")
    for key, actual in (("vx", "vx_b_mean"), ("vy", "vy_b_mean"), ("yaw", "yaw_rate_mean"), ("slip", "contact_slip_mean")):
        number(row[key], walk[actual], "summary actual " + key)


def validate_trajectory(csv_path, trace, physics, case):
    """Bind commands to all900 physical records, not only summary labels."""
    speed, yaw, push = COMMANDS[case]
    f32 = lambda value: struct.unpack("f", struct.pack("f", value))[0]
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 900 and len(trace["steps"]) == 900 and trace["schema"] == REPORT_PROTOCOL
            and trace["physics_mode"] == physics, "Incomplete/wrong900-step actual trace")
    number(trace["interface"]["observation_dim"], 48, "observation dimension")
    number(trace["interface"]["step_dt"], .02, "control dt")
    number(trace["interface"]["physics_dt"], .005, "physics dt")
    done_intervals = 0
    for i, (r, t) in enumerate(zip(rows, trace["steps"])):
        phase = "stand" if i < 200 else "walk" if i < 600 else "stop"
        command = [f32(speed), 0., f32(yaw)] if phase == "walk" else [0., 0., 0.]
        require(int(r["step"]) == i + 1 and abs(float(r["time_s"]) - (i+1)*.02) < 1e-9
                and r["phase"] == phase and t["step"] == i and t["physics_mode"] == physics, "Wrong actual phase/time identity")
        require([float(r[k]) for k in ("cmd_x", "cmd_y", "cmd_yaw")] == command, "CSV actual command differs")
        require(len(t["observation"]) == 1 and len(t["observation"][0]) == 48
                and t["observation"][0][9:12] == command, "Policy actual observed command differs")
        number(float(r["push_delta_vy"]), ({100: push, 400: -push, 750: push}.get(i, 0.)), "physical push")
        # A real done is a behavior failure, not automatically damaged evidence.
        # Actual trace stores integer flags; bool flags are semantically valid too.
        require(type(t["done"]) is list and len(t["done"]) == 1 and
                type(t["done"][0]) in (int, bool) and t["done"][0] in (0, 1), "Invalid measured trace done flag")
        require(r["done"] in ("0", "1") and int(r["done"]) == int(t["done"][0]), "CSV/trace done disagreement")
        done_intervals += int(t["done"][0])
    require(sum(r["phase"] == "walk" and float(r["time_s"]) > 5 for r in rows) == 350, "Wrong350-frame duty window")
    return {"control_records": 900, "done_intervals": done_intervals}


def validate_reset_check(report, trajectory):
    expected = trajectory["done_intervals"] == 0
    require(report["acceptance"]["no_reset"] is expected, "Reported no_reset differs from actual measured done flags")


def audit(path, arm):
    path = Path(path).resolve()
    summary = read_json(path)
    validate_summary(summary, arm)
    hashed = {}
    def verify(file, digest):
        file = Path(file).resolve()
        require(type(digest) is str and len(digest) == 64, "Missing SHA256")
        actual = hashed.setdefault(file, sha(file)) if file not in hashed else hashed[file]
        require(actual == digest.lower(), "Artifact/source changed: " + str(file))
        return file
    def inventory(items):
        seen = set()
        for item in items:
            file = verify(item["path"], item["sha256"])
            require(file not in seen, "Duplicate source inventory path")
            seen.add(file)
        return seen
    for name, digest in PINS.items():
        verify(ROOT / "scripts" / name, digest)
    sources = inventory(summary["source_snapshot"])
    required = {ROOT / "scripts" / name for name in PINS if name != "compare_balanced_gait_screens.py"}
    # Original metric source is inventoried by every child, not the screen's old
    # minimal snapshot in unrelated historical schemas. New screen includes it.
    require({p.resolve() for p in required}.issubset(sources), "Missing reviewed screen source identity")
    checkpoint = verify(summary["checkpoint"], summary["checkpoint_sha256"])
    receipt_path = verify(summary["training_receipt"], summary["training_receipt_sha256"])
    require(checkpoint.name == "model_4246.pt" and checkpoint.parent == receipt_path.parent, "Wrong formal checkpoint location")
    receipt = read_json(receipt_path)
    require(receipt["protocol"] == "balanced_gait_common_physics_training_v1" and receipt["arm"] == arm
            and receipt["completion_verified"] is True and receipt["parent"]["sha256"] == PARENT_SHA, "Wrong actual training receipt")
    number(receipt["balanced_duration_weight"], 0. if arm == "control" else -10., "receipt weight")
    for key, value in (("num_envs",128),("updates",300),("actual_environment_steps",921600),("actual_control_steps",7200)):
        number(receipt[key], value, key)
    path_equal(receipt["checkpoint"], checkpoint, "actual training checkpoint")
    require(receipt["checkpoint_metadata"]["sha256"] == summary["checkpoint_sha256"]
            and receipt["checkpoint_metadata"]["iter"] == 4246 and receipt["checkpoint_metadata"]["adam_steps"] == [85120]*17,
            "Wrong final checkpoint ledger")
    inventory(receipt["source_snapshot"])
    smoke_path = verify(receipt["smoke_evidence"]["path"], receipt["smoke_evidence"]["sha256"])
    smoke = read_json(smoke_path)
    require(smoke["arm"] == arm and smoke["source_snapshot"] == receipt["source_snapshot"] and
            smoke["completion_verified"] is True and smoke["updates"] == 2 and smoke["num_envs"] == 16,
            "Wrong independent same-arm/source smoke")
    seen_artifacts, output_rows = set(), []
    for row in summary["rows"]:
        directory = path.parent / f"{row['physics']}-{row['case']}-seed{row['seed']}"
        report_path = verify(row["report"], row["report_sha256"])
        invocation_path = verify(row["invocation"], row["invocation_sha256"])
        require(report_path == directory / "model_4246_stand_walk_stop.json" and invocation_path == directory / "invocation.json",
                "Actual artifacts escape/misidentify prescribed case directory")
        report, invocation = read_json(report_path), read_json(invocation_path)
        path_equal(invocation["report"], report_path, "invocation report")
        require(invocation["report_sha256"] == row["report_sha256"], "Invocation/report digest differs")
        child_sources = inventory(invocation["source_and_inputs"])
        require({checkpoint,receipt_path,(ROOT/"scripts/evaluate_balanced_gait_candidate.py").resolve(),
                 (ROOT/"scripts/evaluate_balanced_gait_candidate.ps1").resolve()}.issubset(child_sources), "Missing critical invocation input")
        validate_case(row, report, invocation, arm, checkpoint, summary["checkpoint_sha256"], receipt_path, summary["training_receipt_sha256"])
        csv_path = verify(report["artifacts"]["csv"], row["csv_sha256"])
        exp = report["retention_experiment"]
        trace_path = verify(exp["trace_path"], row["trace_sha256"])
        generated = verify(directory / "retention_generated.py", row["generated_sha256"])
        require(exp["trace_sha256"] == row["trace_sha256"] and exp["generated_sha256"] == row["generated_sha256"], "Conflicting artifact digests")
        require(csv_path == directory / "model_4246_stand_walk_stop.csv" and trace_path == directory / "retention_interface.json", "Wrong case CSV/trace path")
        for file in (report_path, invocation_path, csv_path, trace_path, generated):
            require(file not in seen_artifacts, "Repeated actual case artifact")
            seen_artifacts.add(file)
        trajectory = validate_trajectory(csv_path, read_json(trace_path), row["physics"], row["case"])
        validate_reset_check(report, trajectory)
        output_rows.append({"physics":row["physics"],"case":row["case"],"seed":row["seed"],"behavior_passed":row["passed"],
                            "actual_done_intervals":trajectory["done_intervals"],
                            "report_sha256":row["report_sha256"],"csv_sha256":row["csv_sha256"]})
    for file, digest in hashed.items():
        require(sha(file) == digest, "Evidence changed during audit: " + str(file))
    return {"protocol":PROTOCOL,"audit_passed":True,"arm":arm,"summary":str(path),"summary_sha256":sha(path),
            "source_sha256":sha(__file__),"unique_case_artifacts":len(seen_artifacts),"actual_control_records_checked":18900,
            "rows":output_rows,"promotion_performed":False,"behavior_accepted":False,
            "scope":"Identity, hashes and physical command binding only. Checkpoint tensor/config verification remains the pinned wrapper's responsibility; no new behavior, gait, visual or hardware acceptance."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--arm", required=True, choices=("control", "balanced"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(args.output.resolve().drive.upper() == "E:" and not args.output.exists(), "Use a fresh E-drive audit output")
    try:
        result = audit(args.summary, args.arm)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        result = {"protocol":PROTOCOL,"audit_passed":False,"error":str(error),"arm":args.arm,
                  "promotion_performed":False,"behavior_accepted":False}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k != "rows"}, indent=2, allow_nan=False))
    raise SystemExit(0 if result["audit_passed"] else 1)
