"""Read-only guard for ONE speed-weight block; no torch, simulator, or writes.

A=1.5 and B=2.0 differ only in linear tracking weight. Both independently resume
the frozen control3947 full checkpoint. This is an engineering dose comparison,
not a paper reproduction, an acceptance test, or exact RNG/simulator continuity.
"""

import argparse
import collections
import copy
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import math
from pathlib import Path, PureWindowsPath
import pickle
import re
import struct
import zipfile

from check_smith_standweight_config import differences, read, require, _log_path


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = Path("E:/IsaacLab/repo/logs/rsl_rl/unitree_go2_flat")
PARENT_RUN = "2026-09-17_03-10-21_20260917-speedretention128x300"
PARENT = RUN_ROOT / PARENT_RUN / "model_3947.pt"
PARENT_SHA = "3dfbe03e0ac332fee688c970b420167affcdb6e8cacee16e40fc12120751df3f"
PARENT_CONFIG = ROOT / "configs/20260917-speedretention128x300"
CONFIG_SHA = {
    "env.yaml": "c13ceaf5b455bea105a6139840e4e8540d5d796ce476d867ca31fbfb4841ae14",
    "agent.yaml": "f919a8fac4cae0a55403c831621eff52dcb1849561f582e70f558050d1790546",
}
LEDGER = Path("E:/IsaacLab/artifacts/recovery-20260917/20260917-speedretention128x300/source-snapshot.json")
LEDGER_SHA = "979588A30F690580730AD13CEEEB39D7B1817F215F185CFA2E4B00EC5E2F9387".lower()
INSTALLED = Path("E:/IsaacLab/repo/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2")
WEIGHTS = {"A": Decimal("1.5"), "B": Decimal("2.0")}
BUDGETS = {(16, 2), (128, 50)}
PRESERVATION_SCRIPT = ROOT / "scripts/verify_overnight_baselines.py"
PRESERVATION_MANIFEST = ROOT / "configs/overnight_preservation_20260917.json"
PRESERVATION_SHA = {
    PRESERVATION_SCRIPT: "2fbd670d819278caa19142d77bc65ecc74f960e4afc19e8142fee3e507cba41c",
    PRESERVATION_MANIFEST: "0250a8c60db31fb21df03ffd16591d8c8e9bfe61e1c0d85a76e91c2929648e55",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def e_path(path):
    path = Path(path)
    windows = PureWindowsPath(str(path))
    require(windows.is_absolute() and windows.drive.upper() == "E:" and ".." not in windows.parts,
            f"Not an absolute E: path: {path}")
    require(PureWindowsPath(str(path.resolve())).drive.upper() == "E:", f"Path escapes E: {path}")
    return path


def number_is(value, expected):
    try:
        number = Decimal(value)
        return number.is_finite() and number == expected
    except (InvalidOperation, TypeError, ValueError):
        return False


def arguments_ok(arm, num_envs, iterations, run_name):
    require(arm in WEIGHTS, "Arm must be A or B")
    require(type(num_envs) is int and type(iterations) is int and (num_envs, iterations) in BUDGETS,
            "Only 16x2 smoke or one 128x50 block is allowed; no continuation block")
    require(isinstance(run_name, str) and re.fullmatch(r"[A-Za-z0-9_-]+", run_name), "Invalid run name")


def provenance():
    """Pin the complete parent ledger, not just the task's directly edited file."""
    require(sha(LEDGER) == LEDGER_SHA, "Parent source ledger changed")
    sources = json.loads(LEDGER.read_text(encoding="utf-8-sig"))
    require(isinstance(sources, list) and len(sources) == 39, "Unexpected frozen ledger schema")
    for item in sources:
        require(set(item) == {"path", "sha256"}, "Unexpected source record")
        require(sha(e_path(item["path"])) == item["sha256"].lower(), f"Parent source drift: {item['path']}")
    pinned_installed = {str(Path(s["path"]).resolve()).lower() for s in sources
                        if Path(s["path"]).is_relative_to(INSTALLED)}
    actual_installed = {str(p.resolve()).lower() for p in INSTALLED.rglob("*.py")}
    require(actual_installed == pinned_installed, "Installed Go2 source set changed")
    additions = [(LEDGER, LEDGER_SHA), (PARENT, PARENT_SHA)]
    for name, digest in CONFIG_SHA.items():
        additions += [(PARENT_CONFIG / name, digest), (RUN_ROOT / PARENT_RUN / "params" / name, digest)]
    for path, digest in additions:
        require(sha(e_path(path)) == digest, f"Frozen parent/config drift: {path}")
        sources.append({"path": str(path), "sha256": digest})
    for path in (Path(__file__).resolve(), ROOT / "scripts/run_speed_tracking_weight_block.ps1"):
        sources.append({"path": str(e_path(path)), "sha256": sha(path)})
    for path, digest in PRESERVATION_SHA.items():
        require(sha(e_path(path)) == digest, f"Preservation guard/manifest changed: {path}")
        sources.append({"path": str(path), "sha256": digest})
    manifest = json.loads(PRESERVATION_MANIFEST.read_text(encoding="utf-8"))
    require(manifest["schema_version"] == "overnight_preservation_v1"
            and manifest["never_overwrite"] is True and len(manifest["files"]) == 16,
            "Expected complete 16-model preservation manifest")
    preserved_paths = set()
    for item in manifest["files"]:
        path = e_path(item["path"])
        canonical = str(path.resolve()).lower()
        require(canonical not in preserved_paths, "Duplicate preserved model")
        preserved_paths.add(canonical)
        require(path.stat().st_size == item["bytes"] and sha(path) == item["sha256"],
                f"Preserved baseline changed: {path}")
        sources.append({"path": str(path), "sha256": item["sha256"]})
    return sources


class Storage:
    def __init__(self, key, count, location):
        self.key, self.count, self.location = key, count, location


class Tensor:
    def __init__(self, storage, offset, shape, stride, requires_grad=False, hooks=None):
        self.storage, self.offset = storage, offset
        self.shape, self.stride = tuple(shape), tuple(stride)
        self.requires_grad = requires_grad


class MetadataUnpickler(pickle.Unpickler):
    """Deny arbitrary globals; reconstruct metadata, never torch/CUDA objects."""
    def find_class(self, module, name):
        allowed = {("collections", "OrderedDict"): collections.OrderedDict,
                   ("torch", "FloatStorage"): "float32",
                   ("torch._utils", "_rebuild_tensor_v2"): Tensor}
        require((module, name) in allowed, f"Unexpected pickle global: {module}.{name}")
        return allowed[module, name]

    def persistent_load(self, record):
        require(isinstance(record, tuple) and len(record) == 5, "Invalid storage reference")
        kind, dtype, key, location, count = record
        require(kind == "storage" and dtype == "float32" and isinstance(key, str)
                and key.isdecimal() and type(count) is int and 0 < count < 1000000,
                "Unexpected storage type/key/size")
        require(isinstance(location, str) and re.fullmatch(r"cpu|cuda:\d+", location), "Invalid storage location")
        return Storage(key, count, location)


def checkpoint_metadata(path):
    """Bounded CPU-only scan (~1 MB checkpoint); all 68 float tensors checked."""
    with zipfile.ZipFile(e_path(path)) as archive:
        infos = archive.infolist()
        require(len(infos) < 128 and sum(i.file_size for i in infos) < 10000000, "Oversized checkpoint archive")
        names = [i.filename for i in infos]
        require(len(names) == len(set(names)), "Duplicate ZIP entries")
        pickles = [n for n in names if n.endswith("/data.pkl")]
        require(len(pickles) == 1, "Expected one checkpoint metadata pickle")
        base = pickles[0].rsplit("/", 1)[0]
        require(archive.read(base + "/byteorder") == b"little", "Unsupported checkpoint byte order")
        raw = MetadataUnpickler(io.BytesIO(archive.read(pickles[0]))).load()
        require(isinstance(raw, dict) and set(raw) == {"model_state_dict", "optimizer_state_dict", "iter", "infos"},
                "Unexpected checkpoint top-level schema")
        finite_storages = {}
        tensor_count = 0

        def tensor(t):
            nonlocal tensor_count
            require(isinstance(t, Tensor) and isinstance(t.storage, Storage), "Expected float tensor metadata")
            require(type(t.offset) is int and t.offset == 0 and len(t.shape) == len(t.stride), "Unexpected tensor layout")
            expected_stride = 1
            for size, stride in zip(reversed(t.shape), reversed(t.stride)):
                require(type(size) is int and size > 0 and stride == expected_stride, "Expected contiguous tensor")
                expected_stride *= size
            require(expected_stride == t.storage.count, "Tensor/storage size mismatch")
            key = t.storage.key
            data = archive.read(base + "/data/" + key)
            require(len(data) == t.storage.count * 4, "Float storage length mismatch")
            if key not in finite_storages:
                require(all(math.isfinite(v[0]) for v in struct.iter_unpack("<f", data)), "Non-finite checkpoint tensor")
                finite_storages[key] = t.storage.count
            require(finite_storages[key] == t.storage.count, "Inconsistent shared storage")
            tensor_count += 1
            return data

        model = raw["model_state_dict"]
        require(isinstance(model, dict) and len(model) == 17, "Expected 17 actor/critic/std parameters")
        shapes = {}
        for key, value in model.items():
            tensor(value)
            shapes[key] = list(value.shape)
        require("std" in model and model["std"].shape == (12,), "Expected learned 12-joint std")
        std_values = [v[0] for v in struct.iter_unpack("<f", archive.read(base + "/data/" + model["std"].storage.key))]
        require(all(v > 0 for v in std_values), "Action std must be positive")
        optimizer = raw["optimizer_state_dict"]
        require(isinstance(optimizer, dict) and set(optimizer) == {"state", "param_groups"}, "Bad Adam schema")
        groups, states = optimizer["param_groups"], optimizer["state"]
        require(isinstance(groups, list) and len(groups) == 1 and isinstance(states, dict) and len(states) == 17,
                "Expected one Adam group and 17 parameter states")
        params = groups[0]["params"]
        require(len(params) == 17 and len(set(params)) == 17 and set(params) == set(states), "Adam parameter ID mismatch")
        rate = groups[0]["lr"]
        require(type(rate) in (int, float) and math.isfinite(rate) and rate > 0, "Invalid actual optimizer LR")
        steps, state_shapes = [], {}
        for key, state in states.items():
            require(set(state) == {"step", "exp_avg", "exp_avg_sq"}, "Unexpected Adam state fields")
            require(state["step"].shape == (), "Adam step must be scalar")
            step = struct.unpack("<f", tensor(state["step"]))[0]
            require(step >= 0 and step.is_integer(), "Adam step must be a nonnegative integer")
            steps.append(int(step))
            tensor(state["exp_avg"])
            tensor(state["exp_avg_sq"])
            require(state["exp_avg"].shape == state["exp_avg_sq"].shape, "Adam moment shape mismatch")
            state_shapes[str(key)] = list(state["exp_avg"].shape)
        require(tensor_count == 68, "Expected all 68 model/optimizer tensors")
        require(type(raw["iter"]) is int and raw["iter"] >= 0, "Invalid saved iteration")
        return {"iter": raw["iter"], "sha256": sha(path), "tensor_count": tensor_count,
                "model_shapes": shapes, "optimizer_group": groups[0], "adam_steps": steps,
                "adam_state_shapes": state_shapes, "std": std_values, "all_tensors_finite": True}


def parent_metadata():
    metadata = checkpoint_metadata(PARENT)
    require(metadata["sha256"] == PARENT_SHA and metadata["iter"] == 3947, "Wrong actual parent checkpoint")
    require(metadata["optimizer_group"]["lr"] == 1e-5 and metadata["adam_steps"] == [79120] * 17,
            "Parent actual Adam LR/update counters changed")
    expected = {"std": [12]}
    for name, output in (("actor", 12), ("critic", 1)):
        for layer, out_size, in_size in ((0, 128, 48), (2, 128, 128), (4, 128, 128), (6, output, 128)):
            expected[f"{name}.{layer}.weight"] = [out_size, in_size]
            expected[f"{name}.{layer}.bias"] = [out_size]
    require(metadata["model_shapes"] == expected, "Parent actor/critic architecture changed")
    return metadata


def check_documents(env, agent, parent_env, parent_agent, arm, num_envs, iterations, run_name):
    arguments_ok(arm, num_envs, iterations, run_name)
    require(parent_agent["run_name"] == "20260917-speedretention128x300"
            and parent_agent["max_iterations"] == "300" and parent_agent["resume"] == "true"
            and parent_agent["load_checkpoint"] == "model_3648.pt"
            and parent_agent["load_run"] == "2026-09-09_15-50-37_natural_robust_push_20260909"
            and parent_agent["seed"] == "42", "Parent saved configuration identity changed")
    require(parent_env["commands"]["base_velocity"]["ranges"]["lin_vel_x"] == ["-0.5", "1.0"]
            and parent_env["commands"]["base_velocity"]["rel_standing_envs"] == "0.3", "Parent commands changed")
    expected_env = copy.deepcopy(parent_env)
    for scene in (expected_env["scene"], expected_env["scene"]["terrain"]):
        scene["num_envs"] = str(num_envs)
    reward = env["rewards"]["track_lin_vel_xy_exp"]
    require(number_is(reward["weight"], WEIGHTS[arm]), "Wrong tracking-weight arm")
    expected_env["rewards"]["track_lin_vel_xy_exp"]["weight"] = reward["weight"]
    _log_path(env["log_dir"], "env.log_dir")
    _log_path(env["sim"]["log_dir"], "env.sim.log_dir", allow_null=True)
    expected_env["log_dir"] = env["log_dir"]
    expected_env["sim"]["log_dir"] = env["sim"]["log_dir"]
    delta = differences(expected_env, env, "env")
    require(not delta, f"Unexpected command/reward/physics/actions change: {delta}")
    expected_agent = copy.deepcopy(parent_agent)
    expected_agent.update(run_name=run_name, max_iterations=str(iterations), load_run=PARENT_RUN,
                          load_checkpoint="model_3947.pt")
    rate = agent["algorithm"]["learning_rate"]
    require(number_is(rate, Decimal("0.00001")), "Resume LR scalar must match parent actual 1e-5")
    expected_agent["algorithm"]["learning_rate"] = rate
    delta = differences(expected_agent, agent, "agent")
    require(not delta, f"Unexpected PPO/full-resume configuration change: {delta}")


def check_final(metadata, parent, iterations):
    require(type(iterations) is int and iterations in (2, 50), "Invalid finite budget")
    require(metadata["iter"] == 3947 + iterations - 1, "Wrong final iteration label")
    require(metadata["adam_steps"] == [79120 + iterations * 20] * 17, "Wrong actual Adam update count")
    require(metadata["model_shapes"] == parent["model_shapes"]
            and metadata["adam_state_shapes"] == parent["adam_state_shapes"], "Model/Adam schema changed")
    group, parent_group = copy.deepcopy(metadata["optimizer_group"]), copy.deepcopy(parent["optimizer_group"])
    # Adaptive LR is allowed to evolve after the explicitly synchronized start.
    group.pop("lr")
    parent_group.pop("lr")
    require(group == parent_group, "Adam group/options/parameter ordering changed")
    require(metadata["all_tensors_finite"] and metadata["tensor_count"] == 68, "Incomplete finite scan")


def self_test(parent):
    """In-memory mutation tests; no simulator imports, temp files or test outputs."""
    pe, pa = read(PARENT_CONFIG / "env.yaml"), read(PARENT_CONFIG / "agent.yaml")
    rejected = 0

    def must_fail(call):
        nonlocal rejected
        try:
            call()
        except (AssertionError, KeyError, ValueError, TypeError):
            rejected += 1
        else:
            raise AssertionError("Mutation unexpectedly passed")

    for arm in WEIGHTS:
        for envs, updates in sorted(BUDGETS):
            env, agent = copy.deepcopy(pe), copy.deepcopy(pa)
            env["scene"]["num_envs"] = env["scene"]["terrain"]["num_envs"] = str(envs)
            env["rewards"]["track_lin_vel_xy_exp"]["weight"] = str(WEIGHTS[arm])
            agent.update(run_name="selftest", max_iterations=str(updates), load_run=PARENT_RUN, load_checkpoint="model_3947.pt")
            check_documents(env, agent, pe, pa, arm, envs, updates, "selftest")
            for keys, value in ((["commands", "base_velocity", "ranges", "lin_vel_x"], ["-0.5", "1.2"]),
                                (["commands", "base_velocity", "rel_standing_envs"], "0.1"),
                                (["rewards", "track_lin_vel_xy_exp", "params", "std"], "0.4"),
                                (["rewards", "track_lin_vel_xy_exp", "weight"], "nan"),
                                (["scene", "num_envs"], "512"), (["log_dir"], "C:/bad")):
                bad = copy.deepcopy(env)
                node = bad
                for key in keys[:-1]:
                    node = node[key]
                node[keys[-1]] = value
                must_fail(lambda: check_documents(bad, agent, pe, pa, arm, envs, updates, "selftest"))
            for key, value in (("load_run", "wrong_parent"), ("load_checkpoint", "model_3948.pt"),
                               ("resume", "false"), ("max_iterations", "100"), ("run_name", "other")):
                bad = copy.deepcopy(agent)
                bad[key] = value
                must_fail(lambda: check_documents(env, bad, pe, pa, arm, envs, updates, "selftest"))
            bad = copy.deepcopy(agent)
            bad["algorithm"]["learning_rate"] = "0.0003"
            must_fail(lambda: check_documents(env, bad, pe, pa, arm, envs, updates, "selftest"))
            final = copy.deepcopy(parent)
            final.update(iter=3947 + updates - 1, adam_steps=[79120 + updates * 20] * 17)
            check_final(final, parent, updates)
            for key, value in (("iter", 3947 + updates), ("adam_steps", [79120] * 17),
                               ("model_shapes", {}), ("all_tensors_finite", False)):
                bad = copy.deepcopy(final)
                bad[key] = value
                must_fail(lambda: check_final(bad, parent, updates))
    must_fail(lambda: arguments_ok("A", 128, 100, "bad"))
    must_fail(lambda: arguments_ok("B", 16, 50, "bad"))
    must_fail(lambda: MetadataUnpickler(io.BytesIO(b"cos\nsystem\n.")).load())
    return {"positive_config_cases": 4, "rejected_mutations": rejected}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("A", "B"), required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--run-name", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--candidate-dir", type=Path)
    mode.add_argument("--self-test", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    arguments_ok(args.arm, args.num_envs, args.iterations, args.run_name)
    require(bool(args.candidate_dir) == bool(args.checkpoint), "Candidate config and final checkpoint must be checked together")
    sources = provenance()
    parent = parent_metadata()
    result = {"status": "passed", "quality_accepted": False, "parent": parent, "sources": sources,
              "arm": args.arm, "weight": str(WEIGHTS[args.arm]), "num_envs": args.num_envs,
              "additional_updates": args.iterations, "expected_final_label": 3947 + args.iterations - 1,
              "expected_adam_step": 79120 + args.iterations * 20}
    if args.self_test:
        result["self_test"] = self_test(parent)
    elif args.candidate_dir:
        directory = e_path(args.candidate_dir)
        check_documents(read(directory / "env.yaml"), read(directory / "agent.yaml"),
                        read(PARENT_CONFIG / "env.yaml"), read(PARENT_CONFIG / "agent.yaml"),
                        args.arm, args.num_envs, args.iterations, args.run_name)
        require(args.checkpoint.name == f"model_{3947 + args.iterations - 1}.pt", "Unexpected endpoint filename")
        result["checkpoint"] = checkpoint_metadata(args.checkpoint)
        check_final(result["checkpoint"], parent, args.iterations)
    print(json.dumps(result, indent=2))
