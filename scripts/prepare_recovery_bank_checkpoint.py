"""Create an auditable pilot bootstrap without changing the deterministic actor.

Only the scalar-action noise vector and optimizer moments are reset. Parent
weights are never overwritten. This is an exploration experiment, not a model
performance improvement or a promoted checkpoint.
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--noise_std", type=float, default=0.6)
    args = parser.parse_args()
    parent, output = args.parent.resolve(), args.output.resolve()
    lineage = output.with_suffix(".lineage.json")
    if output.drive.upper() != "E:" or output == parent:
        parser.error("Use a distinct output checkpoint on E:")
    if output.exists() or lineage.exists():
        parser.error("Refusing to overwrite checkpoint or lineage file")
    if not math.isfinite(args.noise_std) or not 0 < args.noise_std <= 1.0:
        parser.error("noise_std must be finite in (0,1]")
    # These are locally generated and previously loaded RSL-RL checkpoints.
    source = torch.load(parent, map_location="cpu", weights_only=False)
    target = copy.deepcopy(source)
    state = target["model_state_dict"]
    if "std" not in state or tuple(state["std"].shape) != (12,) or "log_std" in state:
        parser.error("Expected this project's 12-action, state-independent scalar std policy")
    state["std"].fill_(args.noise_std)
    if "state" not in target.get("optimizer_state_dict", {}):
        parser.error("Expected an Adam optimizer state dictionary")
    target["optimizer_state_dict"]["state"] = {}
    if any(not torch.equal(value, source["model_state_dict"][key])
           for key, value in state.items() if key != "std"):
        raise RuntimeError("Deterministic actor/critic tensors unexpectedly changed")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(target, output)
    loaded = torch.load(output, map_location="cpu", weights_only=False)
    for key, value in state.items():
        if not torch.equal(value, loaded["model_state_dict"][key]):
            raise RuntimeError(f"Checkpoint readback mismatch: {key}")
    manifest = {
        "status": "experimental_training_bootstrap_not_promoted",
        "parent": str(parent), "parent_sha256": hashlib.sha256(parent.read_bytes()).hexdigest(),
        "output": str(output), "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "iteration_preserved": source.get("iter"),
        "original_action_std": source["model_state_dict"]["std"].tolist(),
        "new_action_std": args.noise_std,
        "changes": ["state-independent sampling std", "clear Adam moments for new-start-distribution pilot"],
        "unchanged": "all deterministic actor and critic tensors, optimizer parameter groups, saved iteration",
        "purpose": "Test exploration from actual settled fallen starts; parent recovery hip std was below0.10",
        "scope": "simulation only; no hardware or evaluation noise injection",
    }
    lineage.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
