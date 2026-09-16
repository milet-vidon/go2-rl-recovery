"""Create an explicitly untrained matched-pair PPO seed, never a recovered model."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from rsl_rl.modules import ActorCritic
from tensordict import TensorDict


def fresh_checkpoint(seed=42):
    torch.manual_seed(seed)
    obs = TensorDict({"policy": torch.zeros(1, 48)}, batch_size=[1])
    model = ActorCritic(obs, obs_groups={"policy": ["policy"], "critic": ["policy"]},
                        num_actions=12, actor_hidden_dims=[128]*3,
                        critic_hidden_dims=[128]*3, activation="elu", init_noise_std=1.0,
                        actor_obs_normalization=False, critic_obs_normalization=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    return {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "iter": 0, "infos": {"untrained": True, "seed": seed,
                                 "purpose": "matched action-reference ablation, NOT old-policy continuation"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.resolve().drive.upper() != "E:":
        parser.error("The fresh bootstrap must be stored on E:")
    if args.output.exists():
        parser.error("Refusing to overwrite an existing bootstrap")
    checkpoint = fresh_checkpoint(args.seed)
    assert checkpoint["optimizer_state_dict"]["state"] == {}
    assert torch.equal(checkpoint["model_state_dict"]["std"], torch.ones(12))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    loaded = torch.load(args.output, map_location="cpu", weights_only=False)
    assert loaded["iter"] == 0 and loaded["optimizer_state_dict"]["state"] == {}
    assert all(torch.equal(v, loaded["model_state_dict"][k]) for k, v in checkpoint["model_state_dict"].items())
    print(json.dumps({"path": str(args.output), "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      "untrained": True, "seed": args.seed, "optimizer_state_entries": 0}))


if __name__ == "__main__":
    main()
