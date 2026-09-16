"""CPU-only regression for an opt-in hypothesis; does not evaluate recovery.

Run with Isaac Lab's Python and a trusted local model_4899.pt checkpoint.
No simulator, training loop, model export, registration at import, or task
configuration change is performed. Registration tests clean up their namespace.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import math
from pathlib import Path

import torch
from rsl_rl.modules import ActorCritic
from rsl_rl.runners import on_policy_runner
from tensordict import TensorDict


MODULE_PATH = Path(__file__).parents[1] / "src/go2_recovery/recovery_back_exploration.py"
DEFAULT_CHECKPOINT = (
    Path(__file__).parents[2]
    / "repo/logs/rsl_rl/unitree_go2_recovery"
    / "2026-09-16_13-16-04_bank_nominalpd_from3900_20260916/model_4899.pt"
)


def _equal_tree(actual, expected) -> None:
    if isinstance(actual, torch.Tensor):
        assert isinstance(expected, torch.Tensor) and torch.equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _equal_tree(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for item, reference in zip(actual, expected):
            _equal_tree(item, reference)
    else:
        assert actual == expected


def _raises(exception, function) -> None:
    try:
        function()
    except exception:
        return
    raise AssertionError(f"Expected {exception.__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    class_name = "BackExplorationActorCritic"
    prior = vars(on_policy_runner).copy()
    spec = importlib.util.spec_from_file_location("go2_back_exploration_candidate", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert vars(on_policy_runner).keys() == prior.keys(), "Import must not register the policy"
    assert all(vars(on_policy_runner)[key] is value for key, value in prior.items())
    candidate_class = module.BackExplorationActorCritic

    torch.manual_seed(20260916)
    x = torch.randn(8, 48) * 0.01
    # Exact side, positive/negative side gravity noise, transition, back, and
    # nonunit gravity explicitly check normalization and the noisy boundary.
    x[:, 6:9] = torch.tensor([
        [0.0, 0.0, -1.0], [0.04, -0.03, -0.95], [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.05], [0.0, 1.0, -0.05],
        [0.0, math.sqrt(0.9375), 0.25], [0.0, 0.0, 1.0], [0.0, 0.0, 1.05],
    ])
    obs = TensorDict({"policy": x}, batch_size=[len(x)])
    kwargs = dict(
        obs_groups={"policy": ["policy"], "critic": ["policy"]}, num_actions=12,
        actor_hidden_dims=[128] * 3, critic_hidden_dims=[128] * 3, activation="elu",
    )
    with contextlib.redirect_stdout(io.StringIO()):
        reference = ActorCritic(obs, **kwargs)
        candidate = candidate_class(obs, **kwargs)
    # This trusted project checkpoint contains the Adam state as well as tensors.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    assert checkpoint["iter"] == 4899, "This regression specifically audits the 4899 continuation"
    assert reference.load_state_dict(checkpoint["model_state_dict"], strict=True)
    assert candidate.load_state_dict(checkpoint["model_state_dict"], strict=True)
    _equal_tree(candidate.state_dict(), reference.state_dict())
    assert list(candidate.state_dict()) == list(reference.state_dict())
    assert [name for name, _ in candidate.named_parameters()] == [name for name, _ in reference.named_parameters()]
    assert [tuple(p.shape) for p in candidate.parameters()] == [tuple(p.shape) for p in reference.parameters()]

    with torch.no_grad():
        assert torch.equal(candidate.act_inference(obs), reference.act_inference(obs))
        assert torch.equal(candidate.evaluate(obs), reference.evaluate(obs))
        torch.manual_seed(17)
        reference_actions = reference.act(obs)
        torch.manual_seed(17)
        actions = candidate.act(obs)
        assert torch.equal(candidate.action_mean, reference.action_mean)
        # Identical RNG and unchanged std imply bit-identical unaffected samples.
        unchanged = [0, 1, 2, 4]
        assert torch.equal(actions[unchanged], reference_actions[unchanged])
        assert torch.equal(candidate.action_std[unchanged], reference.action_std[unchanged])
        gravity = x[:, 6:9]
        gate = ((gravity[:, 2] / gravity.norm(dim=-1)) / 0.5).clamp(0, 1)
        expected_sigma = torch.maximum(reference.action_std, 0.6 * gate[:, None])
        assert torch.equal(candidate.action_std, expected_sigma)
        assert torch.equal(candidate.action_std[6:], torch.full_like(candidate.action_std[6:], 0.6))
        assert torch.all(candidate.action_std[3] >= reference.action_std[3])
        assert torch.all(candidate.action_std[3] <= torch.maximum(reference.action_std[3], x.new_tensor(0.06)))
        # This mirrors PPO's stored old sigma, log prob, and minibatch replay.
        old_sigma = candidate.action_std.detach().clone()
        old_mean = candidate.action_mean.detach().clone()
        old_logprob = candidate.get_actions_log_prob(actions).detach().clone()
        expected_logprob = (
            -((actions - old_mean) ** 2) / (2 * old_sigma ** 2)
            - old_sigma.log() - 0.5 * math.log(2 * math.pi)
        ).sum(-1)
        expected_entropy = (0.5 * (1 + math.log(2 * math.pi)) + old_sigma.log()).sum(-1)
        torch.testing.assert_close(old_logprob, expected_logprob, rtol=0, atol=2e-6)
        torch.testing.assert_close(candidate.entropy, expected_entropy, rtol=0, atol=2e-6)
        stored_obs = obs.clone()
        candidate.act(stored_obs)
        assert torch.equal(candidate.action_std, old_sigma)
        assert torch.equal(candidate.get_actions_log_prob(actions), old_logprob)
        ratio = (candidate.get_actions_log_prob(actions) - old_logprob).exp()
        assert torch.equal(ratio, torch.ones_like(ratio))
        analytic_kl = (
            (candidate.action_std / old_sigma).log()
            + (old_sigma.square() + (old_mean - candidate.action_mean).square())
            / (2 * candidate.action_std.square()) - 0.5
        ).sum(-1)
        assert torch.equal(analytic_kl, torch.zeros_like(analytic_kl))

    optimizer = torch.optim.Adam(candidate.parameters(), lr=0.0005)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    _equal_tree(optimizer.state_dict(), checkpoint["optimizer_state_dict"])
    steps = [float(state["step"]) for state in optimizer.state.values()]
    assert len(steps) == 17 and min(steps) == max(steps) == 20000
    # Backward is not an optimizer/training step: it checks floor-active std
    # gradients and mean gradients are finite without changing any parameter.
    before_backward = {key: value.clone() for key, value in candidate.state_dict().items()}
    candidate.act(obs)
    loss = -candidate.get_actions_log_prob(actions).mean() - 0.01 * candidate.entropy.mean()
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in candidate.parameters())
    assert candidate.std.grad is not None and torch.isfinite(candidate.std.grad).all()
    _equal_tree(candidate.state_dict(), before_backward)
    _equal_tree(optimizer.state_dict(), checkpoint["optimizer_state_dict"])

    invalid_options = [
        {"actor_obs_normalization": True}, {"critic_obs_normalization": True},
        {"state_dependent_std": True}, {"noise_std_type": "log"},
        {"num_actions": 11}, {"obs_groups": {"policy": ["other"], "critic": ["policy"]}},
        {"back_std_floor": 0}, {"back_std_floor": 1.1}, {"back_std_floor": float("nan")},
        {"back_full_activation_gz": 0}, {"back_full_activation_gz": 1.1},
        {"gravity_norm_epsilon": 0}, {"gravity_norm_epsilon": 0.01},
        {"init_noise_std": 0}, {"unknown_option": True},
    ]
    for options in invalid_options:
        _raises(ValueError, lambda options=options: candidate_class(obs, **(kwargs | options)))
    bad_shape = TensorDict({"policy": torch.zeros(8, 47)}, batch_size=[8])
    _raises(ValueError, lambda: candidate_class(bad_shape, **kwargs))
    for gravity_bad in ((0.0, 0.0, 0.0), (float("nan"), 0.0, 1.0)):
        malformed = obs.clone()
        malformed["policy"][0, 6:9] = torch.tensor(gravity_bad)
        _raises(ValueError, lambda: candidate.act(malformed))

    assert class_name not in prior, "Run this isolated test in a fresh process"
    try:
        assert module.register_back_exploration_actor_critic() is candidate_class
        assert module.register_back_exploration_actor_critic() is candidate_class
        assert getattr(on_policy_runner, class_name) is candidate_class
        assert on_policy_runner.ActorCritic is prior["ActorCritic"]
        conflict = object()
        setattr(on_policy_runner, class_name, conflict)
        _raises(RuntimeError, module.register_back_exploration_actor_critic)
        assert getattr(on_policy_runner, class_name) is conflict
    finally:
        if hasattr(on_policy_runner, class_name):
            delattr(on_policy_runner, class_name)
    assert vars(on_policy_runner).keys() == prior.keys()
    assert all(vars(on_policy_runner)[key] is value for key, value in prior.items())

    print(json.dumps({
        "result": "passed", "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_iteration": checkpoint["iter"], "device": "cpu",
        "mean_and_unaffected_samples_bit_exact": True,
        "normal_sampling_logprob_entropy_and_old_sigma_consistent": True,
        "model_keys_and_parameter_order_unchanged": True,
        "adam_state_unchanged": True, "adam_steps": sorted(set(steps)),
        "invalid_configuration_checks": len(invalid_options) + 3,
        "registration_explicit_idempotent_collision_safe": True,
        "simulator_started": False, "recovery_effectiveness": "not_evaluated",
    }, indent=2))


if __name__ == "__main__":
    main()
