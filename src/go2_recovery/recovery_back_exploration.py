"""Opt-in, unvalidated back-recovery exploration hypothesis for the 48-D Go2 policy.

This changes the stochastic Normal distribution only. It does not change the
actor mean, nominal-pose action reference, inference, rewards, or success tests.
Importing this module does not register or activate the candidate. A caller must
explicitly register it and select ``BackExplorationActorCritic`` in a new run.

Required fixed actor layout (no history or observation normalization): linear
velocity [0:3], angular velocity [3:6], projected gravity [6:9], commands [9:12],
relative joints [12:24], joint velocities [24:36], previous actions [36:48].
The gravity slice must be unscaled/unclipped; other terms keep their task scales.
Gravity may contain the existing observation noise. The SAME stored observation
must be used for rollout and PPO probability recomputation. Shape validation
cannot detect reordered terms: the integrating task must verify their names.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from rsl_rl.modules import ActorCritic
from tensordict import TensorDict
from torch.distributions import Normal


class BackExplorationActorCritic(ActorCritic):
    """Keep checkpoint-compatible parameters; floor std only when tipped back.

    The default floor is ``0.6 * clamp(normalized_gravity_z / 0.5, 0, 1)``.
    Exact upright/side observations therefore retain their learned standard
    deviations, while a >=120 degree tilt gets a 0.6 action-space std floor.
    This is an exploration hypothesis, not evidence of successful recovery.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = (128, 128, 128),
        critic_hidden_dims: tuple[int] | list[int] = (128, 128, 128),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        back_std_floor: float = 0.6,
        back_full_activation_gz: float = 0.5,
        gravity_norm_epsilon: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            raise ValueError(f"Unknown candidate policy options: {sorted(kwargs)}")
        if actor_obs_normalization or critic_obs_normalization:
            raise ValueError("This candidate requires actor/critic observation normalization disabled")
        if state_dependent_std or noise_std_type != "scalar":
            raise ValueError("This candidate requires the checkpoint's state-independent scalar std parameter")
        if num_actions != 12:
            raise ValueError("This candidate requires the verified 12-action Go2 joint order")
        if obs_groups.get("policy") != ["policy"] or obs_groups.get("critic") != ["policy"]:
            raise ValueError("This candidate requires policy and critic to read the single 'policy' group")
        if "policy" not in obs or obs["policy"].ndim != 2 or obs["policy"].shape[-1] != 48:
            raise ValueError("This candidate requires the verified 48-D policy layout, gravity at [6:9]")
        for name, value, upper in (
            ("back_std_floor", back_std_floor, 1.0),
            ("back_full_activation_gz", back_full_activation_gz, 1.0),
            ("gravity_norm_epsilon", gravity_norm_epsilon, 1e-3),
        ):
            if isinstance(value, bool) or not isinstance(value, (float, int)):
                raise ValueError(f"{name} must be a finite number in (0, {upper}]")
            if not math.isfinite(value) or not 0 < value <= upper:
                raise ValueError(f"{name} must be a finite number in (0, {upper}]")
        if not math.isfinite(init_noise_std) or init_noise_std <= 0:
            raise ValueError("init_noise_std must be finite and positive")

        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            num_actions=num_actions,
            actor_obs_normalization=False,
            critic_obs_normalization=False,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type="scalar",
            state_dependent_std=False,
        )
        # Plain immutable configuration only: no parameters/buffers are added,
        # so strict model loading and Adam's parameter order remain unchanged.
        self.back_std_floor = float(back_std_floor)
        self.back_full_activation_gz = float(back_full_activation_gz)
        self.gravity_norm_epsilon = float(gravity_norm_epsilon)

    def _update_distribution(self, obs: torch.Tensor) -> None:
        if obs.ndim != 2 or obs.shape[-1] != 48:
            raise ValueError("Expected the unnormalized 48-D concatenated actor observation")
        gravity = obs[:, 6:9]
        gravity_norm = torch.linalg.vector_norm(gravity, dim=-1)
        if not torch.isfinite(gravity).all() or (gravity_norm <= self.gravity_norm_epsilon).any():
            raise ValueError("Projected gravity must be finite and have a nonzero norm")
        if not torch.isfinite(self.std).all() or (self.std <= 0).any():
            raise ValueError("Learned scalar std must remain finite and positive")

        super()._update_distribution(obs)
        normalized_gz = gravity[:, 2] / gravity_norm
        gate = (normalized_gz / self.back_full_activation_gz).clamp(0.0, 1.0)
        std = torch.maximum(self.distribution.stddev, self.back_std_floor * gate[:, None])
        # All inherited sampling/log-prob/entropy/action_sigma accessors now
        # use this exact same distribution, including PPO minibatch updates.
        self.distribution = Normal(self.distribution.mean, std)


def register_back_exploration_actor_critic() -> type[BackExplorationActorCritic]:
    """Explicitly register only this new name in the local RSL eval namespace.

    Idempotent for this class object; a conflicting pre-existing name fails
    instead of replacing another policy. No third-party source is modified.
    """
    from rsl_rl.runners import on_policy_runner

    name = "BackExplorationActorCritic"
    if hasattr(on_policy_runner, name):
        if getattr(on_policy_runner, name) is not BackExplorationActorCritic:
            raise RuntimeError(f"RSL runner already has a different object registered as {name}")
    else:
        setattr(on_policy_runner, name, BackExplorationActorCritic)
    return BackExplorationActorCritic
