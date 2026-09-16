"""Isolated action-reference experiment with 50 Hz sample-and-hold targets.

Both references use the same soft-limit clamp. This is deliberately NOT a
drop-in change for historical checkpoints: action semantics have changed.
"""

from __future__ import annotations

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass


class ControlStepJointPositionAction(JointPositionAction):
    """Cache q_ref + scale * a once per control step, not per physics substep."""

    def __init__(self, cfg, env):
        if cfg.reference not in ("nominal", "current"):
            raise ValueError("reference must be nominal or current")
        if cfg.use_default_offset or cfg.offset != 0.0 or cfg.clip is not None:
            raise ValueError("Requires offset=0, use_default_offset=False and clip=None")
        super().__init__(cfg, env)
        self._target = self._asset.data.joint_pos[:, self._joint_ids].clone()

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        data = self._asset.data
        reference = data.default_joint_pos if self.cfg.reference == "nominal" else data.joint_pos
        limits = data.soft_joint_pos_limits[:, self._joint_ids]
        self._target[:] = torch.clamp(
            reference[:, self._joint_ids] + self._processed_actions,
            min=limits[:, :, 0], max=limits[:, :, 1],
        )

    def apply_actions(self):
        # No joint-state read here: all four physics substeps share one target.
        self._asset.set_joint_position_target(self._target, joint_ids=self._joint_ids)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._raw_actions[ids] = 0.0
        self._processed_actions[ids] = 0.0
        self._target[ids] = self._asset.data.joint_pos[:, self._joint_ids][ids]


@configclass
class ControlStepJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = ControlStepJointPositionAction
    reference: str = "nominal"
    scale: float = 0.25
    offset: float = 0.0
    use_default_offset: bool = False
