"""Go2 adapter for independently implemented Smith-inspired reward shaping.

Only the isolated SmithNominal task uses this term. It is not a success test:
the existing external four-foot/geometry/continuous-hold evaluator is unchanged.
"""

from isaaclab.managers import ManagerTermBase

from .recovery_smith_math import smith_recovery_terms


class SmithRecoveryReward(ManagerTermBase):
    """Resolve joint-type weights by name, never by an assumed simulator order."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene["robot"]
        expected = {f"{leg}_{kind}_joint" for leg in ("FL", "FR", "RL", "RR")
                    for kind in ("hip", "thigh", "calf")}
        names = self.robot.joint_names
        if len(names) != 12 or set(names) != expected:
            raise ValueError(f"Smith Go2 adapter requires the 12 named Go2 joints, got {names}")
        weights = {"hip": 1.0, "thigh": 0.75, "calf": 0.5}
        self.joint_weights = self.robot.data.joint_pos.new_tensor(
            [weights[name.split("_")[1]] for name in names])

    def __call__(self, env, mode: str, target_height: float = 0.32):
        if mode not in ("roll", "stand"):
            raise ValueError(f"Unknown Smith reward component: {mode}")
        data = self.robot.data
        terms = smith_recovery_terms(
            -data.projected_gravity_b[:, 2],
            data.root_pos_w[:, 2] - env.scene.env_origins[:, 2],
            data.joint_pos - data.default_joint_pos,
            data.joint_vel,
            self.joint_weights,
            target_height=target_height,
        )
        return terms[mode]
