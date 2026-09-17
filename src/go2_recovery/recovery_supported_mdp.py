"""Dense joint-posture shaping for one isolated recovery experiment.

Not a contact/support or standing-success test. Historical tasks, observations,
actuator semantics and external success criteria remain unchanged.
"""

from isaaclab.managers import ManagerTermBase

from .recovery_supported_math import supported_pose_score, validate_supported_joint_soft_range


class SupportedPostureReward(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene["robot"]
        expected = {f"{leg}_{kind}_joint" for leg in ("FL", "FR", "RL", "RR")
                    for kind in ("hip", "thigh", "calf")}
        if len(self.robot.joint_names) != 12 or set(self.robot.joint_names) != expected:
            raise ValueError("Dense posture requires the named twelve-joint Go2 asset")
        limits = self.robot.data.soft_joint_pos_limits
        self.joint_range = (limits[..., 1] - limits[..., 0]).clone()
        validate_supported_joint_soft_range(self.joint_range)
        default = self.robot.data.default_joint_pos
        if not ((default >= limits[..., 0]) & (default <= limits[..., 1])).all().item():
            raise ValueError("Nominal posture must be inside actual soft limits")

    def __call__(self, env):
        data = self.robot.data
        return supported_pose_score(-data.projected_gravity_b[:, 2],
                                    data.joint_pos - data.default_joint_pos,
                                    self.joint_range)
