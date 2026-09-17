"""Supported, near-upright symmetry refinement; no force during inversion.

This is an engineering shaping hypothesis, not animal-motion imitation.
All tensors use ordered FL/FR/RL/RR feet in the physical robot body frame.
"""
import torch


def supported_symmetry_score(feet_b, gravity_b, height, foot_fz, base_force, command):
    if feet_b.ndim != 3 or feet_b.shape[1:] != (4, 3):
        raise ValueError("Expected ordered N x 4 x 3 feet")
    mirrored_right = feet_b[:, [1, 3]] * feet_b.new_tensor([1., -1., 1.])
    error = feet_b[:, [0, 2]] - mirrored_right
    # Bounded positive reward: lifting a foot cannot evade a new penalty.
    score = torch.exp(-(error / feet_b.new_tensor([.06, .04, .04])).square().sum(-1).mean(-1))
    cos_up = -gravity_b[:, 2] / gravity_b.norm(dim=-1).clamp_min(1e-6)
    gate = ((cos_up - .94) / .04).clamp(0., 1.)
    gate *= ((height - .27) / .03).clamp(0., 1.)
    support = (foot_fz > 5.).all(-1) & (base_force <= 1.)
    stationary_command = command.norm(dim=-1) < 1e-6
    return score * gate * support * stationary_command


def supported_symmetry_reward(env):
    from isaaclab.utils.math import quat_apply_inverse
    robot = env.scene["robot"]
    sensor = env.scene.sensors["contact_forces"]
    if not hasattr(env, "_natural_symmetry_ids"):
        names = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
        env._natural_symmetry_ids = (
            robot.find_bodies(names, preserve_order=True)[0],
            sensor.find_bodies(names, preserve_order=True)[0],
            sensor.find_bodies("base")[0])
    body_ids, contact_ids, base_ids = env._natural_symmetry_ids
    a = robot.data
    quat = a.root_quat_w[:, None, :].expand(-1, 4, -1)
    feet = quat_apply_inverse(quat, a.body_pos_w[:, body_ids] - a.root_pos_w[:, None, :])
    return supported_symmetry_score(
        feet, a.projected_gravity_b, a.root_pos_w[:, 2] - env.scene.env_origins[:, 2],
        sensor.data.net_forces_w[:, contact_ids, 2],
        sensor.data.net_forces_w[:, base_ids].norm(dim=-1).amax(-1),
        env.command_manager.get_command("base_velocity"))
