"""Pinned 3547 continuation: one supported-symmetry reward addition only.

Inherited full model/Adam/std, receipt/train-split/physics/reset/config checks.
Frozen old files remain untouched. Smoke16x2 or independent formal128x200.
"""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "scripts/train_handoff_stand.py"
TEMPLATE_SHA = "aee036d6cb700fb953f5627b7d37b93bc3bdbda9756591d4efb20f5869fbc88c"
PARENT_SHA = "5527c4053cb6ef8e6616da65b7dc32a9464eb8d1366f4966a4eb4b65bc913fdb"


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("Pinned training anchor count differs: " + old[:90])
    return source.replace(old, new, 1)


def build_source():
    if hashlib.sha256(TEMPLATE.read_bytes()).hexdigest() != TEMPLATE_SHA:
        raise ValueError("Frozen training template changed")
    source = TEMPLATE.read_text(encoding="utf-8")
    edits = [
        ('PARENT = WORKSPACE / "repo/logs/rsl_rl/unitree_go2_recovery/2026-09-10_17-15-48_aligned_from_uncrossed2849/model_3448.pt"',
         'PARENT = WORKSPACE / "repo/logs/rsl_rl/unitree_go2_handoff_stand/20260917-handoff-stand128x100/model_3547.pt"'),
        ('PARENT_SHA = "1f546523baa57c7997ad2883689667d6e738eac098f14fb5fa595aae778a2de2"', f'PARENT_SHA = "{PARENT_SHA}"'),
        ('PARENT_ITER, PARENT_ADAM_STEP = 3448, 69120', 'PARENT_ITER, PARENT_ADAM_STEP = 3547, 71120'),
        ('PARENT_LR = 2.2500000000000008e-05', 'PARENT_LR = 1e-05'),
        ('"formal": (128, 100)', '"formal": (128, 200)'),
        ('"recovery/recovery_aligned_model3448.pt": PARENT_SHA,',
         '"recovery/recovery_aligned_model3448.pt": "1f546523baa57c7997ad2883689667d6e738eac098f14fb5fa595aae778a2de2",'),
        ('"protocol": "handoff_stand_training_v1"', '"protocol": "supported_symmetry_handoff_training_v1"'),
        ('    files.update([Path(__file__), ROOT / "scripts/validate_handoff_replay.py",',
         '    files.update([Path(__file__), ROOT / "scripts/train_handoff_stand.py", ROOT / "scripts/natural_stance_reward.py", ROOT / "scripts/validate_handoff_replay.py",'),
        ('    for field in ("rewards", "observations", "terminations", "commands", "curriculum"):',
         '''    from isaaclab.managers import RewardTermCfg
    from natural_stance_reward import supported_symmetry_reward
    expected_rewards = aligned.rewards.to_dict()
    expected_rewards["natural_symmetry"] = RewardTermCfg(func=supported_symmetry_reward, weight=2.0).to_dict()
    require(cfg.rewards.to_dict() == expected_rewards, "Only the supported-symmetry reward addition is permitted")
    for field in ("observations", "terminations", "commands", "curriculum"):'''),
        ('"aligned_rewards_observations_terminations_commands_curriculum_equal": True,',
         '"aligned_original_rewards_preserved": True, "supported_symmetry_weight": 2.0, "observations_terminations_commands_curriculum_equal": True,'),
        ('    cfg, agent = configs.HandoffStandEnvCfg(), UnitreeGo2RecoveryPPORunnerCfg()',
         '''    cfg, agent = configs.HandoffStandEnvCfg(), UnitreeGo2RecoveryPPORunnerCfg()
    from isaaclab.managers import RewardTermCfg
    from natural_stance_reward import supported_symmetry_reward
    cfg.rewards.natural_symmetry = RewardTermCfg(func=supported_symmetry_reward, weight=2.0)'''),
    ]
    for old, new in edits:
        source = replace_once(source, old, new)
    return source


if __name__ == "__main__":
    source = build_source()
    exec(compile(source, str(Path(__file__)) + "::<3547-symmetry>", "exec"),
         {"__name__": "__main__", "__file__": str(Path(__file__).resolve())})
