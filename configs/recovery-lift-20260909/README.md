# Recovery Lift continuation

- Parent checkpoint: `recovery_stage30_model2200.pt`
- Run: `2026-09-09_22-27-44_recovery_lift_from2200_20260909`
- Task: `Isaac-Recovery-Lift-Flat-Unitree-Go2-v0`
- PPO continuation: 100 iterations, 512 environments, 1,228,800 environment steps
- Curriculum: `ISAACLAB_RECOVERY_CURRICULUM_STEPS=1`
- Added terms: `low_height_support_penalty=-2.0`, `low_height_lift_velocity=1.5`
- Selected artifact: `models/recovery/recovery_lift_model2300.pt`

The added terms activate only for an upright, three-foot-supported body below
0.30 m. This is an experimental continuation and is not a claim of arbitrary
fall recovery; see the reports under `evaluations/recovery-lift-*`.
