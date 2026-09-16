# Go2 training status — 2026-09-16

Status: **in progress; user expectations are NOT yet met**. Simulation only. No claim of full reference-video reproduction or real-robot safety. All new code, models, reports and runtime data stay on E:.

## Acceptance priorities

1. Natural standing, walking and stopping, with no crossed limbs or ground-supported torso.
2. Recovery must finish in anatomically valid, quiet four-foot stance and retain it; height/contact counts alone are insufficient.
3. Distinguish airborne righting from getting up after an actual settled fall. Do not count an already-standing start as fallen recovery.
4. Compare checkpoints under identical tasks, seeds and protocols. New candidate promotion requires upright/30-degree regression, harder poses, and visual inspection of both front and oblique views. Do not relax criteria to make a run pass.

## Current training (do not duplicate)

- Run directory: `E:\IsaacLab\repo\logs\rsl_rl\unitree_go2_recovery\2026-09-16_11-46-18_rehearsal60_from3448_20260916`
- Log: `E:\IsaacLab\artifacts\recovery-20260916\train-rehearsal.log`
- Current-turn exec session: `38810` (may not survive a later app session; inspect process/log/checkpoints instead).
- Parent: `models/recovery/recovery_aligned_model3448.pt`, SHA256 `1f546523baa57c7997ad2883689667d6e738eac098f14fb5fa595aae778a2de2`.
- Parent load run: `2026-09-10_17-15-48_aligned_from_uncrossed2849`, checkpoint `model_3448.pt`.
- Task/stage: `Isaac-Recovery-Rehearsal-Flat-Unitree-Go2-v0` / `recovery_rehearsal`.
- 512 environments, 24 rollout steps, **1,000 additional iterations requested**, 12,288,000 additional environment steps if completed. Expected last checkpoint: `model_4447.pt`; verify the actual log/file before calling it complete.
- Latest observed saved checkpoint at this record's initial creation: `model_3800.pt`. Training still running; this is not the final result.
- Final pose mix: upright 25%, side 35%, fore/aft 30%, true inversion 5%, random SO(3) 5%. Pose-mixture curriculum 4,800 control steps.
- Tilt curriculum expands the hard subset from 20–30 degrees to 20–60 degrees over 12,000 control steps (500 PPO iterations); 35% of side/fore-aft samples retain 20–30-degree rehearsal. Independent +/-0.20 rad reset noise still applies.
- True inversion is now pi roll, unlike the historical capped `2 * fall_angle` class. Random SO(3) remains unrestricted.
- Preserves Aligned stance penalties, self-collision and 0.33 m stand target. These are still **controlled-drop training starts**, not a settled-fallen-state bank.

Launch already performed (do not run a second copy):

```powershell
Set-Location E:\IsaacLab\go2-rl-open-source
./scripts/train.ps1 -Stage recovery_rehearsal -NumEnvs 512 -MaxIterations 1000 -LoadRun 2026-09-10_17-15-48_aligned_from_uncrossed2849 -Checkpoint model_3448.pt -RunName rehearsal60_from3448_20260916 -Headless -CurriculumSteps 4800
```

## Measured parent baselines, not new-model results

Seed `20260916`, checkpoint3448, task `Isaac-Recovery-Aligned-Flat-Unitree-Go2-Play-v0`, 20 trials per pose. Success requires stance_geometry_v1 plus continuous 3 s valid support and quietness.

| Protocol / pose | Valid recovery or stand acquisition | Valid final stands | Actual settled-fallen eligible starts |
| --- | ---: | ---: | ---: |
| Controlled drop, upright | 20/20 | 20/20 | Not measured |
| Controlled drop, side 45 degrees | 16/20 | 16/20 | Not measured |
| Controlled drop, fore/aft 45 degrees | 14/20 | 14/20 | Not measured |
| Controlled drop, full inversion | 0/20 | 0/20 | Not measured |
| Pre-settled PD, side 90 degrees | 0/20 overall; 0/18 eligible fallen | 0/20 | 18/20 |
| Pre-settled PD, full inversion | 0/20 overall; 0/20 eligible fallen | 0/20 | 20/20 |

Reports:

- `evaluations/20260916-baseline3448-angle45/recovery_aligned_model3448.pt_recovery_metrics.json`
- `evaluations/20260916-baseline3448-settled90/recovery_aligned_model3448.pt_recovery_metrics.json`
- Single-trial pre-settling smoke test is separately retained in `evaluations/20260916-baseline3448-settled-smoke/`; do not add it to the batch denominator.

The 2 s preparation uses **nominal-joint-position PD with zero policy action, NOT zero-torque passive settling**. A start is eligible only with supported quietness for >=0.25 s, an actually fallen classification, and no already-standing classification. Two side trials were not eligible, so true settled-side recovery is 0/18, not 0/20. Reports retain release and handoff root/joint/contact state.

## Current validation tools and pending checks

- `evaluate_recovery.ps1 -SettleSeconds 2` records actual fallen status and does no automatic reset or policy action during preparation. `-SettleSeconds 0` preserves the historical drop protocol.
- `evaluate_go2_stand_walk_stop.py` now reports `stand_walk_stop_stance_geometry_v2`: every settled stand/stop sample must have valid foot/knee/joint geometry and no current base contact; >95% of samples must have four simultaneous vertical foot forces >5 N. Walking is not constrained to four contacts or static geometry.
- Pure math tests, seven stance-screen regression tests, and two command-protocol tests passed on 2026-09-16. Real simulation recovery smoke and 20-trial batch completed.
- Locomotion2250 enhanced-geometry simulation regression **passed** (seed20260909): all13 acceptance gates true; settled stand/stop valid geometry, four vertical contacts and clear base each100% of samples; average body heights0.306/0.308m; stand/stop speed P95 0.0053/0.0107m/s. Walking is separately evaluated and not required to meet static geometry. Report: `evaluations/20260916-locomotion2250-geometry/natural_stop_diagonal_model2250_stand_walk_stop.json`. Log: `E:\IsaacLab\artifacts\recovery-20260916\locomotion2250-geometry.log`. This is one additional seed screen, not broad robustness evidence.
- Intermediate checkpoint3900 controlled-drop45 evaluation **completed** with task Aligned Play, seed20260916,20 trials per pose: upright20/20, side16/20, fore/aft20/20, full inversion0/20; final valid stands match. Fore/aft improves from the parent's14/20, side is unchanged, inversion still fails. Report: `evaluations/20260916-rehearsal3900-angle45/model_3900.pt_recovery_metrics.json`. This limited result does not pass the overall task.
- Checkpoint3900 pre-settled side90/full-inversion batch is running: output `evaluations/20260916-rehearsal3900-settled90/`, log `E:\IsaacLab\artifacts\recovery-20260916\rehearsal3900-settled90.log`, session42524. Read it before duplicating evaluation.
- No new rehearsal checkpoint has passed the complete regression suite or visual verification. Existing recommended locomotion2250 and limited-domain recovery3448 are unchanged. Never overwrite them with an incompletely tested candidate.
- Code audit found no blocking training or preparation bug. Twelve mocked old/new reset combinations preserved default outputs and Torch RNG states exactly. New launch guards reject `recovery_rehearsal` combined with `-FallAngleDeg` or `-FocusSide`, which otherwise silently override/are ignored by the new curriculum. The current run uses neither and explicitly supplies `-CurriculumSteps 4800`.

## Next actions

1. Inspect training process/log and checkpoint timestamps; avoid duplicate jobs. Prefer one simulator at a time after this finite run, because concurrent evaluation slows this 8 GB GPU / 16 GB RAM machine.
2. Locomotion2250 enhanced-geometry regression passed; keep it as an independent reference, not evidence that the recovery policy can walk.
3. Evaluate saved rehearsal checkpoints (e.g.4000 and final4447) with the same parent evaluation task and seed. Check upright/30-degree regression, 45/60-degree starts, then pre-settled side90/full-inversion. Record actual eligible-fallen denominators separately. Add held-out seeds before promotion.
4. If actual settled recovery remains absent, do not simply prolong high-drop training. Implement a separate fallen-state-bank task: generate physical settled states offline, store local root pose/velocity and full joint positions/velocities, then replay validated states without stepping physics inside a subset reset.
5. Important reset-order hazard: the inherited `reset_robot_joints` event follows `reset_base`. A new bank task must own the combined joint/root reset and disable the later joint randomization; otherwise it destroys the collision-consistent fallen state.
   - Store root-link pose relative to source env origin, root-CoM world linear/angular velocities, full joint positions/velocities and exact joint-name order. Load once; add the target env origin on replay. Only XY translation/world-yaw augmentation initially; rotate velocities too. No fresh joint/roll/pitch noise or arbitrary raising of the saved root.
   - Generate with self-collision and require >=0.5s quiet fallen support; reject already-standing or moving states. Replay from cleared sensor/actuator history under the same controller for0.5–1s before accepting a state. This is collision-consistent physical screening, not a mesh-penetration proof.
   - Inherited startup base-mass randomization changes dynamics. First validate collection/replay with matched fixed parameters, then separately validate domain randomization. Split train/held-out original state IDs before yaw augmentation.
   - Test non-contiguous subset reset IDs and assert untouched environments, global simulation time and common-step counters do not change. Keep the original no-bank path and RNG behavior unchanged.
6. Visually inspect front and oblique candidate videos before reporting natural posture. New video artifacts must not be the old September10 videos relabeled as current results.
7. Update this document with actual checkpoints, measured outcomes, and exact next run. Archive configs and provenance before selecting a model.

## Continued follow-up

An app thread heartbeat named **Go2 训练与严格验收**, automation ID `go2`, is active every20minutes. It should remain quiet for unchanged/non-actionable progress, continue evidence-based local training/evaluation, and notify only meaningful changes, completion, failure or required decisions. It must not duplicate an active trainer. Local scheduling needs the computer and app available; it is not a cloud training service.

If sandbox execution fails with `helper_sandbox_lock_failed` / `SetNamedSecurityInfoW ... 5`, request a scoped elevated execution for the E-drive task; do not change machine security settings. For file edits the approved `apply_patch` entrypoint has been `C:\Users\ojo\AppData\Local\OpenAI\Codex\bin\12219cbfbcbddde7\codex.exe --codex-run-as-apply-patch`. Preserve existing user edits and avoid broad staging or deletion.

## Research context

- [Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning (2019)](https://arxiv.org/abs/1901.07517).
- [Learning to Recover for Quadrupedal Wheel-Legged Robots (2025)](https://arxiv.org/html/2506.05516v1) motivates physically settled fallen initial states and checking final posture separately from exploration. Its wheeled platform and reported results are not reproduced here and cannot be claimed for this Go2.
