# First bounded speed-expansion pilot

## Measured baseline, not a fast-running claim

The15rollout frozen probe completed00:44:50, with all16preserved model hashes unchanged.2250 passed0.5m/s on3/3seeds and0.8m/s on2/3(one slip failure).3648 passed0.5and0.8 on3/3each. At1.0 it failed3/3: seed20260909 actualvx0.843118m/s exceeded tracking-error0.12; seeds10/11 actualvx0.888817/0.888869 passed tracking but contact-slip0.131371/0.128045 exceeded0.12. The latter three all retained normal stand/stop geometry and current four-foot support100%, and passed the separate straightness checks.1.5/2.0were deliberately NOT tested after the failed prerequisite. These are simulation diagnostics, not video/hardware acceptance.

Actual baseline reports and immutable report hashes: evaluations/20260917-speed-baselines/summary.json. The goal is high-speed tracking/slip improvement WITHOUT disrupting standing or stopping; no evidence justifies changing the standing reward here.

## Primary-source rationale and limited adaptation

[legged_gym's original command curriculum](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py) expands command bounds conditional on tracking performance. [Rapid Locomotion](https://arxiv.org/abs/2205.02824) and its [author repository](https://github.com/Improbable-AI/rapid-locomotion-rl) motivate velocity-command curricula for agile locomotion. Our experiment is only one manually bounded Go2 command-range stage, NOT reproduction of their complete adaptive curriculum, adaptation module, dynamics, simulator, robot or reported speeds. Inference: a small exposure expansion may improve the edge-of-range skill; it remains falsifiable.1.0was already in the old training range, not an out-of-distribution command.

## Declared experiment

- Frozen parent: original run2026-09-09_15-50-37_natural_robust_push_20260909/model_3648.pt, SHA475f07bf0b05101dd4157212b4e3fd48a04731d452d7a349fe467c508dbaaac8.
- Same Natural-Robust-Push task. Only intended behavioral intervention is uniform commandvx upper bound1.0->1.2m/s, lower bound-.5 unchanged. This is NOT the more complex proposed high-speed/old-distribution mixture sampler.
- Preserve30%zero-command standing, old lateral/yaw ranges,4–8s command switches, push events, rewards (including static3,orientation-7,slip-.5,tracking1.5), observation/action/physics and PPO configuration otherwise.
- Explicitly set runner learning-rate scalar1e-5 to match the parent's actual saved adaptive optimizer LR. Historical initial config3e-4 differs from saved state; silently resetting that scalar could confound retention. Full model/optimizer/std are restored. Iteration label3648is repeated and RNG/simulator reset; not bitwise uninterrupted continuation.
- Smoke16environments x2updates, final3649. Formal128 x300updates =921,600environment steps, expectedfinal3947, parent Adam73120->79120. Reduced from the initially proposed512environments BEFORE formal launch because free RAM during the separate40environment collection was around0.55GB. The environment count is a declared batch/resource difference from the512environment parent; dynamics/rewards are unchanged. Source/config/hash/checkpoint guards fail closed. New files/runs onEonly; no promotion or automatic extension. Keep oldsmoke source snapshot unchanged; launcher/guard later only changed the permitted formal budget.

The smoke completed successfully: full actual YAML comparison passed;68model/optimizer tensors finite; final3649 SHA19185df27b38ec6b8f9beb32920bf124ec3fc581a9b5b853a69f9faaf1e83758. This validates the interface, not capability. Formal training must be explicitly logged after launch.

## Acceptance and next decision

After formal training, evaluate18fresh independent scenarios:3seeds x(normal0.5,retained0.8,target1.0,left/rightyaw±.5,pushdelta-v0.5), all including stand/walk/stop. Keep unchanged v2 geometry/current contacts/height/stop/track/slip criteria, plus separately reported unperturbed straight-line drift checks. Do not require four-foot contact during dynamic gait. Training slip reward's historical contact/sum definition differs from the evaluator's current contact/mean; do not convert reward into the0.12m/s criterion.

Entry scripts: run_speed_curriculum_pilot.ps1 trains only; evaluate_speed_candidate.ps1 is the finite18case regression. Neither promotes or declares full fast running. Frozen turn/push controls must be rechecked under identical current criteria if needed for a no-regression claim. Only if target improves and old functions remain valid proceed to new seeds, actual front/oblique video, then a next bounded speed stage. If worsened, preserve the failure and old parent; do not increase speeds or blindly extend. Recovery is an independent still-unfinished branch.

## Completed result — 2026-09-17 02:20 local; candidate NOT promoted

Formal128environment training completed300updates/921,600steps in E:/IsaacLab/repo/logs/rsl_rl/unitree_go2_flat/2026-09-17_01-22-52_20260917-speedcurriculum128x300. Final3947 SHAaaa3fcdd796bfb99feb95a4940889c8fd4ebeff61282aac0b258ec856b1d7604. Actual full-YAML guard and68finite model/optimizer tensor checks passed; Adamsteps73120->79120. All16frozen exported baselines stayed unchanged. Integrity is NOT behavioral acceptance.

All18serial regressions finished01:49:48. Every report/hash, model identity and summary metric was independently checked against [summary.json](../evaluations/20260917-speed-candidate3947/summary.json):

| Scenario | Passed / 3 seeds | Limitation |
| --- | --- | --- |
| Normal0.5m/s | 3/3 | Extra drift checks pass, but normal drift/yaw/slip worsened versus matched3648. |
| Retained0.8m/s | 3/3 | Actualvx0.725789/0.785438/0.773093m/s; not2m/s running. |
| Target1.0m/s | 0/3 | seed09tracking failure;seed10/11slip failures. |
| Leftyaw+0.5rad/s | 2/3 | seed09yaw0.335796690;error0.164203310>0.15. |
| Rightyaw-0.5rad/s | 3/3 | These finite scenarios only. |
| Lateral delta-v0.5m/s | 3/3 | Limited disturbance screen, not fallen-state recovery. |

At1.0m/s,actualvx0.868040/0.903671/0.902390 is only0.0135–0.0249m/s faster than3648 with the same task,seed,command. Seed09error0.131960fails the unchanged0.12tracking threshold;seed10/11slip0.128212/0.140854fails the unchanged0.12slip threshold. Target remains0/3;1.5/2.0were NOT tested. Do not round failures into passes, weaken criteria or present speed-p95 as sustained tracked speed.

Left-turn regression is CONFIRMED by a NEW [matched3648/left/seed20260909/current-v2 control](../evaluations/20260917-speed3648-left09-control/natural_robust_push_model3648_stand_walk_stop.json): all criteria passed withyaw0.415083711, versus3947's0.335796690failure. This is no longer merely a legacy-protocol warning. Preserving old model bytes allows rollback, but does not make3947functionally non-regressing. Do NOT replace2250/3648 or advance3947to a higher-speed stage.

Rest posture remains normal in the numeric screen: all18scenarios have100%settled stand/stop geometry; no-push current four-foot support100%,push minimum98.4%(unchanged>95%criterion). Minimum rest height0.30644m; maximum rest speed-p950.00976without pushes and0.07505with pushes. All9unperturbed straight-line drift gates pass. These retained abilities do not excuse left-turn and1.0failures or the worse0.5drift/yaw/slip versus matched3648.

Actual18-second diagnostics: front (local-only artifact, not included in this GitHub snapshot: `evaluations/20260917-speed3947-video-target10-front/model_3947_stand_walk_stop.mp4`) and oblique (local-only artifact, not included in this GitHub snapshot: `evaluations/20260917-speed3947-video-target10-oblique/model_3947_stand_walk_stop.mp4`). Main agent inspected3,5.12,8.12,16second frames in both: initial/restored-stop stance is not prone and front legs are not crossed in inspected frames. These independent camera replays remain a FAILED target-speed demonstration, not simultaneous views or continuous walk/fall/recover/walk. Sampled visual checks do not certify every frame or override numeric failures.

Decision: retain failed3947,all reports/videos and frozen baselines; do not blindly extend unchanged or move to1.5/2.0. A future speed intervention must address tracking/slip and turn retention. Separate RAW79-state CPU history checks do not make the recovery dataset training-ready: real simulator input/state replay validation remains pending.
