# Smith-inspired reward-only Go2 pilot

Status19:04local: formal128-environment/2000-iteration training is ACTIVE after successful training and evaluator smokes. Run `E:/IsaacLab/repo/logs/rsl_rl/unitree_go2_recovery/2026-09-16_19-02-40_20260916-smithnominal128x2000`, orchestrator26126, PID119728. Actual saved formal YAMLs also pass the reward-only comparison. This is a method-level reward adaptation, not full paper reproduction and not a successful recovery claim.

## Exact change

The completed NominalTarget2000 control failed all9 strict pose groups (0/20 each). The new `Isaac-Recovery-Bank-SmithNominal-Flat-Unitree-Go2[-Play]-v0` task inherits that control's physics, observations, sample-and-hold nominal action targets, initial-state bank,60% bank mixture and PPO. It starts from the SAME fresh seed42 bootstrap, not either failed trained actor.

Following the [official ResetTask equations at pinned commit583f1de](https://github.com/lauramsmith/fine-tuning-locomotion/blob/583f1de43e91cdd24d632d783872528eb1337480/motion_imitation/envs/env_wrappers/reset_task.py), the independent tensor implementation computes:

- roll: `((1 + cos_up) / 2)^2`;
- stand gate: `cos_up > cos(0.2*pi)` (strictly under36degrees tilt);
- gated stand: `.2*clamp(height/.32,0,1) + .6*exp(-.6*sum((joint_weights*joint_delta)^2)) + .2*exp(-.02*sum(joint_velocity^2))`;
- weighted training shaping: `10*roll + 10*gated_stand`. Two separate logged terms equal20times the source's equally weighted combination.

Go2 target height0.32m and its existing default joint positions are used; hip/thigh/calf weights1/.75/.5 are mapped by actual joint names, not assumed Isaac joint order. No A1 height, gains, torque limits or weights are loaded.

All historical recovery task shaping is removed ONLY in this isolated task: upright/height, orientation progress, low-height lift/support, crossed-limb penalty and strict stand bonus. Existing generic vertical/angular velocity, torque, acceleration, action-rate and soft-joint-limit regularizers remain unchanged. The source ResetTask does not add these regularizers; this is an intentional remaining difference, not an exact reproduction. Near inversion the roll reward is very flat, so those penalties may still suppress the first rolling motion. If this experiment fails, examine logged roll/stand and regularizer terms before a separate regularizer-curriculum ablation.

## Unchanged acceptance and limitations

Reward values do NOT certify normal stance: the dense formula has no foot-contact or crossing test, and its height term saturates. External acceptance remains four simultaneous current vertical foot forces>5N, no base contact, body height0.30–0.55m, proper leg geometry, low body speeds and at least3continuous seconds of valid standing including a valid finish. Side/back bank starts are rechecked as actually settled-fallen after1second nominal-position PD; this is NOT passive zero-torque settling. Controlled drops are reported separately. The same development heldout set is reused, not a new independent final benchmark.

No new integrated walk/fall/recover/walk controller exists. Locomotion2250 and limited-domain recovery3448 remain unchanged and cannot be replaced without numerical and front/oblique visual checks.

## Reproducibility and launch

Bootstrap: `E:/IsaacLab/repo/logs/rsl_rl/unitree_go2_recovery/bootstrap_fresh42_target_v1/model_0.pt`, SHA256 `daa28dd10ef200121bdd6f2a14a03fcd856859ebb5f3ef510d5d946a10f11cb9`.

Bank: `datasets/recovery_states/nominal_pd_v1_20260916/states.npz`, SHA256 `71b882e92035b4c248e5980339c980dbb242dcb1ec711c05252c33249db48f5a`.

Smoke training run: `E:/IsaacLab/repo/logs/rsl_rl/unitree_go2_recovery/2026-09-16_18-53-36_20260916_smithnominal_smoke16x2`, final`model_1.pt`, finite parameters. Two iterations are an interface test, not performance evidence. Saved env/agent YAMLs pass the reward-only config comparison with the completed nominal control. Initial evaluator wrapper preflight refused the new task before launching a simulator; its allowlist was extended without changing acceptance, and the real evaluation smoke is separate.

Installation backup: `E:/IsaacLab/artifacts/install-backup-20260916-185317`. CPU reward math11tests, adapter8tests, full-schema config guard19tests and action control9tests pass, along with historical posture math regressions. Real evaluator smokes completed: heldout side/back0/2each, controlled30degree upright/side/fore-aft0/2each; every bank start was eligible. This is interface validation, not a claim that a2-iteration model recovers.

Formal finite command ALREADY RUNNING (do not execute again):

```powershell
./scripts/run_recovery_smith_pilot.ps1 -RunTag 20260916-smithnominal128x2000 -Iterations 2000 -NumEnvs 128
```

This runs6,144,000 environment steps, then sequential20-trial heldout side/back, controlled30degree and45degree suites under the new matching action task. No automatic promotion or extra training. It rejects duplicate outputs, records and rechecks source hashes, checks saved configs, and rejects invalid physics logs. Keep a single simulator running and every output on E:. See the [live status](training-status-20260916.md) before resuming; do not rerun the whole launcher over an existing run.
