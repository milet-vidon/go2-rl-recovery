# Go2 training status — 2026-09-16

Status: **in progress; user expectations are NOT yet met**. Simulation only. No claim of full reference-video reproduction or real-robot safety. All new code, models, reports and runtime data stay on E:.

## CURRENT JOB — memory-reduced matched pair and source audit

Updated2026-09-16 17:31 local. **The NominalTarget arm and its three evaluations are complete; the CurrentTarget arm is active.** Do not restart the old512-environment launcher or launch a second simulator.

- Active orchestrator session **51383**; log `E:/IsaacLab/artifacts/recovery-20260916/target128x2000-orchestrator.log`.
- Completed first arm: `E:/IsaacLab/repo/logs/rsl_rl/unitree_go2_recovery/2026-09-16_16-14-17_20260916-target128x2000_nominal`, final `model_1999.pt`, SHA256 `58f3231a1959a3d5fa82142e9dcfb2aee2b7cc86f1575e3cee25e36c9efc6ca2`. Training finished16:54; all three evaluations finished17:00. Heldout side/back0/20 each; all upright/side/fore-aft30degree and upright/side/fore-aft/back45degree tests also0/20, with zero valid final stands. This fresh actor FAILED and must not replace recommended models.
- Active second arm: `E:/IsaacLab/repo/logs/rsl_rl/unitree_go2_recovery/2026-09-16_17-00-18_20260916-target128x2000_current`, log `E:/IsaacLab/artifacts/recovery-20260916/20260916-target128x2000-current-train.log`; PID41600 at17:13,1,775,616 environment steps. Both arms start from identical fresh seed42 weights. The21 recorded source hashes still matched at17:13; the actual CurrentTarget YAML comparison remains pending.
- USER DELIVERY PRIORITY: after this pair and its evaluations finish, record and deliver the latest full-scenario results BEFORE another long training branch, including failures. There is currently NO integrated walk/fall/recover/walk switching controller. Clearly label any compilation as separate-policy scenario tests: retained locomotion2250 stand/walk/stop and limited lateral delta-v disturbances, plus BOTH new recovery candidates' full side/back diagnostics. Do not present the old locomotion model as the newly trained recovery actor, conceal resets, or imply that separate seeded views are simultaneous cameras. Inspect front/oblique keyframes. Recovery videos begin AFTER nominal-PD handover; do not claim they show that settling interval.
- Video preparation now implemented (NOT yet simulator-rendered): stand/walk/stop supports `-View front|oblique`, yaw-following articulated bounds, warmup renders, explicit view labels. Its wrapper calls the portfolio evaluator directly, so no task installation is required; the default legacy view remains available. Camera4/4, command2/2 and rest-stance7/7 CPU tests passed. `pair_recovery_views.py` now checks bank/source ordering, action/handover metadata, start eligibility and final outcomes, while preserving matching historical non-bank reports; initial8/8 CPU tests and the actual40-trial NominalTarget report schema check passed. Installed training files were NOT changed. The actual nominal/current YAMLs matched after normalizing only reference/run_name/log_dir.
- READY-TO-RECORD COMMAND once BOTH arms and all six evaluations finish and no simulator remains: `./scripts/record_action_pair_review.ps1 -RunTag 20260916-target128x2000 -Iterations 2000`. Use `-PreflightOnly` to check without recording. It verifies reports/checkpoint hashes/physics logs/source snapshot, refuses existing output, then sequentially records baseline2250 normal/push in both views and BOTH latest candidates' heldout side/back plus upright views. Output `evaluations/20260916-target128x2000-video-review/`. This is a finite recorder, not an automatic policy promotion or a continuous switching controller. If pairing fails, keep both original reports/videos and inspect; do not conceal divergent replays. On partial recorder failure, inspect the exact stage and resume missing commands manually rather than overwriting the full output. Contact sheets and actual visual inspection remain REQUIRED before user delivery.
- Intermittent approval-service capacity errors blocked previous access and the attempted heartbeat prompt update; the existing heartbeat remains active and unchanged. No new demonstration video has been rendered yet. Do not wait for a future successful model indefinitely before showing this pair's actual outcomes.
-17:31 update: CurrentTarget has4,356,096 environment steps (~1418/2000 iterations) and has not yet produced final reports. NominalTarget bank denominators were rechecked:20/20 eligible side starts (16 unique),20/20 eligible back starts (14 unique),zero true recoveries and zero final stands. Final pair-check tests9/9 passed, including reordered multi-trial IDs and cross-level bank consistency; the actual40-trial report remains compatible. Recorder AST validation and `-PreflightOnly` passed: it reported3 missing CurrentTarget reports and active PID41600, started no simulator and created no output. A PowerShell interpolation parse error was fixed before any recording execution. Visual runtime checks remain pending.
- Each arm:128 environments ×24 rollout steps ×2000 iterations =6,144,000 planned environment steps, expected final `model_1999.pt`. This is a NEW batch-size-matched experiment, not an exact continuation/completion of the interrupted512-environment pair. Old model500 is retained only as a separately measured partial run.
- Launch: `scripts/run_recovery_action_ablation.ps1 -RunTag 20260916-target128x2000 -Iterations 2000 -NumEnvs 128`. Runtime configuration, physics buffers, joint/actuator limits, self-collisions and strict success criteria are unchanged apart from the SAME reduced environment count in both arms. Early total GPU use around4.73/8.19GB; this snapshot does not guarantee future memory availability.
- Launcher now rejects logs containing PhysX/CUDA failures, tracebacks, buffer overflows or discarded-contact messages, even if the batch wrapper returns zero. The guard passed the clean smoke log and correctly rejected the recorded native failure. Do not accept a result that needed dropped contacts or weakened dynamics.
- Outputs: `configs/20260916-target128x2000-{nominal,current}/`; `evaluations/20260916-target128x2000-{nominal,current}-{heldout,angle30,angle45}/`. No automatic promotion/extension. Preserve recommended locomotion2250 and limited-domain recovery3448.
- Independent read-only review confirmed matching task inheritance and evaluation arguments. Do NOT run `install.ps1` or change installed rewards/actions/reset/physics during this serial pair: its second arm starts in a new interpreter. Mid-first-arm source snapshot: `configs/20260916-target128x2000-source-snapshot.json`; all recorded source modification times predate this launch. This records21 files, not a runtime lock or complete environment archive. Before accepting the comparison, recheck these hashes and compare the two actual saved YAMLs, allowing only the intended action reference and run identifiers/paths to differ.
- Metadata-only evaluator correction: new target tasks now correctly report DIRECT nominal-position PD during handover, not zero policy action; controlled drops report no handover. Historical model500 JSON is retained unchanged as evidence; its `action_representation` has the correct description but its old `settle_controller` text is misleading. No simulation dynamics or acceptance counts changed. CPU control-target tests9/9, stochastic-diagnostic tests5/5 and video-annotation tests4/4 passed. No overlay was installed during this pair.
- New user direction: if recovery keeps failing, audit paper AND actual source code before further tuning. The read-only audit is COMPLETE: main read Lee2019, Deng2025 and the official Smith reset source; the delegated audit traced its reset/action/training call chain and pinned commits. [Source comparison and bounded next decision](reproduction-audit-20260916.md). Next candidate if this pair stalls is Smith-inspired roll/stand reward separation with all other task variables held fixed, not another unchanged extension. Do not install other simulators/replace this environment or run hardware deployment commands from external READMEs.

## INTERRUPTED JOB — earlier512-environment action-reference pair

**Superseding outcome:** this job stopped2026-09-16 16:06 with PhysX CUDA out-of-memory near iteration584; its last intact checkpoint is500. Orchestrator51917 exited1 because expected999 was missing. CurrentTarget512 never started; old queued plans below are historical and no longer active. Model500 SHA256 `34390bd30f0a2822bb1219f12f7da1a80e71deeba47e785d72ae6f4ed3cfb270`. Saved config archive: `configs/20260916-target1000-nominal-interrupted/`.

Strict partial-model evaluation completed in `evaluations/20260916-target-nominal500-interrupted-heldout/`: side0/20, back0/20, no valid final stands; all starts eligible,16/14 unique sources. Final side mean tilt61.82degrees,height0.14898m; back179.989degrees,height0.057m. This is neither successful recovery nor a completed1000-iteration result. No earlier model is overwritten.

Historical launch record at2026-09-16 15:53 local: a finite512-environment pair was planned, each arm1,000 fresh PPO iterations (12,288,000 environment steps), followed by bank side/back, controlled-drop upright/30-degree and upright/45-degree evaluations. It was interrupted as stated above; it is NOT RUNNING and must not be restarted over existing output.

- Orchestrator exec session **51917**, log `E:/IsaacLab/artifacts/recovery-20260916/target1000-orchestrator-retry.log`.
- Interrupted first arm: `E:/IsaacLab/repo/logs/rsl_rl/unitree_go2_recovery/2026-09-16_15-52-50_20260916-target1000_nominal`; log `E:/IsaacLab/artifacts/recovery-20260916/20260916-target1000-nominal-train.log`. Expected final `model_999.pt` was NOT produced; model500 was the last intact save.
- Second arm `20260916-target1000_current` never started. Its old queue was cancelled when the orchestrator exited1; do not launch it separately as if still part of a running pair.
- Historical launch: `scripts/run_recovery_action_ablation.ps1 -RunTag 20260916-target1000 -Iterations 1000`. Exact bootstrap/bank hashes were checked. Initial launcher parse failure occurred before training and was fixed; the retry log above records the subsequent OOM interruption, not an active job.
- Outputs will be `configs/20260916-target1000-{nominal,current}/` and `evaluations/20260916-target1000-{nominal,current}-{heldout,angle30,angle45}/`. No automatic promotion or further training is performed by this finite script.
- New tasks `Isaac-Recovery-Bank-NominalTarget-Flat-Unitree-Go2[-Play]-v0` and `...CurrentTarget...` MUST be used with their respective models. Both clamp targets to soft joint limits and hold them for four physics substeps. Only reference differs: default q versus control-boundary measured q. All old tasks/recommended weights remain unchanged.
- Smoke tests: each task16 environments/two training iterations, followed by2-trial side/back evaluation. Handover always direct nominal-position PD for1 second. Same source IDs and eligibility; tiny contact-solver differences in one back start, documented in the experiment note. Neither fresh untrained bootstrap passed recovery, as expected.
- Previous BackExplore100 stochastic diagnostic completed too:0/20back,14 unique states, physical joint spans0.587–1.447rad but minimum tilt still167.20degrees. It is explicitly NON-ACCEPTANCE and is not a demo or promoted policy.

[Design, source paper, test coverage and limitations](action-reference-ablation-20260916.md). After completion, compare BOTH arms under these matching fixed-physics action tasks. Repeated heldout bank is development validation. Any promising candidate still needs fresh unseen validation and actually viewed front/oblique videos. Preserve3448 and locomotion2250. On failure inspect the exact stage/log; do not rerun the entire launcher over existing outputs. Heartbeat `go2` remains active.

## PREVIOUS FOLLOW-UP — back-conditioned exploration (completed)

Historical14:54 update: GateControl100, GateRelease100, BackExplore100 and all deterministic evaluations finished without grounded side/back recovery. BackExplore100:30degree20/12/19;45degree20/0/7/0; back179.982degrees/0.057m/zero feet. Old4899 packed-trace regression is bitwise equal (results and CSV SHA256). The later stochastic diagnostic is now completed and the action-reference pair above supersedes this branch. Do not restart the completed100-round branches. [Details](back-exploration-20260916.md) · [completed gate ablation](recovery-gate-ablation-20260916.md).

- Completed4899 evaluation: heldout side0/20,back0/20;16 and14 unique states respectively, exactly the same bank/hash/IDs/order as3900. All starts eligible, no valid final stands.
- Aligned30degree seed20260918: upright20/20,side10/20,fore/aft18/20. Aligned45degree seed20260916: upright20/20,side0/20,fore/aft10/20,inversion0/20. Do not overwrite recommended3448 or locomotion2250.
- Side stops at61.26–71.00degrees(mean65.72),height0.1503m,one supporting foot. Back remains180degrees,height0.057m,zero feet. Raw aggregate reward concealed these failures.
- Hypothesis, NOT established cause: old posture-penalty gate begins at60degrees, immediately requiring narrow/symmetric/nominal stance while still rolling. All20 side finals sit just outside this gate. A conservative partial raw-penalty estimate is1.61; at50degrees this can add at least2.30 weighted penalty versus about0.3 orientation/height shaping gain if leg geometry has not changed. Back has gate0 already, so this change is not expected to directly solve inversion.
- Planned paired control/release: same4899 checkpoint and optimizer/std, same seed42, fixed60% bank mix (do not restart the20→60% curriculum), same214 train states, physics and PPO. Control retains old penalty. Release only delays it using continuous cos(up)0.85→0.95 and height0.24→0.28 ramps, with >=2 CURRENT vertical-foot forces>5N. All stand-success rewards and final geometry/four-contact/height/quietness criteria remain unchanged.
- CompletedControl andRelease100:side mean finaltilt66.37/66.11degrees,stilllowground-supported.30degree resultsControl20/10/17 andRelease20/11/14(upright/side/fore);45degree20/0/5/0 and20/0/6/0(upright/side/fore/back). Final validstands equal successes. No model promoted;no extension ofthisbranch. Next sampling experiment resets neithermean norAdam anddoesnot reuse the Release policy.

## COMPLETED JOB — grounded-state-bank pilot (historical launch record)

Started2026-09-16 13:16 local; completed13:40:11,1,000 iterations,12,288,000 environment steps, final4899. **Do not start a duplicate.** Details below retain the historical launch plan, superseded by measured outcomes above.

- Run: `E:\IsaacLab\repo\logs\rsl_rl\unitree_go2_recovery\2026-09-16_13-16-04_bank_nominalpd_from3900_20260916`
- Log: `E:\IsaacLab\artifacts\recovery-20260916\train-bank-pilot.log`; current-turn session71234.
- Stage `recovery_bank`,512 environments,1,000 additional PPO iterations completed, final `model_4899.pt`,12,288,000 additional environment steps. No training runtime error, but the checkpoint FAILED acceptance as recorded above.
- Loaded `bootstrap_bank3900_std060/model_3900.pt`; deterministic actor/critic equal original3900, sampling std0.6 and fresh Adam moments for this exploration pilot. Original parent and accepted models are untouched.
- Actual saved configs and bootstrap lineage: `configs/recovery-bank-pilot-20260916/`.
- Bank: `datasets/recovery_states/nominal_pd_v1_20260916/`,SHA256 `71b882e92035b4c248e5980339c980dbb242dcb1ec711c05252c33249db48f5a`;214 training states,30 heldout states. Nominal-PD-prepared and cold-replay-validated; not passive zero-torque falls.
- Before launch:19 data/math tests,10 mocked subset-reset tests,37 real-simulator subset-reset checks passed. Real reset changed only env IDs1,7,11; PhysX and cached joints match the recorded train states; global simulation clock unchanged. Report `evaluations/20260916-bank-live/subset-reset.json`.
- Parent3900 heldout baseline: side0/20 trials from16 unique states; back0/20 from14 unique states. Every trial was confirmed settled-fallen. These40 rollouts cover only30 independent source states. Report `evaluations/20260916-bank3900-heldout/model_3900.pt_recovery_metrics.json`.
- Visually inspected front and oblique side/back diagnostics: real ground-supported initial bodies, old policy remains/ends back-down, no recovery. Framed videos are in `evaluations/20260916-bank3900-framed-front/` and `...-oblique/`; these are FAILURE evidence, not result demos. First frame still uses the wide initial render view; later tracking keeps all feet in frame. For final deliverables, initialize the offscreen render camera before setting the first tracked view and verify it again.

Completed follow-up:4899 was evaluated against3900 with the same Bank Play task,hash,heldout split,IDs/order,seed20260918,20 trials and1s nominal-PD handover; Aligned upright/30/45degree regressions also completed. Reports are in `evaluations/20260916-bank4899-{heldout,angle30,angle45}/`. The limited-domain and locomotion baselines are preserved. Even a future successful bank test will not establish arbitrary-fall robustness or an integrated standing/walking/stopping/recovery controller.

## Acceptance priorities

1. Natural standing, walking and stopping, with no crossed limbs or ground-supported torso.
2. Recovery must finish in anatomically valid, quiet four-foot stance and retain it; height/contact counts alone are insufficient.
3. Distinguish airborne righting from getting up after an actual settled fall. Do not count an already-standing start as fallen recovery.
4. Compare checkpoints under identical tasks, seeds and protocols. New candidate promotion requires upright/30-degree regression, harder poses, and visual inspection of both front and oblique views. Do not relax criteria to make a run pass.

## Completed controlled-drop rehearsal run

- Run directory: `E:\IsaacLab\repo\logs\rsl_rl\unitree_go2_recovery\2026-09-16_11-46-18_rehearsal60_from3448_20260916`
- Log: `E:\IsaacLab\artifacts\recovery-20260916\train-rehearsal.log`
- Current-turn exec session: `38810` (may not survive a later app session; inspect process/log/checkpoints instead).
- Parent: `models/recovery/recovery_aligned_model3448.pt`, SHA256 `1f546523baa57c7997ad2883689667d6e738eac098f14fb5fa595aae778a2de2`.
- Parent load run: `2026-09-10_17-15-48_aligned_from_uncrossed2849`, checkpoint `model_3448.pt`.
- Task/stage: `Isaac-Recovery-Rehearsal-Flat-Unitree-Go2-v0` / `recovery_rehearsal`.
- **Completed** at2026-09-16 12:19 local:512 environments,24 rollout steps,1,000 additional iterations,12,288,000 additional environment steps. Log ends atiteration4447/4448 with32min16s training time; `model_4447.pt` saved. No training process remained at12:39. Do not restart this completed run.
- Final serial evaluation session10915 completed:30-degree seed20260918;45-degree and settled90 seed20260916; parent/final60-degree comparison. See the active grounded-bank pilot above for the current trainer.
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
- Checkpoint3900 pre-settled side90/full-inversion batch **completed its report**: side0/18 eligible fallen (0/20 overall), full inversion0/20; no valid final stands. Identical eligible denominators and no improvement over3448. Report: `evaluations/20260916-rehearsal3900-settled90/model_3900.pt_recovery_metrics.json`; log `E:\IsaacLab\artifacts\recovery-20260916\rehearsal3900-settled90.log`. Process42524 was in final shutdown after writing the complete report; confirm it exits before starting another simulator.
- No new rehearsal checkpoint has passed the complete regression suite or visual verification. Existing recommended locomotion2250 and limited-domain recovery3448 are unchanged. Never overwrite them with an incompletely tested candidate.
- Code audit found no blocking training or preparation bug. Twelve mocked old/new reset combinations preserved default outputs and Torch RNG states exactly. New launch guards reject `recovery_rehearsal` combined with `-FallAngleDeg` or `-FocusSide`, which otherwise silently override/are ignored by the new curriculum. The current run uses neither and explicitly supplies `-CurriculumSteps 4800`.

## Next actions

1. Inspect training process/log and checkpoint timestamps; avoid duplicate jobs. Prefer one simulator at a time after this finite run, because concurrent evaluation slows this 8 GB GPU / 16 GB RAM machine.
2. Locomotion2250 enhanced-geometry regression passed; keep it as an independent reference, not evidence that the recovery policy can walk.
3. Final4447 has completed30-degree regression: upright/side/fore_aft each20/20;45-degree upright20/20,side15/20,fore_aft18/20,inversion0/20. Its pre-settled side/inversion remains0/18 and0/20. Final45-degree scores are below intermediate3900 (16/20 side,20/20 fore/aft), so do not select the last checkpoint solely because training lasted longer. Complete the60-degree comparison and keep all original reports.
4. If actual settled recovery remains absent, do not simply prolong high-drop training. Implement a separate fallen-state-bank task: generate physical settled states offline, store local root pose/velocity and full joint positions/velocities, then replay validated states without stepping physics inside a subset reset.
5. Important reset-order hazard: the inherited `reset_robot_joints` event follows `reset_base`. A new bank task must own the combined joint/root reset and disable the later joint randomization; otherwise it destroys the collision-consistent fallen state.
   - Store root-link pose relative to source env origin, root-CoM world linear/angular velocities, full joint positions/velocities and exact joint-name order. Load once; add the target env origin on replay. Only XY translation/world-yaw augmentation initially; rotate velocities too. No fresh joint/roll/pitch noise or arbitrary raising of the saved root.
   - Generate with self-collision and require >=0.5s quiet fallen support; reject already-standing or moving states. Replay from cleared sensor/actuator history under the same controller for0.5–1s before accepting a state. This is collision-consistent physical screening, not a mesh-penetration proof.
   - Inherited startup base-mass randomization changes dynamics. First validate collection/replay with matched fixed parameters, then separately validate domain randomization. Split train/held-out original state IDs before yaw augmentation.
   - Test non-contiguous subset reset IDs and assert untouched environments, global simulation time and common-step counters do not change. Keep the original no-bank path and RNG behavior unchanged.
6. Visually inspect front and oblique candidate videos before reporting natural posture. New video artifacts must not be the old September10 videos relabeled as current results.
7. Update this document with actual checkpoints, measured outcomes, and exact next run. Archive configs and provenance before selecting a model.

## Continued follow-up

### Grounded-state pilot preparation (12:39 follow-up)

- Final rehearsal evaluation finished. At60degrees, parent3448 scored side5/20,fore/aft7/20; final4447 scored side5/20,fore/aft12/20. The larger course improves some fore/aft drops but not actual settled recovery. Do not extend the same drop-only run blindly.
- New isolated Bank task: `Isaac-Recovery-Bank-Flat-Unitree-Go2[-Play]-v0`. Root/joint resets are owned by one `RecoveryBankReset` term; the later joint reset is disabled. Collection/training share fixed mass/CoM and friction0.8/0.6,restitution0 with self-collision; this is a fixed-physics pilot, not domain-randomized robustness.
- Collector first exposed a CPU default-mass / CUDA contact-force mismatch; fixed by moving and caching the weight denominator. Failure log is preserved in `E:\IsaacLab\artifacts\recovery-20260916\bank-smoke.log`. No failed dataset was used for training.
- Small physical smoke:16 candidates,14 accepted,2 rejected because they were not confirmed fallen. Every accepted state passed an additional1s cold replay after sensor/actuator reset.
- Pilot bank: `datasets/recovery_states/nominal_pd_v1_20260916/states.npz` and `manifest.json`;256 candidates,244 accepted,12 rejected.214train and30heldout,split by original generation batch before yaw augmentation. Actual totals:left66,right64,back114. Collection + cold replay completed; log `E:\IsaacLab\artifacts\recovery-20260916\bank-v1.log`. Counts describe valid START STATES, not model recovery successes.
- New bank data/math CPU tests19/19; mocked subset-reset tests10/10; real16-env subset-reset verification passed37 checks. Parent-heldout evaluation completed at0 recoveries; the grounded pilot was then launched as recorded in ACTIVE JOB above.
- Prepared experimental bootstrap `E:\IsaacLab\repo\logs\rsl_rl\unitree_go2_recovery\bootstrap_bank3900_std060\model_3900.pt`: original3900 deterministic actor/critic unchanged; sampling std set0.6,Adam moments cleared. Original hip std was0.087–0.096,too narrow for a useful new exploration trial. This is a hypothesis being tested,not a performance claim. SHA256 `ca445aa39bdca62fe18e04924e967feda2c2af496f7263238a4c2bf22947a3fa`; complete changes/parent hash in adjoining `model_3900.lineage.json`.
- Planned bank mixture20%→60% over12000control steps; other resets retain upright/shallow20–30degree rehearsal. Bank root/joint states are not lifted or perturbed; only world-yaw and target-origin transforms. Do not call old `-FallAngleDeg/-FocusSide/-CurriculumSteps` with this stage.
- Bank evaluation uses `-StateBankPath ... -StateBankSplit heldout -SettleSeconds 1`,Bank Play task,and actual side/back classes. Unique heldout IDs are reported; repeats needed to fill a20trial batch are not additional independent source states. Parent/candidate must use the same IDs,bank hash and fixed physics.

An app thread heartbeat named **Go2 训练与严格验收**, automation ID `go2`, is active every20minutes. It should remain quiet for unchanged/non-actionable progress, continue evidence-based local training/evaluation, and notify only meaningful changes, completion, failure or required decisions. It must not duplicate an active trainer. Local scheduling needs the computer and app available; it is not a cloud training service.

If sandbox execution fails with `helper_sandbox_lock_failed` / `SetNamedSecurityInfoW ... 5`, request a scoped elevated execution for the E-drive task; do not change machine security settings. For file edits the approved `apply_patch` entrypoint has been `C:\Users\ojo\AppData\Local\OpenAI\Codex\bin\12219cbfbcbddde7\codex.exe --codex-run-as-apply-patch`. Preserve existing user edits and avoid broad staging or deletion.

## Research context

- [Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning (2019)](https://arxiv.org/abs/1901.07517).
- [Learning to Recover: Dynamic Reward Shaping with Wheel-Leg Coordination for Fallen Robots (2025)](https://arxiv.org/html/2506.05516v1) motivates physically settled fallen initial states and checking final posture separately from exploration. Its wheeled platform and reported results are not reproduced here and cannot be claimed for this Go2. The [gate ablation notes](recovery-gate-ablation-20260916.md) distinguish its episode-time formula from our state-based gate and document the2019 paper's alternative current-joint-relative action representation for a future separate test.
