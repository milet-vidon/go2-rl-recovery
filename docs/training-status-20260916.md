# Go2 training status — 2026-09-16

Status: **in progress; user expectations are NOT yet met**. Simulation only. No claim of full reference-video reproduction or real-robot safety. All new code, models, reports and runtime data stay on E:.

## CURRENT JOB — stand-weight paired continuation is ACTIVE

Updated2026-09-16 20:18local. User asked to keep training. **One simulator only. Do not duplicate, install/change runtime sources, or restart completed stages.**

- Active finite orchestrator session39467; armA(w10) Python PID33584, launched20:16:24. Run `E:/IsaacLab/repo/logs/rsl_rl/unitree_go2_recovery/2026-09-16_20-16-37_20260916-smithstandpair128x1000_w10`. Outer log `E:/IsaacLab/artifacts/recovery-20260916/20260916-smithstandpair128x1000-orchestrator.log`; arm log `20260916-smithstandpair128x1000_w10-train.log` in the same artifact directory.
- `scripts/run_smith_standweight_pair.ps1` runs w10 training+three20-trial suites, THEN w30 training+the same suites. Both128env×1000extra iterations load the SAME complete Smith1999 parent (including optimizer), not the other arm. Expected finals `model_2998.pt`; each arm adds3,072,000environment steps. Second arm has NOT started yet. No auto-promotion/extension. Do not precreate output/config directories or rerun the whole launcher if a later stage fails.
- The ONLY experiment variable is gated `smith_stand.weight`10versus30; roll10, Go2 physics, nominal+.25 targets, bank, initial mixture, PPO and seed42 remain fixed. No installed task code changed. Both actual16env×2iteration smokes completed with `model_2000.pt`; full-schema config guards passed and each checkpoint's17model/51optimizer tensors were finite. Wrapper67,config15,report16tests passed; identity-block12tests rejected seed/hash/trial mutations. [Design and falsifiable decision](smith-standweight-pair-20260916.md).
- Parent completed2000rounds/6,144,000steps. Final1999 SHA `71e4c7a39ff81836d2ead1c4533988c696e5bd4a0fc939b8cef6b4f6daec1b1c`. All9formal20-trial groups are0ever-valid/0final-valid/0geometry. However side/back final mean height0.147/0.156m and tilt18.66/8.20degrees show body righting followed by low/abnormal support.39/40bank endings have only one supported foot; none have four. This evidence changes the next priority from blindly releasing rolling regularizers to the isolated stand-weight test.
- Parent videos COMPLETE, recorder9644exited successfully: six raw views plus three full paired clips in `evaluations/20260916-smithnominal128x2000-video-review/`. Main ACTUALLY VIEWED all three paired scenes at0/1/3/10.92s, both angles: back and side roll upright then remain folded/low; upright collapses similarly. All nine videos fully decoded; no cropping/failure removal. [Latest videos, hashes and limits](smith-video-review-20260916.md). Do not rerun the completed recorder over this directory. These are not a continuous locomotion/recovery controller.
- Keep locomotion2250 and limited-domain recovery3448 unchanged. After both arms finish: verify actual hashes/configs/protocol/sourceIDs, compare ever-valid versus final-valid especially upright, then record/view the best OR failed diagnostics honestly. More reward is not success, and rewards are not directly comparable across weights. If w30 remains all-zero or abnormal, do not extend unchanged; separately consider validated stand-up initial-state curriculum. No full reference-video reproduction claim.
- Existing heartbeatgo2 continues to read this document; no duplicate automation is needed. Old-video cleanup is pending user clarification; nothing was deleted. All outputs remain on E:.

## COMPLETED JOB — Smith-inspired reward-only training (historical launch notes)

Historical19:04local launch record below. This run and all three evaluations finished20:03; its videos finished20:12. The ACTIVE job is the stand-weight pair above; do not restart this completed run.

- Orchestrator session **26126**, Python PID **119728**, launched19:02:28. Outer log `E:/IsaacLab/artifacts/recovery-20260916/smithnominal128x2000-orchestrator.log`; train log `E:/IsaacLab/artifacts/recovery-20260916/20260916-smithnominal128x2000-train.log`.
- 19:27 continuation: still the SAME single training process, around 1220/2000 iterations and 3,747,840 steps; checkpoint 1000 is finite and all 11 runtime source hashes are unchanged. Interim reward audit warns roll improves while stand decreases and joint-limit penalty increases; this is NOT recovery evidence. Host memory is tight, so no second simulator, heavy parallel analysis or system-setting changes. [Detailed interim evidence](smith-interim-audit-20260916.md). Continue the existing 2000-round job and automatic evaluations; do not restart or modify its sources.
- Active run `E:/IsaacLab/repo/logs/rsl_rl/unitree_go2_recovery/2026-09-16_19-02-40_20260916-smithnominal128x2000`. Expected final `model_1999.pt`; budget128×24×2000=6,144,000environment steps. At19:03:40 roughly52iterations/159,744steps had completed; this is progress, not recovery evidence.
- Launch `scripts/run_recovery_smith_pilot.ps1 -RunTag 20260916-smithnominal128x2000 -Iterations 2000 -NumEnvs 128`. The finite launcher automatically runs strict20-trial heldout side/back, controlled30degree and45degree evaluations AFTER training. All run under `Isaac-Recovery-Bank-SmithNominal-Flat-Unitree-Go2-Play-v0`; configs will be archived in `configs/20260916-smithnominal128x2000/`, reports in `evaluations/20260916-smithnominal128x2000-{heldout,angle30,angle45}/`. Do not create those outputs early or rerun the whole script over existing files. On failure inspect exact stage and resume only missing work.
- Reward-only design: new isolated task inherits NominalTarget physics/actions/bank/PPO. SAME fresh42 bootstrap SHA `daa28dd10ef200121bdd6f2a14a03fcd856859ebb5f3ef510d5d946a10f11cb9`, not continuation of failed1999. All old recovery task shaping removed only in this new task; use Smith-inspired10×roll+10×gated stand withGo2 height0.32and name-resolved joint weights. Generic action/motor/velocity/softlimit regularizers stay identical and may still inhibit initial inverted rolling. Do not claim full Smith reproduction or modify multiple variables to hide failure. [Exact design and source](smith-reward-pilot-20260916.md).
- Main re-read pinned official ResetTask source; independent review found no blocking design issue. Math11, adapter8, config-guard19, prior action9CPU tests passed plus historical posture regressions. Smoke16env×2iterations produced finite `model_1.pt`; roll/stand logs nonzero. Both heldout2-trial and controlled30degree2-trial evaluation smokes completed with valid protocols; all successes0(as expected for near-untrained smoke), all bank starts2/2eligible. This proves interfaces, NOT recovery performance.
- Both smoke and ACTUAL formal saved env/agent YAMLs passed the reward-only whitelist against the completed nominal2000control, allowing only rewards, environment count/budget and run paths/names. Scene/terrain counts checked. Actions, contacts,torque limits,bank/reset,PPO,bootstrapunchanged. Installation backup `E:/IsaacLab/artifacts/install-backup-20260916-185317`; run source hashes recorded in `E:/IsaacLab/artifacts/recovery-20260916/20260916-smithnominal128x2000-source-snapshot.json` and rechecked by launcher at end.
- One wrapper preflight initially rejected the new task before any evaluator simulator started; it was fixed and both real evaluation smokes then passed. No acceptance threshold was changed. Training stdout is now unbuffered for progress monitoring.
- AFTER COMPLETION: inspect all9pose groups, current/final stand geometry and starting-state eligibility. Compare against the completed nominal control with matching seed/sourceIDs/physics; repeated bank is development validation. Never promote on reward alone. Preserve locomotion2250/recovery3448. Record and ACTUALLY VIEW front+oblique latest successful OR failed diagnostics; show true full-scenario results with separate-policy labels, not a nonexistent integrated walk/fall/recover/walk controller.
- Video follow-up is now prepared in `scripts/record_smith_recovery_review.ps1`. Main checked PowerShell AST and `-PreflightOnly`: correctly NOT READY (3 missing formal reports, active PID119728), no output created. Run it only AFTER the orchestrator finishes and all three suites are valid. Four serial simulator calls produce six raw videos and three paired upright/side/back diagnostics in `evaluations/20260916-smithnominal128x2000-video-review/`. Existing source/seed/eligibility/result checks remain; raw mismatches must be preserved. The videos have NOT been recorded yet.
- If again no recovery, first audit logged roll/stand versus unchanged regularizers, then isolate a regularizer curriculum/gate experiment; do not automatically extend this failed branch or simultaneously change bank, actions and PPO. Existing heartbeatgo2 remains active and reads this handoff. All outputs remain on E:.

## COMPLETED PREVIOUS JOB — action-reference pair and verified video

Historical18:34local snapshot, superseded by the ACTIVE job above. **Both128-environment action-reference arms, all six batch evaluations and all twelve serial video runs are COMPLETE. Neither actor passes acceptance. Do not restart these completed jobs.**

- Completed training session51383/log `E:/IsaacLab/artifacts/recovery-20260916/target128x2000-orchestrator.log`; recorder39086/log `E:/IsaacLab/artifacts/recovery-20260916/target128x2000-video-review.log`; CPU assembly35803 all exited0.
- Each arm:128 environments ×24 rollout steps ×2000 iterations =6,144,000 environment steps. Same fresh seed42 network/PPO/rewards/bank/physics; only action reference differs. This replacement pair is NOT a continuation of the interrupted512-environment run.
- Nominal run `E:/IsaacLab/repo/logs/rsl_rl/unitree_go2_recovery/2026-09-16_16-14-17_20260916-target128x2000_nominal/model_1999.pt`; SHA256 `58f3231a1959a3d5fa82142e9dcfb2aee2b7cc86f1575e3cee25e36c9efc6ca2`.
- Current run `E:/IsaacLab/repo/logs/rsl_rl/unitree_go2_recovery/2026-09-16_17-00-18_20260916-target128x2000_current/model_1999.pt`; SHA256 `5479e23fd0617026a2d913eb509356cc5c8d2324a2601dd0e26b4556e9cb8710`.
- BOTH arms: all nine20-trial pose groups have0 successes,0 final valid stands and0 final geometry passes. Suites: heldout side/back, controlled30-degree upright/side/fore-aft, controlled45-degree upright/side/fore-aft/back. Configs and reports in `configs/20260916-target128x2000-{nominal,current}/` and `evaluations/20260916-target128x2000-{nominal,current}-{heldout,angle30,angle45}/`.
- Independent audit: corresponding policy-start root/joint states/velocities match bitwise (max difference0); all40 bank starts eligible per arm but only16 unique side +14 unique back states. Bank SHA `71b882e92035b4c248e5980339c980dbb242dcb1ec711c05252c33249db48f5a`. Bank/30-degree seed20260918,45-degree seed20260916. Handover is1second direct nominal-position PD, NOT zero torque. Controlled drops are NOT settled-fallen starts.
- CurrentTarget back starts end near64.75degree tilt,height0.1512m,two feet; NominalTarget near155.10degrees,height0.0714m,zero feet. Partial rolling is NOT recovery; maximum joint offset worsens to3.27rad. Neither candidate is promoted or extended unchanged. Preserve locomotion2250 and limited-domain recovery3448.
- FINAL VIDEO: `evaluations/20260916-target128x2000-video-review/full_scenario_review_front_oblique.mp4`,1920×700,H.264,25fps,2954frames,118.16seconds,91,536,884bytes. SHA `3eeb61cd7194643565f86f70408c2e928a95b3d33c60b6cf73742ccc197b20e2`. Matching JSON contains source/report/model hashes and exact chapter offsets. Full final decode and SHA check passed; all16 source videos fully decoded before assembly.
- ACTUAL VISUAL QA COMPLETE: main viewed both-angle phase sheets for normal2250, disturbed2250, and upright/side/back for both new actors, then all final compilation chapters and ending. Normal2250 stand/walk/stop passes. Lateral delta-v0.75m/s makes2250 fall inverted and remain down. Nominal upright falls onto its side, nominal back remains inverted; Current upright collapses and its side/back settle into contorted low side support. All new recovery videos correctly show FAIL. This is simulation only, with no hardware validation.
- Honest delivery: old locomotion2250 is explicitly separate from the new recovery actors. Independent matched-condition front/oblique replays are NOT simultaneous cameras. The eight chapters are NOT a continuous walk/fall/recover/walk controller. All original frames retained; bank videos include one pre-policy frame, upright clips do not.1second nominal-PD handover precedes the bank footage. Failures and renderer startup noise are not cut.
- NEXT BOUNDED WORK: this batch's requested video is now delivered before another long branch. Do not run the old recorder again over its output. Implement an isolated Smith-inspired roll/stand reward task AFTER reading `docs/reproduction-audit-20260916.md` and its pinned official source. Hold Go2 physics, action representation, bank and PPO fixed for the reward comparison; no arbitrary torque increase, weakened success criterion or multiple simultaneous sims. Test reward math and upright regressions, smoke first, then a finite128-environment pilot and identical heldout evaluation. Reward direction remains a hypothesis, not proven recovery.
- Subsequent independent initial-state experiment: genuinely zero-drive-torque passive falls with verified actuator outputs and cold replay, NOT zero policy action under PD. Do not combine this with the first reward-only comparison.
- Existing heartbeat `go2` remains active. Earlier prompt-update attempt failed due approval-service capacity and was NOT applied; this status document is the current handoff. All writes remain on E:. No new branch has started yet.

[Video chapter/results and limits](video-review-20260916.md) · [paired experiment](action-reference-ablation-20260916.md) · [paper/source audit](reproduction-audit-20260916.md).

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
