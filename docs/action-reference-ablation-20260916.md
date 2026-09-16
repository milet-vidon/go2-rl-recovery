# Fresh action-reference comparison

Status 2026-09-16: BOTH replacement 128-environment arms completed 2000 iterations and 6,144,000 environment steps each, with final checkpoint 1999. Each arm FAILED all nine strict 20-trial pose groups: heldout side/back; controlled 30-degree upright/side/fore-aft; controlled 45-degree upright/side/fore-aft/back. In every group, successes, final valid stands and final geometry passes are all 0/20. Neither candidate is promoted or extended unchanged. These completed results supersede the earlier 17:13 status saying CurrentTarget128 was still training.

- NominalTarget checkpoint SHA256: `58f3231a1959a3d5fa82142e9dcfb2aee2b7cc86f1575e3cee25e36c9efc6ca2`.
- CurrentTarget checkpoint SHA256: `5479e23fd0617026a2d913eb509356cc5c8d2324a2601dd0e26b4556e9cb8710`.
- Both hashes were verified against the actual checkpoint files. Both actual saved YAMLs match except intended reference/run identifiers; installed source hashes matched.
- The earlier 512-environment attempt is retained as an INTERRUPTED historical run: native PhysX CUDA out-of-memory near iteration 584, intact 500 checkpoint side 0/20 and back 0/20; CurrentTarget512 never started. It is not part of the completed 128-environment pair.

These negative fresh-policy results must not overwrite working locomotion2250 or limited-domain recovery3448. Latest front/oblique recordings and actual visual inspection are complete: the118.16-second full scenario review preserves all frames, failures and separate locomotion/recovery model identities. Normal2250 passes; disturbed2250 and all six new recovery chapters fail. This is not a continuous closed loop. [Latest video review and delivery status](video-review-20260916.md) · [live handoff status](training-status-20260916.md). Completed launcher log: `E:/IsaacLab/artifacts/recovery-20260916/target128x2000-orchestrator.log`.

## Why this experiment

BackExplore100's deterministic policy failed every settled side/back trial. Its separate stochastic back diagnostic also failed 0/20 (14 unique source states), despite physical joint spans of 0.587–1.447 rad across trial/joint pairs. At 50 Hz control-boundary samples, joint-speed peaks were 9.65–23.40 rad/s and applied-torque peaks 4.45–18.36 Nm; substep extrema may be missed. All action standard deviations were 0.6. Best upright tilt reached only 167.20 degrees (180 is inverted), maximum body height only 0.07085 m. This rules out literally motionless exploration in that diagnostic, but does not establish the cause of failed righting. Random thrashing is not recovery.

Source: `evaluations/20260916-back-explore100-stochastic-diagnostic/model_4998.pt_recovery_metrics.json`; `acceptance_eligible=false`.

This experiment tested action reference, inspired by the current-joint-angle righting target in [Lee et al. (2019), section II-D2](https://arxiv.org/html/1901.07517). This is an isolated Go2 experiment, not a reproduction of ANYmal results.

## Matched pair

Both arms start from the SAME freshly seeded ActorCritic, not weights4899 or an old walking actor. 48 inputs, 12 outputs, three128-unit ELU layers, standard deviation1.0, empty Adam state, seed42, checkpoint iteration0. Bootstrap `bootstrap_fresh42_target_v1/model_0.pt`, SHA256 `daa28dd10ef200121bdd6f2a14a03fcd856859ebb5f3ef510d5d946a10f11cb9`.

- NominalTarget: target = default joint position + 0.25 * action.
- CurrentTarget: target = joint position measured at the 50 Hz policy boundary + 0.25 * action.
- BOTH clamp the resulting position target to the same per-joint soft limits. Therefore neither is identical to the old, unclamped nominal task.
- The target is cached once per control step and held for four200 Hz physics substeps. Isaac Lab's built-in RelativeJointPositionAction reads the current joint state every physics substep; this experiment deliberately does not use that behavior.
- Reset changes only the selected environments' action/history cache, initialized to the newly reset joint positions. No physics step or joint-state write occurs in the action reset.
- Both inherit BankControl: fixed60% validated nominal-PD bank,40% upright/shallow rehearsal, same rewards, PPO, material, PD gains, self-collisions and actuator limits. Only the reference differs within the pair.
- The214 train states are unchanged. No passive/zero-torque preparation is introduced here; that is a separate possible future experiment.

## Verification and evaluation

Eight CPU tests cover the reference difference away from clamps, substep sample/hold, identical clamps, ordered joint subsets, subset/all/empty resets, invalid options, configuration isolation, reproducible untrained bootstrap and identical nominal-PD handover targets for old/new action paths.13 gate regressions,5 stochastic-diagnostic guards,3 trace-packing tests,4 video-annotation tests,2 command-protocol tests and the recovery-math regression also pass. Each new task completed a16-environment/two-iteration real training smoke, with no recovery-performance claim. The smoke policies are NOT formal parents.

The evaluator uses the appropriate action task. Bank handover commands the SAME default joint targets directly for at least1 second for both arms; it does not assume a zero action is nominal under CurrentTarget. History is reset afterward without changing the physical pose. Reports explicitly record action reference, scaling, clamp and sample/hold period. Historical evaluator paths retain their original action processing. Both two-trial side/back evaluation smokes completed. Side starts were bitwise identical. One back start had small contact-solver differences after the preceding different rollout (joint-position difference below0.00005 rad, height difference below0.000001 m); classifications and source IDs were identical. Matched protocol does not imply bitwise identical PhysX contact dynamics across policy trajectories.

Original plan was serial 512-environment, 1000-iteration arms; this was interrupted by native CUDA OOM before a complete pair existed. The replacement 128-environment pair and all six evaluation suites are now complete: 2000 iterations per arm (6,144,000 environment steps each), followed by strict 20-trial side/back bank, upright/30-degree and upright/45-degree evaluations. The finite launcher `scripts/run_recovery_action_ablation.ps1` checks exact bootstrap/bank hashes, refuses duplicate outputs and invalid physics logs, and does not automatically promote or extend either model. Controlled-drop angle tests use matched fixed Bank physics, NOT the randomized historical Aligned task; do not compare their counts as if identical. [Audited paper/source differences and the next bounded design change](reproduction-audit-20260916.md).

## Completed pair results

Every row below has 20 trials per arm. Entries are successes / final valid stands / final geometry passes.

| Start protocol | Pose | NominalTarget | CurrentTarget | Eligible settled-fallen starts per arm |
| --- | --- | --- | --- | --- |
| Heldout bank | Side | 0 / 0 / 0 | 0 / 0 / 0 | 20 (16 unique source states) |
| Heldout bank | Back | 0 / 0 / 0 | 0 / 0 / 0 | 20 (14 unique source states) |
| Controlled 30-degree drop | Upright | 0 / 0 / 0 | 0 / 0 / 0 | 0 |
| Controlled 30-degree drop | Side | 0 / 0 / 0 | 0 / 0 / 0 | 0 |
| Controlled 30-degree drop | Fore/aft | 0 / 0 / 0 | 0 / 0 / 0 | 0 |
| Controlled 45-degree drop | Upright | 0 / 0 / 0 | 0 / 0 / 0 | 0 |
| Controlled 45-degree drop | Side | 0 / 0 / 0 | 0 / 0 / 0 | 0 |
| Controlled 45-degree drop | Fore/aft | 0 / 0 / 0 | 0 / 0 / 0 | 0 |
| Controlled 45-degree drop | Back | 0 / 0 / 0 | 0 / 0 / 0 | 0 |

The bank/30-degree seed is 20260918; the 45-degree seed is 20260916. Within each corresponding suite, criterion, protocol, source-ID order and all recorded policy-start root poses, root velocities and joint positions match exactly. Both bank arms use the same heldout bank SHA256 `71b882e92035b4c248e5980339c980dbb242dcb1ec711c05252c33249db48f5a` (verified against the file) and 1-second direct nominal-position PD handover, NOT zero torque. All 40 bank starts per arm were settled-fallen and none already standing; controlled-drop starts were unsettled and are not counted as settled-fallen recovery.

CurrentTarget did change the back-start outcome: mean final tilt 155.10° → 64.75° and height 0.0714 m → 0.1512 m, but ended with only two supporting feet and invalid geometry. Its maximum final joint offset in bank trials was approximately 3.27 rad versus NominalTarget 1.69 rad. Side-start mean final tilt stayed approximately 64° for both. Thus there is evidence of leaving the inverted posture, but no normal recovery or standing retention; unchanged extension is not justified.

Raw reports: [Nominal bank](../evaluations/20260916-target128x2000-nominal-heldout/model_1999.pt_recovery_metrics.json), [30°](../evaluations/20260916-target128x2000-nominal-angle30/model_1999.pt_recovery_metrics.json), [45°](../evaluations/20260916-target128x2000-nominal-angle45/model_1999.pt_recovery_metrics.json); [Current bank](../evaluations/20260916-target128x2000-current-heldout/model_1999.pt_recovery_metrics.json), [30°](../evaluations/20260916-target128x2000-current-angle30/model_1999.pt_recovery_metrics.json), [45°](../evaluations/20260916-target128x2000-current-angle45/model_1999.pt_recovery_metrics.json).

The repeatedly used30 heldout source states are development validation, not a fresh final test. Any promising candidate requires new unseen starts/seeds, normal standing/stop regressions, and actually viewed front/oblique videos before promotion. All previously recommended models remain untouched.
