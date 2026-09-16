# Fresh action-reference comparison

Status updated17:13: the512-environment run crashed near iteration584 with PhysX CUDA out-of-memory; only the intact500 checkpoint was evaluated (side0/20,back0/20). CurrentTarget512 never started. The replacement128-environment NominalTarget arm completed2000 iterations/6,144,000 environment steps, final1999, SHA256 `58f3231a1959a3d5fa82142e9dcfb2aee2b7cc86f1575e3cee25e36c9efc6ca2`. It FAILED every strict20-trial pose screen: heldout side/back, controlled30degree upright/side/fore-aft, and45degree upright/side/fore-aft/back; zero final valid stands in every group. CurrentTarget128 started17:00 and is still training. Both actual saved YAMLs match except intended reference/run identifiers; installed source hashes matched. This negative fresh-policy result must not overwrite the older working locomotion2250 or limited-domain recovery3448. Active log `E:/IsaacLab/artifacts/recovery-20260916/target128x2000-orchestrator.log`; [live handoff status and pending video delivery](training-status-20260916.md).

## Why this experiment

BackExplore100's deterministic policy failed every settled side/back trial. Its separate stochastic back diagnostic also failed 0/20 (14 unique source states), despite physical joint spans of 0.587–1.447 rad across trial/joint pairs. At 50 Hz control-boundary samples, joint-speed peaks were 9.65–23.40 rad/s and applied-torque peaks 4.45–18.36 Nm; substep extrema may be missed. All action standard deviations were 0.6. Best upright tilt reached only 167.20 degrees (180 is inverted), maximum body height only 0.07085 m. This rules out literally motionless exploration in that diagnostic, but does not establish the cause of failed righting. Random thrashing is not recovery.

Source: `evaluations/20260916-back-explore100-stochastic-diagnostic/model_4998.pt_recovery_metrics.json`; `acceptance_eligible=false`.

The next hypothesis is action reference, inspired by the current-joint-angle righting target in [Lee et al. (2019), section II-D2](https://arxiv.org/html/1901.07517). This is an isolated Go2 experiment, not a reproduction of ANYmal results.

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

Original plan was serial512-environment,1000-iteration arms; this was interrupted by native CUDA OOM before a complete pair existed. The active replacement is128 environments and2000 iterations per arm (6,144,000 environment steps each), then strict20-trial side/back bank, upright/30-degree and upright/45-degree evaluations. The finite launcher `scripts/run_recovery_action_ablation.ps1` checks exact bootstrap/bank hashes, refuses duplicate outputs and invalid physics logs, and does not automatically promote or extend either model. Controlled-drop angle tests use matched fixed Bank physics, NOT the randomized historical Aligned task; do not compare their counts as if identical. [Audited paper/source differences and the decision if this pair still fails](reproduction-audit-20260916.md).

The repeatedly used30 heldout source states are development validation, not a fresh final test. Any promising candidate requires new unseen starts/seeds, normal standing/stop regressions, and actually viewed front/oblique videos before promotion. All previously recommended models remain untouched.
