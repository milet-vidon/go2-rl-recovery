# Frozen roll-policy mirror: actual six-case audit, 2026-09-17

Status: **bounded simulator diagnostic; not promoted, not training data, not visual or hardware acceptance**. The six-case simulation chain completed with exit 0. A separate read-only coordinate auditor then passed all six real reports. This note was prepared afterward using PowerShell-only reads of reports, audit artifacts, source hashes and existing research notes; no Python, simulator, training, installation or active-source modification was performed while writing it.

## Outcome and complete denominators

The intervention is a frozen-policy input/output reflection for selected right-side starts, not additional learning. It improved the existing side development cases without changing the unselected cases. It did **not** fix ordinary upright startup.

| Arm / actual requested set | Trials | Confirmed settled-fallen | Mirror selected | Stand acquired / final valid stand | Final geometry only | Real handoff triggered |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OFF / side | 20 | 20 | 0 | 9 / 9 | 20 | 20 |
| ON (`initial_right`) / side | 20 | 20 | 11 | 20 / 20 | 20 | 20 |
| OFF / upside_down | 20 | 20 | 0 | 20 / 20 | 20 | 20 |
| ON / upside_down | 20 | 20 | 0 | 20 / 20 | 20 | 20 |
| OFF / upright | 20 | 0 | 0 | 12 / 12 | 18 | 19 |
| ON / upright | 20 | 0 | 0 | 12 / 12 | 18 | 19 |

Each displayed count retains all 20 trials, including the untriggered upright trial and every failure. Geometry is only one necessary condition: OFF side had geometry 20/20 while strict final standing was only 9/20. The selected 11 right-side trials went from 0/11 to 11/11 final valid stands; the other nine side trials remained exactly unchanged. Across the three sets the dual-controller counts are OFF 41/60 and ON 52/60, **not** a pooled recovery-generalization rate: ordinary upright starts are not fallen-recovery trials.

The separate, same-start [stand3547-only upright reference](../evaluations/20260917-handoff-stand100-upright20/model_3547.pt_recovery_metrics.json) achieved 20/20. Its report SHA-256 is `1c3728c48cb7b2a8ad36ae11f1aae97c03ad53719bca56ddcda372a3c87ad463`. The present roll-first dual controller still achieves only 12/20. These must not be conflated into a claim that ordinary standing was preserved. The actual ordinary upright starts were supported but not settled; all had `standing_at_policy_start=false` and fallen eligibility false. The label “upright” alone does not establish an already-completed valid stand.

Side uses **16 unique bank IDs** for 20 trials; upside_down uses **14 unique IDs** for 20 trials. Repeated draws are not additional independent source states. Both use the bank's nominal `heldout` split, but these cases have been repeatedly inspected and used to choose this development intervention. They are therefore reused development evidence, not a fresh blind held-out test. Twenty successful side trials do not certify generalization. Ordinary upright uses no fallen bank.

## What changed, and what stayed fixed

At the real post-preparation policy-start boundary, ON latches the mask once:

`eligible_settled_fallen_at_real_policy_start && normalized_projected_gravity_body_y < -cos(30 degrees)`.

OFF selects none. The selection does not inspect the requested pose, bank ID, later trajectory or outcome. The mask is fixed throughout each episode. Actual measured starts selected 11 side rows and zero upside_down/upright rows; zero selection for those controls is a measured result, not a pose-label exception.

For selected rows before the unchanged one-way gate, the native issued roll candidate is `J12(roll_actor(M48(real_obs)))`. Unselected rows use `roll_actor(real_obs)`. The standing actor always receives the original real observation dictionary. `TensorDict.clone(recurse=True)` creates the roll-input copy; a newly allocated transformed policy tensor replaces only that copy's policy leaf. The real observation and action-manager history are not rewritten. Both actors are deterministic feed-forward policies with actual Identity observation normalization.

Native joint order is FL, FR, RL, RR for each of hip, thigh, calf. Define:

```text
P = [1,0,3,2,5,4,7,6,9,8,11,10]
S = [-1,-1,-1,-1,1,1,1,1,1,1,1,1]
J(x)[i] = S[i] * x[P[i]]
```

| Actual 48-D observation slice | Roll-input copy transformation |
| --- | --- |
| 0:3 body linear velocity | signs `[1,-1,1]` (polar vector) |
| 3:6 body angular velocity | signs `[-1,1,-1]` (axial vector) |
| 6:9 projected body gravity | signs `[1,-1,1]` |
| 9:12 command `[vx,vy,yaw_rate]` | signs `[1,-1,-1]` |
| 12:24 joint-position residual from nominal | J |
| 24:36 joint velocity | J |
| 36:48 actual previous issued raw action | J on the copy only |
| Actor output, 12-D | J back to physical native coordinates |

Actual native nominal angles and actual hard/soft joint limits satisfy the checked reflection contract. The nominal-plus-0.25-raw soft-clamped target map commutes algebraically with J. This is a representation result, not proof that the physical robot's dynamics are perfectly symmetric.

The frozen gate remains tilt `<30 degrees` and angular speed `<1 rad/s` continuously for 0.2 s, one-way roll-to-stand. There is no retry, ramp, startup actor selector, hidden standing PD at the switch, state/history reset, physics change or extra learning. The existing 1 s nominal-position PD preparation remains explicit; “no hidden PD at switch” does not mean no PD preparation. Policy control remains 0.02 s, 8 s acquisition horizon plus the unchanged 3 s hold observation budget, 550 policy steps total. Success still requires all original gravity, height, linear/angular speed, four simultaneous vertical foot contacts, no base contact and foot/knee/joint-geometry criteria continuously for 3 s. No threshold, timeout or denominator was relaxed.

## Independent coordinate and unchanged-cohort audit

The new stdlib-only auditor is `scripts/audit_handoff_mirror_coordinates.py`, with regression tests in `scripts/test_audit_handoff_mirror_coordinates.py`. All **16 tests passed** (41.115 s, exit 0), including deliberate corruption of masks, polar/axial signs, virtual/real observations, model outputs, history, targets, switch records, missing rows, initial states and tiny unselected differences. Synthetic ON fixtures are explicitly coordinate-test fixtures, not actor inference or simulated recovery evidence. The subsequent six real audits all exited 0; neither the frozen entry nor helper was edited for the audit.

| Real audit case | Stored rows checked | Switch records checked | Baseline | Unselected trials compared | Common scalar values compared exactly |
| --- | ---: | ---: | --- | ---: | ---: |
| off-side | 1,785 | 20 | hard-side | 20 | 597,183 |
| off-upside_down | 2,020 | 20 | hard-upside_down | 20 | 674,732 |
| off-upright | 1,969 | 19 | hard-upright | 20 | 657,302 |
| initial_right-side | 1,993 | 20 | off-side | 9 | 415,954 |
| initial_right-upside_down | 2,020 | 20 | off-upside_down | 20 | 922,012 |
| initial_right-upright | 1,969 | 19 | off-upright | 20 | 898,290 |
| Total | **11,756** | **118** | | **109 comparisons** | **4,165,473** |

There are 120 trial records across the six cases; the 109 unselected comparisons include the 60 OFF controls plus the 49 unselected ON trials, not 109 independent physical source states. OFF-to-hard comparisons total 1,929,217 scalar values; ON-to-OFF unchanged-cohort comparisons total 2,236,256. Scalars include metadata as well as recorded physics/actions, so these counts are not an independent sample size.

For every stored row and switch record, the auditor verified:

- `real_policy_observation` equals the legacy real `policy_observation`; both are finite native 48-vectors. The roll-input vector is exactly M48(real) for selected rows and exactly identical for unselected rows.
- The model-coordinate roll output maps exactly through J12 to the recorded physical-native roll output for selected rows, and is identical for unselected rows. The native output also equals the retained legacy roll-action field.
- Actual real observation command and last-action slices equal recorded real command/history; previous and previous-previous histories retain their native meanings. The hard latched choice matches the issued action, action-manager advancement and actual target.
- An independent per-operation float32 computation of `soft_clamp(default + 0.25 * issued_raw)` and target deltas matches the stored expected and actual target exactly. No FMA-related discrepancy occurred and no action/target tolerance was introduced by this audit.
- Every trial has its required neighborhood: the bounded pre/post-switch rows, or the explicit untriggered tail. Every triggered trial has its actual switch record and single one-way transition. All release and actual policy-start states, including the 11 selected rows, remain exactly equal across controls.
- All applicable old common fields for unselected trials match recursively with the same scalar types, list lengths/order and zero numeric tolerance: real physical diagnostics, observations/actions/history, switch records, target clipping and trace rows. Entire result aggregates also match for fully unselected back/upright sets. For partially selected side, changed aggregate fields are explicitly enumerated in the audit output; unknown common result fields are rejected rather than silently skipped.

Selection is independently recalculated from actual start evidence. Only the float64 reconstruction of normalized gravity-y versus the logged float32 normalization uses the predefined `2e-7` metadata tolerance; maximum observed error was `8.179463206747783e-8`. Actual mask decisions use the strict runtime float32 threshold, and ambiguous near-threshold cases are rejected. This tolerance is not applied to coordinate, action, target or unchanged-trajectory comparisons.

Standing-input evidence has an explicit limit: the audit verifies the frozen entry's `stand_policy(obs)` call and the runtime-checked real48 record. There is **no separately logged standing-input48 argument capture**, so it does not claim one. Similarly, the auditor verifies the recorded model-output coordinate relationship, not an independent fresh evaluation of the network on every row.

The JSON traces cover switch neighborhoods or untriggered tails, not all 550 policy steps. Every-step runtime assertions are supported by the pinned source and completed simulator reports; the offline trace audit must not be described as full-trajectory replay. Old reports lack per-trial ever-success/legacy-success time series; no such series is reconstructed from a short trace. Final validity and complete denominators are checked against the actual final diagnostics.

## Artifact locations and fingerprints

Real reports: `evaluations/20260917-handoff-mirror-v1/<case>/model_1999.pt_recovery_metrics.json`. Each corresponding sibling trace is `model_1999.pt_handoff_mirror_trace.json`, schema `handoff_mirror_neighborhood_v1`; its byte hash is checked against the report. Hard baselines are under `evaluations/20260917-handoff-transition-v1/hard-<pose>/`.

Read-only audit outputs are new external local artifacts at `E:/IsaacLab/artifacts/recovery-20260917/20260917-mirror-audit-<case>.json`, schema `handoff_mirror_coordinate_audit_v1`. They explicitly set `audit_pass=true`, `acceptance_eligible=false`, `training_collection_eligible=false`, `promotion_performed=false`.

| Case | Report SHA-256 | Trace SHA-256 | Audit-output SHA-256 |
| --- | --- | --- | --- |
| off-side | `cd0e31c99b384395aebc71577c1e8c944bb82eb99d81e454045fc0aee2d01c47` | `17bf938cd16ee4c79b4112cf2b9d685d29d92e76959632bd51d0e80fc5405606` | `ccbef25596445943efeb04c5aca4542096eb20977eedb4e2e2decbffae044a66` |
| off-upside_down | `0f858a435e915b231888c411b8b3b2b84178835b6b7601338a3e3bc99a155bdf` | `c72ae61157a0330f3ac21b7d0605f37083ecde41ecbc29b8d84d0f5e12760958` | `45d4751206637608627f0e72ddf5b2e59f5be913fd55071063f481e96d59a4cf` |
| off-upright | `d5102f3772087e7434827ed82d5747fc416adcc5be10b13447fac31c40d4d87c` | `4e3edf31457e9678a397b9c054eaf2d93ac9a210f60fae8745cc40c3f264702c` | `d1f9656712735c25683b4bd8a2bf85e47440017e33ea33613e5b52a9686bd939` |
| initial_right-side | `8ba9ab81b8bd0d94cb6a1ab3e1d5b7f1f9c95165de7d9734932f01afb1576924` | `fbb2874756360a72154fe6602f7614bb0ee15722883c0c6d887ef0cc44b0cac5` | `350661f9dc3fa99bbf24129636ae2217ade3716b476a12064824f06749e3eb48` |
| initial_right-upside_down | `54dfe9670c4684b72e48100224061af48d13d9094c1f866e3f98a82505520ac0` | `30ff665f9d5bb5ed35e7b181513867dd566d3b2d8128ddef067af0c7be630896` | `cb6ced3d593496d840298a4a92d67f074aca0d1cd552b3c4bc7e3b506f3906e6` |
| initial_right-upright | `eedfbdec893688914e8c0af8fc677ccaf70789178c704c60fcf7fcfbea66b31f` | `12fd88dbeb037f6a421fb05cbd4f59ebdd85c7c3bbb969425119ba09111b2b3f` | `d213544be3cec989044ac08d937f2c4b41265ffb09ad9fe7020cb4ab73424b0a` |

Frozen implementation and input identities:

| Item | SHA-256 |
| --- | --- |
| `scripts/audit_handoff_mirror_coordinates.py` | `6337b44e1a9eae37b7697f4b9f4c6a7c984ab6577d1f08b6c1463c1228e3a281` |
| `scripts/test_audit_handoff_mirror_coordinates.py` | `bcaeb5d3ce119b876ccf56fdcd2fbfb36784a59d0de20263964693002f049044` |
| `scripts/evaluate_handoff_mirror.py` | `cf2fbbdceae0e460727a969bf080e1e230ed716f00f914f47381a167df50ba95` |
| `src/go2_recovery/roll_mirror_math.py` | `9e3edf7cd402c745242fa41e8dcf3d4ccb3818cb474344cff52c104c00426d91` |
| `scripts/test_roll_mirror_math.py` | `5d9db62714195696df7e78c1f539ca3cca467f43e90a7518f2cd9a0bbefaeb01` |
| `scripts/evaluate_handoff_mirror.ps1` | `238f67a9fb47f1639a19f4ff481c3db5d813a80e464b26961d7445c208d0d85f` |
| `scripts/compare_handoff_mirror_baseline.py` | `4e51e599436a4212a5bca151115ea0c8c1c8b10d6b947cd3c185e71c106bd368` |
| Original RAW collector `scripts/evaluate_go2_recovery.py` | `4024ba713e3ed616225d64e1d59de97fd31c8369bba83ac3f4235d93552e9e2f` |
| Hard-transition evaluator | `c3a433568561fbe0893d61a570cf420304d44dcf986440e8d7efebd85ff7f10a` |
| Transition math | `b66167039b4698d66f3cc412b19bacd0b1b2f37279f7ce0856ec6c09b196aa4e` |
| Roll1999 model | `71e4c7a39ff81836d2ead1c4533988c696e5bd4a0fc939b8cef6b4f6daec1b1c` |
| Stand3547 model | `5527c4053cb6ef8e6616da65b7dc32a9464eb8d1366f4966a4eb4b65bc913fdb` |
| `datasets/recovery_states/nominal_pd_v1_20260916/states.npz` | `71b882e92035b4c248e5980339c980dbb242dcb1ec711c05252c33249db48f5a` |

The report identifies `protocol_version=handoff_mirror_experimental_v1` and `controller_type=experimental_roll_only_initial_side_mirror_hard_handoff`. Its baseline metric protocol remains `stance_geometry_v1_state_bank_PD_v1_dual_policy_diagnostic_v1`; its baseline transition protocol is separately identified. Different labels do not silently change the metric.

## Relation to primary references and remaining limits

The [earlier proposal and implementation addendum](roll-side-mirror-proposal-20260917.md) preserve the reasoning and the superseded proposed selector; the implemented selector is the stricter real eligible-fallen plus normalized-y rule above, with no pose-label condition.

Previously inspected primary references support different levels of the design:

- [Official Isaac Lab ANYmal symmetry implementation](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/symmetry/anymal.py) supplies the polar/axial/command transformation pattern and consistent joint/history transformation. Its ANYmal-specific joint permutation is not copied as Go2 indices; the actual Go2 order/default/limits were separately checked.
- [Official RSL-RL symmetry extension](https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/extensions/symmetry.py) provides symmetry augmentation and consistency-loss machinery for learning. This experiment did not enable it or update the policy. A moving upstream API is not a pin for the installed version.
- [Mittal et al., arXiv:2403.04359v1](https://arxiv.org/html/2403.04359v1) grounds compatible state/action symmetry and distinguishes learning methods from merely choosing an inference representation. The present frozen inference wrapper is an engineering diagnostic, **not a reproduction of that symmetry-training method**, and it has no paper-derived guarantee of recovery.

Exact observation/action algebra, compatible nominal targets and bounds, and invariance of the geometric acceptance formula do not prove full physical symmetry of the local binary USD. Actual mesh collisions, link inertias, actuator/contact dynamics and hardware have not received a reflection-equivalence proof. The [official Unitree Go2 URDF](https://raw.githubusercontent.com/unitreerobotics/unitree_ros/master/robots/go2_description/urdf/go2_description.urdf) even records nonzero base cross-inertia terms; that separate URDF is not asserted to be identical to this local USD. See the proposal for the local asset fingerprints and this boundary.

**Limited visual QA update before this note was finalized:** the main task completed both ON replays of the previously failed right-side bank state 153 (session 45703, exit 0) and actually inspected front/oblique keyframes at 0, 0.5, 1, 1.5, 2, 4, 8 and 10.96 s. The inspected final stance has separated front legs, without the earlier crossed-leg defect, but some left/right asymmetry remains. This is the main task's visual inspection record; the author of this read-only documentation step did not independently re-inspect images.

Both corresponding real reports give 1/1 final valid stand, first valid-hold onset at 1.3200000524520874 s, final height 0.3235992193222046 m, four vertical foot contacts and a continuous final valid interval of 9.68 s. Their coordinates were also audited by the main task. The 1.32 s metric is the onset of an interval later confirmed to meet the required 3 s hold, not a claim that a 3 s hold had already elapsed at 1.32 s. Final front-foot body-y is +0.17415782809257507 / -0.12457910180091858 m; final front-knee body-y is +0.16235265135765076 / -0.13269227743148804 m.

The paired full-duration video (local-only artifact, not included in this GitHub snapshot: `evaluations/20260917-mirror-right153-paired/model_1999_side_front_oblique.mp4`) has 276 frames at 25 fps. Its [pairing record](../evaluations/20260917-mirror-right153-paired/paired_views.json) explicitly identifies **two independent matched-condition replays, not synchronized cameras**, with no time cropping. Sources are the [front report](../evaluations/20260917-mirror-right153-front/model_1999.pt_recovery_metrics.json) and [oblique report](../evaluations/20260917-mirror-right153-oblique/model_1999.pt_recovery_metrics.json). This is only the same preselected development example, not an additional independent source state, exhaustive frame-by-frame QA, a generalization test, or a full standing/walking/stopping/disturbance demonstration. Coordinate/metric audit is not a substitute for visual inspection of other trajectories.

The [supported, nonfallen startup preregistration](handoff-supported-startup-preregistration-20260917.md) and [startup diagnosis](handoff-upright-startup-selector-analysis-20260917.md) already exist as proposals only at this point. That next independent diagnostic has **no mirror, ramp, retry or training**, and is not implemented by this note. Do not combine it with this mirror arm and attribute results to one factor. Fresh independent starts, broader perturbations, upright retention, normal walking/stopping and visually checked continuous runs still remain before any claim of meeting the user's full reference-video objective.
