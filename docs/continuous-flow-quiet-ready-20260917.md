# Continuous recovery-to-walking: stronger readiness, not relaxed acceptance

Status at 2026-09-17 18:08 local: v1 completed physically but failed behavior;
the two matched v2 full cases have now completed and passed their17 behavior
checks. Independent trace audit and front/oblique video checks are next.
No trot, fast-running or promotion claim follows from this document.

## Actual v1 result

`evaluations/20260917-continuous05-side-full-v1/continuous_flow_report.json`
records one real state132/left episode with original1999/3547/3947:
50 initial nominal-PD intervals,241 recovery,400 movement and300 stopping,
991 total (19.82 s). No reset, source change or interface discontinuity occurred.
The entire actual sequence completed, but the authoritative result is
`behavior_failed`, not PASS.

Two failures:

- Last150 recovery samples had original-strict geometry/support but linear
  speedp95=.116567716 m/s, above the existing quiet-rest .06 limit.
- At command.5, movement mean vx=.608531944,vy=-.120265828 m/s. Lateral drift
  is outside the unchanged abs(vy)<.12 limit, however small the exceedance.

Contact slip=.065134574 m/s and stopping speedp95=.003834366 m/s pass;
the ending150 stop samples retain strict stance. All other16-check v1 gates
except the two listed fail fields pass. Strict recovery's speed<.5 is not a
quiet-rest certificate. Neither failure is hidden by discarding samples.

Root independently replayed the SHA-bound archived v1 metrics against all991
saved rows: the result equals the report exactly. All artifact hashes match;
the first52 rows equal the prior successful smoke. The v1 source archive is
`E:/IsaacLab/artifacts/recovery-20260917/continuous-flow-v1-source-archive`.

## Predeclared v2 changes

V2 keeps the original physical state,old roll->stand gate,action history,
models1999/3547,strict-stance definition,550-interval premovement budget,
400movement/300stop schedule and every numerical behavior threshold.
No smoothing,refiner,additional PD,pose transplant or reset is introduced.

The new **separate** next-action readiness condition is150 consecutive
completed samples that are BOTH original-strict and quiet:
3D linear speed<.06 m/s AND3D angular speed<.15 rad/s. The saved original
`strict_valid_after` is not redefined. Separate strict/ready counters and
`quiet_for_handoff` preserve the distinction. A nonquiet sample resets only
readiness; a nonstrict sample resets both. All counts remain inside550,
so insufficient settling must fail rather than extending the horizon.
The final150 recovery samples are independently checked for actual quiet
speeds as well as the old criteria. This strengthens admission to walking.

The second explicit diagnostic choice is the already-trainedbalanced4246
locomotion actor. Each actor must use its OWN actually passed common-physics
.5 report/CSV/interface. balanced4246 is not promoted: its21-case result13/21
and duty-gap improvement12.35% both fail the earlier registered experiment.
An explicit .5-only integration experiment does not revise those outcomes.
The original3947 case remains the matched control. No new training occurs here.

## Finite next execution

1. Run all new pure tests and preflights; freeze sources. Verify the selected
   actor's checkpoint/training/evidence and unchanged historical exports.
2. One52-interval no-video smoke per selected interface, originalstate132,
   seed20260918. Require actual receipt/source/interface/counter checks;
   smoke always has overall`passed=false` because it is not a complete flow.
3. Run one fullsame-start v2case forcontrol3947 and one forbalanced4246.
   Both use the same stronger readiness and unchangedcommon physics. Compare
   their actual recovery prefixes,readiness boundary and movement/stop, not
   independent stand/reset rollouts. Retain everyfailure.
4. Only a fully accepted diagnostic can proceed to front/oblique render checks.
   One sidecase is not left/right/back/upright coverage or generalization.
   Rendering must include the recorded initial1sPD and allphysical intervals,
   with no temporal cut,reset or splice. Separatecamera runs are labeled
   matched replays,not simultaneousviews.

This first bounded flow intentionally omitsrefinement,.8/1.0,movingpush/fall
and repeatedrecovery. Even if it passes,it does NOT satisfy the user's entire
naturalrecovery+trot+fastrun+fullreferenceworkflow request.

## Actual v2 completion

Serial4594 exited0. Both full cases contain1004 real control intervals:
50initial nominal-PD,254recovery,400movement,300stop (20.08s physical time).
Both reports have `diagnostic_passed_not_promoted`, all17checks true,
live-interface/source-end checks true and no reset attempts.

| Locomotion actor | Recovery-rest speed p95 | Movement vx | Movement vy | Stop speed p95 |
| --- | ---: | ---: | ---: | ---: |
| control3947 | .025052517 | .606867235 | -.118534556 | .003689851 |
| balanced4246 | .025052517 | .543320817 | -.048587775 | .007244752 |

All values are m/s. These are one state132/left start per actor, not independent
trials across falls. Both still use old1999/3547 recovery without the posture
refiner. Reports are in
`evaluations/20260917-continuous05-{control3947|balanced4246}-side-full-v2`.
This new speed-specific diagnostic does not reverse balanced4246's failed
13/21 wider screen or its failed gait-improvement criterion.

## Sensor identity audit and v3 replay

The first independent v2 audit failed: it used articulation body order to index
raw contact-sensor forces. Runtime measurements correctly used separately
resolved sensor IDs, but their identities were not written in v2's report.
No force-series matching or tolerance relaxation was used to hide this gap.
V2 sources and its failed audit receipt are retained. V2 is not an eligible
training prerequisite under the new audit.

V3 (`continuous_recovery_flow_diagnostic_v3_sensor_identity`) adds the actual
sensor body names and resolved foot/base indices, records articulation indices
separately, and requires exact start/end topology equality. Actual sensor foot
indices are[4,8,14,18], articulation indices[15,16,17,18]. It changes no dynamics,
controller, command, readiness or behavioral threshold.

Balanced4246 v3 nonvideo replay passed all17 checks and the complete independent
1004-interval audit. Its measured_rows.json SHA
`8edc7d57e81fc5558eed7da44faf7c375a974afaa9e66e89d6a2029bc36d5a6f`
equals v2 byte-for-byte. Report and independent_trace_audit_v1.json are in
`evaluations/20260917-continuous05-balanced4246-side-full-v3`.
The audit checks recorded bookkeeping/targets/phase/contact mappings and
metrics; it does not independently rerun the neural networks or reconstruct
body meshes. Front/oblique full recordings and visual review are still pending
at this update (18:14 local).
