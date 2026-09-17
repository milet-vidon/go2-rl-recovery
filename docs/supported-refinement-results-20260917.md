# Measured preserved recovery then posture refinement

All six OFF/ON development cases (ordinary upright, side, back;20draws each) completed and passed their full-trajectory audits. Each OFF andON case retains20/20strict ending holds; eachON case also has20/20ending3s controlled by the refinement actor. New side/back front-oblique posture-only videos and source-view visual review are now complete; see [visual review](refinement-video-visual-review-20260917.md). A separate [continuous recovery-walk-stop diagnostic video](continuous-flow-video-review-20260917.md) was completed on2026-09-17 at command0.5m/s, for one bank-initialized side-fall state. It uses roll1999/stand3547/balanced4246, **not posture refiner3746**. Neither that diagnostic nor these posture-only replays establish full-task, natural-recovery, trot or fast-running acceptance.

Original roll1999/stand3547 perform the entire recovery and150consecutive strict valid control intervals. Only the following action switches to trained3746, without PD/ramp/reset/history clearing at that boundary. Initial1s nominal-PD preparation is unchanged and disclosed. The three-actor combination is not a single newly learned recovery policy.

OFF back20 exactly reproduces original results,934063common scalars,trial0CSV and all old stored neighborhoods. Full11000actual rows independently satisfy action/target/history/geometry/support/new150-intervalgate consistency. ON back20 also retains20/20ending strict holds; every trial has236–257consecutive valid intervals actually controlled by the new posture actor. ON firstrefinement action is index293–314(5.86–6.28s policytime), after old recovery has already stood strictly for3s. No performance inference from an unexecuted actor proposal.

Current-source ON evidence: [back-recovery full-trajectory audit v3](../evaluations/20260917-supported-refinement-v1/supported-upside_down/full_trajectory_audit_v3.json). It re-audits OFF in the same process and checks actual complete prefixes up to each trial's first new action. The v2 remains historical evidence from the preceding auditor version; v3 was generated for the current-source video pairing without repeating the simulation. The older v1 audit is superseded: code was strengthened during its execution, so its end-of-run source hash was not a reliable loaded-code identity. The auditor now checks its own source at start/end. No simulation/report was edited for this correction.

Terminal3s mean geometric differences include ALL3000samples, not selected valid frames:

|Measure|Original|Refinement|Interpretation|
|---|---:|---:|---|
|Front left/right lateral bias|62.00mm|12.47mm|79.9% reduction|
|Hind lateral bias|9.30mm|7.42mm|20.2% reduction|
|Front paired fore/aft offset|39.80mm|48.57mm|22.0% worse|
|Hind paired fore/aft offset|69.85mm|81.19mm|16.2% worse|

The result improves lateral symmetry while preserving these recovery holds, but does NOT establish all-axis natural stance. Further style training/visual inspection remains necessary. Repeated development bank draws are not fresh generalization evidence. Old roll-to-stand gate consistency relies on frozen source and exact OFF/ON prefixes; the added150-interval refinement gate is independently recomputed. Torque statistics only measure50Hzcontrol boundaries, not all200Hzphysical substeps.

## Completed ordinary and side comparisons

All values are terminal3s means across all3000samples per case, in millimetres; no valid-frame filtering. OFF→ON:

| Pose | Front x mismatch | Front lateral bias | Front z mismatch | Hind x mismatch | Hind lateral bias | Hind z mismatch |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary upright |19.94→47.95|45.18→3.72|2.65→4.75|37.44→26.97|10.14→7.39|2.48→4.21|
| Side |43.45→56.28|55.45→8.21|7.27→1.64|31.25→34.53|14.09→8.60|7.70→1.60|
| Back |39.80→48.57|62.00→12.47|4.55→2.64|69.85→81.19|9.30→7.42|5.08→3.88|

The substantial front lateral improvement is therefore partly offset by worse fore/aft alignment in every pose. A user-visible claim of more natural overall stance still needs visual review and further targeted training. Ordinary upright is a standing-start control, not a successful fallen recovery. Side20 andback20 reuse16and14unique development states respectively; this is not60independent unseen falls.

Authoritative additional audits: `off-upright/full_trajectory_audit_v2.json`, `supported-upright/full_trajectory_audit_v1.json`, `off-side/full_trajectory_audit_v1.json`, `supported-side/full_trajectory_audit_v1.json` in the same evaluation root. Current auditor SHA `a2d44a311b12ed14bcc7d220ab78389cb67ffab6a413297636994b362a6171b8`,14CPU tests passed. The first OFFupright audit failure was an incorrect requirement for a fallen-state-bank key in an ordinary-upright report; only that distinction was corrected, with the completed simulation reused unchanged.

The legacy neighborhood event text may say `raw ramp completed` at the new phase edge although no ramp exists. Full-trace actor masks, actual actions and chronology are authoritative, and the audit explicitly discloses this descriptive logger limitation. Any future complete workflow including this posture refiner must show actual uninterrupted recovery, refinement, motion and stopping, not concatenate these independent trials. The separate low-speed diagnostic does not validate that untested combination.
