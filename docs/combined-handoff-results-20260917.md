# Combined startup and roll reflection: measured development results

The 12-case frozen-controller experiment completed on September17. All predeclared common-result/CSV/stored-neighborhood comparisons passed with zero mismatches. Original actors and all16preserved model exports remain unchanged. This is an inference/controller composition, not further neural-network training or a single-policy result.

| Startup selector | Roll reflection | Ordinary upright | Side falls | Back falls |
|---|---|---:|---:|---:|
| OFF | OFF |12/20|9/20|20/20|
| ON | OFF |20/20|9/20|20/20|
| OFF | ON |12/20|20/20|20/20|
| ON | ON |20/20|20/20|20/20|

All counts are final valid standing: unchanged body-height/orientation/motion checks, noncrossing foot/knee and joint geometry, four simultaneous current vertical foot contacts, no base contact, and at least3seconds continuous valid hold at the end. Ordinary upright cases are not fallen recovery. The same BOTH-ON implementation actually ran all three pose cases; these are not sums of separate controllers' successes.

For BOTH-ON, side20draws include16unique bank states, with11mirror selections; back20draws include14unique states and no mirror selections. All side/back trials qualified as settled fallen under the explicitly retained1second nominal-position PD preparation. Median onset of a subsequently confirmed valid hold: side1.38s, back3.04s. These onset values do not mean a3s hold was already completed at that time. Final heights: side0.313223–0.326568m; back0.308058–0.326310m.

The bank has been repeatedly inspected during development. Thus these figures are not an independent generalization estimate. No new terrain/friction, arbitrary natural falls, uninterrupted recovery-to-walking, fast running or hardware test has passed here. The speedgate3996 branch separately failed old-function retention and remains rejected.

## Reproducible evidence

- Summary: `evaluations/20260917-handoff-combined-v2/summary.json`, SHA `62db006b67ebac5fa20ef977d3f4a3320afcc8b1c9605358b9307d73ccb5e03a`.
- First OFF-upright simulation is preserved in v1 and explicitly reused after its invocation sources/inputs were rehashed. New strict receipt: `exact_comparison_reaudit_v2.json`,686241common scalars checked.
- Remaining11cases reside under v2. Every receipt records actual baseline/candidate/trace hashes, checked scalar count, exact mismatches and measured interface audit.
- Frozen roll1999 SHA `71e4c7a39ff81836d2ead1c4533988c696e5bd4a0fc939b8cef6b4f6daec1b1c`; stand3547 SHA `5527c4053cb6ef8e6616da65b7dc32a9464eb8d1366f4966a4eb4b65bc913fdb`.

The first attempt stopped on a newly added invalid comparison between two different legacy tilt definitions (`acos(-gz)` versus `acos(-gz/||g||)`). The failed receipt is retained. The correction removed only that cross-definition equality: raw quaternion/velocities/joints/height/contact forces and duplicate first records remain exact, and each derived tilt is still compared to its own historical reference. No physics, source actor, acceptance threshold or numeric trajectory tolerance changed.

## Visual follow-up

All six rendered BOTH-ON replays and three full paired videos are complete. Each11.04-second pair contains276decoded frames at25fps,1920x540, with front left/oblique right. Reports exactly match unrendered BOTH-ON reports except camera label. All550actual control CSV rows match exactly; the frozen video path also writes one pre-action row, explicitly validated rather than mistaken for a control-step difference. The two camera-source551-row CSVs are identical. Independent audit: `E:/IsaacLab/artifacts/recovery-20260917/combined-rendered-audit-v2.json`.

- Side recovery (local-only artifact, not included in this GitHub snapshot: `evaluations/20260917-combined20-side-paired/combined_side_front_oblique.mp4`), SHA `1dc05fe5abaa79e68a58c877ac1b7abb085f29f276d94a5da653405fb0216406`.
- Back recovery (local-only artifact, not included in this GitHub snapshot: `evaluations/20260917-combined20-upside_down-paired/combined_upside_down_front_oblique.mp4`), SHA `83cda7a854ccf313d004222aba97c0aaf5499fcef0b98348c00a63666299e4fe`.
- Ordinary supported startup (local-only artifact, not included in this GitHub snapshot: `evaluations/20260917-combined20-upright-paired/combined_upright_front_oblique.mp4`), SHA `a1b907bec661681767466fa227e1ebce689003494a0795e172637c1c7a059cb3`.

Main actually viewed all6source contact sheets at0/.5/1/2/4/10.96seconds. Final front feet remain separated, body clear of ground, stance retained; residual lateral-width asymmetry remains, particularly after back recovery. During rolling, folded/cross-midline configurations are visible and not counted as valid standing. This sampled visual check does not certify every frame. Each video films only trial0of20; they are separate matched-condition replays, not simultaneous cameras or an uninterrupted recovery/walking/running sequence.

The outer recording command exited1 only after all6successful wrapper diagnostics due to a malformed trailing PowerShell redirection. Per-case evidence is preserved and independently verified; no repeat simulations were needed. A first CSV-audit failure also remains preserved, followed by the explicit boundary-aligned exact comparison described above. Pairer8tests and boundary-mutation test pass; final pairing/QA command exited0. Existing16model files are unchanged; byte preservation alone does not establish integrated locomotion retention.
