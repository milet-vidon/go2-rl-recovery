# Supported startup: preserve ordinary standing without changing fallen recovery

The fixed, measurement-only startup selector completed its six development comparisons on September17. This is a two-actor simulation controller change, **not new training, a single-policy result, or complete recovery/walking/running reproduction**. The separately tested roll-mirror intervention is deliberately absent here.

| Identical20-trial development set | Selector OFF | Selector ON | Starts selecting stand immediately |
| --- | ---: | ---: | ---: |
| Ordinary supported near-upright startup |12/20|20/20|20|
| Settled side bank |9/20|9/20|0|
| Settled upside-down bank |20/20|20/20|0|

Every value is a strict **final valid standing hold**, not merely upright orientation or foot-contact count. All original trials stay in the denominator. Side/back use only16/14unique source states, repeatedly examined during development; these are not independent unseen20-state tests. The upright starts are still moving after the original1s nominal-position PD preparation and are not eligible settled-fallen starts.

## What changed

At the actual policy-start boundary, choose the frozen standing actor only if the [predeclared measurement predicate](handoff-supported-startup-preregistration-20260917.md) passes: fresh current four-foot support, valid original geometry, no base contact, small tilt, bounded height and root motion, and explicitly not an eligible settled-fallen state. Latch this selection. All other starts use the original rolling actor and unchanged one-way handoff gate.

Startup selection is not a handoff, not success, and not a state/history reset. The standing actor remains selected throughout selected episodes, with zero genuine handoff events. No extra PD, mirror, retry, target ramp, changed forces or relaxed scoring was added. The original1s preparation remains explicitly disclosed.

Strict final standing still requires normal foot/knee layout and per-joint offsets, all four CURRENT vertical foot forces>5N, no base contact, height0.30–0.55m, low linear/angular velocity and upright error, simultaneously held for at least3s and valid at the ending.

## Observed improvement and limits

The20 ordinary starts all reach and retain the original standing actor's established behavior. Median onset of the eventual qualifying standing interval is approximately0.12s after policy start; this is **not** a completed3s hold at0.12s and not a fallen-recovery time. Final uninterrupted holds range10.82–10.94s, with final heights0.321747–0.324371m. All20 pass the original geometry check. For trial0, front-foot body-y is+0.175700/−0.126687m; the feet are on their correct sides, though perfect left/right symmetry is not claimed.

This fixes the observed loss from unconditionally invoking a roll expert on a supported ordinary start. It does not resolve this no-mirror controller's11failed side trials. Do not sum this experiment's upright20 with a different mirror controller's side20 and call it one tested60/60controller. A separately implemented and tested combination is still required, followed by fresh initial states/seeds, continuous locomotion integration and video review. Fast-running training remains unsuccessful; the speed-weight A/B candidates were both rejected under their existing criteria.

## Exact controls and provenance

Reports are under [the six-case output directory](../evaluations/20260917-supported-startup-v1/). Every case contains model_1999.pt_recovery_metrics.json, the full trial0 CSV, all-trial first-action and stored transition-neighborhood diagnostics, generated source, invocation/source hashes and simulator log.

- OFF upright/side/back compare exactly to frozen hard controls:686,242/626,513/704,062common scalar values, zero mismatches.
- ON side/back have no selected starts and preserve626,513/704,062common values exactly, including the stored action/physical neighborhoods.
- ON upright matches the same-start stand3547-only reference on31,911common values: all original result fields, applicable metadata, initial states and trial0 CSV. The old stand-only reference has no all-trial step trace, so full all-trial trajectory equivalence is **not** claimed.
- New input/predicate tests18/18 and entry/logger/serialization tests12/12 passed. Runtime assertions check actual issued action, naturally advanced history and executed targets. Explicit first-action records exist for every trial; selected logger rows cannot fabricate a handoff/ramp event. Invalid input aborts the batch rather than silently routing it.

Frozen roll1999 SHA `71e4c7a39ff81836d2ead1c4533988c696e5bd4a0fc939b8cef6b4f6daec1b1c`; stand3547 SHA `5527c4053cb6ef8e6616da65b7dc32a9464eb8d1366f4966a4eb4b65bc913fdb`.

New adapter SHA `30e63d6f60261d3e163c212568613d0f8941da9ce2d8b50676ad6fc281800c60`; wrapper `f2c93dd1ed307cf4f490693bfb4dac1fec181b12c83cfaa2ca05ab2260c37c4f`; comparator `0a43eb401871b410465a2cfe506d20b52de7d89f746cf94641d1e5a60a784d61`. The adapter generates a finite exact-anchor derivative of the SHA-pinned old hard entry in memory and saves the generated source separately; it never edits that historical entry. All16 preserved exported models remain unchanged.

Both front/oblique20-environment video replays completed, showing only trial0. Main checked their entire20-trial result objects against the non-rendered supported-upright run: exact equality. Main also actually viewed uncropped samples at0,0.12,0.5,1,2,4,8,10.96s in BOTH views: front feet visibly separated/noncrossing, body clear of ground and stable posture, with mild residual width asymmetry. This is sampled visual inspection, not every-frame certification or an animal-gait/continuous-recovery-flow claim. Original videos remain in `evaluations/20260917-supported-startup-upright20-front` and `evaluations/20260917-supported-startup-upright20-oblique`; source filenames are `model_1999.pt_upright.mp4`.

The full paired demonstration (local-only artifact, not included in this GitHub snapshot: `evaluations/20260917-supported-startup-upright20-paired/supported_startup_front_oblique.mp4`) is complete:276frames,25fps,1920x540,11.04s, SHA `57043886568c2869c3c3370e604a3a5e4d8f6ea3779d0831c7c4dc9bd03a7b78`. Left front/right oblique are independent matched-condition replays, not simultaneous cameras. The new pairing utility verifies every report field except view, byte-identical trial0 CSV/stored transition traces, actual actor/source hashes and every decoded input/output frame count. It does not trim time or alter the source videos. `paired_metadata.json` records provenance and limitations. Main also viewed paired samples at0,0.5,2,10.96s. The eight new implementation/test/pairing source files were archived with copy-hash verification at `E:/IsaacLab/artifacts/recovery-20260917/20260917-supported-startup-v1-source`; all16 exported baseline model hashes remain unchanged after the completed video batch. No simulator remains active at this handoff boundary.

For later locomotion work, primary [Rapid Locomotion](https://arxiv.org/abs/2205.02824) and the author's [RewardThresholdCurriculum implementation](https://github.com/Improbable-AI/rapid-locomotion-rl/blob/main/mini_gym/envs/base/curriculum.py) were revisited: the implementation updates successful command-bin and neighboring-bin sampling weights only when both linear and angular reward thresholds pass. This supports testing a jointly gated command curriculum, not assuming a larger scalar tracking weight will work. Our failed A/B test is not a reproduction of that curriculum, online system identification or the paper's hardware results; no new curriculum has been implemented or trained here.
