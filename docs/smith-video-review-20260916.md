# Latest SmithNominal recovery videos — failed stand-up, real righting progress

Recorded and visually checked2026-09-16, completed around20:12 local. These are NEW recordings of the completed SmithNominal2000 run, not the older18:29 compilation. Model SHA256 `71e4c7a39ff81836d2ead1c4533988c696e5bd4a0fc939b8cef6b4f6daec1b1c`. No candidate has been promoted.

## Measured outcome

All nine20-trial groups have0 ever-valid stands,0 final valid stands and0 final geometry passes. Reports: [settled bank](../evaluations/20260916-smithnominal128x2000-heldout/model_1999.pt_recovery_metrics.json), [30degrees](../evaluations/20260916-smithnominal128x2000-angle30/model_1999.pt_recovery_metrics.json), [45degrees](../evaluations/20260916-smithnominal128x2000-angle45/model_1999.pt_recovery_metrics.json). The standard-library audit verified the aggregate/individual consistency of all three, not success or visual quality by itself.

Bank side/back starts are20/20eligible each, but only16/14unique source states. Their final mean heights are0.147/0.156m, with mean tilts18.66/8.20degrees respectively (derived per trial from the recorded gravity-vector error). Both groups have zero final four-foot-support states;39of40 have only one current foot force above5N. Near-upright orientation is real progress over remaining inverted, but is NOT standing recovery.

## Complete paired clips

Left is front, right is oblique. They are independently replayed, identically seeded conditions, NOT simultaneous cameras. This is a recovery-only diagnostic, not a continuous walk/fall/recover/walk controller and not a replacement for the independent locomotion2250 model.

| Start | Full paired video | Observed result |
| --- | --- | --- |
| Upright | [11.00s,275frames](../evaluations/20260916-smithnominal128x2000-video-review/smith-upright-paired/model_1999_upright_front_oblique.mp4) | Initially raised body collapses; legs fold into abnormal low support. FAIL. |
| Settled side | [11.04s,276frames](../evaluations/20260916-smithnominal128x2000-video-review/smith-heldout-paired/model_1999_side_front_oblique.mp4) | Rolls toward body-up, then remains crouched with incorrect foot placement. FAIL. |
| Settled back | [11.04s,276frames](../evaluations/20260916-smithnominal128x2000-video-review/smith-heldout-paired/model_1999_upside_down_front_oblique.mp4) | Visibly turns over within the shown sequence, but stays low with folded legs rather than standing. FAIL. |

All source frames remain; no failure intervals or endings were removed. Bank clips retain one post-handover/pre-policy frame. The1s nominal-position PD handover occurs BEFORE footage/policy time; it is not passive zero-torque settling. All three paired clips are1920×540,H.264,25fps. All six960×540raw clips and three paired clips were completely decoded and their frame counts checked. Main visually examined both views at0,1,3and10.92seconds, including each failed ending.

Paired video SHA256:

- upright: `d314e1ffd98cfd04644b83160c6fbf673af8c2f36e9db91c31364940e8b734f0`
- side: `0d7294d35908bcffb3f3f4dba779d7174beef39600e3be24fc77b8fd300657c8`
- back: `55b87a50af5e09ace5d051c470edbb013aae756db2aa3c74d39102dd28934be7`

The four original single-trial reports and paired-view provenance remain beside the videos. Each single-trial result is also failure and is not added to the20-trial batch denominator. Local raw videos, traces and QA sheets stay on E:. The recorder output directory already exists; do not rerun it over this evidence.

Next work is the separate [10-versus30 stand-weight continuation pair](smith-standweight-pair-20260916.md), not an assertion that the present policy is normal. Historical clips have not been deleted because the user's requested cleanup range is not yet confirmed.
