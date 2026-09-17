# Post-recovery refinement video review

## Scope

These are recovery/posture-only development replays, **not** a continuous
recovery-to-walking/running demonstration. Each camera records trial 0 from a
20-environment replay at seed20260918. Front and oblique views are separate
deterministic replays, not simultaneous cameras. The pairing tool compares
their measured trajectories and preserves all276frames at25fps.

The videos begin after the existing1s nominal-PD preparation. They do not show
that preparation or a naturally occurring fall. Recovery uses original1999/3547;
only after150 completed strict-standing intervals does posture actor3746 take
over. No physical or history reset occurs at that new actor boundary.

## Side recovery — visually inspected

Root inspected both source-view contact sheets at0,.5,1,3,6,10.96s.
These inspection paths are local-machine artifacts, not public repository links:

- `E:/IsaacLab/artifacts/recovery-20260917/refinement20-side-front.jpg`
- `E:/IsaacLab/artifacts/recovery-20260917/refinement20-side-oblique.jpg`

The robot visibly begins fallen, rolls, rises and remains raised in the terminal
samples. Front feet are on their correct sides, without the previous recovered
cross-legged stance. The final front-view lateral placement is more even than
the early standing sample. The oblique view still shows staggered fore/aft foot
placement; this is not a completely symmetric or certified animal-like stance.
Still frames do not establish smoothness at every intermediate frame, impact
quality, torque peaks or moving gait.

Pair: [side recovery, front + oblique](../evaluations/20260917-refinement20-side-paired/supported_refinement_side_front_oblique.mp4).

SHA256: `68de62c6bfdd45829b0610dcc280c09fb6c31d416510f5d11dac7893065542ae`.
Actual276frames,25fps,1920x636; no cuts, speed change or video concatenation.
The banner explicitly excludes a walking/trotting/running-flow claim.

## Back recovery — visually inspected

Root also inspected both back-recovery source contact sheets at the same six
times (`refinement20-upside_down-front.jpg` and
`refinement20-upside_down-oblique.jpg` in the artifact directory above).
The initial upside-down state, roll and raised terminal stance are visible.
Front feet remain uncrossed in the terminal samples. The final front stance is
laterally more balanced than at6s, but fore/aft foot staggering remains apparent
in both views. This is a useful lateral-symmetry improvement, not a claim of a
fully natural or exactly symmetric recovery motion.

Pair: [back recovery, front + oblique](../evaluations/20260917-refinement20-upside_down-paired/supported_refinement_upside_down_front_oblique.mp4).

SHA256: `91021a2b857a6b455a9a7dfc37f6a2cf68df0f16881f556d0acbb5baee382213`.
Actual276frames,25fps,1920x636. Pairing passed the same complete-frame and
matched-trajectory checks as the side replay. Only trial0 is visually sampled;
the numerical20-trial result is not20independent visual reviews.

## Required before the requested final video

A separate [continuous recovery-walk-stop diagnostic](continuous-flow-video-review-20260917.md)
is now recorded for one bank-initialized side-fall state at command0.5m/s,
within one physical episode without actor-boundary resets or history clearing.
It uses roll1999/stand3547/balanced4246 and does **not** include posture
refiner3746. It is not natural-recovery, trot or fast-running acceptance;
integrating this refiner into a complete moving workflow remains untested.

The locomotion control/balanced21+21-case screens are complete:11/21 and13/21
passed, respectively, and the gait duty-gap improvement was only12.35%, below
the predeclared25% criterion. [Both candidates remain rejected for promotion](balanced-gait-results-20260917.md).
Trot/run labels require measured repeated foot cycles and actual speed, not
just command names. Existing recovery, turns, disturbance handling and stopping
must remain separately validated; these posture replays cannot replace those tests.
