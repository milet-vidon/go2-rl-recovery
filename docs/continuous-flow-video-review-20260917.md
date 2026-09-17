# Continuous low-speed workflow: actual video review

2026-09-17 18:19 local. This is a development diagnostic, not completion of the
user's natural recovery + trot + fast-running request.

## What was actually recorded and checked

- Same bank state132/left, seed20260918, one robot, common recovery physics,
  roll1999 -> stand3547 -> experimental locomotion balanced4246.
- Initial bank placement occurs before t0. Every subsequent physical interval
  is retained:50 nominal-PD preparation,254 recovery,400 command0.5 movement,
  300 zero-command stopping. No pose transplant/reset after t0.
- Separate cold-start front and oblique recordings, not simultaneous cameras.
  Each has1005 frames at50fps,960x540. Every measured row equals the independently
  audited nonvideo v3 trace byte-for-byte (1004 intervals,20.08s).
- All17 behavior checks pass for this one trajectory. This is not coverage of
  other starts, movement-induced falls, repeated recovery, .8/1.0 or fast running.
- All source/video/artifact hashes, frame ledgers, complete decodes and matching
  trajectories were checked before/after pairing. Native frames are placed
  side-by-side, no crop, temporal cuts, speed change or added title frames.

Video: [front + oblique](../evaluations/20260917-continuous05-balanced4246-side-paired-v3/continuous_flow_front_oblique.mp4)

Output SHA256 `9a2c0f277c675fa509f5c4811003b90da6d5b6bf60a2de9888a2fb20f5b477de`.
1005 frames,1920x684 including144px separate banner,20.10s media length.
The final frame depicts20.08s physical time; t0 adds one displayed frame.
Pairing receipt is in the same directory. Full source reports/rows remain in
`evaluations/20260917-continuous05-balanced4246-side-{front|oblique}-v3`.

## Visual observations, not just metric labels

Root actually viewed both uncropped contact sheets at0,1,1.5,2.5,4,6,6.5,8,12,
14,17,20s. Sheets are `artifacts/recovery-20260917/continuous05-{front|oblique}-v3-qa.jpg`.

- Initially on the side. At1.5–2.5s the robot rolls and braces with widely
  extended/asymmetric legs. This transition is still conspicuous and is not an
  animal-like recovery acceptance.
- At4–6s the trunk is raised and the front feet remain on separate sides;
  no terminal front-leg crossing is visible. Fore/aft staggering remains.
- At6.5–14s it moves with alternating foot placements; the sampled frames do
  not establish a natural trot, aerial phase or fast running. Long/asymmetric
  support and slipping remain relevant training targets.
- At17–20s it is supported and stationary with separated front feet. Full
  quantitative ending150-sample strict hold is checked separately.
- Whole robot and feet remain in frame. White robot and gray ground are visible
  despite logged missing decorative grid textures. Those render warnings do
  not demonstrate a dynamics error. The t0 overlay's height0.000 is a display
  placeholder before a completed measurement interval, not a measured base
  height; interval data and acceptance begin with the real first interval.

This flow intentionally uses the retained old recovery/stand actors, not the
posture refiner3746. Previously rejected broader/gait screens remain rejected.
The smoother refinement transition and natural trot/fast-run workflow still
require separate physical training, evaluation and new full-flow recordings.
