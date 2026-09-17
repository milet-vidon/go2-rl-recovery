# Development checkpoint snapshot — 2026-09-17

These are byte-verified copies of the actual simulation checkpoints used in the latest diagnostics, with the corresponding `env.yaml` and `agent.yaml`. They do not replace the retained recommended models. Each `.pt` is an RSL-RL training checkpoint, not a hardware-ready controller or a single combined policy. Only load checkpoints from sources you trust.

| Snapshot | Role and limitations | SHA256 of checkpoint |
| --- | --- | --- |
| [roll1999](roll1999/model_1999.pt) | Initial roll/reorientation actor; insufficient for normal standing alone. | `71e4c7a39ff81836d2ead1c4533988c696e5bd4a0fc939b8cef6b4f6daec1b1c` |
| [stand3547](stand3547/model_3547.pt) | Stand actor used after roll1999 in the supported handoff protocol. | `5527c4053cb6ef8e6616da65b7dc32a9464eb8d1366f4966a4eb4b65bc913fdb` |
| [refiner3746](refiner3746/model_3746.pt) | Posture-only refinement branch; NOT used in the low-speed continuous video. | `f40772c228996400a9813ad244b48509737dabf2c31af001144de34283433b9e` |
| [balanced4246](balanced4246/model_4246.pt) | Experimental actor in one 0.5 m/s continuous diagnostic; broader gait screening rejected, NOT promoted. | `882baefd00193c4b71bef2a47b5cb7382c37dd9e87b48f7f5863eaa364baebdc` |

Original run directories, relative to the development machine's `repo/logs/rsl_rl/`:

- roll1999: `unitree_go2_recovery/2026-09-16_19-02-40_20260916-smithnominal128x2000`
- stand3547: `unitree_go2_handoff_stand/20260917-handoff-stand128x100`
- refiner3746: `unitree_go2_handoff_stand/20260917-natural-stand128x200-first`
- balanced4246: `unitree_go2_flat/2026-09-17_16-33-35_20260917-gait-balanced-128x300-first`

Read the [publication scope](../../../docs/github-snapshot-20260917.md), [continuous-flow review](../../../docs/continuous-flow-video-review-20260917.md), [posture results](../../../docs/supported-refinement-results-20260917.md), and [failed broader gait screening](../../../docs/balanced-gait-results-20260917.md) before using these snapshots. This archive does not itself make the host-bound experiment launchers portable, and does not claim natural trot, fast running, arbitrary-fall recovery, or real-robot validation.
