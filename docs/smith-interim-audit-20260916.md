# Smith reward pilot: interim audit, NOT acceptance

2026-09-16 19:27 local. The single128-environment/2000-iteration run is still active. At this snapshot1220iterations/3,747,840environment steps have completed. No new simulator, training restart, source installation or model promotion was performed during this audit.

An independent read-only CPU audit inspected TensorBoard through iteration1161 and then exited. Main rechecked all11 runtime source hashes and the actual saved reward-only env/agent configuration; both still match. The log contains no current physics/CUDA error or traceback. Previous-crash minidump text is an old startup artifact, not evidence that this run crashed.

## Training terms, not success rates

These are the existing weighted episode-reward log values averaged over100iteration windows. They mix bank and controlled starts and do not identify which pose classes improve. The earliest0–99window includes initialization/incomplete episodes and is not the primary comparison.

| Logged term | 500–599 | 1000–1099 | 1062–1161 |
| --- | ---: | ---: | ---: |
| smith_roll | 4.7844 | 6.2798 | 6.4080 |
| smith_stand | 1.2658 | 0.7012 | 0.7296 |
| dof_pos_limits | -0.00124 | -0.13413 | -0.14978 |
| dof_torques_l2 | -0.01627 | -0.00586 | -0.00623 |
| dof_acc_l2 | -0.23877 | -0.10227 | -0.09335 |
| action_rate_l2 | -0.25677 | -0.20668 | -0.20675 |
| mean_noise_std | 1.1541 | 1.2523 | 1.2615 |

Hypothesis only: the policy may increasingly optimize torso orientation while remaining low or near extreme joint configurations. Higher roll, lower stand and stronger joint-limit penalty are a warning, not proof of this pose. The stand term's decline cannot distinguish fewer36degree-gate crossings from poorer joint posture/velocity with the current two scalar logs. Final deterministic evaluation and actual front/oblique images must answer that question. A600step episode or timeout=1 is not a recovery.

At the same1000–1099window, the old NominalTarget control had limit penalty-0.03314, torque-0.01031, acceleration-0.07913, action-rate-0.11589and noise0.9501. Only these unchanged regularizers are directly comparable; neither total reward nor value loss nor old uncrossed_stand versus new smith_stand can be compared as a success measure. The new method has not demonstrated general improvement.

## Checkpoint and resource check

`model_1000.pt` at `E:/IsaacLab/repo/logs/rsl_rl/unitree_go2_recovery/2026-09-16_19-02-40_20260916-smithnominal128x2000/` has SHA256 `c5a9c0781e7d6945b13e7c6ac5e31bc56e0b33fa0c19b84bee83a7378ee8e556`. Its iteration field is1000; all17model tensors/80,281elements are finite. Action std spans1.10172–1.33343with no nonpositive values. Optimizer tensors were not independently audited, and finite weights do not establish performance.

Host memory was tight: free physical memory around250MB and free virtual memory around1.2–1.6GB in these snapshots; training process private allocation around7.95GiB. GPU usage was around5.16/8.19GB. These are transient measurements, not a diagnosed memory leak or guaranteed future capacity. Keep ONE simulator, avoid heavy parallel analysis and do not close unrelated user apps or change system/pagefile settings. The training still advanced at roughly1.4–1.7seconds/iteration without recorded physics errors. Existing checkpoints are preserved.

Next: finish this fixed-budget run and its automatic strict20-trial suites, then record full failed or successful upright/side/back replays in both views. Do not adjust the active reward mid-run, extend a failed branch unchanged, relax geometry, or claim reference-video reproduction.
