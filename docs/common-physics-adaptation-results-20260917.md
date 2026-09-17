# Actual common-physics adaptation training

Training and the finite21-case screen completed. Candidate4046 passes15/21 and is not eligible to replace the preserved model. No fast-running/full-flow claim.

The independent revised smoke16x2 completed, then formal128x100 independently resumed original3947 with its exact actor/critic/std/Adam. The final block used307200 environment steps,2400 control intervals and actual Adam step81120 for all17 parameter states. All68 model/optimizer tensors are finite. Original16 portfolio exports plus roll1999/stand3547 remain unchanged.

- Formal directory: `E:/IsaacLab/repo/logs/rsl_rl/unitree_go2_flat/2026-09-17_15-40-04_20260917-commonphysics128x100-first`.
- Final4046 SHA: `52469f49ad5f62f632bc94aa8947311ad37a20a5d611ffb3a6c2eaaf9bd79085`.
- Training receipt SHA: `a056f06bf6e0db274a24c27768390cb1ae6fba4f6c83930e61c06faa65de5ae5`.
- Reviewed trainer SHA: `fa6551d82994273ad8d0ea1cd2234cb0743454066af43706a1187a87db21387f`.

The actual full TRAIN configuration changes only self-collision activation, nominal soft-clamped joint targets and removal of additive base-mass randomization. Original commands/rewards/noise/upright reset/pushes/PPO are retained. Material and CoM were already fixed; the receipt lists them as unchanged.

First smoke also completed, but the first formal attempt stopped at CPU preflight because optimizer betas are a Python tuple and deserialize from JSON as a list. Actual metadata values and model SHA matched. Only JSON transport canonicalization was added; tests preserve rejection of changed values. Because executable source changed, a new independent same-source smoke-r2 was run, not relabelled from the old run. Old artifacts were retained.

Fourteen trainer CPU tests and thirteen candidate-evaluator tests pass. The new evaluator authenticates the actual formal final model, full YAML, initial/final ledgers, generated trainer, parent and same-source smoke. Its physical evaluator source is exactly the frozen retention source; only candidate identity/provenance changes in a private module. It verifies complete evidence again after simulation and retains the separate straight-drift gate.

First physical test is the previously failing full-recovery-physics0.8m/s case, fixed seed20260909. If retained, evaluate0.5, turns/pushes and original-physics old functions before integration. Repeated seeds that produce byte-identical physical traces do not supply diverse-start evidence. Higher-speed targets, gait quality and genuine uninterrupted recovery/walk/trot/run/stop still require their own measured tests.

First actual0.8case completed and passed all13old criteria plus the unchanged extra straight-drift gate. Actual bodyvx0.834921,vy-0.040053,yaw-0.098907rad/s,slip0.087185; all4rest verticalcontacts/geometry andquietstop pass. Relative to old3947in the SAME combinedphysics, speederror/lateraldrift/slip improve, but yaw-rate magnitude worsens from0.02556to0.09891rad/s. Threshold pass is not visually straight, natural fast-running certification. This is one deterministic physical trajectory. Full finite21case screen now follows, with no model promotion:6commonphysics command cases at one seed and15originalphysics oldfunction cases at three seeds; the already-completed.8case is revalidated and reused, not simulated twice.

## Final screen — completed16:20:59local

Serial session97924 exited0; summary `evaluations/20260917-commonphysics4046-first/summary.json` has21actual cases,15passes. Recovery-compatible physics passes3/6(.8,left,1.0), original physics passes12/15. Failed common cases: .5 andpushed.5 fail foot lift; right fails lateral tracking. Failed original cases: .8seed09 fails tracking andseparate straight drift; leftseed09 fails speed/yaw tracking; leftseed11 fails yaw tracking. No thresholds changed.

The 1.0common-physics development case reaches bodyvx.972257m/s andslip.104876m/s, but this one measured speed success does not certify fast running or overcome lost old functions. It is partial physical adaptation, not evidence to abandon common physics. A separate matched128x300 control/one-term timing-variance experiment is specified in `commonphysics-gait-next-experiment-20260917.md`; final checkpoints only, no cherry-picked intermediate selection. Baselines remain unchanged.
