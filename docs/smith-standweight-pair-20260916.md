# Smith stand-weight continuation: bounded paired experiment

Status20:18local: ACTIVE. Both16env×2iteration smokes completed and their actual configs passed;17model and51optimizer tensors per checkpoint are finite. Formal orchestrator39467 started20:16:24, w10 PID33584/run `2026-09-16_20-16-37_20260916-smithstandpair128x1000_w10`; w30 is queued in the same finite script. The preceding2000-iteration run and all new videos are complete and visually checked. Check [live status](training-status-20260916.md) before continuing; do not start a duplicate. Simulation only; expectations are not met.

## Evidence and hypothesis

The completed SmithNominal final1999 has SHA256 `71e4c7a39ff81836d2ead1c4533988c696e5bd4a0fc939b8cef6b4f6daec1b1c`. All nine 20-trial groups (heldout side/back, controlled30 upright/side/fore-aft, controlled45 upright/side/fore-aft/back) have zero ever-valid stands, zero final valid stands and zero final geometry passes. These are failures, not successful recovery videos.

Nevertheless, the fallen-bank final body orientation improves: mean gravity error is0.323 for side and0.143 for back, while mean body height remains0.147/0.156m. Mean per-trial maximum joint offset is1.728/1.233rad.39/40 final states have only one foot above the current vertical-force threshold; none have four. Controlled30 upright also collapses: mean height0.155m and mean maximum joint offset1.297rad. The immediate bottleneck is standing/support after righting, not simply an inability to move out of inversion.

The [official pinned Smith reward](https://github.com/lauramsmith/fine-tuning-locomotion/blob/583f1de43e91cdd24d632d783872528eb1337480/motion_imitation/envs/env_wrappers/reset_task.py) gives equal roll and gated-stand weights. This experiment changes their relative weighting; **it is our ablation, not the paper's original setup or a complete reproduction**. Multiplying stand also multiplies height, joint-pose and joint-velocity shaping, so a quiet low crouch can also receive more reward. Improvement is a hypothesis to test, not an assured fix.

## Fixed comparison

| Arm | Roll weight | Gated stand weight | Additional iterations | Environments |
| --- | ---: | ---: | ---: | ---: |
| A, continuation control | 10 | 10 | 1000 | 128 |
| B, stand emphasis | 10 | 30 | 1000 | 128 |

Both load the SAME full parent1999 checkpoint, including optimizer, critic and policy noise, from `2026-09-16_19-02-40_20260916-smithnominal128x2000`. Neither loads the other arm. Seed42, nominal+.25 action targets at50Hz held across four physics substeps, soft target clamps, Go2 physical limits,60% bank+24% upright+16% shallow-drop mixture, observation noise, generic regularizers and PPO remain unchanged. Each arm adds3,072,000 environment steps. RSL-RL resumes at iteration1999; the expected final filename is `model_2998.pt`, not2999.

Only the optional `train.ps1 -SmithStandWeight 10|30` Hydra scalar differs. Omission preserves existing commands exactly. No installed task code is changed. The saved complete env/agent YAML schema is checked by `check_smith_standweight_config.py`, including exact resume identity; only the declared scalar, count/budget and runtime paths/name may differ. Archived parent configs, model/bank bytes and runtime sources are hash-checked. A single simulator runs at a time; existing outputs must never be overwritten.

First run a16-environment/two-iteration smoke for EACH weight, verify the saved configs and final `model_2000.pt`. This tests continuation and parameter routing, not recovery. The finite formal launcher then runs A and its three strict suites, followed by B and the same suites. Smoke weights are not used to initialize either formal arm.

```powershell
# Only after all current evaluation/video simulators have exited.
./scripts/run_smith_standweight_pair.ps1 -RunTag 20260916-smithstandpair-smoke16x2 -Iterations 2 -NumEnvs 16
./scripts/run_smith_standweight_pair.ps1 -RunTag 20260916-smithstandpair128x1000 -Iterations 1000 -NumEnvs 128
```

## Falsifiable evaluation

Use exactly the existing deterministic SmithNominal Play physical/action environment,20trials each,3s continuous hold,8s recovery-onset horizon, four simultaneous current vertical foot forces>5N, no base contact, normal leg geometry and height0.30–0.55m. Training reward weight is not part of inference and is not an acceptance threshold. Bank handover remains1s direct nominal-position PD, not zero torque. Seeds are20260918 for bank/30degrees and20260916 for45degrees. Bank side/back have16/14unique sources, repeated to20trials; this reused development set is not a final blind benchmark.

Primary screen: upright `final_valid_stands` and `final_geometry_passes`, followed by actual settled-fallen final-valid counts. A higher total reward is incomparable across weights and is not progress. Report ever-valid and final-valid separately. The new standard-library report auditor checks internal consistency, not physical truth or model acceptance; actual front/oblique inspection is still required.

If B remains all-zero, or grows taller with crossed legs/abnormal support, the hypothesis is unsupported. Do not prolong it unchanged or promote it. If there is a signal, require independent seeds/unseen initial states and preservation of normal stance before any promotion. Retain locomotion2250 and limited-domain recovery3448 unchanged. A future stand-up initial-state curriculum is a separate experiment, not bundled here; [Lee2019](https://arxiv.org/html/1901.07517) motivates that decomposition but does not establish this Go2 result.
