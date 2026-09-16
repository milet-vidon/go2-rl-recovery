# Back-conditioned exploration — bounded candidate

Status2026-09-16 14:54 local:100-iteration training andall four evaluations COMPLETED;candidate REJECTED for recovery. Simulation only;all outputs onE:. Do not restart either completed job. Next is an explicitly non-acceptance stochastic diagnostic,not an automatic300-round extension.

The paired gate experiment failed to improve true grounded recovery. A read-only audit found the deterministic4899 back-down rollout almost motionless after2s,while learned hip action std0.055–0.059 maps through gain0.25 to only0.014rad target-angle std. There is no±0.25 action hard clip:the gain is not a bound. The recorded joints remain at least0.267rad inside their soft limits. The CSV covers onlytrial0/state155 and contains positions,not true joint velocity or torque,so it does not prove absence of dynamic torque saturation.

## The single changed variable

Use BankControl physics,reward,fixed60% bank,seed42 andthe same parent4899/std/Adam. Do not include the Release gate change. For a stored noisy actor observation,normalize its projected-gravity vector and compute:

```text
gate = clamp(normalized_gravity_z / 0.5, 0, 1)
effective_std = maximum(learned_std, 0.6 * gate)
distribution = Normal(original_actor_mean, effective_std)
```

Exact upright and90degree side observations retain original std;120degrees or greater inversion gets floor0.6,corresponding to0.15rad target-angle std. Existinggravity±0.05 observation noise can slightly activate the ramp near90degrees. This is not inference-time noise:deterministic evaluation uses the actor mean. Sampling,logprob,entropy andPPO's stored sigma all use the SAME Normal distribution; no noise is appended after sampling.

This is our falsifiable exploration hypothesis,not a claim of reproducing a paper or solving recovery. Constant large noise might merely produce random motion; success must remain deterministic and satisfy the original stand criteria. New learned weights may still regress,so the good controllers remain unchanged.

The policy explicitly requires12 actions,48 actor features,gravity at[6:9],unnormalized observations andthe existing scalar std parameter. Shape checks alone cannot verify term order; actual smoke env.yaml confirms the layout and unscaled/unclipped gravity. Registration is opt-in under a unique RSL class name; no third-party files are edited. State-dict keys and Adam parameter order are unchanged.

## Validation and run provenance

- CPU regression passed strict4899 loading,bitwise-equal initial means/unaffected samples,consistent Normal probabilities/entropy/old sigma,unchanged model keys/parameter order,17 preservedAdam states atstep20000,18 invalid-input/config cases andregistration collision protection.
- Real simulator smoke:16 environments,2 iterations,768 environment steps;run `2026-09-16_14-40-31_back_explore_smoke_20260916`,log `E:/IsaacLab/artifacts/recovery-20260916/back-explore-smoke.log`,completed with exit0 andsaved4900. This validates wiring,NOT recovery performance. The smoke weights are not used as the formal parent.
- Installed overlay backup `E:/IsaacLab/artifacts/install-backup-20260916-144019`.
- Formal run `2026-09-16_14-42-59_back_explore100_from4899_20260916`,512envs,100additional iterations,1,228,800 completed environment steps. Final4998 saved14:45:33;training session2445 exited0;log `E:/IsaacLab/artifacts/recovery-20260916/back-explore100-train.log`. Mean effective action std atfinal training iteration0.21,not itself a recovery result. Actual configs archived in `configs/20260916-back-explore100/`.
- Parent `2026-09-16_13-16-04_bank_nominalpd_from3900_20260916/model_4899.pt`,SHA256 `a021ba7a25d5a9803f7f905b4a242df8582a4d2730de5843303e76beaa26b4e2`. Same fixed bank/train split as Control100; heldout states never enter gradients.
- Resume preserves Adam/std but not the adaptive scheduler's unsaved Python scalar; this matches the earlier Control100 protocol. See [paired experiment](recovery-gate-ablation-20260916.md).

Launch already performed:

```powershell
./scripts/train.ps1 -Stage recovery_bank_back_explore -NumEnvs 512 -MaxIterations 100 -LoadRun 2026-09-16_13-16-04_bank_nominalpd_from3900_20260916 -Checkpoint model_4899.pt -RunName back_explore100_from4899_20260916 -BankPath E:/IsaacLab/go2-rl-open-source/datasets/recovery_states/nominal_pd_v1_20260916 -Headless
```

Evaluation session55053 completed:(1)unchanged4899 with packed trace in `evaluations/20260916-tracepacking4899-heldout/`;then(2–4)candidate grounded heldout,Aligned30,Aligned45 in `evaluations/20260916-back-explore100-{heldout,angle30,angle45}/`. Logs have corresponding names in `E:/IsaacLab/artifacts/recovery-20260916/`. SAME standard Bank Play deterministic evaluator,20side/back trials,seed20260918,1sPD handover,same bank IDs/hash. Standard ActorCritic can load this compatible checkpoint because inference means andstate keys are unchanged. Aligned Play30degree seed20260918 and45degree seed20260916. Compare against Control100,not a differently randomized task.

Trace-only performance refactor packs47 floating values and4 integer/boolean values into two CPU copies instead of51 scalar reads per step. Three CPU tests cover50 float32/64 states,large int64 counts,archived schema,exact field order/types/values andidentical CSV bytes. No physics,RNG,success logic ortrace sampling frequency changed. Real4899 regression PASSED:the entire results object andCSV bytes are EXACTLY identical to the old evaluation. Both traceSHA256 values are `0afdc861957081ad52084fda9699b1b97513d0d7ec6a8239464eaf5fab80c766`. No controlled wall-clock speedup estimate is claimed.

## Deterministic results

Candidate4998 SHA256: `7c3143cd3c1cba058604cc7e364fd1d4ee063a588cd99fd07defe2793e5aec80`.

- Grounded validation side0/20,back0/20;all starts eligible,16/14 unique states. No final valid stand. Side final tiltmean67.33degrees,height0.15090m,one supporting foot in every trial. Back final tilt179.982degrees,height0.057m,zero feet supporting in every trial.
- Aligned30degrees:upright20/20,side12/20,fore/aft19/20. Aligned45degrees:upright20/20,side0/20,fore/aft7/20,inversion0/20. Final valid stands equal success counts.
- Slight30degree improvements over Control100 do not establish settled recovery. Do not promote,overwrite3448/2250,or automatically continue this branch merely because more time is available.
- Next inspect the genuinely stochastic distribution to measure actual joint motion/torque,clearly marked `acceptance_eligible=false`. If physical exploration remains too local,we must revisit action reference andfallen-state preparation; the2019 paper's control-step current-joint targets and2025 paper's zero-torque initialization are different from this nominal-PD pipeline. A new task/bank must not be mislabeled as behavior-preserving continuation of4899.

Stop or redesign if the trained deterministic policy still never leaves a near180degree back-down pose; do not call stochastic thrashing a success. At most300 additional rounds require measured progress and retained stance. If promising,gather a new unseen validation bank/seed and visually check front/oblique videos before promotion. Current repeated heldout trials are a development-validation set,not an untouched final generalization test.

## Completed stochastic diagnostic (not acceptance)

Session76548 completed under BackExplore Play, seed20260918,20 back trials from14 unique bank states. Report `evaluations/20260916-back-explore100-stochastic-diagnostic/model_4998.pt_recovery_metrics.json` is marked `acceptance_eligible=false`. Success0/20 and final valid stands0/20. All joints sampled with std0.6; physical joint spans0.587–1.447rad, sampled speed peaks9.65–23.40rad/s, sampled applied-torque peaks4.45–18.36Nm. These are control-boundary samples, not guaranteed physics-substep peaks. Minimum tilt over all trials remained167.20degrees and maximum height0.07085m. Therefore the diagnostic is not motionless but fails to turn the torso over. No extension/promotion of BackExplore100; a fresh matched [action-reference experiment](action-reference-ablation-20260916.md) now tests a different hypothesis.
