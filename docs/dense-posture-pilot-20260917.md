# Dense posture continuation: bounded experiment, not an accepted recovery policy

## Completed result — rejected, 2026-09-17 00:24

The finite128-environment600-update run completed1,843,200steps. Final2598 SHA2564f4003621ecb2b9e443255fb51673ca3652fbc6d86f790091627e3a028c78602;17model/51optimizer tensors finite,17Adam step counters52000. All53 pinned source/checkpoint/bank hashes and the actual full saved configuration guard passed. These checks validate implementation/integrity only.

Each of the five20-trial suites produced0ever-valid and0final-valid stands: settled-bank side, settled-bank back, upright-PD compatibility,30degree side and30degree fore-aft. Final geometry alone passed1/20fore-aft and0/20elsewhere; it is insufficient for standing. Reports are under evaluations/20260917-denseposture128x600-<scene>/. Do not promote or extend this branch unchanged. Preserve failed evidence. Reward improvement did not produce a normal supported stance; the next experiment must protect a known standing subskill explicitly.

## Why this experiment

The completed Smith parent1999 and stand-weight continuations did not produce a normal single-policy stand. Raising the existing exponentially shaped stand reward alone did not resolve folded legs. A frozen two-policy diagnostic with parent1999 followed by aligned3448 achieved final valid holds in 9/20 side trials and 20/20 back trials. These are 16 and 14 unique bank states, respectively, with a 1-second nominal-PD handover; not arbitrary falls, independent unseen validation, hardware tests, or a single integrated controller. The actual front/oblique videos of these new results have not yet been reviewed.

The 11 side failures switch with folded asymmetric joints despite a nearly upright torso. CPU replay of their recorded 48-value observations matches the real stand actor within 2.86e-6. Replacing only the previous-action observation with the physically executed target encoding changes the actor output but leaves the average first target jump almost unchanged (2.560 to 2.549 radians). Clearing history increases it to 2.949 radians. These are offline input sensitivity probes, not simulated counterfactual recoveries; neither a history-only root cause nor a runtime reset is justified.

[Lee2019, II-D](https://arxiv.org/html/1901.07517) trains self-righting, standing-up and locomotion separately and explicitly includes joint arrangement in the righting goal. Our present experiment is a small Go2 reward adaptation motivated by this missing posture quality, not a reproduction of its ANYmal/TRPO hierarchy, actuator model or learned selector. No synthetic low-root sitting states are injected.

## Exact change

New task: Isaac-Recovery-Bank-DensePosture-Flat-Unitree-Go2-v0. Inherit SmithNominal and add only:

`6 * clamp(cos_up, 0, 1)^2 * (1 - mean(clamp(abs(q - q_default) / soft_joint_range, 0, 1)))`

Actual per-joint limits and nominal positions come from the asset. The target default pose lies within the limits and can physically stand under the existing3448 policy. The extra score distinguishes large within-range joint errors without exponential whole-body saturation. It is not a contact, support, crossing, or success test. Keeping roll10 and stand10 unchanged isolates this new shaping term. It can still learn an undesired local optimum; only external evaluation decides.

Physics, self-collisions, actuator gains/limits, 50Hz nominal+.25 action targets and their soft-limit clamp, 48-dimensional observations including genuine raw-action history, bank TRAIN mixture, zero commands, PPO and termination remain unchanged. No history clearing, target blending, hidden manual-PD handoff or evaluation threshold change.

## Budget and provenance

- Parent: Smith1999, SHA25671e4c7a39ff81836d2ead1c4533988c696e5bd4a0fc939b8cef6b4f6daec1b1c.
- Smoke:16 environments,2 updates,768 environment steps, final2000. This is interface validation, not recovery acceptance.
- Formal:128 environments,600 additional updates,1,843,200 environment steps, expected final2598. New outputs, never overwrite parent/2250/3448.
- Standard RSL full-model/optimizer resume; the adaptive learning-rate scalar starts from configured0.0005 and simulator/RNG reset. This is a new continuation experiment, NOT bitwise uninterrupted resumption or actor-only warm-start. Saved iteration label1999 is reused for the first new update.
- All output/log/config/cache/backup paths stay on E:. Only one simulator at a time; no hardware or cloud use.

The whole serialized environment/PPO configuration guard allows only the extra reward, environment count, run paths, parent identity and finite budget. Runtime sources/checkpoints/bank have a pinned SHA snapshot. Installation backup: E:/IsaacLab/artifacts/install-backup-20260916-235956.

## Evaluation and stopping decision

After training the launcher runs five independent simulator processes: heldout-bank side, heldout-bank back, near-upright preparation with1s nominal PD, controlled30-degree side drop, controlled30-degree fore-aft drop. Each formal scene has20 trials. Independent processes avoid the previously observed previous-scene influence on PD-settled starts. The repeated heldout bank is development validation, not a fresh generalization test.

Single-policy evaluation uses SmithNominal Play, identical action/observation/physics configuration; training-only reward changes do not drive evaluation actions. Require unchanged normal geometry, four simultaneous CURRENT vertical foot forces>5N, no base contact, height0.30-0.55m, low velocities, continuous3s valid hold and valid ending. Preserve ever-valid and final-valid separately. Upright-PD compatibility is not fallen recovery.

Do not promote on reward, model finiteness, passing schema audits or successful dual-policy composition. Compare actual geometry, height and final holds; inspect front and oblique real simulation videos before presenting normal stance. If final standing remains zero or normal-standing regresses, do not extend this branch unchanged. Investigate supported-sitting roll targets, physically collected TRAIN-only stand-up states or a separately controlled symmetry experiment. Do not train on the development heldout switching records.

See [current progress](training-status-20260916.md) for the only current process and completed stages.
