# Proposed next speed experiment: one tracking-weight change, retained command distribution

Status: design only, 2026-09-17. No runtime edits, training, simulator launch, model promotion or acceptance change were performed for this proposal. Recovery/handoff work remains separate and has priority. The proposed experiment is falsifiable; it does not guarantee faster, non-slipping locomotion.

## Evidence checked, not inferred from training reward

The [control audit](speed-retention-control-audit-20260917.md) now covers the completed18-case screen. For this proposal, both current summaries are completed and all36 actual report SHA values, checkpoint identities, acceptance aggregates and core walking metrics were checked against their summaries:

| Development screen, three evaluation seeds | Expanded-range 3947, 01:22 run | Original-distribution control 3947, 03:10 run |
| --- | --- | --- |
| Normal 0.5 m/s | 3/3 | 3/3 |
| Retained 0.8 m/s | 3/3 | 3/3 |
| Left / right turns | 2/3 / 3/3 | 3/3 / 3/3 |
| Lateral delta-v 0.5 m/s | 3/3 | 3/3 |
| Target 1.0 m/s | 0/3: one tracking, two slip failures | 0/3: three tracking failures only |
| Target actual mean vx, seeds 09/10/11 | 0.868040 / 0.903671 / 0.902390 m/s | 0.840405 / 0.842149 / 0.832786 m/s |
| Target contact-slip mean, same seeds | 0.110574 / 0.128212 / 0.140854 m/s | 0.080700 / 0.112386 / 0.108686 m/s |

Sources: [expanded summary](../evaluations/20260917-speed-candidate3947/summary.json), SHA256 `14ba5a9d31c5cc2d166ec59fa104f1497a0c6cb7cf641dfb72296af7f5f08ba8`; [control summary](../evaluations/20260917-speed-retention-control3947/summary.json), SHA256 `cf33b2809e700d834c195b5dc8317310ccd1bd168f81671729d85c44bff09379`. Both checkpoint basenames are model_3947.pt; identify them by full path and SHA, never just by that basename.

The control retains the 15 old-function screens but is not perfect: normal 0.5-m/s seed10 has vy=-0.091705 m/s and yaw=0.065499 rad/s, within the existing drift screen but not zero. Its target-speed slip margin is only 0.007614 m/s in seed10. This motivates a small, stop-loss-controlled intervention, not a claim that faster learning is safe. These are three evaluation seeds of one trained policy per branch, not independent training replicates or proof of causality.

## What the actual code and saved configuration say

The [control env configuration](../configs/20260917-speedretention128x300/env.yaml) and [PPO configuration](../configs/20260917-speedretention128x300/agent.yaml) use independent uniform nonstanding commands vx[-0.5,1.0], vy[-0.2,0.2], yaw[-0.6,0.6], 30% all-zero standing, and 4-8-second command resampling. The prior expansion changed only the vx upper bound to 1.2, with shared LR synchronization. It left the yaw marginal unchanged but reduced density in every old joint command region by 1-1.5/1.7=11.76%. That is exposure dilution, not proof that dilution alone caused the left-turn regression.

The installed [tracking reward implementation](E:/IsaacLab/repo/source/isaaclab/isaaclab/envs/mdp/rewards.py:304) is `weight * exp(-((command_vx-vx)^2 + (command_vy-vy)^2) / std^2)`, with weight1.5/std0.5. Angular tracking is a separate weight0.75/std0.5 term. An isolated 0.16-m/s forward error still earns kernel value0.902668, so a high tracking reward can coexist with the unchanged 0.12-m/s acceptance failure. This is a reward/acceptance distinction, not proof that the kernel is the cause.

The [slip penalty](E:/IsaacLab/repo/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/mdp/rewards.py:237) has weight-0.5 and sums current foot XY speeds gated by maximum contact-force norm over sensor history >1N. The evaluator instead computes a conditional mean of current >5N-contact foot speeds, requiring <0.12m/s. Do not convert training slip reward into that acceptance statistic. Keep both definitions and all thresholds unchanged here. Other important unchanged terms include gait2, air-time2, static-stance3, flat-orientation-7, base-height-30 and action-rate-0.015.

The actual PPO configuration is the same 48-input/12-action, 128x128x128 ELU actor/critic, 24 steps/env, 5 epochs x4 minibatches, adaptive LR, gamma0.99, lambda0.95, entropy0.005, desired-KL0.005, clip0.2, with no actor/critic observation normalization. This proposal does not change PPO, action targets, noise, pushes, assets, physics, reset rules or evaluation.

## Parent choice and the single intervention

Use the ORIGINAL-DISTRIBUTION control as the shared frozen parent for both new arms:

- Run: `E:/IsaacLab/repo/logs/rsl_rl/unitree_go2_flat/2026-09-17_03-10-21_20260917-speedretention128x300`.
- Model: `model_3947.pt`, SHA256 `3dfbe03e0ac332fee688c970b420167affcdb6e8cacee16e40fc12120751df3f`.
- Direct checkpoint metadata inspection, without importing Torch or a simulator, found iter3947, saved optimizer LR1e-5 and Adam step79120. Preserve all optimizer/model/std state; initialize the runner LR scalar from that actual saved value.

Why this parent: it has a complete current-protocol 15/15 old-function screen and lower target slip in all three seeds. Frozen3648 has higher target vx in two seeds but already fails target slip in those two, and its current-protocol parent turn/push comparison is incomplete. Returning to3648 would discard the low-slip/retention state and ask two questions at once. The tradeoff is explicit: control3947 is slower and has limited slip headroom; success is not assured. Keep3648 and2250 immutable reference models. Never warm-start from the failed expanded3947.

Arm A: continue this parent with the existing tracking weight1.5. Arm B: change only `env.rewards.track_lin_vel_xy_exp.weight` from1.5 to2.0. Preserve std0.5, yaw reward, slip reward, all command ranges and the 30% standing mixture. This preserves the original command distribution rather than adding a new sampler or diluting turn rehearsal. The 2.0 value is a prespecified engineering test dose, not a parameter claimed from a paper.

Hypothesis: a 33.3% increase in the existing XY tracking term's relative weight can improve the conservative parent's 1.0-m/s tracking before slip or old-function regression appears. For a fixed state it multiplies that term and its local sensitivity by4/3; PPO normalization and policy dynamics mean this is NOT a guarantee of a 33% behavioral change. XY tracking also affects lateral tracking and zero-command rewards, so ordinary walking/stopping and turning still require explicit regression tests. Changing std as well, weakening slip, changing gait rewards, changing the sampler, or extending the velocity range would invalidate this one-factor experiment.

## Finite budget, smoke and checkpoint stop-loss

1. CPU/static tests first: full YAML comparison permits only the A/B tracking-weight scalar plus run/log metadata and the shared declared budget/LR synchronization. Unit-check the 4/3 term ratio on fixed synthetic states, unchanged std and independent yaw/slip outputs. Pin the parent, actual/archive configs, inherited reward/command sources and installed overlays. Never modify the historical pilot or frozen models.
2. Separate smoke arms:16env x2updates from the frozen control parent, complete optimizer resume; expected label3948 and Adam79160. The smoke verifies interface/config/finite state only and is discarded as a formal starting point. Recheck actual LR before loading. A smoke is not a quality result.
3. Formal:128env, at most100new updates per arm, organized as two declared 50-update blocks with a complete regression between blocks. Each block is153,600environment steps; each arm max307,200; both arms max614,400. Training seed42 and both arms' block/reset protocol match. Only one simulator at a time; all files onE, wait for the independent recovery job and reject overlapping replay/validation/training processes. No automatic third block or higher speed.
4. Ordinary RSL resume repeats a saved iteration label: block1 runs labels3947..3996, Adam79120->80120; block2 loads that arm's3996 and ends4045, Adam81120. Block2 must synchronize the runner LR scalar from its own just-saved optimizer value, not assume1e-5 forever. Simulator/RNG reset between blocks is a declared shared procedural difference, not uninterrupted training. Both arms must use this same procedure; budgets are counted by actual optimizer updates, not subtracting checkpoint labels.
5. After EACH50-update block, close training and run the unchanged18fresh-scene matrix: seeds20260909/10/11 x normal0.5, retained0.8, target1.0, left/rightyaw+/-0.5 and lateral delta-v0.5. Include complete stand4s/walk8s/stop6s. At most72such development rollouts across the two arms/two blocks. Keep every failure and intermediate checkpoint; no selective video or best-frame scoring.
6. Stop that arm immediately on source/hash/config inconsistency, nonfinite state, invalid physics logs, any new old-function report failure, failed unperturbed straightness gate, or any target slip >=0.12m/s. Do not alter thresholds or resume around a failed checkpoint. An arm stopped for regression is rejected, not silently replaced by its earlier checkpoint.
7. Prespecified B futility screen after block1: all old-function and slip screens must pass, target mean absolute vx error across the three seeds must improve by at least0.01m/s versus the frozen control, and no target seed may worsen by more than0.02m/s. These margins are engineering decision rules, not paper thresholds or acceptance. If unmet, stop B; do not add more updates. A may complete its second block if its safety/retention screens pass, to distinguish plain continuation from B's intervention.
8. Final experimental success requires all18unchanged report criteria and all unperturbed straightness gates, especially target tracking error<0.12AND slip<0.12 in each seed. Report all continuous old-function differences, even when pass flags are unchanged. To attribute a benefit to B, compare B against matched-budget A, not only its frozen parent. If A also succeeds without a meaningful B advantage, prefer the simpler unchanged-reward continuation and do not credit the weight change. One A/B pair cannot establish a robust causal effect.

Passing these development screens is still not promotion. Freeze the selected checkpoint and design, then run a predeclared fresh three-seed18case confirmation matrix (seeds first checked unused in development), inspect complete front/oblique evidence, and only then discuss changing recommendations. Failed/partial stages do not authorize1.5/2.0m/s, hardware tests, or an integrated recovery controller.

## Primary-source connection and limits

[Rapid Locomotion via Reinforcement Learning](https://arxiv.org/abs/2205.02824) studies adaptive velocity-command curricula and adaptation for agile locomotion. Its [author-provided RewardThresholdCurriculum](https://github.com/Improbable-AI/rapid-locomotion-rl/blob/main/mini_gym/envs/base/curriculum.py) updates command-bin weights and neighboring bins only when BOTH linear and angular tracking rewards clear their thresholds. The primary [legged_gym implementation](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py) likewise conditions range expansion on tracking performance. These sources support jointly checking tracking and retained maneuver capability before advancing difficulty; they do not supply evidence that our proposed2.0weight will work.

This is a local, deliberately small reward-sensitivity experiment after a failed range expansion. It does not implement either paper's complete curriculum, online adaptation, training scale, robot/dynamics, hardware validation or reported speed. No animal-motion imitation,2m/s running, arbitrary-fall recovery, full reference-video reproduction or guaranteed non-forgetting is claimed. If the B arm fails, preserve the negative evidence and revisit command-conditioned sampling or another separately controlled hypothesis; do not combine changes post hoc in this experiment.
