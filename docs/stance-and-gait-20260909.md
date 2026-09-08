# Standing, walking and stopping: validation protocol

The priority is a supported stand, real foot swings during locomotion, and return to a quiet four-foot stance after stopping. Recovery from a fallen pose is a separate task and remains unvalidated for the new locomotion policies.

## Why the earlier videos were inadequate

The previous `robust_model799.pt` collapsed to about 0.10 m base-link height and failed to move at the requested 0.5 m/s. Four detected foot contacts alone did not imply a correct pose. `standard_stance` and `robust` also inherited a sparse multiplicative standing reward but had no explicit swing timing or contact-slip objective.

A zero-action PD rollout held the nominal joint pose at about 0.28 m once settled. This establishes a usable physical baseline. It does not establish a unique optimal standing height. An initial 0.40 m root height is an above-ground drop initialization, not a desired settled height.

USD inspection and Isaac Lab's `ArticulationData.root_pos_w` implementation confirm that root position is the base-link position. The asset root and base have no authored translation offset. Both the local and online grid assets place the physical collision plane at z=0. An earlier explanation attributing the problem to a root-frame/chassis height offset was incorrect and has been withdrawn.

The 20260909 standard experiment used a 0.24 m shaping target and restarted from the official flat Go2 checkpoint. It restored supported locomotion, but subsequent foot-position measurements showed dragging: foot centers stayed near their 0.023 m contact height. This experiment is not the final normal-gait result, and its improvement cannot be attributed solely to the changed height target.

## Current implementation

`Isaac-Natural-Flat-Unitree-Go2-v0` is a new task, keeping historical task definitions available. It adapts the following functions from Isaac Lab's upstream Spot environment:

- `air_time_reward`: rewards bounded swing/contact durations; zero commands favor contact.
- `GaitReward`: synchronizes FL/RR and FR/RL while separating the two diagonal pairs.
- `foot_slip_penalty`: penalizes tangential foot velocity during contact.
- `foot_clearance_reward`: encourages foot clearance while feet move.
- `joint_position_penalty`: increases the nominal-joint preference during standing.

A base-height penalty uses a 0.30 m target, between the measured compliant PD stand and the upright learned stands. Commands resample every 4–8 s with 30% zero-command samples, explicitly training move/stop transitions. The 48 observations, 12 actions, action scale 0.25 and original PD settings are retained.

This is PPO with explicit gait rewards. It is not AMP and it does not reproduce all behavior in the reference video.

## Evaluation

Each rollout is continuous: 4 s stand, 8 s at 0.5 m/s, 6 s stop. Initial base velocity and joint velocity are zero. Heading control, command resampling, random pushes and fall-triggered reset are disabled. A separate disturbance run adds lateral velocity increments of +0.5, -0.5 and +0.5 m/s at 2, 8 and 15 s. These are simulated velocity impulses, not measured physical pushes in newtons.

The evaluator records every 20 ms: base position and velocity, tilt, each foot's body-relative horizontal position, world height, vertical contact force, instantaneous contact flag, tangential speed, and reset status. Metrics exclude the first second of each phase for settling; complete videos retain those transitions.

Screening thresholds are: no reset or base contact; base height >=0.25 m after settling; tilt <10 degrees (20 under disturbance); body-frame mean walking speed within 0.12 m/s of the command; quiet-phase speed P95 <0.06 m/s (0.20 under disturbance); each foot height P95 >0.04 m while walking; contact slip mean <0.12 m/s; and four-foot contact fraction >95% at rest. Passing these screens still requires visual inspection and does not prove hardware readiness or arbitrary-fall recovery.

## Literature checked

1. Wu et al., [Learning Robust and Agile Legged Locomotion Using Adversarial Motion Priors](https://doi.org/10.1109/LRA.2023.3290509), RA-L 2023. Bibliographic metadata was verified against Crossref; this is the reference identified from the supplied video. No exact implementation reproduction is claimed.
2. Escontrela et al., [Adversarial Motion Priors Make Good Substitutes for Complex Reward Functions](https://arxiv.org/abs/2203.15103), 2022. Sections III and IV distinguish velocity tracking from natural movement and describe 4.5 seconds of German Shepherd motion retargeted to Unitree A1, a discriminator over state transitions, and reference-state initialization. The repository's [A1 configuration](https://github.com/escontra/AMP_for_hardware/blob/main/legged_gym/envs/a1/a1_amp_config.py), [motion loader](https://github.com/escontra/AMP_for_hardware/blob/main/rsl_rl/rsl_rl/datasets/motion_loader.py), and [discriminator](https://github.com/escontra/AMP_for_hardware/blob/main/rsl_rl/rsl_rl/algorithms/amp_discriminator.py) identify the components still required for an AMP implementation. A1 reference motions require a verified Go2 joint-order and geometry mapping; renaming the robot is insufficient.
3. [Isaac Lab Spot reward implementation](https://github.com/isaac-sim/IsaacLab/blob/37ddf626871758333d6ed89cf64ad702aef127d0/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/mdp/rewards.py). This is the direct implementation source for the new gait task, with Go2 names and task-specific weights.

The paper's hardware results and energy-efficiency numbers are not measurements of this project.

## Completed natural-gait evaluation

Run `2026-09-09_01-15-32_natural_gait_20260909` resumed calibrated standard checkpoint 598 with 512 environments for 1,000 additional PPO iterations (12,288,000 environment steps). Parameters, including training seed 42, are in `configs/natural-20260909/`. Iteration labels are inherited on resume; checkpoint 950 is from this run, not the old standard-stance model with the same number.

Checkpoint 950 passed all nine screens on seeds 20260909, 20260910 and 20260911. Seeds change startup mass/material randomization; initial pose remains nominal upright. This is not an arbitrary-pose or large-sample reliability test.

| Seed | Stand height (m) | Walk height (m) | Stop height (m) | Walk speed (m/s) | Stop speed P95 (m/s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20260909 | 0.300 | 0.329 | 0.305 | 0.566 | 0.038 |
| 20260910 | 0.308 | 0.342 | 0.310 | 0.603 | 0.032 |
| 20260911 | 0.307 | 0.340 | 0.309 | 0.601 | 0.030 |

The first continuous one-second quiet four-foot hold begins 0.86–1.04 seconds after the stop command. Walking diagonal-support fraction is 0.831–0.917. For seed 20260909 the foot-center height P95 values are 0.079, 0.117, 0.084 and 0.079 m (contact height is about 0.023 m). This supports visible foot swings, but also exposes left/right asymmetry.

The same checkpoint and seed 20260909 passed three lateral delta-v tests (+0.5, -0.5, +0.5 m/s). The continuous video shows support maintained. This is a bounded flat-ground disturbance test, not rough-terrain or full-fall recovery. All transitions remain in the videos. Phase and gait-cycle contact sheets were inspected; residual roll/pitch of several degrees remains visible.

## Longer training regression and follow-up

Final checkpoint 1597 failed quiet standing and stopping. Although upright with foot swings, it kept moving at about 0.22–0.23 m/s under zero command. Its report and CSV are preserved in `evaluations/natural-20260909/natural1597_failed_stop.*`.

The inherited Spot rewards enable gait/air-time mode for either nonzero command OR measured body speed >0.1 m/s. Moving also removes the larger standing joint-position penalty. Thus a policy can continue gaining gait reward by maintaining uncommanded movement. This mechanism is consistent with the regression, but no isolated causal ablation has been completed.

The separate `Isaac-Natural-Stop-Flat-Unitree-Go2-v0` task disables the measured-speed branch by setting its threshold to infinity for gait, air-time and joint-posture terms. Nonzero command still enables locomotion. It also doubles the flat-orientation penalty and quiet-stance reward and uses a smaller PPO KL target/entropy coefficient. These change training only: inference runs the learned policy at 50 Hz without a scripted posture override or stop-state action latch. The original Natural task remains available for comparison.

## Selected command-gated model

Run `2026-09-09_01-37-53_natural_stop_20260909` resumed Natural 950 for 400 iterations, 512 environments (4,915,200 additional environment steps). Intermediate checkpoint 1100 and final checkpoint 1349 both passed the basic rollout. Final 1349 was then tested with the same three seeds; all nine screens passed in each run.

| Seed | Stand height (m) | Walk height (m) | Stop height (m) | Walk speed (m/s) | Stop speed P95 (m/s) | Quiet hold begins after stop (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20260909 | 0.307 | 0.352 | 0.309 | 0.569 | 0.021 | 0.54 |
| 20260910 | 0.316 | 0.366 | 0.321 | 0.572 | 0.021 | 0.70 |
| 20260911 | 0.314 | 0.364 | 0.318 | 0.549 | 0.006 | 0.52 |

For seed 20260909, maximum settled stand tilt decreased from 4.6 to 2.7 degrees and mean walking contact slip from 0.084 to 0.046 m/s relative to Natural 950. Walking pitch still reaches about 7.8 degrees, and foot swing heights remain unequal. Contact sheets spanning all phases and a full gait cycle were inspected. These are observed improvements for this test set; there is no claim that the reward loophole cannot recur after further training.

Selected checkpoint 1349 also passed the three-impulse rollout on seed 20260909, with no base contact or reset. Settled walking speed was 0.557 m/s, maximum phase tilt 7.89 degrees, and stop speed P95 0.048 m/s. Four-foot support fraction was 97.3% in the disturbed stand and 99.6% in the disturbed stop; brief corrective foot unloading is visible. The continuous video and frames immediately after each impulse and after settling were inspected. This report uses the separately documented disturbance speed/tilt thresholds; it is not a claim of arbitrary-push recovery.
