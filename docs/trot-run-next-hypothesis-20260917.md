# Trot/run direction review: measured diagonal support bias

Read-only review on 2026-09-17. This note does not start training, alter a running task, promote a model, or claim a paper reproduction. The proposed intervention below has NOT been implemented or tested. The independent natural-recovery-standing experiment is a different branch.

## What actually failed

I re-read all 72 original JSON reports for control3947 / weight-A3996 / weight-B3996 / speed-gate3996, checked each report SHA against its summary, checked checkpoint identities, and checked `passed == all(acceptance.values())`. All matched. These are four single-training-run development screens, not independent training replicates.

| Branch | Old-function reports | Target 1.0 m/s reports | Target actual vx, seeds 20260909 / 10 / 11 (m/s) | Target contact slip (m/s) |
| --- | ---: | ---: | --- | --- |
| Original-distribution control3947 | 15/15 | 0/3 | .840405 / .842149 / .832786 | .080700 / .112386 / .108686 |
| Weight-A3996, unchanged tracking weight 1.5 | 14/15 | 1/3 | .901569 / .915274 / .903240 | .114336 / .150306 / .130915 |
| Weight-B3996, tracking weight 2.0 | 15/15 | 0/3 | .810049 / .833245 / .835683 | .080301 / .138663 / .127630 |
| Bounded command-cap gate3996 | 10/15 | 0/3 | .712774 / .785815 / .792117 | .061156 / .116883 / .108106 |

Control fails only `walk_tracking` at the target: the unchanged mean-vx error limit is strictly .12 m/s. A's other target failures are slip; A additionally loses right-turn seed11 `lateral_tracking`. B loses target tracking and two target-slip checks. Gate regresses ordinary/0.8/right09 tracking; normal09 also fails its separately declared straight-drift gate. Lower slip at lower achieved speed is not a matched-speed improvement.

All 144 settled stand/stop phase records across those 72 reports have geometry fraction 1, zero invalid geometry samples, and current base-clear fraction 1. The minimum current four-vertical-contact fraction is .973333, above the unchanged .95 exclusive gate. Thus these speed failures are not evidence that the robot stopped being able to stand. Conversely, these locomotion reports contain no fallen-recovery evaluation; they cannot certify preservation of the rolling/standing expert chain.

Summaries containing exact per-case JSON paths and hashes:

- [Control](../evaluations/20260917-speed-retention-control3947/summary.json)
- [A](../evaluations/20260917-speed-weightA3996/summary.json)
- [B](../evaluations/20260917-speed-weightB3996-restart0958/summary.json)
- [Gate](../evaluations/20260917-speed-gate3996-first/summary.json)

## New measurement: already diagonal, but strongly unequal support duration

For each of the 36 straight-case CSVs (four branches x three commands x three seeds), select exactly `phase == "walk" and time_s > 5.0`: 350 samples, timestamps 5.02 through 12.00 s, 50 Hz, after the evaluator's original one-second walking settling exclusion. No stand/stop samples or cherry-picked subwindows are used.

For foot i, `c_i = (current net_forces_w[i,2] > 5 N)`, exactly the recorded `*_foot_vertical_contact`. Duty is `mean(c_i)`. This is NOT history-max force norm, `foot_contacts`, or the older `*_foot_contact` norm-based field. The CSV also records the actual current `*_foot_fz`, enabling the sensitivity check below. The legacy walking report's `diagonal_support_fraction` uses different contact semantics, so the new statistic is kept separate.

Diagonal agreement is the mean of `c_FL == c_RR` and `c_FR == c_RL`. Exact alternating diagonal support is the fraction of `1001` or `0110` in FL/FR/RL/RR order. These are descriptive diagnostics, not a new acceptance criterion or a complete gait classifier. Agreement alone can be gamed by four-foot standing; exact two-contact patterns and repeated transitions must also be inspected.

Control3947, percentages (rounded only for display):

| Command / seed suffix | FL duty | FR duty | RL duty | RR duty | Diagonal agreement | Exact alternating diagonal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| .5 / 09 | 79.43 | 27.43 | 27.43 | 80.29 | 98.14 | 90.86 |
| .5 / 10 | 76.29 | 28.86 | 27.71 | 76.29 | 97.43 | 92.86 |
| .5 / 11 | 77.71 | 28.86 | 27.14 | 77.71 | 98.29 | 92.57 |
| .8 / 09 | 75.14 | 31.43 | 31.71 | 75.43 | 98.86 | 92.00 |
| .8 / 10 | 71.71 | 28.00 | 29.43 | 76.57 | 96.29 | 92.86 |
| .8 / 11 | 72.29 | 28.00 | 30.57 | 74.29 | 95.71 | 91.14 |
| 1.0 / 09 | 72.29 | 29.71 | 32.29 | 74.57 | 97.57 | 93.14 |
| 1.0 / 10 | 69.71 | 28.00 | 30.86 | 75.71 | 93.86 | 89.71 |
| 1.0 / 11 | 70.29 | 27.71 | 30.86 | 76.00 | 93.86 | 89.71 |

Control's diagonal-pair duty difference `abs((d_FL+d_RR-d_FR-d_RL)/2)` ranges .424286 to .524286. Recomputing directly from current force at 1 N gives .422857–.530000; at 10 N gives .420000–.522857. The large bias is not explained by choosing the 5 N threshold. Raw touchdown counts are 12–21 per foot over seven seconds; force chatter can inflate counts, so they must not be called debounced stride frequencies. A future cycle-based diagnostic should debounce and report its rule separately.

Three-seed means of exact alternating-diagonal support:

| Branch | .5 m/s command | .8 | 1.0 |
| --- | ---: | ---: | ---: |
| Control | 92.10% | 92.00% | 90.86% |
| A | 91.72% | 91.62% | 91.33% |
| B | 91.33% | 91.91% | 91.33% |
| Gate | 91.81% | 92.00% | 89.52% |

All 36 files have zero sampled four-foot-flight frames under the 5 N definition. This does not exclude a flight shorter than the 20 ms sampling interval, nor establish a universal definition of running. It does establish that commanding 1.0 m/s, or having diagonal contact pairs, is insufficient evidence of fast running. This review has not visually accepted these 36 videos; most cases have no video.

The narrow conclusion is **a diagonal alternating structure with pronounced diagonal support-duration bias**, not absence of any trot-like structure. It is a concrete naturalness problem and a candidate contributor to slip/speed tradeoffs; its causal role in target-speed failure is not established.

## Why the existing gait term does not remove this bias

The actual saved control [env configuration](../configs/20260917-speedretention128x300/env.yaml) already uses `GaitReward`, weight 2, pairs FL/RR and FR/RL, std .1, max_err .2. Its `velocity_threshold=inf` disables reward triggered only by residual measured speed: command, not drifting at zero command, selects gait. Air-time reward has weight 2 and mode_time .35. Static stance is 3, base-height penalty -30, orientation -7, slip -.5; A keeps them and B changes only XY tracking 1.5→2.0. All four archived PPO configurations have `symmetry_cfg: null`, the same 48-input/12-action 128x128x128 ELU architecture, and no actor observation normalizer.

The installed [Spot reward source](E:/IsaacLab/repo/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/mdp/rewards.py) was read directly, not inferred from the class name. Its asynchronous terms compare the **current** air time of one pair with the **current** contact time of the other. That enforces simultaneous opposite contact modes, but does not equate the duration of the two alternating half-cycles. The same intended sync/async structure is visible in [upstream Isaac Lab](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/mdp/rewards.py).

Analytic local-code counterexample, not a physically realized rollout: assume instantaneous switching between perfect opposite diagonal pairs, cycle length T=a+b, and both half-periods below .35 s. All six GaitReward component errors are zero, so its raw reward is 1 whether a=b=.24 or a=.34,b=.14. Every foot's current air/contact timer equals elapsed time since the most recent switch. The air-time term is then `4*t`; its cycle-average raw value is `2*(a*a+b*b)/(a+b)`: .480000 for the balanced case versus .563333 for the unequal case. Thus the existing two timing terms do not identify balanced duty as better; in this idealized fixed-period case the air-time term even favors inequality. Other physical costs and tracking terms may offset this, so this is an identified objective loophole, not proof of which term created the learned bias.

## Primary-source cross-check: curriculum and symmetry are different levers

The [Rapid Locomotion paper](https://arxiv.org/abs/2205.02824) reports velocity curriculum plus online identification on Mini Cheetah, not this Go2 controller. Its [author curriculum](https://github.com/Improbable-AI/rapid-locomotion-rl/blob/main/mini_gym/envs/base/curriculum.py) samples a multidimensional weighted grid and increases successful/neighbor weights only when linear and angular scores both pass. The [training robot configuration](https://github.com/Improbable-AI/rapid-locomotion-rl/blob/main/mini_gym/envs/mini_cheetah/mini_cheetah_config.py) and [robot reward/command implementation](https://github.com/Improbable-AI/rapid-locomotion-rl/blob/main/mini_gym/envs/base/legged_robot.py) also differ in observations, domain randomization and timing. The rejected local scalar-cap pilot is not that algorithm. It reached only .9, with one eligible .9 window after promotion; extending it unchanged would confound exposure with its observed retention regression.

[Mittal et al., Symmetry Considerations for Learning Task Symmetric Robot Policies](https://arxiv.org/abs/2403.04359) distinguishes task symmetry from requiring every motion itself to be symmetric, and analyzes on-policy augmentation and mirror loss. [Su et al., Leveraging Symmetry in RL-based Legged Locomotion Control](https://arxiv.org/abs/2403.17320) compares data augmentation with equivariant/invariant networks; its findings do not certify this controller or imply that imposing bilateral loss fixes our measured cycle bias.

The installed RSL-RL metadata is **3.1.2**. Its PPO already accepts `symmetry_cfg` with `use_data_augmentation`, `use_mirror_loss`, `mirror_loss_coeff`, and `data_augmentation_func`; a library upgrade is unnecessary. [Official v3.1.2 PPO](https://github.com/leggedrobotics/rsl_rl/blob/v3.1.2/rsl_rl/algorithms/ppo.py) augments minibatches and/or compares the actor's mirrored action means; current [main's symmetry extension](https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/extensions/symmetry.py) exposes the same two conceptual mechanisms but has a different class layout. Do not paste main's API into the pinned installation. This review does not assert the local file is byte-identical to the online tag.

| Option | What it tests | Main confound/risk here | Decision for the next bounded test |
| --- | --- | --- | --- |
| RSL-RL left/right data augmentation only | Whether symmetric experience improves task equivariance | Must transform all 48 observations, actions and real history correctly; verify robot/randomization symmetry and PPO likelihood handling. Global task equivariance still allows two reflected, individually biased periodic attractors. | Worth a separate later experiment, not automatically a cure for unequal half-periods. |
| Mirror loss only | Whether actor outputs commute with reflection | A low MSE is not balanced support on a realized trajectory; can oppose useful phase-dependent actions or degrade the pretrained policy. | Do not enable alongside another intervention now. |
| One command-gated completed-duration variance reward | Whether removing the timing loophole reduces the observed within-trajectory support bias | Can shorten/alter cycles, affect turning, or improve duty while sacrificing speed/slip. Requires actual diagnostics and stop-loss. | Most direct falsifiable first experiment for this specific finding; proposal below. |

If symmetry is tested later, use left/right reflection only after joint/body mapping checks: linear velocity/gravity use polar-vector signs (+,-,+); angular velocity uses axial signs (-,+,-); command (vx,vy,wz) maps (+,-,-); swap FL↔FR and RL↔RR with hip signs inverted, and consistently transform q offsets, qdot and previous action. Verify applying the mapping twice is identity, policy/critic observation keys are preserved, and nominal target mapping agrees in physical joint coordinates. Do not assume front/back symmetry: the nominal thigh angles and head/body geometry differ. Do not add phase observations or replace the actor architecture merely to obtain a gait label; that would change the pretrained interface and invalidate a one-factor continuation.

Neither symmetry regularization nor hand-shaped reward changes reproduce [animal-motion imitation](https://xbpeng.github.io/projects/Robotic_Imitation/index.html), which uses reference motions. No retargeted motion data, imitation discriminator, or paper-level reproduction was performed here.

## One finite proposed experiment, not an automatic next launch

Preconditions: finish the current stand experiment and the separate control3947 recovery-physics/action compatibility screen. Do not overlap GPU simulators. If that screen fails, isolate physics/action differences first; do not mask them by changing reward or disabling recovery self-collision.

Hypothesis H1: under the unchanged command distribution, adding a penalty on disparity of completed foot-contact/air durations reduces the diagonal duty gap without losing standing, stopping, turning, push retention or .5/.8 tracking. H1 does **not** predict that 1.0 m/s or fast running automatically succeeds.

1. Freeze original-distribution control3947 as parent, full path `E:/IsaacLab/repo/logs/rsl_rl/unitree_go2_flat/2026-09-17_03-10-21_20260917-speedretention128x300/model_3947.pt`, SHA `3dfbe03e0ac332fee688c970b420167affcdb6e8cacee16e40fc12120751df3f`. Preserve all actor/critic/std/Adam state and synchronize initial runner LR to the actual saved optimizer LR. Do not start from rejected A/B/gate.
2. Control arm: unchanged parent task. Treatment arm: add exactly one term, **-10 × near_straight_command × air_time_variance_penalty**, using the installed upstream function on ordered four feet. Its expression is `var(clamp(last_air_time,max=.5)) + var(clamp(last_contact_time,max=.5))`, with the implementation's sample variance. Proposed command gate: vx>.1, abs(vy)<.1, abs(wz)<.3. It is zero at stand/stop and the ±.5 turn commands, never requires four-foot contact during motion, and uses no desired periodic phase. Leave all existing timing/velocity/slip/stance rewards intact. This weight and gate are an engineering test dose, not author-paper settings.
3. Dose rationale is limited to the analytic example above: unequal .34/.14 durations give variance sum .026667; -10 contributes -.266667, outweighing the +.166667 weighted air-time advantage relative to .24/.24 by .1. This does not predict PPO improvement. Check the exact tensor function against balanced, unequal, stopped, turning and shuffled-order synthetic examples before any simulation; log raw existing air/gait/variance terms and actual term occupancy rather than just total reward.
4. Smoke each arm at 16 env x 2 updates from the original parent, then discard smoke weights. Formal budget: **one 128 env x 50 update block per arm**, 153,600 environment steps each, matched seed42 and 24 steps/env. No automatic second block. Freeze sources/config hashes; only the declared added term plus run metadata may differ. Save actual new checkpoint SHA, finite tensors and Adam-step delta. Existing A50 is useful historical context, but a new matched control is preferable; reused results cannot certify current execution equivalence.
5. Record completed command-window exposure and current/last air-contact durations. If fewer than 32 completed near-straight windows survive resets with at least two observed completed cycles per foot, label the experiment insufficiently exposed; do not silently extend or tune its gate after seeing results. The term's raw history values immediately after command changes must be logged; do not pretend they are fresh reference motions. This bookkeeping must not overwrite contact/action history.
6. Evaluate both final checkpoints on the unchanged 18-case development screen. Stop/reject on any old-function or straight-drift failure, any target slip >=.12 m/s, finite/config/source mismatch, or invalid physics. Analyze the same 350-frame window plus debounced complete-cycle diagnostics for all nine straight cases. H1 requires at least 25% reduction in the mean absolute diagonal duty gap over the nine cases, no case worsening by more than .05 absolute duty, no lost per-foot swing cycles, and no new short-period contact chatter. These are new **diagnostic selection margins**, not replacements for old acceptance. A duty improvement alone is not a successful speed candidate.
7. To claim 1.0 m/s support, require all three unchanged target tracking/slip gates in addition to old-function retention; show each actual mean vx. If no checkpoint meets all conditions, keep the original model and report a rejected hypothesis. Do not raise speed to 1.5/2.0 or alter slip/standing thresholds to rescue it. If development succeeds, freeze the selection before fresh-seed confirmation and inspect front/oblique videos with a contact timeline. Faster commands require separate staged validation.
8. Preserve frozen roll1999 and the independently accepted standing expert and its source hashes. Before integration promotion, rerun actual ordinary/side/back recovery and genuine continuous recovery→strict stand→walk→stop under one physical/action configuration, without reset, extra PD, history clearing or terminal-state transplantation. Keeping model files byte-identical is fallback preservation, not proof that a newly switched controller retains behavior. Full-flow video must be one uninterrupted trajectory; independent replays can supply matched camera views but cannot be spliced into a success sequence.

## Reproducibility appendix

Inspected local source SHA256:

- Evaluator `E:/IsaacLab/go2-rl-open-source/scripts/evaluate_go2_stand_walk_stop.py`: `bbba369177d95bc24163bdd2b5ba96506f5db7d4b1ce38875f8a8885c49e22e4`.
- Installed Spot rewards `E:/IsaacLab/repo/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/mdp/rewards.py`: `14e2d61f0061f27191e23228469e590c27ffef001646812993b9d4e0cce929a6`.
- Installed PPO `E:/IsaacLab/env/Lib/site-packages/rsl_rl/algorithms/ppo.py`: `4373ac1b2f9fdf14d9da57516968fc95d8f605d2967fee01dc61bf0d09423478`.

CSV paths are exactly `E:/IsaacLab/go2-rl-open-source/evaluations/<branch>/<case>-seed<seed>/<file>`. The filename is `model_3947_stand_walk_stop.csv` for control and `model_3996_stand_walk_stop.csv` for A/B/gate. Each listed file has 900 total rows and the selected walking window has 350; SHA values identify the full file, not a filtered derivative.

| Branch | Case / seed | CSV SHA256 |
| --- | --- | --- |
| 20260917-speed-retention-control3947 | normal / 20260909 | 700b67843445c5e844e83a13a7f453ce16c59c0f6498cabeda01982cfa1f1ec2 |
| 20260917-speed-retention-control3947 | normal / 20260910 | f22d3085cbc047e933e6b19deb2943d8b2ccccc9f7c8bcf82444a4cf6a63fce4 |
| 20260917-speed-retention-control3947 | normal / 20260911 | e97818863876735a3c955fc24588580144da95d4683f85c1eb1ce5e0c3495653 |
| 20260917-speed-retention-control3947 | retained08 / 20260909 | 65b47addb153059d48c82958b4f55ababca3a90e40f0edff966c6f261a0b2001 |
| 20260917-speed-retention-control3947 | retained08 / 20260910 | 59f97d867c0ff38b081a0b54add2698007d1c501177df5d1c939daa0fc3a2398 |
| 20260917-speed-retention-control3947 | retained08 / 20260911 | 041e5c3c61fa5081f71f8b557ed61d152b56a0dbf30e643b29bc266f8465eca0 |
| 20260917-speed-retention-control3947 | target10 / 20260909 | a92868824cc102707770594830f0a82003731dc6c8a9a22fafed58a8b1c6a417 |
| 20260917-speed-retention-control3947 | target10 / 20260910 | 156ea1bfba994865493446274939a723fae535b0682cbc9bc061ef54bfebc526 |
| 20260917-speed-retention-control3947 | target10 / 20260911 | 18569d06f8a8d8b9931a4dc262263c9a0c840385b1456f4f6c029390e7a6ef8d |
| 20260917-speed-weightA3996 | normal / 20260909 | 65224025dedecf6f99cfc4c0f514ef3ad23c7ec99a54d8239a5f6d6c51722d02 |
| 20260917-speed-weightA3996 | normal / 20260910 | 8f2a4ef60546250a013faf1390dac400f6c3181941bc607e3176cd4b90dd0cc0 |
| 20260917-speed-weightA3996 | normal / 20260911 | 1d4c5e21fa6ed5cd449ba4cb337f1c5703665a4d979b8442ae0cd7fde17d7894 |
| 20260917-speed-weightA3996 | retained08 / 20260909 | b44a79467ad1f762a72b5e64be69e959c5d30b3007fb727a031d140f3eb832a4 |
| 20260917-speed-weightA3996 | retained08 / 20260910 | 6e7c6a850cf9f708e55736860629ed836bd7c21e85c320332c37c96f92211ee5 |
| 20260917-speed-weightA3996 | retained08 / 20260911 | 0026acb7689d2d731dd334caaa9c800cc805c81fe971badc785c337cfd3c7c34 |
| 20260917-speed-weightA3996 | target10 / 20260909 | 11446c67ea1ddaa0256e4ef51b9e9fd741742e4c64656c062049002e6b13b1b7 |
| 20260917-speed-weightA3996 | target10 / 20260910 | 97573e3897d6822d84c3eb4c7f6d0b2f647cce486de2b9b67068012a34b5dc49 |
| 20260917-speed-weightA3996 | target10 / 20260911 | 1756e460cd28fd7817d8a69f5bf68aba24c0317adc61fdde0110a5a42f375b04 |
| 20260917-speed-weightB3996-restart0958 | normal / 20260909 | a672837e83461f22bec1b7f228a8634d96bfa5fe6d675a4d3eabb5175c407f9f |
| 20260917-speed-weightB3996-restart0958 | normal / 20260910 | 14a4a0877c91d707bd1c4ec54c871ed8f1e4dae928cd865d41898a0c5ed7aa22 |
| 20260917-speed-weightB3996-restart0958 | normal / 20260911 | 6e2d3833e1b4d838430d639ecb5ccb307c19965803f7e2abf51f481ba415ef9f |
| 20260917-speed-weightB3996-restart0958 | retained08 / 20260909 | 5233dec070e0e3877101fc26fd689369b0ae624e5f3518f363c076f9677ac1bf |
| 20260917-speed-weightB3996-restart0958 | retained08 / 20260910 | bc554bb72966381a8859209d501785629d36362875d7a16fb6374062a5222da9 |
| 20260917-speed-weightB3996-restart0958 | retained08 / 20260911 | 1f6dcedc20fce960dc5914f37480cf49afc13c559743ec14cb26478dce491241 |
| 20260917-speed-weightB3996-restart0958 | target10 / 20260909 | 85b211aba4a82bbc26e268f3816fd81747701723e8bc45ad9d84ef11cdfa2af8 |
| 20260917-speed-weightB3996-restart0958 | target10 / 20260910 | fac1c73973ffe714e11376765357cc3f900aafabe56096a71bde83549c321f82 |
| 20260917-speed-weightB3996-restart0958 | target10 / 20260911 | c3096f73901935b5be173e6f07b7ddc89e1803d4249ae7b15097810bc4777e6d |
| 20260917-speed-gate3996-first | normal / 20260909 | 05f4af67ef34cb2e7fa82d879f56a7a2eda8400c0f948b5f8e7e33bbd6aeb5a4 |
| 20260917-speed-gate3996-first | normal / 20260910 | 86f08fa06e4911be863360d7e9e5a58067ecce58dc0efbabec2588ac28844d0a |
| 20260917-speed-gate3996-first | normal / 20260911 | c4f84332cd8313f5e8759a8b5991ba0498d04ab6ad97737e959f9ef019aafb93 |
| 20260917-speed-gate3996-first | retained08 / 20260909 | 381e09f4a9ea6c864bb74c8a9eb9c0f32685a76b893188fb8116bb6ffcb00f56 |
| 20260917-speed-gate3996-first | retained08 / 20260910 | 96fb1a387cb1de4e1ffa17d16f8a9faabae85d3c2f5ba8ae141268e418592722 |
| 20260917-speed-gate3996-first | retained08 / 20260911 | c5a176ff05fcb95bcbbca32b322ec7acf73a0dd67f7e4809e82ed8b05453f904 |
| 20260917-speed-gate3996-first | target10 / 20260909 | 9ff2e7367e682472e5a5f21f654bcc481c05087121b6a7f26037aa70eb89087c |
| 20260917-speed-gate3996-first | target10 / 20260910 | 71c5b59c3528cdcab815a0607144892b605ca2194f15708a709fde3cbbdb2ed0 |
| 20260917-speed-gate3996-first | target10 / 20260911 | 21bd08adb3055f71f8133ba20d0d5c2cf9c8ed10dab2b9eb4186d4ab3cf07442 |
