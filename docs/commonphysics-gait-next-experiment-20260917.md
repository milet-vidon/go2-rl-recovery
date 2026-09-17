# One next experiment: balanced completed support durations

Review of final4046, 2026-09-17. The completed21-case screen gives3/6 common-physics cases and12/15 original-physics old-function cases. This is a finite training proposal, not model promotion.

## Evidence and interpretation

Final4046 is **partial adaptation**, not evidence that common physics is the wrong direction. After 100 updates from3947 it passes the .8 case (vx .834921, vy -.040053, slip .087185 m/s), whereas the parent failed common-physics lateral drift. It also passes this one 1.0 development case (vx .972257, slip .104876). That does not establish fast running or fresh-seed generalization.

The actual failures are .5 and pushed .5 FL foot-height p95 .038149/.037957 m, below the unchanged strict .04 threshold, and right-turn lateral speed -.176611 m/s, outside the unchanged .12 error threshold. Left turn passes. All six cases retain settled standing/stopping geometry fraction1; simultaneous current four-vertical-contact fraction is at least .973333. A failed moving gait is not a failed standing pose.

The original-physics regressions are .8/seed09 (vx .624683, yaw -.193711), left/seed09 (vx .361847, yaw .212095), and left/seed11 (yaw .337032). Thus4046 must not replace the retained locomotion model even though common-physics straight-speed tracking improved. The next A/B must retain both physics screens; a straight-gated penalty has no established remedy for this physics/turning tradeoff.

Each CSV has900 rows. The unchanged walking selection `phase == walk && time_s > 5` contains exactly350 samples (5.02–12.00 s). Recomputed from each CURRENT foot vertical force >5 N, with exact agreement to the saved vertical-contact columns:

| Case | FL duty | FR duty | RL duty | RR duty | Diagonal duty gap | Exact diagonal-pair support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| .5 | .831429 | .314286 | .260000 | .802857 | .530000 | .842857 |
| .8 | .728571 | .362857 | .317143 | .751429 | .400000 | .885714 |
| left | .777143 | .325714 | .254286 | .771429 | .484286 | .877143 |
| right | .774286 | .348571 | .314286 | .771429 | .441429 | .877143 |
| pushed .5 | .822857 | .300000 | .245714 | .808571 | .542857 | .857143 |
| 1.0 | .708571 | .385714 | .342857 | .714286 | .347143 | .900000 |

Gap = abs((d_FL+d_RR-d_FR-d_RL)/2). Exact support is1001 or0110 in FL/FR/RL/RR order. None of these descriptive measurements proves a natural trot; the long FL/RR support and short FR/RL support remain conspicuous. Duration balancing targets this measured bias, but does not directly guarantee enough lift or correct turning.

Source: [running screen and per-case paths/hashes](../evaluations/20260917-commonphysics4046-first/summary.json). All six report hashes were checked. CSV hashes in case order above: `7cc6d23f70b935b690e6c6ff5b8200ffd08a09fa46c957435307acdc07e3ee5f`, `def24b90d271b29066883a574b16dc4d59527c4352620f4bbb908e7e1e86211f`, `45ac2701089df4a28be70af35a8b0fcb3821d9c2c5284bccc6f03166ed1c9821`, `fba07f86a5f9020f916081900cfbc1279ddb6c830bde2e93d47b68734cd52b9c`, `a1399f2fb9509e67cdbfb6bbb472eca1f4367cf726ec6525db8c20aee07c7433`, `9924dd4db4acad2e746f1551ba3a0b7476bceab0a8f654c921478517b23bb389`.

## Precisely one new reward

Keep existing air_time weight2/mode_time .35 and GaitReward weight2/std .1/max_err .2 unchanged. Their implementation uses current timers (installed Spot `rewards.py` lines32–58 and166–185). Opposite diagonal pairs can remain perfectly synchronized/anti-synchronized despite unequal alternating half-periods. The earlier [timing analysis](trot-run-next-hypothesis-20260917.md) supplies a counterexample, not a proof of learned causation.

Add positive penalty `I[vx>.1, abs(vy)<.1, abs(wz)<.3] * (sample_var(min(last_air_time,.5)) + sample_var(min(last_contact_time,.5)))` over exactly the four foot bodies. Caller reward weight is -10. No new phase input, motion reference, symmetry loss, sampler change, clearance reward, or altered existing weight. Stationary and ±.5 yaw commands disable the term, but shared-network updates can still affect those behaviors; evaluation remains necessary.

The ungated expression matches installed official `air_time_variance_penalty`, lines203–215, SHA `14e2d61f0061f27191e23228469e590c27ffef001646812993b9d4e0cce929a6`, and the [upstream source](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/mdp/rewards.py#L203). Use `torch.var(..., correction=1)`, not population variance. Balanced .24/.24 durations yield0; .34/.14 yields .0266667, hence -.266667 weighted reward-rate units. The earlier air-time imbalance advantage is +.166667 weighted units, leaving a -.1 difference in that idealized comparison. RewardManager multiplies *all* terms by dt=.02; the new term in that example contributes -.005333 per control step, not -.266667 per step.

Important semantic boundary: sensor timers use CURRENT force NORM >1 N (`contact_sensor.py` lines422–443 and saved force_threshold1), whereas the evaluation duty above uses CURRENT VERTICAL force >5 N. Do not relabel them as identical. Last durations persist between completed events and across command changes; zeros after reset mean no completed event yet. No timer/history mutation or fabricated fresh cycles is allowed. Raw variance, command eligibility and gated penalty must be separately inspectable. Clipping at.5 also means this is a bounded disparity penalty, not a complete gait detector.

## Fixed matched A/B, then a decision

1. Independently resume the **same frozen3947 full model/critic/std/Adam state**, not selected intermediate4046 weights. Both arms use the combined recovery physics and original TRAIN noise, reset, pushes, command distribution and PPO. Keep roll1999/stand3547/refiner3746 untouched.
2. After a16-env×2-update smoke per arm (discard smoke weights), run **128 env×300 updates per arm**, seed42,24 steps/env:921600 environment steps per arm. A declares the new term at weight0 (RewardManager skips it); B uses weight-10. Original terms are identical. Initial scalar LR must equal actual resumed optimizer LR. Run serially. Evaluate only the two predeclared final checkpoints for selection; midpoints are diagnostic/safety-only, not another search over tiny checkpoints.
3. Existing uniform commands imply nominal gate occupancy .7×(.9/1.5)×(.2/.4)×(.6/1.2)=.105. At average6 s resampling this budget is roughly322 eligible command windows per arm before resets, versus roughly54 for50 updates. This is an exposure estimate, not a measured count. Log actual eligible control samples and completed timer events; if exposure is poor, report it rather than silently extending the budget.
4. Apply unchanged .5/.8/left/right/push/1.0 common-physics screen and original-physics old-function retention to **both finals**, including the full current21-case protocol. Require no standing/stopping regression, all prior retention gates, and actual lift/slip/turn checks. For the causal gait hypothesis, additionally require ≥25% lower mean diagonal duty gap over common .5/.8/1.0 than matched A, no case worsening by>.05, and visible repeated swing by all feet. Duty balance without lift/tracking/retention is rejected. Right-turn improvement is measured, not promised by a straight-only gate.
5. If both fail, retain3947 and distinguish A's insufficient adaptation from B's ineffective/adverse shaping. If A passes and B fails, prefer A. If B passes all requirements, freeze it before fresh-start confirmation and front/oblique visual inspection. No lowering thresholds, disabling self-collision, speed-cap escalation, or cherry-picked videos to rescue a result. Recovery→strict stand→walk→stop must then be tested continuously in the same physics without reset/history clearing before full-flow claims.

This is an engineering reward ablation using an upstream timing penalty, **not animal-motion imitation or a paper reproduction**.

## Implemented and smoke-tested

Root reviewed the trainer/launcher, reward and read-only exposure logger. The logger never reads the sensor's lazy `data` getter: it observes cached `_data`, verifies the pinned .005s/history3/timer/1N configuration, and cannot refresh reset rows. It excludes initial partial windows and all reset-interrupted windows, records actual eligible complete windows with at least2observed touchdown/liftoff events per foot, and batches GPU statistics every24steps. Raw switch counts are not debounced gait-frequency measurements. A minimum32such complete eligible windows is an exposure check, not acceptance.

CPU tests passed: reward19,trainer8,exposure8,candidate19. Serialsmokes completed with768actualenvsteps each. Control smoke SHA250a62617a8ef24eeed4c99180cfabf5bae3b85c62cd23ef49cd75748d328a1d exactly equals the historical commonphysics16x2model despite the new zero term/observer. This limited exact comparison supports nonperturbation of the instrumentation; it does not certify300updateequivalence. Balanced smoke SHA3d9bb483729bcab853ff2a90e49f4839228d55bd9815ef482d2c87dd903e0a50. Neither1s-smoke contains completed eligible windows; no gait claim follows.

Formal serialsession63935 is now active, control then balanced. Training receipt and retained files keep their inherited `common_physics_*` filenames inside NEW arm-specific run directories, but protocol is `balanced_gait_common_physics_training_v1` and arm/weight are explicit. NewtrainerSHA574e9cac8a5413293df898b6802fe5c0d1a3da9d34a00c0bb084b7e396dc79c5. Old trainers/evaluators/models were not edited.

## Formal completion, before behavior selection

Session63935 completedexit0. Each arm actually300updates/921600envsteps/7200controlintervals/17Adamcounters85120. Both full actual receipt verifications passed; all16preserved exports remain byte-identical.

|Arm|Run under repo/logs/rsl_rl/unitree_go2_flat|Final4246 SHA|Complete eligible windows with2rawtouch/lift events perfoot|
|---|---|---|---:|
|Control|2026-09-17_16-26-13_20260917-gait-control-128x300-first|ea9a8c19af0cdbcfbde3841551b7039be9ced62927fedc2790e7835aba6ee935|248|
|Balanced|2026-09-17_16-33-35_20260917-gait-balanced-128x300-first|882baefd00193c4b71bef2a47b5cb7382c37dd9e87b48f7f5863eaa364baebdc|247|

Training receipt SHAcontrol6ff3cb73fdb78ec33ac5ba7e66082903186692f5779dc45c6065217a8da6df6a; balanced299b45ea7c89e10107c4273e1d4867e546c3017bb8e18ee48cc9664427c91b9e. Raw all-command nonreset variance means.0356444/.0345256; these training-distribution measurements are not matched physical gait tests and cannot establish naturalness orspeed improvement.

Each final is queued for the same21actual cases, allnew executions. The comparison consumes all350walking samples at5.02–12.00s for common .5/.8/1.0, retains failures, and separately debounces contact state with2consecutive50Hzsamples. It excludes initial/terminal partial cycles; periods<.15s are an explicit short-cycle diagnostic, not a gait definition. The balanced arm must have>=2completecycles perfoot and noadditionalshortcycles relative tocontrol, in addition to>=25%mean duty-gap reduction/no>.05individualworsening andall21oldphysicalcriteria. Four comparisonCPUtests pass, including a measured4046CSVforce/flag regression. No behavior selection or promotion has occurred yet.
