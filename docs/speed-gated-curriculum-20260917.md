# Bounded reward-gated velocity sampling pilot

Predeclared before any new simulation on September17. User authorized continued local training. This is a separate locomotion branch, not integrated fallen recovery, animal imitation, or a replacement for frozen models.

## Evidence and single-factor hypothesis

Control3947 retains15/15 old-function development checks but fails all3 command1.0m/s cases. Original-distribution additional50update armA loses a turning check; reward-weight2.0 armB does not improve target speed sufficiently and is rejected. Neither is continued.

[Rapid Locomotion](https://arxiv.org/abs/2205.02824) identifies an adaptive velocity curriculum as one component. The author's [RewardThresholdCurriculum source](https://github.com/Improbable-AI/rapid-locomotion-rl/blob/main/mini_gym/envs/base/curriculum.py) uses jointly passing linear/angular reward thresholds to expand successful/neighbor sampling weights. Our simpler forward-cap gate is explicitly an engineering hypothesis inspired by that principle, NOT the author's multidimensional bin algorithm, system identification or hardware reproduction.

## Fixed intervention

Independently resume full control3947, SHA3dfbe03e0ac332fee688c970b420167affcdb6e8cacee16e40fc12120751df3f. Keep PPO, actor/critic/std, Adam moments, initial actualLR1e-5, rewards, physics, pushes, command timing and seed42. Preserve30% standing, original negative-vx draws, lateral/yaw draws and resampling RNG calls. Only positive moving vx draws are multiplied by a global cap starting0.8, then0.9, finally1.0. Original configured range remains[-0.5,1.0]; actual positive range is[0,cap]. No added random draws does NOT imply whole-rollout RNG/physics trajectories remain equal after changed commands.

Measure unweighted exp(-squared_xy_error/0.25) and exp(-squared_yaw_error/0.25), matching the actual parent reward kernels with std0.5. Do not compare weighted reward sums against raw thresholds. On each naturally completed4–8s command window, discard its first0.5s. Any reset/termination/timeout interrupts and discards that window. Only nonstanding forward windows sampled at the CURRENT cap, with command in[cap-0.2,cap] and at least3.5s measured data, enter the gate. Both average raw scores must be strictly>0.8. A nonoverlapping32-window block with at least80% successful windows raises cap by0.1, max1.0; unsuccessful blocks leave cap unchanged. No shortened/reset window or old-cap observation can promote. All window records and decisions are retained. These thresholds are development training controls, not behavioral acceptance, and will not be tuned to rescue this candidate.

## Budget, checks and stopping rule

Run16env x2update interface smoke first (768steps, not quality evidence); then, only if guards/tests pass, independently restart the same parent for ONE128env x50update block (153600steps,24simulated seconds per environment). This may be insufficient to open the full course; report actual exposure and cap instead of silently extending. No second block or higher speed is authorized by this specific experiment plan. Exact parent weights/moments and full actual saved configuration are checked before learning; actual final finite tensors and Adam counters are checked afterward. New entry layers onto SHA-pinned official trainer without editing historical or installed sources. Preserve all16 exported models throughout.

After training, run the unchanged18case stand/walk/stop/0.8/1.0/turn/push screen with seeds20260909/10/11. Compare both frozen control3947 and already completed matched-budget armA; reused development controls are not fresh generalization. Any old15 failure or straight-line drift regression rejects promotion. Target1.0 must also pass original speed and contact-slip(<0.12m/s) criteria; command cap or training reward is never proof of actual running. Record per-seed differences. A promising result needs actual front/oblique videos, fresh confirmation, and separately integrated recovery regressions before replacing any functional baseline.

Implementation files: scripts/train_speed_gated_curriculum.py, scripts/speed_gated_command.py, src/go2_recovery/speed_gate_math.py, scripts/run_speed_gated_curriculum.ps1. See CURRENT status for live execution.

## Actual execution (complete; candidate rejected)

All20pure-stdlib gate tests and6CPU mocked command-lifecycle/official-source generation tests passed. First smoke failed BEFORE learning on a read-only CPU/GPU Adam-counter equality comparison. Its failed directory/log remains intact. The corrected comparator transfers only its comparison copies to CPU, never changes runner state. New retry run `2026-09-17_11-12-18_20260917-speedgate16x2-smoke-r2` completed768steps with exact parent model/moments, full actual-config guard,68finite tensors and17Adam counters79160. It had no complete command window and is only an interface smoke.

Formal run `E:/IsaacLab/repo/logs/rsl_rl/unitree_go2_flat/2026-09-17_11-14-21_20260917-speedgate128x50-first` completed independently from3947:128env x50updates=153600steps,1200control steps per environment; final3996 SHA `41db1ebbae4ee466f99adffd48e68a6be907e928008a5aa992e0ef1bc9fb376b`. Its source/config/full-parent-state/finite-checkpoint guards passed. All16exported baselines remain unchanged, which is not a behavioral retention claim.

Observed course:657resamples,190standing/157reverse/310positive draws;378naturally completed windows,138reset-interrupted windows discarded. There were33eligible top-band windows. The first32contained30joint-score passes and promoted cap0.8→0.9; only1eligible0.9window completed afterward, so the1.0stage was NOT reached. The pilot did not contain enough post-promotion exposure for a second gate decision; do not reinterpret this as a gate failure or silently extend the frozen budget. Curriculum counters do not establish robot task success.

The18case evaluation started11:15:37local in `evaluations/20260917-speed-gate3996-first` and finished11:34:05.9786649, exit0. The candidate passes10/18 overall, oldfunctions10/15 and target1.0m/s0/3. Failed walk_tracking cases: normal-seed20260909, all3retained08, all3target10 and right-seed20260909. Normal09 also fails separately declared straight-drift: commandedvx0.5, actualvx0.37337553679943086 and vy0.12492256503697717 (>0.12 bound). ALL18quietstand-stop/restgeometry/current4verticalcontacts/no-currentbase/combinedrestsupport checks pass. Do not misdescribe the speed regression as a standing failure.

| Same18case screen | Old15 passes | Target1.0 passes | Mean target absolute vx error(m/s) |
|---|---:|---:|---:|
| Frozen control3947 |15/15|0/3|0.161553|
| Original-distribution50update A |14/15|1/3|0.093306|
| Weight2.0 B |15/15|0/3|0.173674|
| New bounded speed-gate50 |10/15|0/3|0.236431|

Speed-gate targetvx=0.7127741384506225/0.7858153646332877/0.7921165805203574m/s; corresponding contactslip=0.061156446550620816/0.11688270232687159/0.10810560202533495m/s. Target slip stays below0.12 but actual speed is insufficient. Its mean target error worsens0.07487816674368725m/s versus control. An independent read-only audit checked all72 source reports from control/A/B/gate against summary hashes and reported pass/failure and motion fields; no discrepancies. These are reused development comparisons, not fresh generalization.

The candidate is REJECTED for promotion or unchanged extension. This bounded capped-command experiment did not improve the required behavior; it does not prove that longer training would fix it or that the author's different multidimensional curriculum fails. Old models remain available and unchanged. Previous real standing video remains the latest visually checked artifact; no new candidate video or integrated recovery success is claimed.

Independent read-only source review confirmed command-reset/compute ordering, old-command window attribution, actualbody-frame raw reward kernels and before-learning full-resume validation. It did not independently rerun tests or simulator. Six new source/test/launcher files are copied with byte-hash verification under `E:/IsaacLab/artifacts/recovery-20260917/20260917-speedgate128x50-source`. The curriculum is not saved in the PPO checkpoint for restart; this finite branch starts a new course from3947 and must not be described as resumable simulator/RNG/command-state continuity.
