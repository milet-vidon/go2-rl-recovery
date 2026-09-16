# Grounded recovery: posture-penalty gate ablation

Status: in progress, simulation only. Neither arm is accepted or recommended. This is a bounded diagnostic experiment, not full reference-video reproduction.

## Evidence motivating the change

The1,000-iteration Bank pilot4899 failed every heldout grounded recovery: side0/20 and back0/20, covering16 and14 independent states respectively. The extra trials repeat source states. All starts met the settled-fallen eligibility screen. Side final tilt61.26–71.00degrees(mean65.72),height0.1503m,one vertical foot contact in every trial. Back remains approximately180degrees,height0.057m,zero feet supporting. The30degree side regression fell to10/20 and45degree side to0/20, so4899 cannot replace existing controllers.

The old anatomical penalty starts at60degrees and includes foot/knee crossing, width, symmetry and nominal joint deviation. All side finals lie just outside that boundary. At their unchanged leg shapes, advancing to50degrees would introduce a conservatively estimated weighted penalty of at least2.30, versus roughly0.3 orientation/height shaping benefit. This suggests a barrier along the recovery path; it does NOT establish causality. At the actual final61–71degree states, both old and proposed gates are zero.

Recorded and visually inspected all four front/oblique side/back diagnostics, including initial and final frames. They show genuine ground-supported lying poses and failure to stand. The camera now uses a world-origin viewer plus explicit yaw-relative frame tracking; the previous asset-root render callback was overwriting the custom first-frame view. No physical parameters changed.

- [Heldout4899 report](../evaluations/20260916-bank4899-heldout/model_4899.pt_recovery_metrics.json)
- [30degree regression](../evaluations/20260916-bank4899-angle30/model_4899.pt_recovery_metrics.json)
- [45degree regression](../evaluations/20260916-bank4899-angle45/model_4899.pt_recovery_metrics.json)
- Failure diagnostics: `evaluations/20260916-bank4899-diagnostic-front/` and `...-oblique/`. Their green LEG GEOMETRY OK label refers only to leg shape in body coordinates, not recovery; stand-hold stays0. Future recordings reserve green for a completed valid stand and use amber for leg-shape-only agreement.

## Controlled comparison

Both arms load the exact final4899 checkpoint and preserve its learned Gaussian std and Adam state; no exploration reset. Same seed42,512 environments,24 rollout steps,unchanged PPO. Both fix bank sampling at60%, matching the pilot's plateau; otherwise resuming would reset the global counter and restart the easier20→60% mixture.214 train states only. The30 heldout states never enter training.

| Arm | Anatomical penalty gate | All other rewards/physics |
| --- | --- | --- |
| Control | clamp((cos(up)−.50)/.40,0,1) | Unchanged |
| Release | clamp((cos(up)−.85)/.10,0,1) × clamp((height−.24)/.04,0,1) × indicator(at least2 current feet Fz>5N) | Identical to Control |

Release begins near31.8degrees and reaches full orientation factor near18.2degrees. Ordinary high,near-horizontal supported stance keeps the full original anatomical penalty. The support indicator is discrete; only orientation/height ramps are continuous. A robot can evade shaping by staying tilted/low or lifting feet: external success still rejects that behavior. Back-down states already had gate0, so no direct inversion cure is claimed.

Initial budget100 additional iterations per arm,1,228,800 environment steps each. Maximum400 per arm only if measured trajectory or true recovery improvement justifies extension. Execute arms and evaluations serially to fit the8GB GPU. The orchestration script exits after both arms and does not automatically promote models or extend training.

Parent path: `E:/IsaacLab/repo/logs/rsl_rl/unitree_go2_recovery/2026-09-16_13-16-04_bank_nominalpd_from3900_20260916/model_4899.pt`.

Parent SHA256: `a021ba7a25d5a9803f7f905b4a242df8582a4d2730de5843303e76beaa26b4e2`.

Bank SHA256: `71b882e92035b4c248e5980339c980dbb242dcb1ec711c05252c33249db48f5a`.

Launch performed2026-09-16 14:19 local; do not rerun while active:

```powershell
Set-Location E:\IsaacLab\go2-rl-open-source
./scripts/run_recovery_gate_ablation.ps1 -RunTag 20260916-gate100 -Iterations 100
```

Orchestrator log: `E:/IsaacLab/artifacts/recovery-20260916/gate100-orchestrator.log`, session58543. Control run: `2026-09-16_14-19-08_20260916-gate100_control`. Actual config shows60% fixed bank,old aligned penalty,self-collisions and0.33m stand target. Both branch configs are copied into `configs/20260916-gate100-{control,release}/` when their training completes.

Each final checkpoint is evaluated in the SAME existing Bank Play task:seed20260918,20 trials each side/back,heldout split,1s nominal-PD handover,8s horizon+3s hold. Same IDs/order as the3900/4899 baselines. Also Aligned Play30degree seed20260918 and45degree seed20260916,upright included. Final acceptance still requires correct foot/knee/joint geometry,simultaneous four vertical contacts,clear base,height.30–.55m,small motion and3s continuous retention. Do not count partial righting or mean training reward as success.

Before launch:13 new gate/reward/config tests,19 bank-data tests,10 subset-reset tests and earlier recovery-math tests pass. Legacy reward outputs are bitwise unchanged in regression tests; new task configs differ only bygate mode. Overlay installed with backup `E:/IsaacLab/artifacts/install-backup-20260916-141830`.

## Primary research cross-check; not a claimed reproduction

[Deng et al., Learning to Recover (2025), sectionsIII-D/E andIV-B](https://arxiv.org/html/2506.05516v1) uses episode-time-dependent shaping and a curriculum to allow early recovery motion before stronger posture/smoothness constraints. Its final reward equation weights terms selectively, not simply every term identically. The reported platform is wheeled-legged,including Go2-W; their success rates and0.42m height threshold cannot be transferred to this plain Go2. Our state-based gate is a distinct hypothesis,not their episode-time formula.

[Lee et al., Robust Recovery Controller (2019), sectionsII-A andII-D2](https://arxiv.org/html/1901.07517) separates self-righting,standing-up and locomotion. It uses current-joint-relative position targets for righting/standing and nominal-pose offsets for locomotion. This provides a concrete alternative action-representation hypothesis if the gate test fails; it is NOT implemented in this ablation. Changing the offset of a pretrained nominal-target policy without a verified conversion would change its deterministic behavior, so it must be a separate experiment.
