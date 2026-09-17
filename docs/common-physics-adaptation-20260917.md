# Finite common-physics adaptation

Status: implementation prepared for review, not run or accepted.

Control3947 passed the three individual 0.8 m/s interventions (self-collision,
nominal soft clamp, and grouped fixed mass/CoM/material), but the full combination
failed the independent straight-line drift check. This does not identify a
single culprit or prove a particular interaction mechanism. The proposal is a
bounded adaptation to the required combined physics, not removal of that physics.

`train_common_physics_adaptation.py` uses three checked hooks in the frozen
official Isaac Lab trainer. It does not import or install the experimental
speed sampler. The actual task remains NaturalRobustPush TRAIN, including its
original command ranges, 30% standing commands, observation noise, ordinary
upright resets, pushes, rewards, actor/critic architecture and PPO settings.

Actual pinned TRAIN changes are self-collisions ON; nominal + 0.25 raw action
with control-step soft joint-limit clamping; and removal of the startup additive
base-mass randomization (old uniform range -1 to +3 kg). The TRAIN parent already
has no CoM randomization and already fixed material (static friction 0.8, dynamic
0.6, restitution 0); these remain enforced invariants, not additional changes.
Receipts include every actual YAML difference, the actual changed physical
paths, and declared physical fields that were already equal. Existing training push
events are deliberately retained: this is not a Play configuration substituted
for training. Actual complete YAML is checked against the pinned parent plus
only those physical fields and run metadata.

Both runs independently load original control3947, SHA
`3dfbe03e0ac332fee688c970b420167affcdb6e8cacee16e40fc12120751df3f`.
All actor/critic/std tensors and Adam moments/counters must exactly equal that
parent before learning. Scalar and optimizer LR both start at `1e-5`; adaptive
LR may subsequently evolve. This is full-checkpoint continuation, not exact RNG,
optimizer scheduling, or simulator-state continuity across process launches.

- Smoke: 16 environments × 2 updates × 24 steps = 768 environment steps;
  expected final label3948 and Adam79160.
- Formal: 128 × 100 × 24 = 307200 environment steps;
  expected final label4046 and Adam81120.

The formal launcher requires the successful same-source smoke receipt but does
not start from its model. Neither launcher automatically starts another run.
Unique run/output paths remain on E:. Existing files are not overwritten.
The launcher refuses overlapping E-workspace Python/Kit processes and never
stops them. Both actual step counts and all68 finite model/optimizer tensors are
verified. The16 portfolio exports and frozen roll1999/stand3547 are SHA-protected.

Example commands, to run only after source review and no active simulator:

```powershell
./scripts/run_common_physics_adaptation.ps1 -Mode smoke -RunTag 20260917-commonphysics16x2-smoke
./scripts/run_common_physics_adaptation.ps1 -Mode formal -RunTag 20260917-commonphysics128x100-first -SmokeReceipt <actual-smoke-run>/common_physics_training_result.json
```

`-PreflightOnly` performs read-only CPU/source/checkpoint checks without creating
a run or starting Isaac Sim. It still observes the process-exclusion guard.

Completion is training evidence only. Before promotion, compare the new actor
under full recovery physics at0.5/0.8 m/s, stand/stop, turns and pushes, with the
same original criteria and separate straight-drift gate. Also run the original
physics regression to measure loss of prior capabilities. Exact duplicate
trajectories under different seeds are one physical case, not independent
generalization evidence. No speed expansion, extra gait reward, fast-running,
continuous recovery integration, or hardware acceptance is claimed here.
