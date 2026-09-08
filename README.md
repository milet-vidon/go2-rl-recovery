# Go2 RL Recovery & Locomotion

> A portfolio-grade Isaac Lab project for learning coordinated Unitree Go2 locomotion and reactive self-recovery from controlled falls.

[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-5.1-76B900?logo=nvidia)](https://isaac-sim.github.io/IsaacLab/)
[![Robot](https://img.shields.io/badge/robot-Unitree%20Go2-111827)](https://www.unitree.com/go2)
[![Task](https://img.shields.io/badge/task-quadruped%20RL-2563EB)](https://github.com/leggedrobotics/legged_gym)
[![License](https://img.shields.io/badge/license-MIT-22C55E.svg)](LICENSE)

This project explores a clean separation between two learned behaviors:

1. **Standard locomotion policy** for quiet standing and commanded walking.
2. **Recovery policy** for returning from controlled fore-aft, side, and upside-down starts.

The separation follows the hierarchical recovery idea of Lee et al. and the simulation-first workflow used by modern legged RL repositories. The current artifacts are research demonstrations, not a claim of sim-to-real safety or hardware readiness.

## Portfolio Snapshot

| Area | Implementation |
| --- | --- |
| Simulator | NVIDIA Isaac Sim 5.1 / Isaac Lab |
| Robot | Unitree Go2, 12-DoF position policy |
| Learning | PPO with RSL-RL |
| Locomotion | Velocity tracking, foot air-time, torque and action smoothing |
| Recovery | Orientation, height, support-contact and stable-stand shaping |
| Evaluation | Deterministic pose classes with a 3 s stable hold criterion |
| Hardware | Not tested; simulation only |

## Demonstrations

- [Standard stance video](videos/standard_stance_model950.mp4)
- [Recovery from a 30-degree fore-aft start](videos/recovery_fore_aft_30deg.mp4)
- [Recorded recovery metrics](evaluations/recovery_stage30_30deg.json)

The videos are intentionally kept as experiment artifacts. They make it easy to inspect failure modes instead of presenting an unverified success montage.

## Results So Far

The included recovery checkpoint was evaluated with 20 deterministic trials per pose class. On 30-degree starts it achieved **17/20 fore-aft recoveries** and **1/20 side recoveries**. Full side-fall and upside-down recovery are still open research tasks in this repository.

The standard-stance checkpoint is an intermediate training artifact. Use it to reproduce the current experiment, not as a deployment controller.

## Repository Structure

```text
src/go2_recovery/       Environment configs and reward terms
scripts/                Train, play, evaluate, and regression-test utilities
configs/                Experiment manifest and reproducibility metadata
models/recovery/        Selected self-recovery checkpoint
models/locomotion/      Selected standard locomotion checkpoint
videos/                 Portfolio demonstrations and debug recordings
evaluations/            Machine-readable metrics and criteria
docs/                   Research notes and paper references
```

## Quick Start

### Play the standard policy

```powershell
& E:\IsaacLab\go2-rl-open-source\scripts\play.ps1 `
  -Stage standard_stance -Static `
  -Checkpoint E:\IsaacLab\go2-rl-open-source\models\locomotion\standard_stance_model950.pt
```

### Evaluate recovery deterministically

```powershell
Push-Location E:\IsaacLab\repo
& .\isaaclab.bat -p scripts\environments\evaluate_go2_recovery.py `
  --task Isaac-Recovery-Flat-Unitree-Go2-Play-v0 `
  --checkpoint E:\IsaacLab\go2-rl-open-source\models\recovery\recovery_stage30_model2200.pt `
  --output_dir E:\IsaacLab\go2-rl-open-source\evaluations\local_run `
  --trials 20 --poses fore_aft --angle_deg 30 --headless `
  --kit_args=--/app/vulkan=false
Pop-Location
```

### Train

```powershell
& E:\IsaacLab\go2-rl-open-source\scripts\train.ps1 `
  -Stage standard_stance -NumEnvs 128 -MaxIterations 4000 -Headless
```

The scripts redirect Isaac Sim user data, temporary files, caches, logs, and outputs to `E:\IsaacLab`.

## Method Notes

The standard policy uses the official flat-ground Go2 task as its baseline, with an explicit quiet-stance term added for the zero-command case. The recovery policy uses a controlled reset distribution and conservative rotated body clearance. Stable success requires upright gravity alignment, nominal height, low linear/angular speed, at least two foot contacts, and 3 seconds of continuous hold.

This design is informed by:

- Lee, Hwangbo & Hutter, *Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning* (2019).
- Hwangbo et al., *Learning agile and dynamic motor skills for legged robots* (Science Robotics, 2019).
- Kumar et al., *RMA: Rapid Motor Adaptation for Legged Robots* (2021).
- Margolis et al., *Walk These Ways* (2022).

See [`docs/research-notes.md`](docs/research-notes.md) for links and the engineering interpretation used here.

## Limitations and Roadmap

- Train the standard stance policy to a mature checkpoint before hardware use.
- Replace the single recovery policy with a behavior selector and specialized roll/pitch recovery skills.
- Add randomized contacts, friction, payload, actuator delay, and sensor noise.
- Validate with hardware-specific safety limits and an emergency stop.

## Provenance and License

Task code retains the upstream Isaac Lab BSD-3-Clause headers. New project glue, experiment metadata, and scripts are MIT licensed in [`LICENSE`](LICENSE). Go2 USD assets and Isaac Sim are not redistributed; obtain them from Unitree and NVIDIA under their respective terms.
