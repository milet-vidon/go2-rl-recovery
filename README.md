# Go2 RL Recovery and Locomotion

An Isaac Lab project for Unitree Go2 locomotion and self-recovery experiments.

This repository follows the behavior separation used in quadruped RL work: a
standard locomotion policy handles quiet standing and commanded walking, while
a recovery policy is evaluated separately from fallen poses. It is not a
claim of sim-to-real safety or hardware readiness.

## Repository layout

```text
src/go2_recovery/   Task terms and environment configurations
scripts/             Training, playback, and deterministic evaluation
models/              Selected checkpoints (Git LFS recommended for more)
videos/              Reproducible debug demonstrations
evaluations/         JSON metrics with explicit success criteria
docs/                Research notes and references
```

## Requirements

- Windows + Isaac Lab / Isaac Sim 5.1
- Python environment at `E:\IsaacLab\env` (or adapt the scripts)
- NVIDIA GPU with CUDA support

All generated caches, logs, models, and videos in the supplied scripts are
redirected to `E:\IsaacLab`.

## Run a deterministic evaluation

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

## Play a policy

```powershell
& E:\IsaacLab\go2-rl-open-source\scripts\play.ps1 `
  -Stage standard_stance -Static `
  -Checkpoint E:\IsaacLab\go2-rl-open-source\models\locomotion\standard_stance_model950.pt
```

## Current evidence

The included recovery checkpoint achieved 17/20 on 30-degree fore-aft starts
and 1/20 on 30-degree side starts in the recorded evaluation. The included
standard-stance checkpoint is an intermediate training artifact and should not
be treated as a deployment-ready controller. The videos are demonstrations,
not safety validation.

## Research basis

See [`docs/research-notes.md`](docs/research-notes.md). Key references include
Lee et al. (2019) on hierarchical recovery, Hwangbo et al. (2019) on robust
quadruped locomotion, and Kumar et al. (2021) on RMA.

## License and provenance

The task code is derived from Isaac Lab and retains its upstream BSD-3-Clause
license headers. New project glue, experiment metadata, and scripts are offered
under MIT in [`LICENSE`](LICENSE). Robot assets and Isaac Sim are not
redistributed here; obtain them from their respective vendors.
