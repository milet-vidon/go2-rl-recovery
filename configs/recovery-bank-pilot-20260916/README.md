# Grounded-state-bank pilot — completed, rejected

This experiment has not passed recovery acceptance and is not a recommended controller.

- `env.yaml` / `agent.yaml`: actual configuration dumped by the512-env run `2026-09-16_13-16-04_bank_nominalpd_from3900_20260916`.
- `bootstrap-lineage.json`: original3900 parent hash, sampling-std0.6 change and cleared Adam moments. Deterministic actor/critic tensors were preserved and checked after serialization.
- The original parent is archived at [`models/recovery/recovery_rehearsal_model3900.pt`](../../models/recovery/recovery_rehearsal_model3900.pt), SHA256 `70f1429040ebe3db1f8bdcbe246bff0c791f50c9739a46568932332e1ffe81aa`. Its45-degree drop improvement does not imply settled-fall recovery; the heldout bank baseline is0 recovered states.
- Bank: [`nominal_pd_v1_20260916`](../../datasets/recovery_states/nominal_pd_v1_20260916/manifest.json),214 training states and30 heldout states, fixed physics, nominal-position PD preparation and cold replay. This is NOT zero-torque passive fall sampling.
- Completed1,000 additional iterations (12,288,000 environment steps) at2026-09-16 13:40 local. Final checkpoint4899 is NOT promoted: heldout settled side0/20 (16 unique states), back0/20 (14 unique states); both final valid-stand counts0. Repeats are not independent starting states.
- Aligned controlled-drop regression:30degrees upright20/20,side10/20,fore/aft18/20;45degrees upright20/20,side0/20,fore/aft10/20,inversion0/20. These regressions reject replacing the earlier controllers.
- Side final inclination61.26–71.00degrees (mean65.72), height0.1503m, one supporting foot in every trial. Back remains approximately180degrees and0.057m with zero feet supporting. A body-frame leg-geometry flag while inverted is not standing.
- Reports: `evaluations/20260916-bank4899-heldout/`, `...-angle30/`, `...-angle45/`. A separate single-factor penalty-gate experiment is being prepared; neither more training time nor higher aggregate reward is acceptance evidence.

Use the [live training status](../../docs/training-status-20260916.md) to find the active process and follow-up protocol. Do not duplicate a running job or use the heldout bank for training. Retain normal standing/walking/stopping regression and visually check front/oblique videos before choosing any new model.
