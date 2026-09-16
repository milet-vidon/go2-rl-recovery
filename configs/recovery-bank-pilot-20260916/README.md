# Grounded-state-bank pilot — in progress

This experiment has not passed recovery acceptance and is not a recommended controller.

- `env.yaml` / `agent.yaml`: actual configuration dumped by the512-env run `2026-09-16_13-16-04_bank_nominalpd_from3900_20260916`.
- `bootstrap-lineage.json`: original3900 parent hash, sampling-std0.6 change and cleared Adam moments. Deterministic actor/critic tensors were preserved and checked after serialization.
- The original parent is archived at [`models/recovery/recovery_rehearsal_model3900.pt`](../../models/recovery/recovery_rehearsal_model3900.pt), SHA256 `70f1429040ebe3db1f8bdcbe246bff0c791f50c9739a46568932332e1ffe81aa`. Its45-degree drop improvement does not imply settled-fall recovery; the heldout bank baseline is0 recovered states.
- Bank: [`nominal_pd_v1_20260916`](../../datasets/recovery_states/nominal_pd_v1_20260916/manifest.json),214 training states and30 heldout states, fixed physics, nominal-position PD preparation and cold replay. This is NOT zero-torque passive fall sampling.
- The requested1,000 further iterations are still running. No new pilot checkpoint has been evaluated or promoted at this record's creation.

Use the [live training status](../../docs/training-status-20260916.md) to find the active process and follow-up protocol. Do not duplicate a running job or use the heldout bank for training. Retain normal standing/walking/stopping regression and visually check front/oblique videos before choosing any new model.
