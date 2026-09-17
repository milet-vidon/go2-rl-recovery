# Frozen locomotion in recovery-compatible physics

Actual diagnostic completed on2026-09-17; no training or integrated-flow acceptance. Original control3947 SHA3dfbe03e0ac332fee688c970b420167affcdb6e8cacee16e40fc12120751df3f remains unchanged.

The new read-only instrumented original-physics0.5/seed20260909 rollout matches all18common JSON fields and the entire900-row CSV byte-for-byte against the original control screen. CSV SHA700b67843445c5e844e83a13a7f453ce16c59c0f6498cabeda01982cfa1f1ec2. Actual receipt: evaluations/20260917-locomotion-recovery-retention-v1/original-normal-seed20260909/original_exact_audit.json.

Then retained the original ordinary-upright reset and changed to SmithNominal physical/action semantics: self-collision ON, nominal soft-clamped position targets, fixed mass/CoM and material randomization. It did not use fallen bank starts or a recovery actor. Both configurations verify real policy48, native action history, targets and no auto-reset every step.

|Command|Actual vx|Actual vy|Slip|Old report checks|Separate straight-drift check|
|---|---:|---:|---:|---|---|
|0.5m/s|0.610329|-0.118895|0.063702|pass|pass, close to0.12bound|
|0.8m/s|0.874343|-0.126406|0.111503|pass|FAIL: abs(vy)>=0.12|

Each command was run with seeds20260909/10/11. IMPORTANT: with fixed physical randomization and diagnostic reset, all three seeds produce byte-identical900-row CSVs at a given command. They are repeated deterministic checks, NOT three diverse initial states or robustness evidence. 0.5CSV SHAba4a0eaf216b8ef822356b89e066bc09b5ce750936b70d67cdc1923b1a4136d4;0.8SHA44fbf6be17b891ff5aebd4d4955262a8bcbf27d90581ccf83958a9be1adcd95c. All seven wrappers exited0 with complete interface traces and unchanged protected models; process completion is not behavioral success.

Decision: compatibility screen FAILED, so no integrated recovery/walk/fast-run claim and no reward change used to conceal this difference. Next bounded diagnostic separately tests original+collision-only, original+softclamp-only, and original+fixed-randomization-only, at0.8seed20260909. Keep the original thresholds, actor, reset and18s command sequence. These single-factor variants are attribution experiments, not acceptable substitutes for full recovery physics. The running/trot-duration proposal remains pending this diagnosis.

## Completed single-factor diagnostics

All three actual cases completed with exit0 and separate interface evidence under `evaluations/20260917-locomotion-physics-ablation-v1`. Each uses original control3947, command0.8, seed20260909 and unchanged4s stand/8s walk/6s stop; none changes all recovery physical factors together.

|Only changed factor|Actual vx|Actual vy|Slip|Old checks + straight drift|
|---|---:|---:|---:|---|
|Self-collision enabled|0.748739|-0.060475|0.071359|pass|
|Nominal soft-clamped targets|0.778416|0.010090|0.116786|pass|
|Remove base-mass randomization (CoM/material already fixed)|0.756124|-0.041134|0.088745|pass|

Independent read-only audit checked all report/trace/generated hashes,65 source/input hashes per case and2700 actual action/history/float32 target records. The fixed-randomization label groups mass/CoM/material, but its ACTUAL changed config is only add_base_mass to None; CoM already None and material already .8/.6/0 in original PLAY and TRAIN. Actual base mass changes8.174316 to6.921000kg. Self-collision-only CSV is byte-identical to the original .8seed09 trajectory, which means no measured trajectory effect in this case, not that collisions never matter. Soft-clamp actually clipped417/900 intervals and685/10800 joint-components; its slip .116786 and FL lift p95 .040404 only narrowly pass their .12/.04 thresholds.

The individual passes do not repair or invalidate the combined-physics failure. These are nonlinear closed-loop systems; the evidence does not identify a unique offending factor. Do not disable self-collisions, relax lateral drift or substitute single-factor scores for common-physics performance. Next bounded hypothesis is full-state fine-tuning of control3947 in the ACTUAL combined physical/action configuration, retaining original command sampler/rewards/PPO/reset/pushes, before any new gait reward or higher-speed curriculum. This adaptation is being implemented/reviewed, not yet trained or accepted.
