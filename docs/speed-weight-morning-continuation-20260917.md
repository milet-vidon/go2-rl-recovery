# Speed-weight experiment: morning continuation, not acceptance

The user's September17 morning continuation resumed a stopped workflow. The original pair summary still said `running`, but there was no workspace Python/Kit process. A completed; original B stopped mid-training. No old weights, logs or results were overwritten.

## Complete A result

All18 original report hashes, identities, pass flags and failed criteria agree with [A's summary](../evaluations/20260917-speed-weightA3996/summary.json). The original parent is control3947, SHA `3dfbe03e0ac332fee688c970b420167affcdb6e8cacee16e40fc12120751df3f`. A's50-update endpoint3996 is SHA `121c9c9ef69522a9527d22a2650a8f556f5ec693a14ad22cf6af0c9407bd3d7b`.

| Development screen | Control3947 | A3996 |
| --- | ---: | ---: |
| Original15 walking/turn/push reports |15/15|14/15|
| Target1.0m/s reports |0/3|1/3|
| All reports |15/18|15/18|

A is rejected by the predeclared stop-loss. Its only old-function failure is right-turn seed20260911, **lateral_tracking**, actual body lateral velocity−0.139289502756936m/s versus zero command, beyond the0.12m/s limit. This is lateral drift during turning: that report's static/stop geometry and four-foot support passed. Do not mislabel it a standing/stopping failure.

Target seeds20260910/11 fail only **limited_slip**, with mean contact slip0.15030603777276/0.13091514207851m/s, limit<0.12. Target mean forward-speed absolute error improved to0.09330562216894966m/s, but speed improvement does not compensate for slip or old-function regression. No second A block or promotion.

## Interrupted original B

Original run `E:/IsaacLab/repo/logs/rsl_rl/unitree_go2_flat/2026-09-17_05-34-54_20260917-weightB128x50-first` has only model3950. Its train log ends normally at3972/3997 and79,872steps, without final3996, config/checkpoint receipt or regression. Searches of the training, wrapper and pair logs found no recorded traceback, fatal, runtime exception, keyboard interrupt, CUDA/OOM or error marker explaining termination. The cause is unknown; do not infer shutdown, crash or memory exhaustion from absence of a final report.

Preserved original evidence: `E:/IsaacLab/artifacts/recovery-20260917/20260917-speed-weight-first-pair` and `20260917-weightB128x50-first`. The old summary is retained as evidence of the interrupted process, not silently relabeled as a completed pair.

## Independent B replacement block

A fresh B128env x50update block started10:02 local, from the same frozen control3947, not the partial3950. The tag's `0958` is only an identifier. Only XY tracking weight changes1.5→2.0; original command ranges/30%stand, PPO, other rewards, physics and fixed evaluation criteria remain unchanged. Both completed arms have153,600new environment steps and matching update budgets; the failed partial attempt is preserved but not added to B's replacement actor lineage.

Run: `E:/IsaacLab/repo/logs/rsl_rl/unitree_go2_flat/2026-09-17_10-02-05_20260917-weightB128x50-restart0958/model_3996.pt`.

SHA: `6262f14be50bc8bf435505b5af9258eeec37ec014d7d38f825b8180aa2dde665`.

Full actual/archive configuration comparison and checkpoint guard passed:68finite tensors,17Adam counters80120, expected50updates. Receipt: `E:/IsaacLab/artifacts/recovery-20260917/20260917-weightB128x50-restart0958/config-checkpoint-check.json`. All16 frozen export files remain byte-identical; that is fallback preservation, not proof of behavioral retention.

B's18-case serial regression started10:04:53 and completed10:21:34: [complete summary](../evaluations/20260917-speed-weightB3996-restart0958/summary.json), SHA `b0477c56e240c516c137fcec03ae6552ae7330dd4b8ed7c362e9277498805448`. Main checked all18 original report hashes, identities, failed criteria and continuous metrics against the summary. Old functions15/15 and straight-drift gates pass, target1.0m/s0/3, total15/18. Log: `E:/IsaacLab/artifacts/recovery-20260917/20260917-weightB-restart0958-regression.log`.

All three target cases fail `walk_tracking`: actualvx0.8100486944402967/0.8332453310489655/0.8356825772353581m/s. Seeds10/11 also fail `limited_slip`,0.1386625057388447/0.1276303834390065m/s; seed09 slip0.08030088995795885 passes. Mean target absolute error0.1736744657584599 worsens0.012121327036903023 versus parent. Seed09 alone worsens0.030356368848255766, exceeding the preregistered0.02margin. Therefore B fails both stop-loss and futility. No second block or promotion. A is faster but regresses old turning/slip; B preserves the15 old development pass flags but fails speed/slip. Neither is a successful speed candidate, and this single pair is not a causal/generalization proof. No higher-speed, recovery integration or new visual acceptance follows.

## Recovery work remains separate

The supported-startup predicate passed18CPU tests; a separate simulator entry is in preparation. It investigates why an ordinary supported start was sent to a rolling expert, not a new gate, hidden PD phase or relaxation of standing acceptance. Mirror-assisted side recovery has promising development evidence but is not yet combined with startup retention or walking/running in one uninterrupted controller. See the current [training status](training-status-20260916.md) for actual active processes and next work.
