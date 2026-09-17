# Supported startup plus roll reflection: joint development diagnostic

Declared before the first combined-controller simulator run. User requests autonomous continuation and self-review. Keep every original actor, source, report and failed speed-training candidate. One simulator at a time; current speed-gate18screen must finish first.

## Why this experiment

Independent startup selector retained ordinary supported starts20/20 but side9/20; independent roll reflection retained side20/20 and back20/20 but ordinary starts12/20. These are DIFFERENT controllers and their successes cannot be combined on paper. This experiment implements their real combination and checks exact measured behavior. It is a frozen two-actor inference/controller experiment, NOT further learned recovery training or integration with moving commands.

At the actual policy-start boundary after unchanged1s nominal-position PD, compute S from the existing measured supported-nonfallen predicate and M from eligible settled-fallen plus normalizedgravity-y<-cos30degrees. S requires eligibilityfalse and M eligibilitytrue; assert disjoint rather than silently choose priority. Latch both for the episode. Gate events retain their actual meaning; selected S stays stand from the first action without a fabricated handoff. Native standing actor always gets real observations. Roll mirror remains J12(roll(M48(real))) only on M. Do NOT disable M after the stand actor becomes active: preserve the old candidate/logging semantics. No physical state/history reset, additional PD, new force, retry, ramp, changed reward or loosened success criterion.

## Finite validation matrix

Freeze roll1999 SHA71e4c7a39ff81836d2ead1c4533988c696e5bd4a0fc939b8cef6b4f6daec1b1c and stand3547 SHA5527c4053cb6ef8e6616da65b7dc32a9464eb8d1366f4966a4eb4b65bc913fdb. Four flag combinations, each upright/side/upside_down20trials at seed20260918, in cold sequential processes:

| Startup | Mirror | Exact reference |
|---|---|---|
| OFF | OFF | Original hard diagnostic, all3poses |
| ON | OFF | Startup-only ON, all3poses |
| OFF | ON | Mirror-only ON, all3poses |
| ON | ON | Upright against startup-only ON; side/back against mirror-only ON |

All first9control comparisons must pass before both-ON. The comparator pins all9historical report hashes. Require exact common results/starts/physical metadata, trial0 CSV and every stored all-trial neighborhood (zero tolerance). Only explicit protocol/controller labels and audited artifact hashes/mode annotations may differ. Independently recompute startup/mirror selections, disjointness, first actions, real/virtual48 and12coordinate mapping, actual issued action, natural history and float32 clamped joint targets. The small2e-7 pre-existing tolerance applies ONLY to normalized-gravity recomputation, never trajectories or actions. Invalid data or any mismatch aborts and preserves evidence; do not weaken checks to force equivalence.

Strict final standing criteria stay unchanged: upright/height/motion, noncrossed foot/knee layout and joint offsets, simultaneous CURRENT4vertical foot contacts, no base contact and3s continuous hold at ending. Ordinary supported starts are not settled-fallen recovery. The repeated bank includes only16/14unique side/back states; the20draws are not20independent states. Even exact12-case success would establish this repeatedly used development combination only, not fresh initial-state generalization, normal walking, running or the reference video.

CPU source-generation/mixed-mask tests and comparator mutation tests must pass before simulation. After successful combined cases, record and actually inspect front/oblique full videos; new independent seeds/states and uninterrupted stand/walk/fall/recover/walk/stop still need separate tests. No automatic promotion. Actual execution status belongs in CURRENT, not implied by this plan.
