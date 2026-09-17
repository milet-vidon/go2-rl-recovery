# Independent comparator review

Static review on2026-09-17, while serial job9458 was active. No Python, simulator or active/pinned source was run or changed. Reviewed comparator SHA `7518b1e1431207d7779778348a48ab825f7855d6d4f9f2b2974d3a01c2ece702`; tests SHA `a21eee2544c2e96dd9096476d4baaaa776df0753136bbbeb022492614717c10a`.

## Findings

1. **P1: summary labels are not bound to the actual21 experiments.** In `load_screen` (lines84–105), the21 row-label set is correct, but each report is only checked for checkpoint hash and identical `retention_passed`. Actual report physics, seed, commands, protocol and candidate arm/receipt are unchecked. Repeating one same-checkpoint report under several distinct row labels can therefore pass the screen-identity test. The three straight cases can silently measure the same CSV under different names. Report/CSV hashes alone do not prove distinct case identity. The active screen runner already performs stronger checks; its invocation evidence should be bound independently before treating comparator results as a complete21-case comparison.
2. **P2: a zero control gap falsely satisfies a percentage-reduction claim.** `mean_b <= .75*mean_a` is true for0/0. The selection wording says at least25% reduction, but relative reduction is undefined for a zero denominator. This likely does not affect the currently biased parent, but deserves a boundary test and an explicit zero-baseline outcome. Do not patch the active comparator during this run.
3. **Scope limitation, not a discovered cycle-count bug:** after confirming two consecutive samples, `cycles` uses confirmation times consistently and excludes initial/terminal partial cycles. A rejected one-frame excursion does not create a cycle. However, the short-period check only sees debounced complete periods below150ms; it cannot certify absence of raw one-frame chatter or abnormally short stance/swing phases within a longer period. Equal short-cycle counts may also mask more rejected raw spikes. Report it literally as a short-complete-period diagnostic. “Two consecutive50Hz samples” is more precise than claiming40ms continuously measured force persistence.

## What is correct

The prescribed identity set has6 recovery-physics cases at20260909 plus5 original-physics cases at3 seeds. Together with length21, set equality rules out duplicate **labels**, though not duplicate evidence. `measure` uses all350 samples with `phase == walk` and `time_s >5`, verifies5.02–12.00 timestamps, computes contacts from CURRENT vertical force strictly>5N and cross-checks the saved flags. FL/FR/RL/RR duty, diagonal-pair difference and exact1001/0110 support use the correct leg mapping. This is not history-max contact union. Existing tests cover the known actual4046 FL duty291/350 and gap.53, basic complete cycles, constant states and single-frame chatter. They do not cover `load_screen` or `compare` identity/percentage boundaries.

## Independent supplement, without changing active files

Added `scripts/audit_balanced_gait_screen_identity.py` and its test file. The auditor binds each prescribed label to its unique case directory, report, invocation, CSV and physical interface trace. It verifies reviewed source hashes, actual model/receipt hashes, arm/weight/full formal budget ledger, same-arm smoke, actual invocation argv, report command tuple and every900-step CSV/policy-observation command including the frozen push schedule. Original failure flags stay failures: a recorded done=1 is valid evidence if CSV/trace agree and the report says no_reset=false, rather than being automatically rejected as corrupt identity. It writes a separate audit receipt and never edits source evidence or promotes a policy.

Schema decisions were checked against the actual4046 normal report/invocation/CSV/interface trace and actual control4246 formal receipt. In particular, CSV steps are1–900 while interface steps are0–899; commands occupy observation indices9:12; interface dones are integer0 and CSV done is text0. Speed.8 is represented as float32 in measured commands. The tests deliberately substitute4246 identity metadata in copies of the4046 schema; they are fixtures, not experimental results.

After job9458 exits, run CPU-only `test_audit_balanced_gait_screen_identity.py`, then one audit for each completed arm, for example:

`E:/IsaacLab/env/python.exe -B scripts/audit_balanced_gait_screen_identity.py evaluations/20260917-gait-control-4246-first/summary.json --arm control --output evaluations/20260917-gait-control-4246-first/identity_audit.json`

Use the analogous balanced path/arm. These tests and audits have **not been run during the active serial job**. Audit success means evidence identity consistency, not good gait or new model acceptance. Full checkpoint tensor/config verification remains the already pinned candidate wrapper's responsibility; this supplement verifies those bound receipt/model bytes and the measured command identity without importing Torch.
