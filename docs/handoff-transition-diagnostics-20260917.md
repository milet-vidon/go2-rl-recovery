# Handoff transition diagnostics after stand3547 — proposal only

Read-only investigation dated 2026-09-17. No simulator, training, installation,
hardware command, or pinned Python modification was performed for this note.
The next experiment should diagnose the transition before adding training.

## Verified local finding

The current dual-policy controller is intentionally one-way, not a general
recovery supervisor:

- `src/go2_recovery/recovery_handoff_math.py` updates
  `new_switched = switched | just_switched`. Once true, it never becomes false.
- `scripts/evaluate_go2_recovery.py` lines 760–763 call this gate and select
  `torch.where(switched[:, None], stand_action, roll_action)` every control step.
- `scripts/test_recovery_handoff_math.py` explicitly tests
  `test_latch_never_switches_back_even_with_invalid_state`.
- Evaluation forbids an unexpected automatic environment reset. No other
  stand-to-roll transition occurs during a trial.

Consequently, after a switched robot falls onto its back, the evaluator continues
using the standing actor until the trial ends. This explains why the controller
cannot retry self-righting. It does **not** establish the cause of the first fall:
the immediate target discontinuity and inadequately supported switching state
remain separate hypotheses.

The completed 128×100 continuation preserved **stand-only** upright20/20.
The dual roll1999→stand3547 controller retained back20/20 and side9/20, but a
subsequent cold dual upright control achieved only12/20 (final geometry18/20,
no eligible fallen starts). Thus its real three-bucket baseline is41/60, not
49/60. Roll-first behavior damages ordinary upright retention; the stand-only
baseline must not be substituted for it. The 11 failed side cases ended inverted with zero supporting
feet. Their first clamped target discontinuities remained2.496–2.557rad, versus
2.526–2.585rad before the continuation. The main agent checked that pre-switch
states, observations and roll actions were unchanged. Do not continue the same
training merely because its training reward increased.

Evidence:

- `evaluations/20260917-handoff1999-to3547-side/model_1999.pt_recovery_metrics.json`
- `evaluations/20260917-handoff1999-to3547-upside_down/model_1999.pt_recovery_metrics.json`
- `evaluations/20260917-handoff-stand100-upright20/`
- `evaluations/20260917-handoff1999-to3547-upright-control/model_1999.pt_recovery_metrics.json`
- Frozen stand3547 SHA256:
  `5527c4053cb6ef8e6616da65b7dc32a9464eb8d1366f4966a4eb4b65bc913fdb`.

## What primary sources support — and what they do not

[Lee, Hwangbo and Hutter, 2019](https://arxiv.org/html/1901.07517), sections
II-D/II-E/III-C, separately learn self-righting, standing and locomotion, with a
selector choosing behavior from current observations and prior behavior. Its
selector objective covers repeated loss of balance. Action-difference and torque
costs encourage smooth transitions. The paper explicitly describes a manually
designed selector switching to standing at a bad time and falling; it also warns
that heuristic FSM corner cases and unsmooth transitions remain. Its0.5s wait is
for state-estimator convergence, **not** a published0.5s action blend. Our frozen
two-actor Go2, gates and linear ramps are engineering diagnostics, not a
reproduction of that learned ANYmal hierarchy or its reported success rate.

[Yu et al., arXiv2502.06676v1](https://arxiv.org/html/2502.06676v1), sections2.2
and4.4, freezes skill experts while learning a gating network and multiplicative
Gaussian composition. It contrasts this with problematic discrete switching and
warns that additive blending can create conflicting behavior. This motivates
testing transition compatibility, but gives **no stability guarantee** for our
linear raw-action ramp. We are not implementing its learned composition.

[Unitree's official State_FixStand](https://github.com/unitreerobotics/unitree_rl_lab/blob/main/deploy/include/FSM/State_FixStand.h)
initializes a configured joint-target interpolation from the prior commanded
position. This is an explicit engineering precedent for transition targets, not
evidence that recovery-to-standing policy blending works. That implementation
also sets controller gains; we will not copy its gain changes or nominal-pose
controller into this test. The official
[State_RLBase](https://github.com/unitreerobotics/unitree_rl_lab/blob/main/deploy/include/FSM/State_RLBase.h)
calls its environment reset on entry; that reset must **not** be imported into
our no-history-reset diagnostic. These are moving main-branch references read
on the date above, not a claimed pinned reproduction.

## Experiment A first: one finite raw-action handoff ramp

Keep both weights frozen (roll1999 and stand3547), SmithNominal physics, all
observations,50Hz control, the existing30deg/1rad/s/0.2s entry gate, the one-way
latch, current start preparation and strict acceptance. Change **only** the
handoff ramp duration: baseline T=0, then predeclared T=0.2 and0.5s.

At the first gate edge retain the actual last issued raw action `a_anchor`.
For ramp samples k=1..N, N=T/0.02 (10 or25), compute both actors from the real
current observation and issue

`a_exec(k) = (1 - k/N) * a_anchor + (k/N) * a_stand(obs_k)`.

After N samples, issue the unchanged standing actor output. For T=0 use the
original hard-switch branch exactly, without interpolation/canonicalization.
The current action manager continues applying
`target = soft_clamp(default_q + 0.25 * a_exec)`.
`last_action` on the next step must be the actual issued `a_exec`, naturally
written by the existing action manager. Never substitute a hypothetical actor
output, clear either history buffer, replace the48-vector, teleport joints/root,
zero velocities, change gains, or run extra settling steps at the switch.
Keeping an old *policy-issued action* as the ramp anchor is explicit; it is not
an inserted nominal-standing PD phase.

An offline calculation on the11 actual failed3547 switch records gives:

| Duration | Original target L-infinity jump | First ramped target jump |
| --- | --- | --- |
| 0s | 2.496206–2.556688rad | same |
| 0.2s | same recorded boundary | 0.249621–0.255669rad |
| 0.5s | same recorded boundary | 0.099848–0.102268rad |

These are **first-command arithmetic counterfactuals only**, using recorded
default/soft limits, prior raw action/target and first stand output. They are
not rollouts, stability evidence or predicted success counts. Subsequent policy
outputs change with the new physical state and history; a finite ramp does not
bound every future target jump, and longer ramps can delay needed support.

Why start with raw ramp: it leaves the existing action and history semantics
intact and exactly reaches the original raw standing output. Target-space ramp
is a separate optional arm, not silently equivalent: interpolate between prior
executed target and current clipped standing target, then map back through
`(target-default_q)/0.25`. Saturated raw actions can differ even when targets
match; this changes what the policy sees in last_action. In these11 records its
first target jump is numerically the same as raw ramp, so there is no current
evidence justifying the extra representation change.

Falsifiable primary question: does reducing the first transition discontinuity
reduce the same side re-falls and improve final-valid side count above9/20 while
preserving dual back20/20 and improving (not mislabeling) dual upright12/20?
Any dual upright result below20/20 still fails the intended standing-retention
goal, even if it improves on12/20. If both ramps still fail the same11 despite
verified smaller early target jumps, the first abrupt step alone is not a
sufficient explanation. Do not tune duration continuously on selected clips.

## Experiment B separately: reversible fall detector with hysteresis

Compare hard-switch T=0 with one reversible supervisor; **no A ramp** in this
comparison. Keep the forward gate exactly unchanged. Add only a stand-to-roll
exit when normalized body gravity indicates tilt at least70deg continuously
for0.1s (five completed control intervals). Nonfinite state is a failed trial,
not an opportunity to switch or continue silently. On return to roll, clear
only the supervisor's gate counters and re-arm the unchanged forward gate.

The30deg entry versus70deg exit separation and separate0.2/0.1s persistence are
predeclared engineering hysteresis, not values attributed to either paper.
Do not require low angular speed to exit a genuinely fallen standing mode.
No low-height-only exit: the legitimate standing-up phase starts low. Do not
modify action history, learned policy state, physical state, gains or command.
No extra reset or extra time budget; the fixed original trial horizon already
bounds retries. Record every transition and repetition, including repeated
failures. Keep strict standing acquisition/hold/end checks unchanged.

This arm tests the missing retry path, **not** whether the first handoff is safe.
The initial stand transition and all observations/actions up to the first
return-to-roll event must match hard-switch baseline exactly. A later recovery
can improve bounded robustness but must not erase the initial fall, time loss,
extra contact impulses or retry count. If it loops roll→stand→fall without a
valid finish, report the loop as failure. Do not combine A+B until both individual
arms have results and the interaction is explicitly named as a new experiment.

## New experiment entry without invalidating the RAW collector

Do not edit `scripts/evaluate_go2_recovery.py`, whose current SHA256 remains
`4024ba713e3ed616225d64e1d59de97fd31c8369bba83ac3f4235d93552e9e2f`.
Its hash is provenance for the existing RAW handoff archive. Also preserve all
replay/reset helpers, source receipts, datasets and checkpoints.

Create a separately named `scripts/evaluate_handoff_transition.py` only after
the next decision authorizes implementation. It can be a reviewed standalone
snapshot of the frozen evaluator plus a small separate controller math module.
Do not import the old script as a helper: its top-level argument parsing and
AppLauncher construction have process side effects. Do not runtime-patch the
old collector, install over its task package or rewrite old manifests.

The initial new entry exposes only mutually exclusive modes `hard` and `raw_ramp`.
Retry is a separate future experiment and is **not implemented in this entry**.
Reject incompatible flags, use a new protocol/schema
name, record the old baseline SHA and its own actual source SHA, and never mark
single-policy or hardware acceptance. Initially collect no training data. If
future transition snapshots are wanted they require a new dataset/schema and
new provenance, not an append to the old RAW collection.

Before experimental runs, `hard` (blend0) must reproduce the actual old dual
three20-trial buckets: upright12/20, cold side9/20 and cold back20/20, or41/60.
The separate stand-only upright20/20 does not belong in that aggregate. These
controls are development evidence, not integrated recovery or retention acceptance.
Require field-by-field equality of starts, switch states/times, real48
observations, two-level history, actor outputs, executed targets and final
per-trial results, not just matching the aggregate count. Only after that gate
may the separately labeled0.2/0.5s experiments run. The original forward gate
and one-way latch remain unchanged in both ramp arms. Tests must include T=0 equivalence,
T step counting, no mutation of tensors, per-environment masks, finite checks,
forward-gate continuity and unchanged neighbors. Backward hysteresis/no-chatter
tests belong only to the later independent retry experiment.

For each later arm use separate cold processes and the same complete20-trial
denominators, including untriggered/re-fallen cases. Preserve the original
horizon and3-second final valid hold; no resets or timeout extension. Log the
full per-trial transition list, gate counters/reason, phase, interpolation weight,
both raw actors, actual issued raw/history/target, per-joint target delta,
tilt/height, angular speed, current support forces, base contact, torque peaks
and clamp masks over at least the transition neighborhood. The old CSV only
records trial0, so it cannot establish timing for all11 failed cases.

These previously inspected heldout bank cases are now **development evidence**:
do not relabel repeated selection on them as fresh generalization. After choosing
a promising arm, use a predeclared independent set/seed and visually inspect
front and oblique continuous roll→stand trials before any promotion. Existing
upside/back20/20 does not imply every second-attempt roll state is in distribution.

Recommendation for the next available simulator slot: implement and verify the
new hard baseline, then A raw0.2/raw0.5 as the first finite diagnostic. B remains
an independent follow-up regardless of A's result. Do not start another training
block or combine controller changes without this evidence.

## Completed finite experiment — 04:28:40 local, 2026-09-17

All nine cold-process20-trial cases finished with provenance/trace checks.
New hard controls exactly match original common metadata/per-trial results.
[Batch](../evaluations/20260917-handoff-transition-v1/summary.json).

| Final valid /20 | Side | Back | Upright preparation |
| --- | --- | --- | --- |
| Hard | 9 | 20 | 12 |
| Raw0.2s | 9 | 14 | 11 |
| Raw0.5s | 0 | 0 | 0 |

Both ramps REJECTED; no extension/promotion. The0.2s side arm retains all11same
failures while first target jumps shrink2.496206–2.556688rad to0.249621–0.255669rad.
All20switch times unchanged; the11failures end inverted near0.057m with no foot
support. Smaller first discontinuity alone is insufficient. Geometry pass is
not standing. Actual start quaternion/gravity split left9/9 versus right0/11;
this motivates a separate roll-only mirror diagnostic, not a causal proof.

Four subsequent preselected diagnostic videos were actually inspected in front
and oblique views: right153 flips back after0.68s handoff and fails, left132
stands after1.04s handoff, ending0.3222m with non-crossing feet/mild asymmetry.
Both pairs preserve276frames/25fps and verified actors/starts/source metadata.
They are selected development examples, not fresh generalization or walk/run.
