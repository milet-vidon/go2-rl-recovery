# Future stand-to-refinement raw-action transition

Status: pure function and CPU-test source only, not executed, integrated,
trained or physically evaluated. All previous frozen entry points, models,
videos and reports remain unchanged.

## Precisely what is proposed

Keep the original roll1999->stand3547 selector and strict-standing criterion.
The existing `supported_refinement_gate.advance_after_completed_interval`
alone owns the150-consecutive-completed-interval qualification. Its new latch
can change only the NEXT issued action; t=0 is not a completed sample.

Once that latch is enabled, evaluate stand3547 and refiner3746 on the same
current real48 observation. Both see the previous action actually issued by
the shared manager, including mixed actions during the transition. For the
first25 affected50Hz actions use:

`alpha_j = j / 25; issued_j = (1-alpha_j)*stand3547(real_obs_j) + alpha_j*refiner3746(real_obs_j)`

Here j=1..25. The25th and every later action are EXACTLY the current3746
output. Duration is25 completed control intervals (0.5s), not25 precomputed
frames. First and last command application boundaries are24 intervals apart;
the25th affected interval ends0.5s after the initial edge. No endpoint is
copied from a different physical trajectory.

This follows the latest requested **live old/new endpoints** specification.
It deliberately differs from the fixed-previous-issued-anchor candidate in
`refinement-switch-motion-review-20260917.md`. That earlier document is left
unchanged; the two proposals must not share an experiment identity. Stand3547's
fresh output during mixing is an unissued proposal, not a second actual action.

The helper returns a proposed count as well as the raw action and coefficient.
Commit that count only after the issued action completes one real20ms interval.
Use the old gate's latch/one-time edge without rewriting it. A lost strict
sample afterward does not unlatch refinement or restart the transition; the
existing evaluator must retain the resulting regression as failure evidence.
The helper cannot authenticate a gate or prove an actual simulator step itself.

## Boundaries that must remain unchanged

- Inputs are matching finite native-order12-dimensional raw actions, before
  scale/offset/soft clamp. No manual PD, new limits, filtering or physics change.
- Only the old-standing branch may call this helper. It has no roll action
  input and is not a replacement for the old roll->stand gate.
- Preserve the actual observation, action and sensor history. Never write the
  unused stand/refiner proposals into history and never reset at this boundary.
- Keep the complete150-step gate, original phase evidence and actual command0.
  No supplied pose/ID/outcome can trigger the interpolation.
- The comparison is explicit `hard` versus `linear25`. Hard mode produces the
  exact direct-refiner branch on its first affected action; its progress is one
  completed direct-switch interval, not a fictitious25-step ramp.
- First runtime integration is out of scope. The continuous .5 diagnostic and
  current posture-video protocols must not silently start using this helper.

## Why this may help, and what it cannot prove

The measured hard boundary changed target by0.154800–0.170410rad; the first
actual20ms joint-position change was only0.004883–0.008153rad. Those are different
quantities. A coefficient ramp may reduce the contribution of disagreement
between actors on the first transition step, but the old actor's output and
physical state also change. Therefore this is NOT a slew-rate limiter and does
not guarantee smaller target jumps, torque, slip, acceleration, jerk or more
natural visual movement. Two live endpoints can both jump even while alpha
changes slowly; an explicit test preserves that counterexample.

With fixed nominal/limits, the same action manager applies the existing
`soft_clamp(q_nominal + .25*issued)` after this function. Do not interpolate
already-clamped targets and call it equivalent. The target, actual joint motion,
body motion and contact behavior must be measured separately in the future.

## Smallest later OFF/ON experiment

After the active chain and independent code review, run CPU tests first. If a
future adapter is implemented, use separate new output directories/protocol,
the same actual start selection/seed/physics/models, and all20 side/back trials
plus ordinary supported starts. Do not extend the550-interval budget to rescue
the ramp, hide failures, or count its25 intervals as the final150 intervals
controlled wholly by the refiner. A separate pure-refiner ending hold is still
required; otherwise report the experiment as incomplete/failing.

Before the edge, hard and linear controllers should have byte-identical actual
measurements and issued actions. At and after it, compare alpha, actual/previous
raw actions, both current actor proposals, true executed targets, root/q/qdot,
current contact/slip, control-boundary torque and unchanged final strict holds.
Retain complete traces and videos; examine dynamics through the full0.5s ramp
and afterward, not merely its first sample. Improved command continuity without
preserved physical recovery/standing is rejected. These sources perform no
such comparison and make no learned-policy or physical-success claim.

New source: `scripts/refinement_action_transition.py`.
New tests: `scripts/test_refinement_action_transition.py` (14 independent tests;
CPU-only and not yet run while the serial simulator guard is active).
