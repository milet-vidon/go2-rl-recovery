# Continuous recovery-to-locomotion diagnostic design

Status: design only, 2026-09-17. No integrated simulation or acceptance is implied.
The recovery-physics locomotion compatibility screen must pass first. A failed
screen blocks integration; do not compensate by disabling self-collisions or
changing action semantics halfway through a rollout.

## Smallest honest scope

One simulator, one physical environment, one initialization, one continuous
recording, and three explicitly named frozen actors:

`initial fallen state -> roll -> stand with strict 3 s hold -> locomotion -> zero-command locomotion stop`

Use the evaluated roll1999, the standing actor that actually passes its current
candidate comparison (not automatically the newest checkpoint), and a locomotion
actor that passes the recovery-physics screen. This is a deterministic multi-policy
controller, not a single-policy model or a reproduction of a published selector.
There is no training in this diagnostic.

The first integration should test one 8 s movement command at a measured, passed
speed, followed by 6 s zero-command stopping. Start with 0.5 m/s only if its screen
passes. Additional trot/run segments are separate extensions with preregistered
commands and measured actual speeds. Neither a requested 1.0 m/s command nor an
8 s simulation duration proves fast running or a particular gait.

The minimum version starts from a real fallen state at the single episode's
initialization. A second fall during movement is recorded as a failed continuous
trial; it is not hidden by resetting or restarting recovery. Repeated fall/recovery
within the same physical episode needs a separately tested recovery-session selector.

## Interfaces that can be reused

Source files remain unchanged. A new, SHA-pinned adapter can derive one generated
program from the combined evaluator, with explicit, count-checked transformations.
Do not run two entry-point `main()` functions in succession; both construct/reset
their own environment. Do not import the old evaluation entry points directly:
their top-level argument parsing and AppLauncher construction have side effects.

Useful existing components:

- Combined `build_source()`: existing startup decision, roll-only reflection,
  frozen one-way roll-to-stand gate, real action/history assertions and per-trial
  state classification. Keep these definitions, not the outer fixed-length loop.
- `recovery_handoff_math.update_handoff_gate`: tilt <30 degrees and angular speed
  <1 rad/s for 0.2 s; selector only, never a recovery success test.
- `handoff_transition_math.transition_action(..., total_ramp_steps=0)`: unchanged
  hard recovery switch. Do not add interpolation under the name of the old gate.
- `roll_mirror_math.initial_right_mask`, `roll_input_copy`, and
  `physical_roll_action`: true-start classification, deep-copy model coordinates,
  inverse reflection into the real native action order.
- Recovery `_stance_geometry` plus the complete strict-stability conjunction in
  its loop. `_stable_stand` alone is insufficient: it retains history-max contact
  support and omits the explicit current-foot/base/geometry checks.
- Retention `_set_command`, per-step measurement extraction and rest acceptance
  formulas, adapted into the new lifecycle with their origins recorded.
- `evaluate_locomotion_recovery_physics.validate_interface` and the generic
  portions of `begin_step`/`end_step`: all native 48-D terms, current action/history,
  nominal-plus-0.25 soft-clamped targets. The current helpers assume N=1, and
  `begin_step(step=0)` assumes initial history is zero; never invoke that check
  with a reset local phase clock at locomotion handoff.

Suggested new pure helper API, without modifying the old gate:

`next_phase(phase, elapsed_control_steps, strict_valid, recovery_gate_state, command_schedule)`

This returns phase/counters/edge metadata only. It must not receive mutable
environment buffers or call reset, write_state, set_target, step, or policy.reset.
Minimum states are `recover`, `strict_hold`, `move`, `stop`, `complete`, `failed`.
The old startup and genuine handoff masks remain distinct metadata. Entering
`move` is a new controller edge, not a second old `just_switched` event.

## Physical and observational contract

Construct the SmithNominal physical configuration once, and retain it for all
phases: self-collisions ON, 0.005 s physics, four substeps per control action,
0.020 s control, the same friction/material/mass/CoM, native joints, motor model,
soft limits, and nominal action reference. The locomotion actor's old unclamped
training physics is not silently restored when it becomes active.

One action manager handles all three actors. Every issued native action `a_k`
is executed as `soft_clamp(q_nominal + 0.25 * a_k)`, held over four physics steps.
No action clipping, no per-actor target offset, no target ramp, and no manual PD
may be inserted at actor switches in this first experiment.

Every actor is feed-forward, deterministic mean, and has identity observation
normalization. Load each actor with its own pinned configuration/checkpoint; the
currently read locomotion and standing YAMLs both use 128/128/128 ELU networks,
but their PPO configurations differ and should not be conflated. Isolate all
additional model-construction RNG with `torch.random.fork_rng` so initial-state
sampling remains independent of how many actors were constructed.

Before every issued action, recompute and compare the native observation exactly:

`[v_body(3), omega_body(3), projected_gravity(3), actual_command(3), q-q0(12), qdot-qdot0(12), actual_last_action(12)]`

The last twelve values are the action manager's last actually issued action,
not the current actor's previous proposal and not a zero vector at a phase edge.
Clamping changes the executed target, not the stored raw action. Keep both.

## Command routing without fictitious measurements

Disable random command resampling and heading control at construction, with
flags/ranges recorded. The recovery configuration has `rel_standing_envs=1`:
simply assigning a nonzero command is insufficient because its manager can zero
standing rows during `compute`. Set actual `vel_command_b`, `is_standing_env`,
and `is_heading_env` deliberately before observation recomputation each step.

- During `recover`/`strict_hold`, the real command is exactly zero. The two
  recovery actors can therefore continue to consume their existing observations;
  only the roll actor receives its permitted mirrored copy.
- During `move`, set the real command to the schedule, recompute observations,
  and call only the locomotion actor. Do not evaluate roll/stand on nonzero
  commands. Do not fabricate a zero-command observation copy and describe it as
  the real measurement.
- During `stop`, the real command is zero but the locomotion actor remains
  active for the full six-second stop test. This preserves the meaning of its
  existing stop behavior. Returning to the standing actor is a separate fourth
  edge and is not necessary for the minimum continuous demonstration.

Log the commanded value before inference and the manager's value after the
step; assert they are equal. Writing commands only after inference creates a
one-step delay and an invalid command/observation claim.

## Actual start, history, and existing protocol conflicts

The historical combined evaluator initializes a bank state, performs 1 s direct
nominal-position PD, and then calls `_reset_policy_history`. That helper clears
action/observation/termination managers and resets the episode clock. It must
NOT be copied into a literal uninterrupted/no-history-reset entry.

For the minimum initial-fallen experiment, allow exactly one initial environment
reset and bank-state placement before the continuous clock begins. Record those
as initialization, not as a natural fall. Retain the existing 1 s startup PD only
as an explicitly recorded initial preparation segment, never at later edges.
After initial reset, verify the natural action history is already zero; during
preparation retain it and record the direct nominal target. Remove the later
`_reset_policy_history` call entirely. Do not reset any episode or simulation
counter after preparation. This changes the old startup lifecycle and requires
a new zero-command recovery-prefix control; numerical equivalence must be
measured, not assumed, even if the first policy observation happens to be equal.

If the requirement is no direct PD preparation at all, that is a different
initialization protocol requiring a new recovery evaluation; existing 20/20
settled-under-PD results do not establish it.

Classify startup and mirror masks once from the actual measured start of this
continuous recovery episode, after fresh contacts and the recorded settling
window. Never choose a mirror mask from requested pose, bank ID, previous video,
or previous episode. Keep its fixed value for the recovery session. A rightward
roll action is inverse-reflected exactly once before the action manager.

The existing gate is one-way latched, and the supported-startup mask chooses
standing for the whole old episode. Reusing those old masks after a later fall
would keep the standing actor selected and can prevent recovery. The minimum
design therefore rejects later falls. A future repeated-fall design may reset
only its own selector/session counters and classify a new actual recovery start;
it must not reset action/observation/physical history. It also needs a new valid
settled-fallen predicate or a explicitly new dynamic-fall protocol. The old
`eligible_settled_fallen` API cannot honestly be fed `True` for an unsettled fall.

## Exact step order and strict hold

Use one monotonic global control index and simulation-substep counter. Phase
timers are additional counters, not replacements for environment history.

1. Inspect state after the preceding completed interval; do not count the initial
   t=0 observation as 0.02 s of persistence.
2. Update the old recovery selector if recovery is active. Update the independent
   strict-stand counter only from completed physical intervals.
3. If at least 150 consecutive valid stand samples have completed, latch the
   stand-to-locomotion edge for the next issued action. Reset only that new phase
   timer. Do not clear the old `switched`, startup, mirror, or manager histories.
4. Set the real phase command, recompute observation, assert its native contract,
   evaluate the active actor(s), and retain the real previous action/target.
5. Execute exactly one environment step and assert no `done`, unchanged physical
   configuration, actual action equals issued action, manager.prev_action equals
   the saved prior action, and target equals the same soft-clamp expression.
6. Record after-state, current forces, feet/knees/joints, command, actor identity,
   selector counters, and all edges. Render without an extra physics step.

The strict hold uses the existing acceptance, continuously for 3 s: gravity
error <0.35; height strictly 0.30–0.55 m; linear speed <0.50 m/s; angular speed
<1 rad/s; all four CURRENT vertical foot forces >5 N; no current base contact
>1 N; and every foot/knee/joint passes `normal_stance_geometry`. A failed sample
resets only the strict-hold counter. Gate convergence alone cannot start walking.
An 8 s maximum onset allowance plus the 3 s hold matches the old recovery horizon.

Set the one-time episode timeout long enough for startup preparation + maximum
recovery allowance + all motion/stop segments + margin. Keep base-contact auto
reset disabled and explicitly inventory all active termination terms. Any `done`
invalidates the entire continuous trial; never call `policy.reset(dones)` and
continue rendering it as if uninterrupted. Reward computation is not acceptance.

## Evidence and acceptance

Record every control step, not only the old +/-1 s handoff neighborhoods. For a
small one-robot demonstration, this is manageable and permits auditing all
boundaries. Include actual physical state continuity, actor SHA/name, real48,
roll virtual48 when used, actor proposals, issued raw12, preceding raw12,
executed target12, current contacts, geometry, physical/config hashes and times.

Require recovery hold completion, no later reset/fall, the existing locomotion
tracking/slip/clearance/level checks and separate straight-line drift checks,
and the existing quiet-stop/normal-rest/current-contact checks. Do not impose
four-contact standing geometry throughout the moving phase. Additionally require
the last 150 stop samples to meet strict standing, without replacing the tighter
existing quiet-stop speed criterion (p95 <0.06 m/s for no push).

Run left/right/back initialization plus ordinary supported startup separately,
then render a selected continuous trial with one uninterrupted timestamp. If
front/oblique videos come from separate deterministic replays, audit their traces
and label them as matched replays rather than simultaneous cameras. Never
concatenate three old scene videos into a claimed continuous trial. A failed
trial remains evidence, not a successful demo with its failed segment removed.

Minimum CPU tests: 149 samples cannot trigger movement; invalid sample restarts
the count; exactly 150 completed samples triggers only the next action; t=0 does
not count; command routing does not mutate raw history; all phase edges retain
last issued action; mirror fixed from actual episode start; row permutation if
vectorized; later fall fails; no reset API in pure state machine; invalid/nonfinite
state cannot select a successful transition. Then test frozen recovery-prefix
behavior and retention compatibility before any full-flow claim.

## Sources read for this design

- `scripts/evaluate_handoff_combined.py` SHA `0788beb2e8d4457f6c286d9aff80638b269e080a9b75167f9f7977456380bed0`
- `scripts/evaluate_go2_stand_walk_stop.py` SHA `bbba369177d95bc24163bdd2b5ba96506f5db7d4b1ce38875f8a8885c49e22e4`
- `scripts/evaluate_locomotion_recovery_physics.py` SHA `cc2437ffb0d84da97eda5906333e54070f2d065ea475a148b6be7f1c6cfea4c8`
- `src/go2_recovery/recovery_handoff_math.py` SHA `df8a45a5e64ddc05a3062b73f769134fd3612aa4149f88eae51a67252699369b`
- `src/go2_recovery/handoff_transition_math.py` SHA `b66167039b4698d66f3cc412b19bacd0b1b2f37279f7ce0856ec6c09b196aa4e`
- Corresponding mirror/startup helpers, physical configuration, nominal-target
  implementation, and actual stand/locomotion actor YAMLs.

This note changes no executable source, checkpoint, receipt, or registered task.
