# Continuous-flow integration review, 2026-09-17

This is an implementation review and a new **bounded development diagnostic**,
not a completed physical validation, model promotion, deployment, or trot/run
claim. At review time the two new 4246 arms were still being evaluated; neither
is selected here. No pinned historical evaluator is modified.

## Decision: build a real .5-command flow now

There is enough evidence to try the original frozen control3947 at **command
0.5 m/s only**, with frozen roll1999/stand3547, in one real physical episode.
This is not an actual-speed hard limiter: the measured upright-screen forward
speed was 0.610329 m/s. It is not a claim that the whole compatibility matrix
passes. The earlier design's opening blanket statement that a failed screen
blocks integration is narrowed explicitly for this diagnostic: the exact speed
being tried must pass the existing full-physics screen; candidate promotion
still requires the full registered retention matrix and independent continuous
verification. This changes diagnostic admission scope, **not any numerical
acceptance threshold, old roll/stand gate, physical model or preservation gate**.

Actual supporting evidence is
`evaluations/20260917-locomotion-recovery-retention-v1/recovery-normal-seed20260909/`:

- control3947 SHA `3dfbe03e0ac332fee688c970b420167affcdb6e8cacee16e40fc12120751df3f`;
- report SHA `21c4415673267e6c248a404fd80ff3125577728e04b8fe8560ae7b812faf96fb`;
- interface SHA `985c447375683a8ee174af78a0ee230b2b01db0c686ada367b6ab5db35a08343`;
- CSV SHA `ba4a0eaf216b8ef822356b89e066bc09b5ce750936b70d67cdc1923b1a4136d4`.

All 13 original checks and the additional straight-drift check pass, but margins
are small: forward tracking error 0.110329 < 0.12; lateral speed -0.118895,
only 0.001105 m/s inside the 0.12 threshold. This makes the real handoff test
important: its starting state differs from an ordinary upright-reset screen.
The .8 screen still **fails** lateral drift (-0.126406 m/s), despite its legacy
`passed` field being true. Read `retention_passed`, not legacy `passed` alone.
Each speed's three seed CSVs are byte-identical, hence only one distinct physical
trajectory per speed; they are not three independent robustness trials.

## Smallest executable scope and seams

New separate files `evaluate_continuous_recovery_flow.py` and
`continuous_flow_runtime.py` implement one N=1 episode. Initial scope is
refinement **off**, frozen1999/3547/3947, real .5 command, no push, no retry.
No old `main()` is executed and no old 550-/900-interval recorder is reused.
The runtime extracts only an enumerated set of SHA-pinned function definitions
through AST; their imports/top-level parser/AppLauncher/old main are not executed.

The pure phase helper owns no simulator. Exactly150 completed strict samples
controlled by stand select the **next** locomotion action; move lasts400 control
intervals, stop300, and the final150 stop intervals must be strict. Recovery's
maximum is550 intervals. Its optional-refinement logical branch remains unused
by this first runtime: enabling it later requires a separate150 refinement hold
inside the same shared550 budget, authenticated3746 and a new four-actor report.
Do not call the refinement-enabled flow a three-actor controller.

The implementation boundaries are:

| Component | Reused safely | Not copied unchanged |
|---|---|---|
| Combined recovery | Actual-start startup predicate, fixed mirror, one-way10-sample gate, hard action selection | Pose loop, fixed horizon recorder, post-PD history reset |
| Supported refinement | Reviewed as extension/reference only | Entire550-step `RefinementRecorder`, always-evaluate-all-actors loop, three-actor full-flow title |
| Retention interface | Exact raw48/12 contract, nominal+.25 target assertions, actual mass/CoM/material snapshot | Upright-reset `configure_physics`, old900-step `finish_report`, phase-local step0 |
| Old stand/walk/stop | Foot/current/history measurements and unchanged physical limits | Second environment/main/reset, `policy.reset(dones)`, fixed stand4 schedule |
| New motion metrics | Last150 recovery stand, move350 after first50, stop250 after first50, last150strict | Source authenticity, reset/physics continuity or video claims |

## Initialization and clocks: concrete conflict resolved

`RslRlVecEnvWrapper.__init__` already calls the environment reset. The new runtime
does not call a second `env.reset()` in main. Initial bank placement legitimately
resets native action/sensor buffers **before t=0** and is disclosed as initialized
bank state, not a natural observed fall. Ordinary startup is explicitly separate.

The old preparation is50 manual control-equivalent intervals of4 physics steps.
It directly sets the nominal joint target, never calls `env.step`, never computes
commands/rewards/interval events, advances `_sim_step_counter` and
`episode_length_buf`, but **does not advance `common_step_counter`**.
The new runtime retains that behavior, records every interval, and does not call
`_reset_policy_history`. It never rewrites common/episode counters to make a
seemingly continuous clock. The physical clock is
`(_sim_step_counter - simstep_origin)/4`, checked for exact divisibility and
one-interval progress. Both manager counters are logged independently.

After the wrapper reset and initial placement, a live `_reset_idx`/`reset` guard
raises before a reset can change the robot. A late `done` is a failure, not a
reason to call `policy.reset` and continue. This matters because stock
`ManagerBasedRLEnv.step` auto-resets **before returning** a done to the caller;
checking dones only after `env.step` cannot by itself guarantee no reset.
Timeout is configured once at28s, longer than1+11+8+6s. No later physics/config
change, target ramp, re-placement or fallback stand actor is allowed.

Video warmup is render-only before t=0. The video records t=0 and every subsequent
completed interval at50fps, including all1s nominal-PD preparation. Its first
frame must not silently start after PD as in old recovery videos. A frame ledger
records real timestamps and source-frame hashes. Do not assume the old276-frame
posture-video count; the completed sequence length is variable. Front/oblique
recordings, when added as separate cold runs, must be labeled matched replays,
not simultaneous cameras. No temporal splice is a full-flow demonstration.

## Action, observation and gate order

1. At the actual post-PD start, measure fresh contacts/quiet duration and classify
   startup and mirror once. Bank labels/IDs are provenance, not routing inputs.
   A requested bank start that is not actually settled-fallen is rejected as
   fallen-recovery evidence; an ordinary supported start is identified separately.
2. Preserve the old gate's first-policy-action rule: do not count the final PD
   sample as the first completed recovery interval simply because global time=1s.
3. For recovery only, update the original gate from the preceding completed
   policy interval; startup-selected rows never create a genuine gate event.
4. Set the real command before recomputing the observation: zero for roll/stand,
   .5 for locomotion movement, zero for locomotion stopping. Disable standing
   and heading masks explicitly so the command manager cannot zero .5 afterward.
5. Check raw native48 exactly:
   `[v_body, omega_body, gravity, real_command, q-q0, qdot-qdot0, actual_last_action]`.
   Roll receives only its permitted deep-cloned mirrored input and its output is
   inverse-reflected once. Stand receives native input. During move/stop call only
   locomotion; do not fabricate zero-command views for dormant recovery actors.
6. Preserve action manager `action`/`prev_action` across every edge. Its next
   executed target is exactly `soft_clamp(q0 + .25 * raw_action)` for every actor.
   Record raw/previous/previous-previous action and targets, not only pose.
7. After exactly4 substeps, record current physical state, all contacts/geometry,
   strict flag, actual command, counters and selected actor. The strict150 edge
   affects the next action; the initial t=0 state does not count as persistence.
8. The stop phase keeps locomotion for all300 intervals. A later fall terminates
   the diagnostic instead of recycling the latched old selector or clearing history.

The pure helper's original exact comparison incorrectly rejected real float32
.8 (`0.800000011920929`) against a Python `.8` plan. It now encodes only the
**planned** command with explicit IEEE float32 pack/unpack, while retaining the
unmodified actual measured tuple and exact comparison. No broad tolerance or
rounding of evidence is used. Neighboring float32 values and incorrect commands
remain failures. This fixes transport semantics, not speed qualification: the
new runtime still forbids .8 pending its separate physical prerequisites.

## Evidence, behavior and failure semantics

Every completed interval has one row: `step` is zero-based from initial release,
`time_s=(step+1)*.02`, `phase_interval` is one-based. Phases are preparation,
recovery, move, stop. Each row includes all old stand/walk/stop foot fields,
current force-norm contact **and** current vertical>5N separately, body positions,
knee y, joint offsets, real command, `gravity_b`, root3D speed norms,
`strict_valid_after`, actor role and before/after physical counters. Initial t=0
is a separate record, never a completed interval. JSONL records retain complete
before/after root/q/qdot/action/target/current-and-history-force state.

Only command values may change between completed physical intervals. The runtime
checks the remaining snapshots exactly, all finite measurements, counter progress,
action/target/history, unchanged start/end physical interface, source/input hashes,
and no reset attempts. Initial PD is disclosed separately from policy actions.
Its raw action buffer remains naturally zero; no action is invented to describe
the direct target assignment.

Motion analysis does not require valid standing geometry during movement or
no base contact during initial fallen recovery. It applies the old motion limits
and separate lateral drift, resting quiet-speed p95<.06, normal geometry/current
support, plus the ending strict150. Recovery's looser strict-speed<.5 alone is
not quiet-rest success. Last150 recovery stand replaces the old fixed4s stand
window in this **new** protocol; it must not be labeled the unchanged old900-step
experiment. Independent strict recomputation must agree with measured flags;
ambiguous float32 threshold evidence fails closed, not by loosening limits.

Early failure writes the rows/trace/video already measured, physical failure
snapshot and error receipt. Missing phases never yield PASS. `--smoke` explicitly
records50 PD+2 policy intervals, sets `passed=false` and is not a full-flow test.
Even a complete diagnostic cannot promote a model or assert full reference-video
reproduction, faster running, hardware readiness or broad generalization.

## Minimal review/run order

1. Parent reviews these independent sources, compiles and runs CPU tests after
   the active serial simulator chain exits. No Python/simulator was run by this
   author while the chain was active.
2. Run one N=1 no-video smoke with actual pinned actors and held-out bank.
   Inspect all52 intervals, input/source hashes and no-reset records.
3. Run a finite complete .5 case on the same start. A failure is useful integration
   evidence, not permission to lower a threshold. Compare the zero-command prefix
   with a cold original-lifecycle control: omission of post-PD reset is a real
   protocol change, so old numerical equivalence must not be assumed.
4. Run left/right/back/ordinary starts as separately identified bounded cases;
   select no model based on one convenient video. Render only actual complete
   traces with full PD/transition/stop included and inspect the feet visually.
5. Refinement and .8 are separate later extensions. Full registered old-function
   retention remains necessary for promotion, regardless of .5 diagnostic outcome.

Source review pins: phase helper originally `1e22265e...`, now
`7ea6c73a36678a8240c8f74d3fc89aa0c45cd5b394bd5632de024341d48a7929`;
supported refinement `bdb03cf09fafa9c0ed5aa14ee8818b4f95b6d145300e33b6e5cac8bd1b0ad0ff`;
mirror `cf2fbbdceae0e460727a969bf080e1e230ed716f00f914f47381a167df50ba95`;
retention `cc2437ffb0d84da97eda5906333e54070f2d065ea475a148b6be7f1c6cfea4c8`;
old locomotion `bbba369177d95bc24163bdd2b5ba96506f5db7d4b1ce38875f8a8885c49e22e4`.
The new runtime records its own exact start/end source inventory. The existence
of executable code and CPU tests is not evidence that the physical flow works.
