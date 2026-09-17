# Overnight Go2 work — recovery, locomotion, running, no regressions

User authorization: on 2026-09-17 before sleeping, continue local simulation training toward fallen recovery, normal walking and fast running while retaining all existing functions. Do not promise completion in one night. Existing heartbeat go2 remains attached to this conversation every20minutes; use the current status document to avoid duplicate work. At the first morning follow-up after08:00 Asia/Shanghai, summarize real results and remaining failures with video paths. This is not an authorization for cloud spending, real hardware, system/power changes or closing user applications.

## Preserve before improving

All16 currently exported model files are inventoried by exact absolute path, size and SHA256 in configs/overnight_preservation_20260917.json. Run scripts/verify_overnight_baselines.py before and after each new batch and before any promotion. Byte preservation is NOT a behavioral regression test. Never overwrite/delete these files, historical configs, source backups or failed evidence; save each candidate under a new run/config/report path on E:.

Important frozen controls:

| Model | Established scope | Not established |
|---|---|---|
| natural_stop_diagonal_model2250.pt | Normal standing,0.5m/s command walking/trotting, stopping; historical0.5m/s lateral delta-velocity disturbance evidence |0.75m/s disturbance recovery, arbitrary grounded falls,2m/s running |
| natural_robust_push_model3648.pt | Historical three-seed normal/left/right/fast0.8m/s command tests | Historical0.75m/s disturbance fails stationary four-foot screen; old reports lack newer geometry checks |
| recovery_aligned_model3448.pt | Limited30degree controlled-start recovery and same-Smith-physics upright-preparation-to-stand compatibility | Reliable arbitrary single-policy side/back recovery, locomotion or fast running |
| Smith1999 plus3448 | Frozen dual diagnostic: side9/20, back20/20 final valid holds on repeated development bank;16/14unique states | Single-policy recovery, unseen generalization, reliable both-side recovery, integrated walk/fall/recover/walk loop or new visual acceptance |

DensePosture128x600 was a zero-command recovery experiment, NOT a drop-in locomotion replacement. Its five suites all failed0/20; do not relaunch or extend unchanged. The expanded-speed3947 pilot was rejected for1.0m/s failures and a matched left-turn regression. The original-distribution control3947 completed15/18but1.0m/s still0/3; no promotion or automatic extension. TRAIN-only handoff reset validation and100-update stand3547training completed: stand-only upright20/20,dual side9/back20, no side improvement. Newly measured dual upright is only12/20: the supervisor does not inherit the stand actor's20/20. Read training-status-20260916.md CURRENT for actual active process/provenance and the separate transition diagnosis. Do not change pinned code or overlap simulators during any active finite run.

## Candidate acceptance matrix

Use identical task/physics/seed/protocol for candidate-versus-control comparisons; if changing physics or action/observation semantics, re-evaluate both instead of recycling unrelated old counts. Keep at least the existing three regression seeds20260909/20260910/20260911, and add a genuinely fresh seed/state set only after development decisions are fixed. Confirm actual historical seeds from reports before comparison. Sideways commands have an interface but no dedicated passing report was found: treat lateral walking as unverified/new, not as an already guaranteed feature.

| Capability | Required check |
|---|---|
| Stationary and stop | Unchanged normal geometry, height, low drift/tilt, no base contact, current simultaneous foot support; do not accept crossed legs or a low crouch |
| Normal walking/start/stop | Existing0.5m/s protocol, real forward speed/tracking error, natural diagonal coordination, no falls or auto-resets, return to normal stand |
| Turning | Existing0.5m/s forward plus yaw±0.5rad/s; verify actual yaw sign/tracking and stop behavior |
| Existing faster locomotion | Preserve the historical0.8m/s command regression; this is not proof of fast running |
| Disturbance | Reproduce the validated0.5m/s lateral delta-v protocol; separately report0.75m/s as the known harder failure, not a calibrated force |
| Fallen recovery | Settled side-left/side-right/back, unchanged geometry/current4feet/no base contact/.30-.55m/quiet/3s continuous hold AND valid ending; distinguish controlled drops and PD preparation |
| New fast running | First measure baseline, then staged0.8→1.0→1.5→2.0m/s command curriculum as a working simulation target; report ACTUAL speed, tracking, slip, tilt, falls, joint/torque limits and deceleration/stop; no claim from a speed command alone |
| Complete behavior | Same uninterrupted controller: stand→walk/run→disturbance/fall→recover→walk/run again→stop. Label every actor/selector, do not replace a missing closed loop with edited unrelated clips |

Existing command entry points are scripts/evaluate_stand_walk_stop.ps1 (-Checkpoint,-Task,-OutputDir,-WalkSpeed,-YawRate,-LateralSpeed,-PushSpeed,-Seed,-View,-NoVideo) and scripts/evaluate_maneuvers.ps1. The latter's fast case is ONLY0.8m/s and push case is0.75m/s; avoid treating either name as an acceptance result. It does NOT accept a Task parameter and defaults to Natural-Stop: use the first wrapper directly for any other action/physics task. Before using SmithNominal in the locomotion evaluator, verify its bank-collection setup and action observation compatibility in a small smoke; do not assume the locomotion wrapper handles recovery-bank environment variables. Use fresh E-drive output directories, serial simulations and the exact policy-compatible task. Motion metrics differ from stationary criteria: do not demand all four feet contact simultaneously throughout a trot/run.

An independent read-only review found the old straight-line passed flag does not screen lateral drift or unintended yaw. Add separately reported straight-line checks, provisionally abs(mean lateral velocity)<0.12m/s and abs(mean yaw rate)<0.15rad/s, validate the protocol and compare both frozen baseline and candidate under it. Do not rewrite historical passed flags or call baseline perfect under the stricter screen. These are declared new development screens, not tuned after seeing a new candidate result.

The old locomotion reports alone do not prove newer crossing geometry criteria. Before promotion, add an explicitly tested locomotion geometry audit where missing; keep reward curves, byte hashes, report audits and actual behavioral acceptance separate. Candidate that worsens any established function is not promoted, even if a new skill improves. Keeping old weights provides fallback, not proof a new integrated controller retains them.

## Evidence-driven training loop

1. Inspect current real process/logs; finish any already running finite batch. On CUDA/PhysX errors, non-finite parameters, unsafe resource exhaustion or new authorization requirements, retain last intact checkpoints and do not start another batch until diagnosed. Do not change user applications, power policy or security.
2. Audit complete reports and compare final holds, both-side geometry and upright compatibility. If the current dense-posture branch remains all-zero or regresses, do not extend it unchanged. Consult primary papers/official repos and isolate one falsifiable intervention: supported-sitting righting targets, physically collected TRAIN-only handoff/stand-up curriculum, or a correctly tested left-right symmetry experiment. Do not train on the repeatedly inspected heldout switch snapshots.
3. Preserve separate locomotion and recovery baselines. A learned/hybrid selector may be investigated, but a tilt-only switch is known insufficient and raw-action history/target jumps are real compatibility concerns. Never clear history, teleport states, add invisible manual PD or inflate actuator limits to manufacture success.
4. Extend velocity training as a separate, recoverable locomotion branch with ordinary walk/stop/turn rehearsal; this branch may make progress while a recovery hypothesis still needs analysis, but must not replace recovery or be presented as an integrated success. Only combine/promote after all relevant regressions are credible. Changing high-speed curriculum must not forget zero-command stopping. A fast controller that cannot stop normally is a failure.
5. Save provenance, seed, actual configs, source/model SHA, successes and failures. Record front+oblique real simulator views of full scenarios and actually inspect them. Separate replays are not simultaneous cameras, and separate skills are not a continuous controller. Update CURRENT after material transitions.

No package/model installation, external publication, real robot execution or resource purchase is required by this overnight plan. Keep all new work on E:. If a permission or usage limit prevents continuation, report that limitation honestly; do not silently bypass it or claim training continued.
