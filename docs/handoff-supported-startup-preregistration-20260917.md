# Supported, nonfallen startup selector: preregistration, not implementation

Status: proposal only. This note fixes one later diagnostic experiment before implementing its selector or observing its outcomes. The rule was supplied before this read-only count; none of its thresholds is fitted to trial IDs, failure outcomes, or mirror/ramp results. The existing reports have already been inspected extensively, so this is **development-data preregistration**, not an independent held-out test. No simulator, Python process, training, code change, or model promotion is authorized by this document.

The independent question is whether the frozen roll1999-first/stand3547 controller unnecessarily invokes its rolling expert at ordinary, currently supported nonfallen starts. See the [previous bounded startup diagnosis](handoff-upright-startup-selector-analysis-20260917.md): identical actual initial states yield stand-only 20/20 versus original roll-first 12/20 strict final stands. This experiment has **no mirror, ramp, retry, new training, or change to the existing one-way handoff gate**. It does not predict that the new branch will succeed.

## Frozen intervention

Evaluate the following conjunction exactly once per episode, at the existing actual policy-start boundary after the unchanged nominal-position PD preparation. Use current real physical state and contact measurements, not a requested pose label, bank ID, roll/stand output, future outcome, or injected observation.

| Input | Required predicate |
| --- | --- |
| Contact freshness | `contacts_fresh_since_pose_write == true`; all four measurements belong to the current sample, not a union or maximum over contact history. |
| Feet | Exactly the original four named feet, each current world-vertical contact force strictly `> 5 N`. |
| Stance geometry | The existing frozen `normal_stance_geometry` predicate is true, with unchanged native joint/foot ordering and thresholds. |
| Base contact | The existing measured `base_contact` is explicitly false; its original force threshold is unchanged. |
| Tilt | Existing real tilt from upright strictly `< 20 degrees`. |
| Height | Existing body-height measurement lies in the inclusive interval `[0.20, 0.55] m`. |
| Root angular speed | Euclidean norm of the actual world-frame angular velocity is strictly `< 1 rad/s`. |
| Root linear speed | Euclidean norm of the actual world-frame linear velocity is strictly `< 0.5 m/s`. |
| Fallen eligibility | The unchanged `eligible_settled_fallen_recovery` value is an explicit boolean false. |

Required arrays must have their original finite dimensions; scalar values must be finite and booleans present. Missing, stale, null, nonfinite, or internally contradictory input makes the diagnostic invalid and must be reported as such. It must not silently become false fallen eligibility, zero force/speed, a replaced physical state, or an excluded trial. A valid record that simply fails any predicate follows the unchanged original roll-first path.

If the conjunction is true, select the standing actor from the first policy action and retain that actor for the episode. Otherwise retain the original roll-first initialization, including the step-zero gate behavior and subsequent one-way roll-to-stand gate. Do not re-evaluate the startup selector during rollout, reset its flag to enable retries, or switch a startup-standing episode back to roll.

Do **not** require `settled == true`, an already completed support window, or `standing_at_policy_start == true`. These are not missing parts of this rule: it deliberately admits a supported but still-moving ordinary start. Also do not add a joint-speed limit, a pose-class exception, or an alternative height band after seeing a failure. Such changes would be a new proposal/version, not this experiment.

## Meaning and limits of the physical bounds

The conjunction is coherent as a conservative **simulated initial actor-routing category**: fresh four-foot support, normal geometry, no base contact, and small tilt jointly distinguish these near-upright starts from the frozen side/back fallen starts. False fallen eligibility alone would not do so, because unsettled ambiguous states can also have false eligibility.

Nevertheless, `0.20 m` admits a crouch below the accepted standing height, while the speed limits admit appreciable motion. Four forces above `5 N` at one instant do not prove sustained load bearing, a center-of-mass margin inside the support polygon, contact persistence, low foot slip, all-body collision clearance, or feasible future torques. The geometry predicate is not a mesh-collision test. No joint-speed restriction is included. These are intentional limits of this bounded hypothesis, not evidence of hardware safety or already-stable standing. The lower height bound in particular has not been validated across the full admitted interval by the existing upright starts.

Startup selection and success remain separate. The original success criterion still requires gravity error `< 0.35`, height `0.30–0.55 m`, root speed `< 0.50 m/s`, angular speed `< 1 rad/s`, four simultaneous vertical foot forces `> 5 N`, no base contact, the unchanged foot/knee/joint geometry, and **all conditions continuously held for 3 s**. The original 8 s acquisition horizon plus 3 s observation/hold budget, timestep, deadline accounting, and final-valid-stand test remain unchanged. Selection is not success; a final supported snapshot is not a completed hold.

## What the frozen inputs say, without a new rollout

PowerShell-only read-only evaluation of the exact predicate above gives:

| Frozen policy-start set | Trials | Select stand | `settled` true | `standing_at_policy_start` true | Predicate failures |
| --- | ---: | ---: | ---: | ---: | --- |
| Ordinary upright | 20 | 20 | 0 | 0 | None. |
| Side | 20 | 0 | 20 | 0 | All 20 fail four-current-feet, height, tilt, and false fallen eligibility. |
| Upside down | 20 | 0 | 20 | 0 | All 20 fail those four conditions and also no-base-contact. |

All ordinary upright starts have current geometry-valid four-foot support and no base contact. Their height is `0.26197–0.28747 m`, tilt `0.865–6.473 degrees`, root linear speed `0.06939–0.21250 m/s`, and root angular speed `0.14500–0.43009 rad/s`. The minimum individual foot vertical force across those 20 starts is `15.567994117736816 N`. They are not settled: maximum joint speed is `0.52885–1.11830 rad/s`, above the existing settling limit, and their quiet supported windows are zero. None is an eligible settled-fallen recovery trial.

The side set contains 16 unique bank IDs and the back set 14, so 20 trials are not 20 independent source states in either group. This count only shows how this fixed predicate partitions these already-known inputs. It does not measure the selected controller's behavior, validate the full threshold boundary, establish generalization, or turn the reports into training data.

Source reports and SHA-256:

- [Original upright dual control](../evaluations/20260917-handoff1999-to3547-upright-control/model_1999.pt_recovery_metrics.json): `2f3e3f6f5f2fae44883bf9a03a740404425f5bbfffb4710b60802801a125eeae`.
- [Original side dual control](../evaluations/20260917-handoff1999-to3547-side/model_1999.pt_recovery_metrics.json): `e77c40de96649e454ff60d6f8b681f08c7dac3d61730c81c292605025d942715`.
- [Original back dual control](../evaluations/20260917-handoff1999-to3547-upside_down/model_1999.pt_recovery_metrics.json): `a6e853374d433e375047cac3c40b865736b54535054564ccdaed9cbc6f5f3605`.
- [Same-state stand-only upright reference](../evaluations/20260917-handoff-stand100-upright20/model_3547.pt_recovery_metrics.json): `1c3728c48cb7b2a8ad36ae11f1aae97c03ad53719bca56ddcda372a3c87ad463`.

All 20 release records and all 20 actual policy-start records in the upright dual and stand-only reference were previously compared recursively and exactly. The release records have stale/null contact data and must **not** substitute for policy-start support measurements.

## Fixed comparison and provenance requirements

Later implementation requires a separate authorization and a separate entry point; no pinned evaluator or source should be edited. Freeze:

- Roll checkpoint SHA `71e4c7a39ff81836d2ead1c4533988c696e5bd4a0fc939b8cef6b4f6daec1b1c` and stand checkpoint SHA `5527c4053cb6ef8e6616da65b7dc32a9464eb8d1366f4966a4eb4b65bc913fdb`.
- Original evaluator SHA `4024ba713e3ed616225d64e1d59de97fd31c8369bba83ac3f4235d93552e9e2f`; hard-transition entry SHA `c3a433568561fbe0893d61a570cf420304d44dcf986440e8d7efebd85ff7f10a`; transition math SHA `b66167039b4698d66f3cc412b19bacd0b1b2f37279f7ce0856ec6c09b196aa4e`.
- Original state-bank, initial-state preparation, task/seed/angle per matched pose, physical parameters, command, timestep, native ordering, `nominal + 0.25 * raw_action` soft-clamped targets, and natural action history. Do not alter state, history, gains, target offsets, PD duration, gate thresholds, horizon, score, denominators, or actors' deterministic inference semantics.

The finite proposed comparison is:

1. **Off control first:** the new entry with startup selection disabled must match all old result fields and applicable common metadata exactly, with zero numeric tolerance, against completed frozen hard controls: upright `12/20`, side `9/20`, back `20/20`. An aggregate count alone is insufficient. No on arm if this control fails.
2. **One on arm:** the same three 20-trial sets, original frozen starts, weights and strict timing. No mirror/ramp/training and no threshold sweep. Preserve all failures and stop this finite batch after these comparisons.
3. **Selected ordinary-start comparison:** compare initial inputs and all common outcome diagnostics against the same-state stand-only3547 reference. The development retention target is the reference's `20/20` strict final stands, not a new relaxed definition. The reference CSV contains only trial 0; it cannot establish full per-step, all-trial trajectory equivalence. Any such stronger claim requires a separately authorized all-trial stand-only trace, not an inferred reconstruction.
4. **Unselected retention control:** side and back have no selected starts in this development set. Their old physical/action/result records must remain exactly equal to the original hard controls wherever corresponding observations exist. A difference means the supposedly startup-only intervention changed the unselected path or execution and needs investigation, even if counts improve.

Record selection once as a distinct `startup_selection`: complete real predicate inputs, per-condition booleans, selected actor, and explicit valid/invalid-input status. An initial stand choice is **not** a roll-to-stand switch at time zero. Do not fabricate a gate-trigger record, count it as a triggered handoff, or count an ordinary start as settled-fallen recovery. For originally roll-first cases, preserve the original real switch records and one-way gate. Record actual first actor action, previous action/history, executed target, and all-trial transition neighborhoods without extra stochastic sampling or physical advancement just for logging. Record actor initialization RNG isolation and source/model/report/trace hashes.

The new report must identify the experimental startup protocol and its original metric protocol separately, mark diagnostic/non-promoted status, and retain the old result definitions. It must explicitly distinguish a startup standing choice, a later genuine handoff, and a strict final stand. If retention fails, report the measured failure; do not extend the timeout or change the selector within this experiment.

Even a successful development comparison would support only this bounded ordinary-start retention hypothesis with these two frozen actors. Fresh starts/seeds, broader heights and perturbations, actual settled-standing starts, and visually checked continuous runs remain needed for generalization. This cannot establish normal walking, fast running, an integrated single policy, unrestricted recovery, hardware safety, or full reproduction of the reference video.
