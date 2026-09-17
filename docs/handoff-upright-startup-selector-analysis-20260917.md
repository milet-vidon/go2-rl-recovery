# Upright startup selection: bounded diagnosis, not an implemented selector

This read-only comparison uses the old frozen evaluator's [stand3547-only upright20 report](../evaluations/20260917-handoff-stand100-upright20/model_3547.pt_recovery_metrics.json) and [roll1999-first to stand3547 upright20 control](../evaluations/20260917-handoff1999-to3547-upright-control/model_1999.pt_recovery_metrics.json). It does not inspect or predict the ongoing raw-ramp results, change thresholds, run a simulator, or create training data.

Both reports use the same stand checkpoint SHA `5527c4053cb6ef8e6616da65b7dc32a9464eb8d1366f4966a4eb4b65bc913fdb`. All20 `release_state_before_settling` records and all20 `policy_start_state` records are recursively exactly equal. Task, seed20260918, angle0, nominal-PD preparation1s, physics/action representation, horizon8s, extra hold window3s and strict standing criterion match. The controller, roll checkpoint and associated diagnostic labeling differ. Report SHAs, respectively:

- Stand-only: `1c3728c48cb7b2a8ad36ae11f1aae97c03ad53719bca56ddcda372a3c87ad463`.
- Roll-first dual: `2f3e3f6f5f2fae44883bf9a03a740404425f5bbfffb4710b60802801a125eeae`.

## These are supported ordinary starts, not already-settled standing

At release, nominally upright bodies are0.3814–0.4175m high with5.28–17.66degree tilt. Root/joint velocities are zero and the recorded stance-geometry check passes, but contacts are explicitly **not fresh**: contact/support fields are null, not measured zero or measured support. The report labels every release `controlled_drop_unsettled_contacts_not_yet_valid`.

After1s direct nominal-position PD, every policy-start record has fresh current four-foot vertical support, no base contact, and valid foot/knee/joint geometry. Tilt is0.865–6.473degrees; height is only0.26197–0.28747m. Root linear speed is0.06939–0.21250m/s, angular speed0.14500–0.43009rad/s, and maximum joint speed0.52885–1.11830rad/s. Every joint-speed maximum exceeds the existing0.50rad/s settling limit; each quiet-supported window is0. Every `settled`, `standing_at_policy_start`, `fallen_at_policy_start` and settled-fallen eligibility flag is false. All are classified `other_nonstanding_start_not_confirmed_fallen`.

The defensible description is **near-upright, geometrically normal, currently four-foot-supported, still-moving low-height ordinary starts**. They are neither established3s standing holds nor confirmed fallen recovery starts. Their height is below the strict0.30m standing minimum. Current contact/geometry at one instant cannot certify future stability, a support polygon margin or all collision geometry.

## Roll-first changes the path before the standing expert acts

Stand-only acquires and retains strict standing20/20, with median onset0.12s and P90 0.16s. Roll-first finishes12/20; among its successes the corresponding values are3.53s and4.438s. These are acquisition/retention times from ordinary starts, not settled-fallen recovery times.

The frozen evaluator initializes every `switched` flag false. It deliberately skips gate counting at step0, selects roll actions until the original0.2s continuous tilt/angular-speed gate latches, and cannot switch back afterward. Although all20 initial states satisfy the gate's instantaneous tilt/angular-speed bounds, they do not yet have the required history. Nineteen trials eventually switch; one never does. First switches occur1.14–9.08s after policy start (mean4.18s), not at0.2s. At those boundaries body height is0.15382–0.15843m, roughly0.10–0.13m below the same trial's policy-start height. Thus the standing expert often receives a much lower folded state only after substantial roll-policy execution. This is direct recorded evidence of a startup-path problem; it does not by itself identify each contact event that delayed the gate.

The recorded **first roll-to-stand** raw-action L-infinity jump is8.4125–8.8441; the actual soft-clamped joint-target jump is1.34685–1.37972rad. These are not the initial PD-to-first-policy-action jumps, which are not recorded for all20 trials in these JSONs. Similar handoff jumps occur in both successful and failed trials, so their magnitude alone does not explain the entire20-to12 loss. The full physical trajectory and contact impulses of all trials cannot be recovered from the old CSV: it contains only trial0.

## Eight failures are not eight identical falls

| Trial(s) | Recorded final failure | Evidence |
| --- | --- | --- |
| 0,18 | Inverted again after a switch | Height about0.057m, gravity error about2, no supporting foot contacts, hold0; geometry alone still reports true. |
| 1,2,13,16 | Insufficient final continuous hold | Height0.3116–0.3202m, geometry true, four current feet; final holds1.36/2.46/2.80/2.70s are below3s. Switches at9.08/8.02/7.52/7.76s leave late standing acquisition. Do not extend the horizon to turn these into passes. |
| 3 | Final stance geometry failure | Height0.30750m and four contacts, but FR foot body-y=-0.05700m gives signed width0.05700m, below the strict0.06m requirement; hold0. This is too narrow, not evidence that the front legs cross the midline. Other unrecorded fore/hind coordinates may also fail. |
| 17 | Never switched; low and invalid stance | Height0.21668m, four contacts, RR foot signed width0.03276m and knee width0.03048m fail their gates; maximum joint offset0.73762rad also exceeds0.65rad. Final tilt is small, but no qualifying continuous switch interval was recorded. |

JSON final diagnostics do not include every trial's final root speed/base-contact value. No missing velocity cause or exact re-fall timing is inferred. The four short-hold trials cannot be called full successes merely because their last snapshot looks supported.

## Independent startup-selector hypothesis for a later decision

The falsifiable hypothesis is that **unconditionally invoking the roll expert on supported, nonfallen ordinary starts unnecessarily removes the standing expert's valid starting distribution**. A separately named startup-only selector could choose the standing expert immediately for a predeclared physically supported, nonfallen category and reserve roll-first for confirmed fallen starts. This is not a proposed change to the current forward gate, ramp duration or retry controller. No new numeric threshold is selected in this note, and the category must not be called already-settled standing.

Before any later implementation, declare the selector's observables, freshness/ambiguity handling and fixed rule independently of individual failed-trial IDs. Use actual initial observations/contacts and natural action history; do not recognize the requested pose label, inject states/actions/observations, add PD preparation or tune on these20 outcomes. A selector is a new intervention, not proof that the hard-switch or ramp itself improved. Compare it separately against both this exact dual baseline and stand-only, keeping weights, physical starts, timing and3s criteria fixed; record the branch chosen at time0, both actors' actual initial outputs, executed target and all-trial transition neighborhoods. Require side/back preservation as well as ordinary-start standing retention before broader claims.

These repeatedly examined reports are development evidence only. They cannot train or certify a new selector, generalize to arbitrary falls, substitute for tests starting from actual settled standing, or establish integrated recovery-to-walking/hardware safety. The ongoing ramp experiment remains independent and no result is assumed here.
