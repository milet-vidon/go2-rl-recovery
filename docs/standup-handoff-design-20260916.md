# Standing after righting: conditional next diagnostic

This is a **read-only design audit, not an implemented controller or successful result**. Finish the active w10/w30 pair before changing its runtime sources or launching another simulator. See [current status](training-status-20260916.md).

## Evidence

Smith1999 rights its body but finishes low, with abnormal leg placement and usually one supported foot. Its completed w10 continuation2998 also fails the settled-fallen bank: side0/20 and back0/20 final-valid stands, zero geometry passes and zero four-foot endings. Final mean heights are0.1413/0.1468m. The w30 arm must still be assessed; neither higher reward nor a nearly level torso is acceptance.

[Lee2019, II-D](https://arxiv.org/html/1901.07517) separates self-righting, standing-up and locomotion. The roll behavior also arranges the legs into a supported sitting configuration; the standing-up behavior learns from near-upright sitting starts. This supports testing skill compatibility and start-state coverage, but does not establish our Go2 result or reproduce the paper's complete learned hierarchy.

## Static implementation audit

No definite joint-index, gravity-sign, height-frame or double-action-offset bug was found. Actual bank order is four hips, four thighs, four calves. Default targets are `[0.1,-0.1,0.1,-0.1,0.8,0.8,1,1,-1.5,-1.5,-1.5,-1.5]`;2250,3448 and Smith saved defaults agree. Smith weights are resolved by actual joint names, not assumed leg-major ordering. Root height is root-link height minus environment origin, matching the acceptance coordinate.

The stand reward contains20% height,60% joint-pose and20% quiet-joint terms, gated below36degrees tilt. It does not itself require normal foot support/geometry. A low quiet local optimum remains a learning hypothesis, not a proved implementation error.

Relevant files: `src/go2_recovery/recovery_smith_math.py`, `recovery_smith_mdp.py`, `recovery_control_targets.py`, and the saved bank manifest. Audit found their checked installed counterparts unchanged.

## Conditional follow-up if the completed pair still fails

1. In the SAME Smith Play physics/action environment, first evaluate existing3448 from normal upright starts with zero commands and unchanged strict acceptance.2250 is an optional normal-standing reference. Old-task success does not prove compatibility: Smith additionally soft-clamps targets, and2250 originally used no self-collisions. Record actual default angles/soft limits and target-clamping frequency; the static audit could not verify those runtime limits.
2. Only if compatibility passes, test one transparent, fixed, one-way Smith-to3448 handoff against Smith alone on matching bank states. A predeclared diagnostic switch can require tilt<30degrees and angular speed<1rad/s for0.2s. This switch is NOT a success criterion. Include non-triggering episodes in the denominator.
3. Do not move the body, set joint positions, reset velocities, inject standing PD or reset action history at the switch. Preserve the actual preceding executed policy action in the observation. Record switching time/state and action/target discontinuity. Both actors are non-recurrent48-observation networks, but this alone is not physical compatibility.
4. If upright compatibility passes but handoff fails, train a distinct low-upright/sitting stand-up curriculum using physically validated initial states. Existing3448/2250 were not shown to cover Smith's low, folded, single-foot states. Do not tune switch thresholds to cherry-pick a successful clip.

This would be a multi-policy diagnostic, not single-policy recovery or an integrated walk/fall/recover/walk demo. Unchanged acceptance remains mandatory: normal geometry, height0.30–0.55m, four CURRENT vertical foot forces>5N, no base contact, low body speeds, at least3s continuous valid hold AND valid ending. New candidates require independent starts/seeds and actually viewed front/oblique video. Preserve recommended2250/3448 and all failed evidence.

Actor-only warm-start is a separate alternative, not the same experiment: copy only actor tensors, use identical fresh critic/exploration/optimizer and reset iteration state for both arms. `runner.load(load_optimizer=False)` is insufficient because it still loads critic, noise and iteration state. Do not bundle this alternative with the handoff or current reward-weight pair.
