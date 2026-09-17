# Roll-only left/right mirror — bounded diagnostic proposal

2026-09-17. Read-only PowerShell/JSON/source audit plus primary-source reading.
No Python, simulator, training, installation, or pinned-runtime edit was performed
for this proposal. This is an optional next factor, not an approved experiment.

## Evidence and scope

In `evaluations/20260917-handoff-transition-v1/{hard-side,raw02-side}/`, the
same 11 right-side starts failed and the same nine left-side starts succeeded.
Side classification used the **actual post-settling policy-start** quaternion
and body gravity, not bank ID order: right side toward ground has body gravity
y=-0.932..-0.992 and roll=+97..+111 degrees; left has opposite signs.
All 11 failures finished inverted at about0.057m with zero supporting feet.
The0.2s ramp reduced their first target jumps by approximately90% without
resolving them. This motivates checking directional skill asymmetry; it does
not prove it is the only cause. Stand3547 alone preserved upright20/20, whereas
the roll-first hard dual controller achieved only12/20. Mirroring side starts
must not be sold as a fix for that independent startup-selection problem.

## Exact representation, not ANYmal indices

The runtime report records native order:
`[FL_hip,FR_hip,RL_hip,RR_hip, FL_thigh,FR_thigh,RL_thigh,RR_thigh,
FL_calf,FR_calf,RL_calf,RR_calf]` (each name ends `_joint`).

Let `P=[1,0,3,2,5,4,7,6,9,8,11,10]` and
`S=[-1,-1,-1,-1,1,1,1,1,1,1,1,1]`.
Define `J(x)[i]=S[i]*x[P[i]]`. It is an involution: `J(J(x))=x`.
Official ANYmal uses a different native leg order, so its permutation must not
be pasted into this Go2 pipeline.

For body reflection `R=diag(1,-1,1)`, polar vectors transform by R; angular
velocity is axial and transforms by `det(R)R=diag(-1,1,-1)`.

| Actual48 policy slice | Transformation on a copy |
| --- | --- |
| 0:3 body linear velocity | multiply `[1,-1,1]` |
| 3:6 body angular velocity | multiply `[-1,1,-1]` |
| 6:9 body projected gravity | multiply `[1,-1,1]` |
| 9:12 command `[vx,vy,yaw_rate]` | multiply `[1,-1,-1]` |
| 12:24 measured `q-default_q` | J |
| 24:36 joint velocity minus zero nominal velocity | J |
| 36:48 actual previous issued raw action | J |
| 12-D actor output | J back into real native actuator coordinates |

The current observations are unnormalized/unscaled48 with no height scan and
the current actors are feed-forward. A new entry must assert those actual
contracts rather than silently applying this map to another model/layout.

## What has been proved, and what has not

PowerShell arithmetic on **actual hard-side report metadata** found exact zero
maximum error for `J(default_q)=default_q`, mirrored soft bounds, and mirrored
hard bounds. Actual nominal angles are hips `[+.1,-.1,+.1,-.1]`, front thighs
`.8/.8`, hind thighs `1/1`, and calves `-1.5` throughout. Hip bounds are symmetric;
thigh/calf bounds match their left/right partner (front/hind thigh bounds differ,
which is another reason not to use front/back interchange).

Consequently the current target map
`T(a)=soft_clamp(default_q+.25*a)` commutes algebraically with J. For negative
hip signs, the paired interval transforms as `[-upper,-lower]`; other paired
intervals are unchanged. This also makes transforming joint residuals valid:
`J(q-default)=J(q)-default`. These are representation results, not rollout results.

For existing **geometric acceptance**, exchanging FL/FR and RL/RR while reflecting
foot/knee body-y preserves every lateral bound; x/z stay unchanged, fore/hind
labels stay unchanged, and maximum absolute joint residual is invariant.
Thus the acceptance formula itself does not prefer left over right. This proves
invariance of the formula for reflected coordinates, **not** that the actual
USD forward kinematics or contact trajectories generate those coordinates.

The actual local asset is binary USDC and was not parsed in this no-Python
review. Local hashes: `go2.usd`=`ba171c972b987d8c8fb7157ccad2ba9c0c1fed105755d2d8af46bef96cc11c6d`;
`Props/instanceable_meshes.usd`=`2902646d0f4c13c9ecae3ac9046e32c7d18902eef9679de9829f642a15c93fb5`.
Before claiming morphological/dynamical symmetry, separately check actual USD
joint axes/origins, paired link frames/masses/CoM/inertias, collision shapes,
material and actuator properties. For an invariant center body inertia, y-reflection
requires xy/yz cross terms to vanish. The official Unitree URDF has nonzero base
`ixy=0.00012166` and `iyz=-3.12e-5`, despite paired left/right hip offsets and
leg geometry. This is explicit evidence against casually assuming perfect
physical symmetry; it is **not** proof that this separate local USD matches that
URDF. [Official Unitree Go2 description](https://raw.githubusercontent.com/unitreerobotics/unitree_ros/master/robots/go2_description/urdf/go2_description.urdf).

## One-factor diagnostic if selected

At the real policy-start boundary after unchanged1s settling, latch a per-trial
mask for confirmed lateral fallen starts with gravity-y<-0.5; log raw gravity,
quaternion and the decision. Only declared `side` trials may be selected. Do not
recompute the mask every frame or choose it using outcome/ID. Upright/back and
left-side starts retain the original roll branch.

Before the unchanged one-way gate, selected rows issue
`a_real=J(roll_actor(M48(real_observation)))`; unselected rows issue original
roll output. After the gate, **all rows use stand_actor(real_observation)**.
Always compute the standing actor and gate from real, untouched observations.
Action manager receives only native real output; it updates history naturally.
The next roll input transforms a copy of that real history. Never write mirrored
values back to observation/action buffers, reset history, alter physical state,
reflect the asset, insert PD, change the gate, add retry, enable a ramp, change
startup policy, or train. Keep hard handoff T=0 and existing550 policy steps and
final3s acceptance. Virtual roll inputs are explicitly labeled policy-interface
transforms, never saved as real measured states or accepted reset-bank samples.

Implement only in a **new separately hashed experimental entry/helper** if
authorized. Keep the RAW collector and completed transition experiment immutable.
An identity/no-mask control must first match the completed hard result fields.
Then test the complete20 side starts, with9 left controls and11 right tests,
followed by unchanged upright/back controls. Record native/virtual48 obs,
native/virtual roll actions, real standing actions, fixed masks, limits,
actual issued history/targets and all switch/final results. The9 unselected
left trajectories and upright/back should be unchanged under deterministic
same-batch controls; compare actual records, not just counts. Count all starts,
including untriggered and re-fallen cases. No promotion without later independent
starts and continuous front/oblique inspection.

Before any simulation, add bounded CPU tests for exact native names, M48/J
involutions, no mutation, per-row masks, identity equivalence, finite/shape
fail-closed behavior, default/bound symmetry and target-clamp commutation,
vector axial signs, acceptance-formula invariance, and natural native→virtual
last-action consistency. A separate read-only USD kinematic check must distinguish
approximate asset symmetry from exact algebra. No such CPU or physics test was
executed in this proposal turn. More restrictive supported-sitting gating remains
a competing **separate** factor; do not combine it with mirror/startup changes.

## Primary-source grounding and limits

[Official Isaac Lab ANYmal symmetry](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/symmetry/anymal.py)
uses the same polar/axial/command signs and transforms joint state and last action
consistently, but its joint indices are ANYmal-specific. This supports the mapping
pattern, not a Go2 recovery result.

[Official RSL-RL symmetry extension](https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/extensions/symmetry.py)
provides minibatch augmentation and a mirror-consistency loss; it does not make
this pretrained roll actor symmetric merely because the library supports them.
The current installed version implements symmetry within PPO, not that moving
main-branch extension API. No dependency/API migration is proposed.

[Mittal et al., 2024, sections II-B/III](https://arxiv.org/html/2403.04359v1)
define symmetry using compatible state/action transformations and invariant
reward/dynamics, distinguish task symmetry from periodic motion symmetry, and
warn that real robots need not be perfectly symmetric. Their methods modify
learning through augmentation/loss. This frozen inference-time side selection
is a falsifiable engineering diagnostic, **not reproduction of that training
method**, and offers no guaranteed recovery. Web main-branch sources were read
on2026-09-17 and are not a pin for the local assets/dependencies.

## Implementation decision addendum (same date; supersedes proposal selector only)

The main agent subsequently selected one isolated mirror diagnostic. Its fixed
selector is **real policy-start eligible-settled-fallen AND normalized gravity
body-y < -cos(30deg)**. It does not inspect requested pose labels, state IDs or
outcomes. This replaces the earlier proposed -0.5/declared-side condition above;
that proposal text is retained as decision history, not silently rewritten.
The actual upright/back controls must select zero rows; this is checked from
their real start measurements, not forced by the requested label.

New files are `scripts/evaluate_handoff_mirror.py`,
`src/go2_recovery/roll_mirror_math.py`, and `scripts/test_roll_mirror_math.py`.
They require frozen roll1999/stand3547 identities and hard handoff only.
`TensorDict.clone(recurse=True)` deep-copies all leaves; only the new dictionary's
policy tensor is replaced by a newly allocated masked transformed tensor.
Standing receives the original dictionary. Actor normalization must actually be
Identity and the runtime seven-term48 layout must match. Trace fields distinguish
real measured48, roll input48, model-coordinate output12 and physical-native12.
The implementation must pass off-mode exact controls before on-mode evaluation;
creation of these files does not establish simulation equivalence or success.
