# Stand-preserving recovery experiment — first100updates complete, not accepted

## Progress03:16 — validation and training complete; side success did not improve

All79TRAIN samples passed explicit real input replay (max48-observation error1.788139e-7), and the separate8env/24round subset-reset test passed selected-state/history checks and unaffected-environment checks. A one5ms finite contact probe passed. These do NOT serialize or reproduce full PhysX caches, certify safety, or upgrade the unchanged RAW manifest. Full reports: evaluations/20260917-handoff-replay-all79/handoff_replay_validation.json and evaluations/20260917-handoff-reset-subset8/handoff_reset_subset_validation.json.

After a16env2update smoke and upright20/20 guard, the isolated128env100update run trained307200steps from frozen3448 with verified full model/Adam/std load. Final model3547 SHA5527c4053cb6ef8e6616da65b7dc32a9464eb8d1366f4966a4eb4b65bc913fdb remains experimental. Upright20/20, cold1999->3547 side9/20 and back20/20 match the old pair's success counts; the SAME11side attempts end inverted at about0.057m with zero foot contacts. Faster recovery timing in successful cases is not sufficient: some joint offsets also grew. All16frozen portfolio models remain unchanged.

Do not automatically run the proposed remaining200updates or restart the100-update entry (it only supports fresh3448, not3547 continuation). Investigate the measured approximately2.5rad target jump and verified one-way latch separately. A subsequent finite convergence probe would need an explicit correct continuation entry and unchanged upright/back/side gates. These repeatedly inspected heldout cases are now development regressions, not fresh generalization evidence. No recovery video or unified walking/running controller has been accepted for3547.

## Progress02:40 — first real input replay passed; reset adapter still needs live validation

The real four-environment check at evaluations/20260917-handoff-replay-input-contact4/handoff_replay_validation.json completed with process exit0. Sample IDs0/1/40/41 cover left/right/back TRAIN sources. Explicit rootLINK/CoMvelocity/q/qd/history/target state and recomputed48-value observations matched the recorded sources within the unchanged2e-5 tolerance. A separate single5ms previous-target hold produced finite reported foot forces and heights. This is input/API evidence, NOT recovery or deterministic future-trajectory evidence; all79states are not yet covered by this completed check.

The first run closed Kit before writing its report and is unusable as evidence. The retry wrote a passing input report, then crashed during nondefault full Kit shutdown. Restoring the historical default shutdown resolved that in the four-sample run. All three output/log directories are retained; success requires the report plus a healthy process, not an exit code alone.

New isolated, uninstalled modules handoff_action_restore.py, handoff_reset_env.py, and handoff_env_cfg.py preserve old tasks. The reset adapter runs physical/history restoration only after the normal manager reset.9CPU adapter tests and12history tests pass. Its fixed80/20 mixture means ordinary upright training drops versus actual TRAIN handoff snapshots in expectation, not already-settled standing or exact per-batch quotas. The standalone config retains Aligned3448 rewards with Smith-compatible fixed physics and nominal soft-clamped actions. No training has used these modules yet. Required next gates: all79real input checks, actual subset-reset isolation and observations, finite contact probe, then a finite training smoke and unchanged upright20/20 guard.

The RAW manifest remains training_ready=false. CPU tests, a small input check, and a proposed new environment must not upgrade it implicitly. No full PhysX solver/contact-cache replay is claimed.

## Earlier progress01:29 — data collected, physical replay still unimplemented

New independent cold TRAIN40side/40back rollouts completed with the frozen1999->3448pair, seed20260917.79triggered states from79different TRAIN source IDs were exported;1untriggered back case remains in the80trial source ledger. The sample archive is datasets/handoff_states/train1999_to3448_20260917/handoff_observations.npz (SHA4546db065dad9127b1591440f1248ae4ebaf50f556af9a5533c6a9b2abc3c1b5). Schemahandoff_observation_collection_v1 is deliberately RAW: training_ready=false,replay_validated=false,not compatible with the old nominal_pd_fallen_v1loader.16CPU export/validation tests passed. No3448handoff-policy continuation has launched.

Each sample retains link pose, CoM world velocities,q/qd,actual a[t-1]/a[t-2],last executed target,48-value observation and command. IDs were looked up by original state_id values(not array offsets) and checked split0=TRAIN; noheldout snapshots were used. Recorded q is unchanged despite79soft-limit exceedances(max.106209rad) and39hard-limit micro-exceedances(max.001489rad). Neither silently clamp these measured states nor claim they are certified safe. Export metadata records this explicitly.

Reset implementation must run AFTER super()._reset_idx because the ordinary event executes before action_manager.reset. Restore only chosen environments' two-level history and action-term target caches, without a whole-batch process_actions call or hidden settling PD. Recompute actual observations from simulator state; offline algebraic agreement is not simulation validation. Full PhysX contact/friction/solver caches are unavailable, so cold replay cannot promise identical future trajectories. Ultimately validate continuous1999->candidate transitions, not just reset samples. A pureCPU subset-history helper is being prepared separately and is not installed or used yet.

Following the failed DensePosture600-update experiment, preserve2250/3448/Smith1999 and all evidence. Investigate a COPY of3448 trained to stand from a physically collected train-only handoff distribution, while preserving ordinary standing. Do not extend the failed Dense branch unchanged. This is a proposal for one bounded experiment, not an overnight success claim.

## Evidence and scope

The completed Dense and historical3448 upright-PD compatibility reports have exactly equal20 release states and20 policy-start states under identical SmithNominal Play, seed20260918 and preparation protocol.3448 achieved20/20final valid stands with mean height0.3254m. Dense achieved0/20 with mean height0.1989m and all final heights below0.229m. Dense bank-side/back/30degree-side/30degree-fore likewise all0/20final valid. Training reward202.31 did not translate into standing. The recorded CSV contains only diagnostic trial0; never present its trajectory as all20trials.

Frozen Smith1999->3448 cold-process diagnostic currently gives side9/20 and back20/20 final-valid, including all untriggered attempts in the denominator. This is two-policy development evidence, not a visually accepted universal or single-policy controller. Existing recorded heldout switch observations were already analyzed and MUST NOT become training data.

## One falsifiable intervention

1. Freeze all existing models. Train a separate3448copy with its existing standing objective, not another reward-weight search from a nonstanding policy.
2. Use80% verified physically attainable ordinary-standing/stand-up preparation starts and20% actual handoff states reached by frozen1999 from TRAIN bank only. Validate physics/action/observation semantics against the existing compatible actor before comparing. Do not fabricate a sitting pose by lowering root height.
3. Collect/replay true root and joint velocities, previous raw action and executed target, and all other controller memory that affects the transition. A reset that silently zeros action history is not faithful replay. First test simulator round-trip and actual observation equality; if not supported, implement/test collection before training. No unseen manual PD or history clearing at the learned-policy switch.
4. Proposed budget128environments x300updates with upright compatibility check every100updates. Stop and reject on any loss from the20/20 upright baseline; preserve checkpoints and reports. Pin configs/sources/parent/train-bank identities and use newE-drive paths. Do not overlap the active speed probe.
5. Development goals fixed before training: upright20/20; frozen1999->candidate back at least20/20 and side at least15/20, using the identical cold independent pose process and complete denominator with unchanged final3-second valid stance. This checks a stand-up subskill, not integrated locomotion.
6. Only after development improvement, test new seeds/states and actually inspect front/oblique simulator videos. Ordinary locomotion, turns, pushes and stop remain separate regressions. Never call two independent actors or edited videos a complete continuous controller.

This design still needs implementation, data collection and smoke validation. Do not report it as running before a real training process starts.
