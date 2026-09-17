"""Pure logical tests only: no simulator, Torch, model or physical verification."""

from dataclasses import FrozenInstanceError, replace
import inspect
import unittest

import continuous_flow_phase as flow


def advance(state, *, valid=True, standing=True, fallen=False, done=False, finite=True, quiet=True):
    plan = flow.next_action(state, recovery_stand_active=standing if state.phase == "recovery" else None)
    measured = flow.CompletedInterval(plan.absolute_control_step, plan.absolute_control_step + 1,
                                      plan.real_command, plan.real_command, valid, fallen, done, finite, quiet)
    return flow.after_completed_interval(state, plan, measured)


def repeat(state, count, **kwargs):
    for _ in range(count):
        state = advance(state, **kwargs)
    return state


class ContinuousPhaseTests(unittest.TestCase):
    def test_strict_alone_does_not_start_walking_before_quiet150(self):
        state = repeat(flow.initial_state(), 150, quiet=False)
        self.assertEqual((state.phase, state.recovery_strict_count, state.recovery_ready_count), ("recovery", 150, 0))
        state = repeat(state, 149)
        self.assertEqual(state.phase, "recovery")
        state = advance(state)
        self.assertEqual((state.phase, state.recovery_ready_count), ("move", 150))

    def test_not_quiet_restarts_ready_without_forging_invalid_geometry(self):
        state = repeat(flow.initial_state(), 149)
        state = advance(state, quiet=False)
        self.assertEqual((state.recovery_strict_count, state.recovery_ready_count), (150, 0))
        self.assertEqual(repeat(state, 150).phase, "move")

    def test_quiet_never_substitutes_for_strict_or_extends_budget(self):
        state = repeat(flow.initial_state(), 550, valid=False, quiet=True)
        self.assertEqual((state.phase, state.recovery_ready_count), ("failed", 0))
        state = repeat(flow.initial_state(), 550, valid=True, quiet=False)
        self.assertEqual((state.phase, state.recovery_strict_count), ("failed", 150))

    def test_no_initial_sample_and_exact150_next_action(self):
        initial = flow.initial_state(actual_control_step=50)
        self.assertEqual(initial.completed_intervals, 0)
        state = repeat(initial, 149)
        self.assertEqual(state.phase, "recovery")
        plan = flow.next_action(state, recovery_stand_active=True)
        self.assertEqual((plan.actor_role, plan.real_command, plan.absolute_control_step), ("stand", flow.ZERO_COMMAND, 199))
        after = advance(state)
        self.assertEqual(after.phase, "move")
        self.assertEqual(after.recovery_hold_completed_at, 200)
        self.assertEqual(after.move_intervals, 0)
        self.assertEqual(flow.next_action(after).real_command, (.5, 0.0, 0.0))
        self.assertEqual((initial.completed_intervals, state.recovery_strict_count), (0, 149))

    def test_invalid_standing_sample_restarts_hold(self):
        state = advance(repeat(flow.initial_state(), 149), valid=False)
        self.assertEqual((state.phase, state.recovery_strict_count), ("recovery", 0))
        self.assertEqual(repeat(state, 149).phase, "recovery")
        self.assertEqual(repeat(state, 150).phase, "move")

    def test_roll_samples_do_not_count_as_standing_hold(self):
        state = repeat(flow.initial_state(), 200, standing=False)
        self.assertEqual(state.recovery_strict_count, 0)
        self.assertEqual(repeat(state, 150).phase, "move")

    def test_original_stand_selector_cannot_unlatch(self):
        state = advance(flow.initial_state())
        with self.assertRaises(ValueError):
            flow.next_action(state, recovery_stand_active=False)

    def test_optional_refinement_separate150_and_next_action(self):
        state = repeat(flow.initial_state(flow.FlowConfig(.8, True)), 150)
        self.assertEqual((state.phase, state.refinement_strict_count), ("refinement", 0))
        self.assertEqual(flow.next_action(state).actor_role, "refinement")
        self.assertEqual(flow.next_action(state).real_command, flow.ZERO_COMMAND)
        state = repeat(state, 149)
        self.assertEqual(state.phase, "refinement")
        state = advance(state)
        self.assertEqual((state.phase, state.refinement_hold_completed_at, state.premove_intervals), ("move", 300, 300))
        self.assertEqual(flow.next_action(state).real_command, (0.800000011920929, 0.0, 0.0))

    def test_point8_actual_float32_command_is_exact_not_tolerated(self):
        state = repeat(flow.initial_state(flow.FlowConfig(.8)), 150)
        plan = flow.next_action(state)
        actual = (0.800000011920929, 0.0, 0.0)
        sample = flow.CompletedInterval(plan.absolute_control_step, plan.absolute_control_step + 1,
                                        actual, actual, False, False, False, True, False)
        self.assertEqual(flow.after_completed_interval(state, plan, sample).phase, "move")
        for wrong in ((.8, 0.0, 0.0), (0.8000000715255737, 0.0, 0.0), (.5, 0.0, 0.0)):
            failed = flow.after_completed_interval(state, plan, replace(sample, command_after=wrong))
            self.assertEqual(failed.failure_reason, "command_mismatch")
        self.assertEqual(sample.command_after, actual)

    def test_refinement_invalid_restarts_only_new_hold(self):
        state = repeat(flow.initial_state(flow.FlowConfig(.5, True)), 150)
        state = advance(repeat(state, 100), valid=False)
        self.assertEqual((state.recovery_strict_count, state.refinement_strict_count), (150, 0))
        self.assertEqual(repeat(state, 150).phase, "move")

    def test_premove_budget550_includes_refinement(self):
        failed = repeat(flow.initial_state(), 550, valid=False, standing=False, fallen=True)
        self.assertEqual((failed.phase, failed.premove_intervals), ("failed", 550))
        self.assertEqual(failed.failure_reason, "shared_premove_budget_exhausted")
        state = repeat(flow.initial_state(flow.FlowConfig(.5, True)), 251, valid=False, standing=False)
        state = repeat(state, 150)
        state = repeat(state, 149)
        self.assertEqual((state.phase, state.refinement_strict_count), ("failed", 149))

    def test_hold_on_last_allowed_premove_sample_succeeds(self):
        state = repeat(flow.initial_state(), 400, valid=False, standing=False)
        state = repeat(state, 150)
        self.assertEqual((state.phase, state.premove_intervals), ("move", 550))
        state = repeat(flow.initial_state(flow.FlowConfig(.5, True)), 250, valid=False, standing=False)
        state = repeat(state, 300)
        self.assertEqual((state.phase, state.premove_intervals), ("move", 550))

    def test_exact400_move_then300_stop_no_stand_fallback(self):
        state = repeat(flow.initial_state(), 150)
        state = repeat(state, 399, valid=False)
        self.assertEqual((state.phase, state.move_intervals), ("move", 399))
        state = advance(state, valid=False)
        self.assertEqual((state.phase, state.move_intervals, state.stop_intervals), ("stop", 400, 0))
        self.assertEqual((flow.next_action(state).actor_role, flow.next_action(state).real_command),
                         ("locomotion", flow.ZERO_COMMAND))
        state = repeat(state, 150, valid=False)
        state = repeat(state, 149)
        self.assertEqual((state.phase, state.stop_strict_count), ("stop", 149))
        state = advance(state)
        self.assertEqual((state.phase, state.completed_intervals, state.stop_intervals), ("complete", 850, 300))

    def test_early_stop_hold_does_not_hide_invalid_last_sample(self):
        state = repeat(flow.initial_state(), 150)
        state = repeat(state, 400, valid=False)
        state = repeat(state, 299)
        state = advance(state, valid=False)
        self.assertEqual((state.phase, state.failure_reason, state.stop_strict_count), ("failed", "stop_last150_not_strict", 0))

    def test_fallen_initial_recovery_allowed_later_fall_fails(self):
        state = advance(flow.initial_state(), valid=False, standing=False, fallen=True)
        self.assertEqual(state.phase, "recovery")
        for state in (repeat(flow.initial_state(), 150),
                      repeat(flow.initial_state(flow.FlowConfig(.5, True)), 150),
                      repeat(repeat(flow.initial_state(), 150), 400, valid=False)):
            failed = advance(state, valid=False, fallen=True)
            self.assertEqual((failed.phase, failed.failure_reason), ("failed", "later_fall"))

    def test_done_nonfinite_and_contradictory_strict_fail_closed(self):
        for fields, reason in (({"done": True}, "unexpected_done"), ({"finite": False}, "nonfinite_state"),
                                ({"fallen": True}, "inconsistent_strict_and_fallen")):
            state = advance(flow.initial_state(), **fields)
            self.assertEqual((state.phase, state.failure_reason, state.completed_intervals), ("failed", reason, 1))

    def test_real_command_before_and_after_must_match_actor_routing(self):
        for state in (flow.initial_state(), repeat(flow.initial_state(), 150)):
            plan = flow.next_action(state, recovery_stand_active=True if state.phase == "recovery" else None)
            sample = flow.CompletedInterval(plan.absolute_control_step, plan.absolute_control_step+1,
                                            plan.real_command, plan.real_command, True, False, False, True, True)
            wrong = (.8, .1, 0.0)
            for field in ("command_before", "command_after"):
                result = flow.after_completed_interval(state, plan, replace(sample, **{field: wrong}))
                self.assertEqual((result.phase, result.failure_reason), ("failed", "command_mismatch"))

    def test_t0_repeated_skipped_and_reset_counter_rejected(self):
        state = flow.initial_state(actual_control_step=50)
        plan = flow.next_action(state, recovery_stand_active=True)
        sample = flow.CompletedInterval(50, 51, flow.ZERO_COMMAND, flow.ZERO_COMMAND, True, False, False, True, True)
        for before, after in ((50, 50), (50, 52), (0, 1), (49, 50)):
            with self.assertRaises(ValueError):
                flow.after_completed_interval(state, plan, replace(sample, control_step_before=before, control_step_after=after))
        next_state = flow.after_completed_interval(state, plan, sample)
        with self.assertRaises(ValueError):
            flow.after_completed_interval(next_state, plan, sample)

    def test_no_old_selector_call_after_handoff_or_terminal_action(self):
        state = repeat(flow.initial_state(), 150)
        with self.assertRaises(ValueError):
            flow.next_action(state, recovery_stand_active=True)
        for terminal in (advance(flow.initial_state(), done=True), repeat(repeat(state, 400), 300)):
            with self.assertRaises(ValueError):
                flow.next_action(terminal)

    def test_immutable_no_mutable_action_history_or_environment_api(self):
        state = flow.initial_state()
        with self.assertRaises(FrozenInstanceError):
            state.phase = "move"
        self.assertFalse(hasattr(state, "reset"))
        for fn in (flow.initial_state, flow.next_action, flow.after_completed_interval):
            self.assertNotIn("env", inspect.signature(fn).parameters)
            self.assertNotIn("action_history", inspect.signature(fn).parameters)

    def test_invalid_types_and_forged_phase_or_plan_rejected(self):
        for speed in (1., 0., float("nan"), True):
            with self.assertRaises(ValueError):
                flow.FlowConfig(speed)
        state = flow.initial_state()
        with self.assertRaises(ValueError):
            flow.next_action(replace(state, phase="move"))
        plan = flow.next_action(state, recovery_stand_active=True)
        sample = flow.CompletedInterval(0, 1, flow.ZERO_COMMAND, flow.ZERO_COMMAND, True, False, False, True, True)
        for bad in (replace(sample, strict_valid=1), replace(sample, quiet_for_handoff=1), replace(sample, control_step_after=True),
                    replace(sample, command_after=(float("nan"), 0.0, 0.0))):
            with self.assertRaises(ValueError):
                flow.after_completed_interval(state, plan, bad)
        with self.assertRaises(ValueError):
            flow.after_completed_interval(state, replace(plan, actor_role="locomotion"), sample)


if __name__ == "__main__":
    unittest.main(verbosity=2)
