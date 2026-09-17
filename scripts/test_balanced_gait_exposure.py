import unittest
import io
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
from balanced_gait_exposure import DurationExposure, ExposureWindows, eligible


class ExposureTests(unittest.TestCase):
    def test_gate_boundaries(self):
        commands = torch.tensor([[.5,0,0],[0.,0,0],[.1,0,0],[.5,.1,0],[.5,0,.3],[-.5,0,0]])
        self.assertEqual(eligible(commands).tolist(), [True,False,False,False,False,False])

    def test_actual_complete_cycles_and_reset_discard(self):
        tracker = ExposureWindows(2, "cpu")
        command = torch.tensor([[.5,0,0],[.5,0,0]])
        original = command.clone()
        before = torch.zeros((2,4), dtype=torch.bool)
        # Discard the initially partial command segment, establishing a real boundary.
        tracker.update(torch.zeros((2,3)),command,before,before,torch.zeros(2,dtype=torch.bool))
        for k in range(4):
            after = ~before
            next_command = command.clone()
            done = torch.zeros(2, dtype=torch.bool)
            if k == 3:
                next_command.zero_()
                done[1] = True
            tracker.update(command, next_command, before, after, done)
            before = after
        result = tracker.summary()
        self.assertEqual(result["completed_command_windows"],1)
        self.assertEqual(result["completed_eligible_windows_two_raw_touch_and_lift_per_foot"],1)
        self.assertEqual(result["reset_discarded_windows"],1)
        self.assertEqual(result["unfinished_windows_at_end"],0)
        self.assertTrue(torch.equal(command, original))

    def test_single_contact_is_not_enough(self):
        tracker=ExposureWindows(1,"cpu")
        tracker.update(torch.tensor([[.5,0,0]]),torch.zeros((1,3)),torch.zeros((1,4),dtype=torch.bool),
                       torch.ones((1,4),dtype=torch.bool),torch.zeros(1,dtype=torch.bool))
        self.assertEqual(tracker.summary()["completed_eligible_windows_two_raw_touch_and_lift_per_foot"],0)

    def test_nonfinite_rejected(self):
        tracker=ExposureWindows(1,"cpu")
        tracker.update(torch.tensor([[float("nan"),0,0]]),torch.zeros((1,3)),torch.zeros((1,4),dtype=torch.bool),
                       torch.ones((1,4),dtype=torch.bool),torch.zeros(1,dtype=torch.bool))
        with self.assertRaises(ValueError):
            tracker.summary()
        # Invalid evidence stays invalid even if later measurements are finite.
        tracker.update(torch.zeros((1,3)),torch.zeros((1,3)),torch.zeros((1,4),dtype=torch.bool),
                       torch.ones((1,4),dtype=torch.bool),torch.zeros(1,dtype=torch.bool))
        with self.assertRaises(ValueError):
            tracker.summary()

    def test_initial_partial_excluded_even_if_cycles_present(self):
        tracker=ExposureWindows(1,"cpu")
        tracker.touch[:]=3
        tracker.lift[:]=3
        tracker.update(torch.tensor([[.5,0,0]]),torch.zeros((1,3)),torch.zeros((1,4),dtype=torch.bool),
                       torch.ones((1,4),dtype=torch.bool),torch.zeros(1,dtype=torch.bool))
        self.assertEqual(tracker.summary()["initial_partial_windows_discarded"],1)
        self.assertEqual(tracker.summary()["completed_command_windows"],0)

    def test_observer_never_triggers_lazy_sensor_data_or_changes_result(self):
        class Sensor:
            cfg=NS(history_length=3,update_period=.005,track_air_time=True,force_threshold=1.0)
            _data=NS(current_contact_time=torch.zeros((1,4)),last_air_time=torch.zeros((1,4)),
                     last_contact_time=torch.zeros((1,4)))
            @property
            def data(self):
                raise AssertionError("Lazy sensor property must not be read by telemetry")
            def find_bodies(self,names,preserve_order):
                return list(range(4)),names
        command=torch.tensor([[.5,0,0]])
        result=(object(),object(),torch.ones(1,dtype=torch.long),{})
        env=NS(unwrapped=NS(scene=NS(sensors={"contact_forces":Sensor()}),num_envs=1,device="cpu",
                             command_manager=NS(get_command=lambda name:command)),step=lambda action:result)
        previous=env.step
        with patch.object(Path,"open",return_value=io.StringIO()):
            observer=DurationExposure(env,Path("unused-in-memory"))
            # Non-flush hot path must never convert a tensor scalar or copy a
            # tensor to host; these would impose CUDA synchronization in training.
            with patch.object(torch.Tensor,"__bool__",side_effect=AssertionError("tensor bool sync")), \
                 patch.object(torch.Tensor,"__int__",side_effect=AssertionError("tensor int sync")), \
                 patch.object(torch.Tensor,"__float__",side_effect=AssertionError("tensor float sync")), \
                 patch.object(torch.Tensor,"cpu",side_effect=AssertionError("tensor host copy")), \
                 patch.object(torch.Tensor,"item",side_effect=AssertionError("tensor scalar sync")):
                self.assertIs(env.step(torch.zeros((1,12))),result)
            summary=observer.finish()
        self.assertIs(env.step,previous)
        self.assertTrue(summary["lazy_sensor_property_never_read"])
        self.assertEqual(command.tolist(),[[.5,0.,0.]])

    def observer_fixture(self, stream, nonfinite=False):
        command=torch.tensor([[.5,0,0],[.5,0,0]])
        data=NS(current_contact_time=torch.zeros((2,4)),
                last_air_time=torch.tensor([[.1,.2,.3,.4],[.5,.5,.5,.5]]),
                last_contact_time=torch.tensor([[.4,.3,.2,.1],[.5,.5,.5,.5]]))
        if nonfinite:
            data.last_air_time[0,0]=float("nan")
        sensor=NS(cfg=NS(history_length=3,update_period=.005,track_air_time=True,force_threshold=1.0),
                  _data=data,find_bodies=lambda names,preserve_order:(list(range(4)),names))
        result=(object(),object(),torch.tensor([0,1],dtype=torch.long),{})
        env=NS(unwrapped=NS(scene=NS(sensors={"contact_forces":sensor}),num_envs=2,device="cpu",
                           command_manager=NS(get_command=lambda name:command)),step=lambda action:result)
        with patch.object(Path,"open",return_value=stream):
            observer=DurationExposure(env,Path("unused-in-memory"))
        return env,observer,data

    def test_batch_flush_and_finish_preserve_nonreset_statistics(self):
        stream=io.StringIO()
        env,observer,data=self.observer_fixture(stream)
        expected=float(data.last_air_time[0].var(correction=1)+data.last_contact_time[0].var(correction=1))
        for _ in range(23):
            env.step(None)
        self.assertEqual(stream.getvalue(),"")
        env.step(None)
        rows=[json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]["control_step"],24)
        self.assertEqual(rows[0]["eligible_fraction"],.5)
        self.assertAlmostEqual(rows[0]["raw_mean_nonreset"],expected)
        self.assertEqual(rows[0]["environment_intervals"],48)
        self.assertEqual(rows[0]["eligible_nonreset_environment_intervals"],24)
        self.assertEqual(rows[0]["reset_discarded_windows"],24)
        summary=observer.finish()
        self.assertAlmostEqual(summary["raw_variance_mean_nonreset"],expected)
        self.assertAlmostEqual(summary["gated_variance_sum"],24*expected)
        self.assertEqual(summary["control_steps"],24)
        json.dumps(summary,allow_nan=False)

    def test_nonfinite_variance_fails_at_flush_and_finish(self):
        for steps in (1,24):
            stream=io.StringIO()
            env,observer,_=self.observer_fixture(stream,nonfinite=True)
            for _ in range(steps-1):
                env.step(None)
            if steps==24:
                with self.assertRaisesRegex(ValueError,"Nonfinite measured"):
                    env.step(None)
            else:
                env.step(None)
            with self.assertRaisesRegex(ValueError,"Nonfinite measured"):
                observer.finish()
            self.assertIs(env.step,observer.original_step)
            self.assertTrue(stream.closed)


if __name__ == "__main__":
    unittest.main()
