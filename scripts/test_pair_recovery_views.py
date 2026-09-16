"""CPU metadata checks; no video encoding or simulator initialization."""
import copy
import unittest
from pair_recovery_views import validate_reports


def legacy():
    return {'checkpoint_sha256': 'abc', 'task': 'task', 'seed': 42, 'angle_deg': 30,
            'protocol_version': 'v1', 'criterion': 'strict',
            'results': {'side': {'trials': 1, 'successes': 0}}}


def bank():
    r = legacy()
    r.update({'settle_actual_s': 1., 'settle_control_steps': 50, 'settle_controller': 'nominal_pd',
              'policy_action_mode': 'deterministic_mean', 'acceptance_eligible': True,
              'action_representation': {'reference': 'current', 'scale': .25},
              'state_bank': {'sha256': 'bank', 'schema_version': 'nominal_pd_v1', 'split': 'heldout',
                             'selected_state_ids': [132]}})
    r['results']['side'].update({
        'final_valid_stands': 0, 'settled_fallen_trials': 1, 'settled_fallen_recovery_successes': 0,
        'settled_fallen_final_valid_stands': 0, 'stable_hold_s': 3., 'horizon_s': 8.,
        'state_bank_selection': {'selected_state_ids': [132], 'selected_unique_state_count': 1},
        'policy_start_state': [{'trial': 0, 'settled': True, 'standing_at_policy_start': False,
            'fallen_at_policy_start': True, 'eligible_settled_fallen_recovery': True,
            'classification': 'settled_fallen_under_nominal_pose_PD', 'bank_state_id': 132,
            'bank_saved_pose_class': 'left', 'bank_requested_pose_class': 'left',
            'actual_pose_class': 'left', 'height_m': .15}],
        'final_diagnostics': [{'trial': 0, 'geometry_ok': False, 'stable_hold_s': 0., 'height_m': .08}]})
    return r


class PairTests(unittest.TestCase):
    def test_legacy_and_matching_failed_bank_accepted(self):
        for factory in (legacy, bank):
            report = factory()
            validate_reports(report, copy.deepcopy(report))

    def test_optional_field_only_one_side_rejected(self):
        a = legacy(); b = copy.deepcopy(a); b['action_representation'] = None
        with self.assertRaises(ValueError): validate_reports(a, b)

    def test_required_bank_fields_cannot_be_missing_on_both_sides(self):
        for key in ('settle_actual_s', 'settle_controller', 'policy_action_mode', 'acceptance_eligible'):
            with self.subTest(key=key):
                a = bank(); del a[key]
                with self.assertRaises(ValueError): validate_reports(a, copy.deepcopy(a))
        for key in ('sha256', 'schema_version', 'split', 'selected_state_ids'):
            with self.subTest(key=key):
                a = bank(); del a['state_bank'][key]
                with self.assertRaises(ValueError): validate_reports(a, copy.deepcopy(a))

    def test_different_controller_policy_action_and_bank_rejected(self):
        changes = [('settle_actual_s', 2.), ('settle_controller', 'zero_torque'),
                   ('policy_action_mode', 'stochastic'), ('acceptance_eligible', False),
                   ('action_representation', {'reference': 'nominal', 'scale': .25})]
        for key, value in changes:
            with self.subTest(key=key):
                a = bank(); b = copy.deepcopy(a); b[key] = value
                with self.assertRaises(ValueError): validate_reports(a, b)
        a = bank(); b = copy.deepcopy(a); b['state_bank']['sha256'] = 'other'
        with self.assertRaises(ValueError): validate_reports(a, b)

    def test_eligibility_selection_and_record_ids_rejected(self):
        for key, value in (('bank_state_id', 133), ('eligible_settled_fallen_recovery', False), ('trial', 1)):
            a = bank(); b = copy.deepcopy(a)
            b['results']['side']['policy_start_state'][0][key] = value
            with self.assertRaises(ValueError): validate_reports(a, b)
        a = bank(); a['results']['side']['state_bank_selection']['selected_state_ids'] = [133]
        with self.assertRaises(ValueError): validate_reports(a, copy.deepcopy(a))

    def test_outcomes_and_final_hold_rejected(self):
        for key in ('successes', 'final_valid_stands', 'settled_fallen_recovery_successes',
                    'settled_fallen_final_valid_stands'):
            a = bank(); b = copy.deepcopy(a); b['results']['side'][key] = 1
            with self.assertRaises(ValueError): validate_reports(a, b)
        for key, value in (('geometry_ok', True), ('stable_hold_s', 3.)):
            a = bank(); b = copy.deepcopy(a); b['results']['side']['final_diagnostics'][0][key] = value
            with self.assertRaises(ValueError): validate_reports(a, b)

    def test_minor_physical_drift_does_not_claim_bitwise_trajectory_match(self):
        a = bank(); b = copy.deepcopy(a)
        b['results']['side']['policy_start_state'][0]['height_m'] += 1e-6
        b['results']['side']['final_diagnostics'][0]['height_m'] += 1e-6
        validate_reports(a, b)

    def test_different_pose_or_missing_trial_records_rejected(self):
        a = bank(); b = copy.deepcopy(a); b['results']['back'] = b['results'].pop('side')
        with self.assertRaises(ValueError): validate_reports(a, b)
        a = bank(); del a['results']['side']['policy_start_state']
        with self.assertRaises(ValueError): validate_reports(a, copy.deepcopy(a))

    def test_source_order_preserved_for_multiple_trials(self):
        a = bank(); result = a['results']['side']
        result['trials'] = 2; result['settled_fallen_trials'] = 2
        result['state_bank_selection'] = {'selected_state_ids': [132, 150], 'selected_unique_state_count': 2}
        start = copy.deepcopy(result['policy_start_state'][0]); start.update(trial=1, bank_state_id=150)
        result['policy_start_state'].append(start)
        final = copy.deepcopy(result['final_diagnostics'][0]); final['trial'] = 1
        result['final_diagnostics'].append(final)
        a['state_bank']['selected_state_ids'] = [132, 150]
        validate_reports(a, copy.deepcopy(a))
        b = copy.deepcopy(a); b['results']['side']['state_bank_selection']['selected_state_ids'].reverse()
        with self.assertRaises(ValueError): validate_reports(a, b)
        a['state_bank']['selected_state_ids'].reverse()
        with self.assertRaises(ValueError): validate_reports(a, copy.deepcopy(a))


if __name__ == '__main__': unittest.main()
