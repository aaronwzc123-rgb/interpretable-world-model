"""Regression checks for actual-seed accounting and M3 intervention semantics."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import torch

HERE = Path(__file__).resolve()
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE.parents[1] / 'scripts'), str(REPO / 'scripts'), str(REPO)]
import evaluate_m3_corrected_flip as evaluation
import staged_stage_c_eval as sc


class FakeHead:
    def __init__(self): self.index = 0
    def atoms(self, feat):
        values = [-.3, .2, -.1]
        value = values[self.index % len(values)]
        self.index += 1
        return torch.tensor([[value, .4, -.5, .6, -.7, .8, -.9, .1, -.2, .3]])


def fake_components():
    wm = SimpleNamespace(preprocess=lambda x:x, encoder=lambda x:None,
        dynamics=SimpleNamespace(obs_step=lambda *a, **k:(None,None), get_feat=lambda x:None),
        _sym_head=FakeHead())
    behavior = SimpleNamespace(actor=lambda atoms:SimpleNamespace(mode=lambda:torch.tensor([[1.,0.,0.,0.,0.]])))
    return wm, behavior


class M3EvaluationAuditTests(unittest.TestCase):
    def test_wraparound_is_rejected_before_rollouts(self):
        with self.assertRaisesRegex(ValueError, 'Repeated actual'):
            evaluation.seed_inventory(range(201), 'pool', 3)
        inventory = evaluation.seed_inventory(range(201), 'direct', 3)
        self.assertEqual(len({p['actual_seed'] for p in inventory}),201)
        self.assertEqual(inventory[-1]['actual_seed'],200)

    def test_timeout_near_alternative_endpoint_never_redirects(self):
        row = dict(truncated=True, final_pos=(5,4), key_pos=(1,1), ball_pos=(5,5),cue_is_key=1)
        self.assertEqual(evaluation.outcome(row), 'timeout')
        result = evaluation.summarize([{'outcome':'target'}],[{'outcome':evaluation.outcome(row)}],10,0)
        self.assertEqual(result['baseline_target_to_other_rate'],0)
        self.assertEqual(result['baseline_target_to_timeout_rate'],1)

    def test_timeout_stops_at_exact_environment_limit(self):
        wm, behavior = fake_components()
        rows, result = sc.rollout(wm, behavior, 0, 3, eval_seed_mode='direct')
        self.assertEqual(result['steps'],3)
        self.assertEqual(len(rows),3)
        self.assertTrue(result['truncated'])
        self.assertEqual(result['actual_seed'],0)

    def test_negation_tracks_current_output_and_preserves_other_atoms(self):
        wm, behavior = fake_components()
        _, result = sc.rollout(wm, behavior, 0, 3, flip_idx=0, flip_start=0, eval_seed_mode='direct')
        trace = result['intervention_trace']
        np.testing.assert_allclose([r['policy_cue'] for r in trace],[.3,-.2,.1])
        for row in trace:
            np.testing.assert_array_equal(row['predicted_atoms'][1:],row['policy_atoms'][1:])
        self.assertEqual(result['intervention_operation'],'negate_current_output')

    def test_fixed_value_is_explicit_and_distinct(self):
        wm, behavior = fake_components()
        with self.assertRaisesRegex(ValueError,'requires an explicit'):
            sc.rollout(wm, behavior, 0, 3, flip_idx=0, flip_start=0, flip_mode='fixed_value')
        _, result = sc.rollout(wm, behavior, 0, 3, flip_idx=0, flip_start=0,
            flip_mode='fixed_value', fixed_flip_value=.8, eval_seed_mode='direct')
        np.testing.assert_allclose([r['policy_cue'] for r in result['intervention_trace']],[.8,.8,.8])

    def test_no_intervals_for_descriptive_summary(self):
        result=evaluation.summarize([{'outcome':'target'}],[{'outcome':'timeout'}],0,0)
        self.assertIsNone(result['other_rate_gain_ci95'])
        self.assertIsNone(result['zero_redirection_upper95_one_sided'])


if __name__ == '__main__':
    unittest.main()
