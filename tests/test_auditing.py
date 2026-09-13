import unittest

import numpy as np
import pandas as pd

from morphoraman.auditing import correctness_auc, summarize_audit
from morphoraman.core import center_logits, compute_dg_index, compute_dg_steps, consensus_with_stress
from morphoraman import melanoma, diabetes, bacteria


class AuditTests(unittest.TestCase):
    def test_below_chance_auc_is_not_flipped(self):
        self.assertEqual(correctness_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]), 0.0)

    def test_single_class_auc_is_undefined(self):
        self.assertTrue(np.isnan(correctness_auc([1, 1], [0.2, 0.8])))

    def test_ties(self):
        self.assertEqual(correctness_auc([0, 1], [0.5, 0.5]), 0.5)

    def test_summary_preserves_negative_association(self):
        frame = pd.DataFrame({'run': [0] * 4, 'damage_mode': ['mixed'] * 4,
                              'regime': ['H1'] * 4, 'correct': [0, 0, 1, 1],
                              'eREDG': [0.9, 0.8, 0.2, 0.1]})
        row = summarize_audit(frame, 'run').iloc[0]
        self.assertEqual(row['auc_eREDG_to_correct'], 0.0)
        self.assertEqual(row['cliffs_delta'], -1.0)
        frame['DG'] = 1.0
        row = melanoma.summarize_one_audit_condition(
            frame, dataset='melanoma', repeat_id=0,
            damage_fraction=0.4, damage_mode='mixed', regime='H1')
        self.assertEqual(row['AUC_eREDG_raw'], 0.0)
        self.assertNotIn('AUC_eREDG_unsigned', row)

    def test_dg_shift_invariance(self):
        rng = np.random.default_rng(7)
        history = rng.normal(size=(10, 7, 3))
        shifted = history + rng.normal(size=(10, 7, 1)) * 50
        np.testing.assert_allclose(compute_dg_steps(history), compute_dg_steps(shifted), atol=1e-12)
        self.assertAlmostEqual(compute_dg_index(history), compute_dg_index(shifted))

    def test_common_mode_motion_has_zero_dg(self):
        history = np.array([[[1., 2., 3.]], [[6., 7., 8.]]])
        self.assertEqual(compute_dg_index(history), 0.0)

    def test_centering_does_not_mutate_history(self):
        history = np.arange(12.).reshape(2, 2, 3)
        original = history.copy()
        center_logits(history)
        compute_dg_steps(history)
        np.testing.assert_array_equal(history, original)

    def test_early_and_total_use_same_geometry(self):
        history = np.random.default_rng(8).normal(size=(10, 7, 3))
        steps = compute_dg_steps(history)
        total, early = steps.sum(), steps[:3].sum()
        self.assertLessEqual(early, total)
        self.assertAlmostEqual(early, compute_dg_index(history[:4]))
        self.assertAlmostEqual(total, compute_dg_index(history))
        for module in (melanoma, diabetes, bacteria):
            frame = pd.DataFrame({'DG': [total, 0.], 'DG_early': [early, 0.],
                                  'damage_mode': ['mixed'] * 2, 'regime': ['H1'] * 2})
            result, _ = module.add_eREDG(frame, hard_delta=0.)
            self.assertAlmostEqual(result['eREDG'].iloc[0], early / (total + 1e-12))
            self.assertTrue(np.isnan(result['eREDG'].iloc[1]))

    def test_recorded_band_dg_matches_trajectory(self):
        logits = np.random.default_rng(9).normal(size=(7, 3))
        for regime in ('plain', 'H1', 'H2'):
            result = consensus_with_stress(logits, frozen_flags=np.zeros(7, bool),
                       regime=regime, return_trajectories=True, record_stress=True,
                       rng=np.random.default_rng(3))
            np.testing.assert_allclose(result['step_dg_hist'], compute_dg_steps(result['l_hist']))
        result = melanoma.morphogenetic_consensus_one_audit(logits, rng=np.random.default_rng(3))
        np.testing.assert_allclose(result['band_dg'], compute_dg_steps(result['l_hist']).sum(axis=0))


if __name__ == '__main__':
    unittest.main()
