import unittest

import numpy as np

from morphoraman.core import (
    apply_random_band_damage,
    compute_dg_index,
    morphogenetic_consensus_one,
    softmax,
)


class CoreTests(unittest.TestCase):
    def test_softmax(self):
        p = softmax(np.array([[1000.0, 1001.0], [-1000.0, -1001.0]]))
        np.testing.assert_allclose(p.sum(axis=1), 1.0)

    def test_stationary_trajectory(self):
        self.assertEqual(compute_dg_index(np.ones((4, 7, 3))), 0.0)

    def test_damage_count(self):
        _, _, damaged = apply_random_band_damage(
            np.ones((7, 3)), damage_fraction=0.4,
            mode="mixed", rng=np.random.default_rng(1),
        )
        self.assertEqual(len(damaged), 2)

    def test_reproducibility(self):
        logits = np.random.default_rng(1).normal(size=(7, 3))
        for regime in ("plain", "H1", "H2"):
            results = [
                morphogenetic_consensus_one(
                    logits, frozen_flags=np.zeros(7, dtype=bool),
                    regime=regime, return_trajectories=True,
                    rng=np.random.default_rng(12),
                )
                for _ in range(2)
            ]
            np.testing.assert_array_equal(results[0]["l_hist"], results[1]["l_hist"])


if __name__ == "__main__":
    unittest.main()
