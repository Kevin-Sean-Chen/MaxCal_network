"""Tests for core MaxCal Network functions."""

import unittest

import numpy as np

from maxcal_network import (
    compute_tauC,
    corr_param,
    cos_ang,
    param2M,
    sim_Q,
    spk2statetime,
    word_id,
)


class CoreFunctionTests(unittest.TestCase):
    """Test small deterministic examples."""

    def test_word_id_returns_expected_state(self):
        self.assertEqual(word_id((0, 0, 0)), 0)
        self.assertEqual(word_id((1, 1, 1)), 7)

    def test_param2M_returns_valid_shapes_and_row_sums(self):
        matrix, stationary = param2M(np.ones(24))

        self.assertEqual(matrix.shape, (8, 8))
        self.assertEqual(stationary.shape, (8,))
        np.testing.assert_allclose(matrix.sum(axis=1), 0)

    def test_compute_tauC_counts_time_and_transition(self):
        states = np.array([0, 1, 0])
        times = np.array([0, 2, 5])

        occupancy, transitions = compute_tauC(states, times)

        self.assertEqual(occupancy[0], 2)
        self.assertEqual(occupancy[1], 3)
        self.assertEqual(transitions[0, 1], 1)
        self.assertEqual(transitions[1, 0], 1)

    def test_spk2statetime_accepts_two_arguments(self):
        firing = [
            [0, np.array([], dtype=int)],
            [1, np.array([0])],
            [2, np.array([], dtype=int)],
        ]

        states, times = spk2statetime(firing, 1)

        self.assertEqual(states.shape, times.shape)

    def test_metrics_match_identical_vectors(self):
        first = np.array([1.0, -1.0])
        second = np.array([1.0, -1.0])

        self.assertAlmostEqual(cos_ang(first, second), 1.0)
        self.assertAlmostEqual(corr_param(first, second), 1.0)

    def test_sim_Q_returns_aligned_outputs(self):
        generator = np.array([[-1.0, 1.0], [1.0, -1.0]])

        states, times = sim_Q(generator, total_time=1, time_step=0.1)

        self.assertEqual(len(states), len(times))
        self.assertEqual(times[0], 0)
        self.assertTrue(np.all(np.diff(times) >= 0))


if __name__ == "__main__":
    unittest.main()
