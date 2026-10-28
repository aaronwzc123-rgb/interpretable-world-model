import unittest

from scripts.evaluate_d3_directed_actions import bootstrap_episode_mean


class EpisodeBootstrapTests(unittest.TestCase):
    def test_identical_within_episode_states_do_not_increase_precision(self):
        short = bootstrap_episode_mean([0, 1], [0, 1], [0, 1], 2000, 12)
        repeated = bootstrap_episode_mean([0] * 100 + [1] * 100,
                                          [0] * 100 + [1] * 100,
                                          [0, 1], 2000, 12)
        self.assertEqual(short, repeated)
        self.assertEqual(short, [0.0, 1.0])

    def test_constant_effect(self):
        self.assertEqual(bootstrap_episode_mean([0.4] * 4, [0, 0, 1, 1],
                                               [0, 1], 100, 0), [0.4, 0.4])
