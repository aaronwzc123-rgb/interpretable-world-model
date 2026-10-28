"""Regressions for seed wrapping and evaluation partition leakage."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from envs.minigrid import MiniGrid, TRAIN_SEED_START, TRAIN_POOL
from scripts.evaluate_m1_cuestart import evaluation_seed_plan, grouped_split, fixed_upper_action


class EvaluationSeedTests(unittest.TestCase):
    def reset_provenance(self, seed, mode="pool", train=False):
        env = MiniGrid("memoryS7_cuestart", mode="train" if train else "eval",
                       seed=seed, eval_seed_mode=mode)
        try:
            observation = env.reset()
            self.assertNotIn("actual_seed", observation)
            self.assertNotIn("initial_state_signature", observation)
            return env.episode_provenance
        finally:
            env.close()

    def test_historical_pool_wraps_and_records_both_seed_values(self):
        first = self.reset_provenance(0)
        wrapped = self.reset_provenance(200)
        self.assertEqual(wrapped["requested_seed"], 200)
        self.assertEqual(wrapped["actual_seed"], 0)
        self.assertEqual(first["initial_state_signature"], wrapped["initial_state_signature"])

    def test_direct_mode_does_not_wrap_and_increments_on_reset(self):
        env = MiniGrid("memoryS7_cuestart", mode="eval", seed=1199,
                       eval_seed_mode="direct")
        try:
            env.reset()
            self.assertEqual(env.episode_provenance["actual_seed"], 1199)
            env.reset()
            self.assertEqual(env.episode_provenance["actual_seed"], 1200)
        finally:
            env.close()

    def test_eval_option_does_not_change_training_seed_stream(self):
        pool = self.reset_provenance(123, "pool", train=True)
        direct = self.reset_provenance(123, "direct", train=True)
        expected = TRAIN_SEED_START + np.random.RandomState(123).randint(TRAIN_POOL)
        self.assertEqual(pool["actual_seed"], expected)
        self.assertEqual(pool, direct)

    def test_repeated_eval_seeds_fail_before_rollout(self):
        with self.assertRaisesRegex(ValueError, "Repeated actual"):
            evaluation_seed_plan(500, 0, "pool")
        _, actual = evaluation_seed_plan(200, 1000, "direct")
        self.assertEqual(actual, list(range(1000, 1200)))

    def test_evaluator_rejects_training_seed_overlap(self):
        with self.assertRaisesRegex(ValueError, "training seed pool"):
            evaluation_seed_plan(200, 9990, "direct")

    def test_grouped_split_keeps_repeated_episode_seeds_together(self):
        rows = [{"episode": i, "actual_seed": i % 200,
                 "initial_state_signature": f"configuration-{i % 4}"}
                for i in range(500)]
        train, test, meta = grouped_split(rows, 20260813)
        train_seeds = {r["actual_seed"] for r, keep in zip(rows, train) if keep}
        test_seeds = {r["actual_seed"] for r, keep in zip(rows, test) if keep}
        self.assertEqual(train_seeds & test_seeds, set())
        self.assertEqual(meta["actual_seed_overlap"], 0)
        self.assertEqual(meta["initial_state_overlap"], 4)
        self.assertTrue(np.all(train ^ test))

    def test_fixed_upper_controller_never_reads_cue_or_object_identity(self):
        self.assertEqual(fixed_upper_action({"wall_ahead": False}), 2)
        self.assertEqual(fixed_upper_action({"wall_ahead": True}), 0)
        for wall in (False, True):
            left_cue = {"wall_ahead": wall, "cue_is_key": False, "key_left": True}
            right_cue = {"wall_ahead": wall, "cue_is_key": True, "key_left": False}
            self.assertEqual(fixed_upper_action(left_cue), fixed_upper_action(right_cue))

    def test_fixed_upper_coverage_is_six_step_path_on_all_four_initial_states(self):
        paths = {}
        for seed in range(32):
            env = MiniGrid("memoryS7_cuestart", mode="eval", seed=seed,
                           eval_seed_mode="direct", max_steps=20)
            try:
                env.reset()
                actions = []
                for _ in range(20):
                    action = fixed_upper_action(env.god_state())
                    actions.append(action)
                    _, _, done, _ = env.step(action)
                    if done:
                        break
                self.assertTrue(done)
                self.assertEqual(actions, [2, 2, 2, 2, 0, 2])
                paths[env.episode_provenance["initial_state_signature"]] = actions
            finally:
                env.close()
        self.assertEqual(len(paths), 4)


if __name__ == "__main__":
    unittest.main()
