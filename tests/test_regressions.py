import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from ruamel.yaml import YAML


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import neuraldnf
import tools


class RegressionTests(unittest.TestCase):
    def test_minigrid_presets_use_steps_key(self):
        config = YAML(typ="safe").load(ROOT.joinpath("configs.yaml").read_text())
        for preset in ("crafter", "minecraft"):
            self.assertIn("steps", config[preset])
            self.assertNotIn("step", config[preset])

    def test_training_chunk_never_exceeds_budget(self):
        self.assertEqual(tools.training_chunk(0, 10, 4), 4)
        self.assertEqual(tools.training_chunk(8, 10, 4), 2)
        self.assertEqual(tools.training_chunk(10, 10, 4), 0)
        self.assertEqual(tools.training_chunk(12, 10, 4), 0)

    def test_rule_extraction_preserves_negative_literals(self):
        model = neuraldnf.NeuralDNF(2, 1, 1)
        with torch.no_grad():
            model.conj.w.copy_(torch.tensor([[1.0, -1.0]]))
            model.disj.w.copy_(torch.tensor([[1.0]]))
        self.assertEqual(
            model.extract_rules(["a", "b"], ["out"], w_thr=0.5),
            ["out :- a, not b."],
        )

    def test_rule_extraction_warns_for_negative_disjunctions(self):
        model = neuraldnf.NeuralDNF(1, 1, 1)
        with torch.no_grad():
            model.conj.w.fill_(1.0)
            model.disj.w.fill_(-1.0)
        with self.assertWarns(RuntimeWarning):
            rules = model.extract_rules(["a"], ["out"], w_thr=0.5)
        self.assertEqual(rules, [])

    def test_atomic_checkpoint_keeps_torch_format(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            tools.atomic_torch_save({"value": torch.tensor([1, 2])}, path)
            loaded = torch.load(path, map_location="cpu", weights_only=False)
            self.assertTrue(torch.equal(loaded["value"], torch.tensor([1, 2])))
            self.assertFalse(path.with_name(path.name + ".tmp").exists())

    def test_memory_cuestart_is_explicit(self):
        from envs.minigrid import MiniGrid

        env = MiniGrid("memoryS7_cuestart", mode="eval", seed=0, max_steps=20)
        try:
            env.reset()
            self.assertEqual(int(env._env.unwrapped.agent_pos[0]), 1)
            self.assertEqual(int(env._env.unwrapped.agent_dir), 0)
        finally:
            env.close()

    def test_official_and_cuestart_memory_are_distinct(self):
        from envs.minigrid import MiniGrid

        official = MiniGrid("memoryS7", mode="eval", seed=0, max_steps=20)
        cuestart = MiniGrid("memoryS7_cuestart", mode="eval", seed=0, max_steps=20)
        try:
            official.reset()
            cuestart.reset()
            self.assertNotEqual(
                official._env.unwrapped.agent_pos.tolist(),
                cuestart._env.unwrapped.agent_pos.tolist(),
            )
            self.assertFalse(official.god_state()["cue_visible"])
            self.assertTrue(cuestart.god_state()["cue_visible"])
        finally:
            official.close()
            cuestart.close()

    def test_official_memory_task_is_not_globally_patched(self):
        import minigrid.envs.memory as memory

        self.assertEqual(memory.MemoryEnv._gen_grid.__module__, "minigrid.envs.memory")

    def test_dreamer_symbolic_policy_forward(self):
        import dreamer

        raw = YAML(typ="safe").load(ROOT.joinpath("configs.yaml").read_text())
        merged = {}
        for preset in ("defaults", "minigrid", "minigrid_symbolic", "debug"):
            self._recursive_update(merged, raw[preset])
        merged.update(device="cpu", compile=False, envs=1)
        config = SimpleNamespace(**merged)
        env = dreamer.make_env(config, "eval", 0)
        config.num_actions = (
            env.action_space.n
            if hasattr(env.action_space, "n")
            else env.action_space.shape[0]
        )
        with tempfile.TemporaryDirectory() as directory:
            logger = tools.Logger(Path(directory), 0)
            agent = dreamer.Dreamer(
                env.observation_space, env.action_space, config, logger, iter(())
            ).to("cpu")
            try:
                obs = {key: np.expand_dims(value, 0) for key, value in env.reset().items()}
                output, _ = agent(obs, np.array([True]), training=False)
                self.assertEqual(tuple(output["action"].shape), (1, config.num_actions))
                self.assertIsNotNone(agent._wm._sym_head)
            finally:
                env.close()
                logger._writer.close()

    def test_symbolic_head_bce_is_finite_and_backpropagates(self):
        head = neuraldnf.SymbolicHead(
            8, ["cue_is_key"], n_lit=4, n_conj=2, loss_type="bce"
        )
        feat = torch.randn(2, 3, 8, requires_grad=True)
        data = {"label_cue_is_key": torch.randint(0, 2, (2, 3, 1)).float()}
        loss, _ = head.masked_loss(feat, data)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(feat.grad)

    def test_symbolic_feat_policy_mode_keeps_continuous_dimension(self):
        raw = YAML(typ="safe").load(ROOT.joinpath("configs.yaml").read_text())
        self.assertEqual(raw["minigrid_memory_symbolic_nav_v2_bce"]["sym_loss"], "bce")
        self.assertEqual(raw["minigrid_memory_symbolic_nav_v2_bce"]["sym_policy_input"], "feat")

    def test_doorkey_relational_labels_match_config(self):
        from envs.minigrid import MiniGrid

        raw = YAML(typ="safe").load(ROOT.joinpath("configs.yaml").read_text())
        labels = raw["minigrid_symbolic_nav_rel"]["sym_labels"]
        self.assertEqual(labels, list(MiniGrid.LABELS_DOORKEY_REL))
        self.assertFalse(any(label.startswith("t_") for label in labels))

        env = MiniGrid("doorkey6x6_rel", mode="eval", seed=0,
                       max_steps=300, emit_labels=True)
        try:
            obs = env.reset()
            state = env.god_state()
            self.assertEqual(
                {key for key in obs if key.startswith("label_")},
                {f"label_{label}" for label in labels},
            )
            self.assertTrue(all(isinstance(state[label], (bool, np.bool_)) for label in labels))
        finally:
            env.close()

    @staticmethod
    def _recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                RegressionTests._recursive_update(base[key], value)
            else:
                base[key] = value


if __name__ == "__main__":
    unittest.main()
