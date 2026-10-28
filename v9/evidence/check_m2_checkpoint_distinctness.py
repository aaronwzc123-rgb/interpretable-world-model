"""Verify matching aggregate probe scores do not come from identical checkpoints."""
import hashlib
import json
import pathlib
import sys
import tempfile

import numpy as np
import torch

REPO = pathlib.Path("C:/Users/super/Desktop/test/Xdreamer")
sys.path.insert(0, str(REPO))
from scripts import evaluate_m2_head as evaluator
import dreamer
import tools
from envs.minigrid import MiniGrid
from envs.wrappers import OneHotAction


torch.set_num_threads(1)
states, features, checkpoint_hashes = [], [], []
for seed in (0, 1):
    path = REPO / f"models/memory/memory_experiment_2_m2/seed{seed}/checkpoint_084000.pt"
    checkpoint_hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    state = torch.load(path, map_location="cpu", weights_only=False)["agent_state_dict"]
    states.append(state)
    config = evaluator.P.build_config(evaluator.OVERLAYS)
    config.device, config.compile, config.num_actions = "cpu", False, 5
    env = MiniGrid("memoryS7_cuestart", mode="eval", seed=1000, eval_seed_mode="direct",
                   max_steps=config.time_limit, emit_labels=True)
    logger = tools.Logger(pathlib.Path(tempfile.mkdtemp()), 0)
    agent = dreamer.Dreamer(env.observation_space, OneHotAction(env).action_space,
                            config, logger, dataset=None).to("cpu")
    agent.load_state_dict(state)
    agent.eval().requires_grad_(False)
    collected = []
    with torch.no_grad():
        for episode in range(8):
            obs = env.reset()
            latent = action = None
            for _ in range(config.time_limit):
                data = agent._wm.preprocess({k: np.asarray(v)[None] for k,v in obs.items()
                                             if not k.startswith("log_")})
                embed = agent._wm.encoder(data)
                latent, _ = agent._wm.dynamics.obs_step(latent, action, embed,
                                                        data["is_first"], sample=False)
                collected.append(agent._wm.dynamics.get_feat(latent)[0].numpy())
                index = evaluator.scripted(env.god_state())
                action = torch.nn.functional.one_hot(torch.tensor([index]), 5).float()
                obs, reward, done, _ = env.step(index)
                if done:
                    break
    features.append(np.stack(collected))
    env.close()
    logger._writer.close()
    del agent

comparison = {}
for prefix in ("_wm.encoder", "_wm.dynamics", "_wm._sym_head", "_task_behavior.actor"):
    keys = [k for k in states[0] if k.startswith(prefix)]
    unequal = [k for k in keys if not torch.equal(states[0][k], states[1][k])]
    comparison[prefix] = {"tensor_count": len(keys), "different_tensor_count": len(unequal)}
difference = features[0] - features[1]
result = {"checkpoint_sha256": checkpoint_hashes, "parameter_comparison": comparison,
          "scripted_requested_and_actual_seeds": [1000, 1007],
          "feature_shape_per_checkpoint": list(features[0].shape),
          "feature_sha256": [hashlib.sha256(x.tobytes()).hexdigest() for x in features],
          "feature_max_absolute_difference": float(np.max(np.abs(difference))),
          "feature_root_mean_square_difference": float(np.sqrt(np.mean(difference ** 2))),
          "interpretation": "Different weights and posterior features can give identical thresholded aggregate probe metrics; identical scores do not establish identical representations."}
output = pathlib.Path(__file__).with_suffix(".json")
output.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
