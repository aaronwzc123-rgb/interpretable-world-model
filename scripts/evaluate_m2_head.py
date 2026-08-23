"""Evaluate an online MemoryS7 symbolic head on held-out scripted trajectories.

The privileged controller is used only to obtain balanced state coverage. It never
enters the model observation; the reported atom is produced from the local-grid
posterior. Results are separated into visible-cue and pure-memory time steps.
"""
import argparse
import json
import pathlib
import sys
import tempfile

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import dreamer
import tools
import envs.minigrid as M
import envs.wrappers as wrappers
import probe_memory as P

OVERLAYS = [
    "defaults", "minigrid", "minigrid_memory", "minigrid_memory_cuestart",
    "minigrid_memory_symbolic_nav_v2", "minigrid_memory_symbolic_nav_v2_entropy",
    "minigrid_memory_symbolic_nav_v2_bce",
]
LEFT, RIGHT, FWD = 0, 1, 2


def scripted(g):
    reach, left, right = (
        (g["key_reach"], g["key_left"], g["key_right"])
        if g["cue_is_key"]
        else (g["ball_reach"], g["ball_left"], g["ball_right"])
    )
    if reach or not g["wall_ahead"]:
        return FWD
    if left:
        return LEFT
    if right:
        return RIGHT
    return LEFT


def metrics(y, pred):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    if not len(y):
        return {"n": 0}
    return {
        "n": int(len(y)),
        "accuracy": float(np.mean(y == pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "base_rate": float(max(y.mean(), 1.0 - y.mean())),
        "true_positive_rate": float(y.mean()),
        "predicted_positive_rate": float(pred.mean()),
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config = P.build_config(OVERLAYS)
    config.device = args.device
    config.compile = False
    config.num_actions = 5
    base = M.MiniGrid(
        "memoryS7_cuestart", mode="eval", seed=0,
        max_steps=config.time_limit, emit_labels=True,
    )
    act_space = wrappers.OneHotAction(base).action_space
    logger = tools.Logger(pathlib.Path(tempfile.mkdtemp()), 0)
    agent = dreamer.Dreamer(
        base.observation_space, act_space, config, logger, dataset=None
    ).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    agent.load_state_dict(checkpoint["agent_state_dict"])
    agent.requires_grad_(False)
    agent.eval()
    base.close()

    head = agent._wm._sym_head
    cue_index = head.labels.index("cue_is_key")
    phases = {"all_known": ([], []), "visible": ([], []), "pure_memory": ([], [])}
    successes = []
    for episode in range(args.episodes):
        env = M.MiniGrid(
            "memoryS7_cuestart", mode="eval", seed=episode,
            max_steps=config.time_limit, emit_labels=True,
        )
        obs = env.reset()
        latent = action = None
        success = False
        for _ in range(config.time_limit + 1):
            obs_batch = {
                key: np.asarray(value)[None]
                for key, value in obs.items() if not key.startswith("log_")
            }
            data = agent._wm.preprocess(obs_batch)
            embed = agent._wm.encoder(data)
            latent, _ = agent._wm.dynamics.obs_step(
                latent, action, embed, data["is_first"], sample=False
            )
            feat = agent._wm.dynamics.get_feat(latent)
            pred = int(head.atoms(feat)[0, cue_index].item() > 0)
            g = env.god_state()
            truth = int(g["cue_is_key"])
            if g["cue_known"]:
                phases["all_known"][0].append(truth)
                phases["all_known"][1].append(pred)
            if g["cue_visible"]:
                phases["visible"][0].append(truth)
                phases["visible"][1].append(pred)
            if g["cue_known"] and not g["cue_visible"]:
                phases["pure_memory"][0].append(truth)
                phases["pure_memory"][1].append(pred)
            action_index = scripted(g)
            action = torch.nn.functional.one_hot(
                torch.tensor([action_index], device=args.device), 5
            ).float()
            obs, reward, done, _ = env.step(action_index)
            if done:
                success = bool(reward > 0)
                break
        successes.append(success)
        env.close()

    result = {
        "checkpoint": str(pathlib.Path(args.checkpoint).resolve()),
        "episodes": args.episodes,
        "evaluation_seeds": [0, args.episodes - 1],
        "coverage_controller": "privileged scripted controller; actions only",
        "controller_success_rate": float(np.mean(successes)),
        "cue": {name: metrics(*values) for name, values in phases.items()},
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
