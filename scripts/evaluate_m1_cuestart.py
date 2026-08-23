"""Evaluate CueStart continuous-belief memory with episode-disjoint probes."""

import argparse
import json
import pathlib
import tempfile
import sys

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import dreamer
import tools
import envs.minigrid as M
import envs.wrappers as wrappers
from scripts.probe_memory import build_config, fit_probe


OVERLAYS = {
    "plain": ["defaults", "minigrid", "minigrid_memory", "minigrid_memory_cuestart"],
    "grounded": ["defaults", "minigrid", "minigrid_memory",
                 "minigrid_memory_cuestart", "minigrid_memory_shape"],
}
FEATURES = ("grid", "stoch", "deter", "feat")


def score(y, pred):
    if not len(y) or len(np.unique(y)) < 2:
        return {"n": int(len(y)), "accuracy": None, "balanced_accuracy": None}
    return {"n": int(len(y)), "accuracy": float(np.mean(y == pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "base_rate": float(max(y.mean(), 1 - y.mean())),
            "predicted_positive_rate": float(pred.mean())}


def load_agent(checkpoint, condition, device):
    config = build_config(OVERLAYS[condition])
    if condition == "plain":
        config.ndnf_enabled = config.shape_enabled = config.sym_enabled = False
    config.device, config.compile, config.num_actions = device, False, 5
    emit = condition == "grounded"
    base = M.MiniGrid("memoryS7_cuestart", mode="eval", seed=0,
                      max_steps=config.time_limit, emit_labels=emit)
    action_space = wrappers.OneHotAction(base).action_space
    logger = tools.Logger(pathlib.Path(tempfile.mkdtemp()), 0)
    agent = dreamer.Dreamer(base.observation_space, action_space, config,
                            logger, dataset=None).to(device)
    data = torch.load(checkpoint, map_location=device, weights_only=False)
    agent.load_state_dict(data["agent_state_dict"])
    agent.requires_grad_(False)
    agent.eval()
    base.close()
    return agent, config


@torch.no_grad()
def collect(agent, config, condition, episodes):
    rows, successes = [], []
    emit = condition == "grounded"
    for episode in range(episodes):
        env = M.MiniGrid("memoryS7_cuestart", mode="eval", seed=episode,
                         max_steps=config.time_limit, emit_labels=emit)
        obs, latent, action = env.reset(), None, None
        success = False
        for _ in range(config.time_limit + 1):
            batch = {k: np.asarray(v)[None] for k, v in obs.items()
                     if not k.startswith("log_")}
            model_data = agent._wm.preprocess(batch)
            embed = agent._wm.encoder(model_data)
            latent, _ = agent._wm.dynamics.obs_step(
                latent, action, embed, model_data["is_first"], sample=False)
            feat = agent._wm.dynamics.get_feat(latent)
            action = agent._task_behavior.actor(agent._wm.augment(feat)).mode()
            action_index = int(action.argmax(-1)[0].item())
            g = env.god_state()
            rows.append({"episode": episode, "cue": int(g["cue_is_key"]),
                         "visible": bool(g["cue_visible"]), "known": bool(g["cue_known"]),
                         "decision": bool(g["key_reach"] or g["ball_reach"]),
                         "grid": np.asarray(obs["grid"], dtype=np.float32),
                         "stoch": latent["stoch"][0].reshape(-1).cpu().numpy(),
                         "deter": latent["deter"][0].cpu().numpy(),
                         "feat": feat[0].cpu().numpy()})
            obs, reward, done, _ = env.step(action_index)
            if done:
                success = bool(reward > 0)
                break
        successes.append(success)
        env.close()
    return rows, float(np.mean(successes))


def evaluate(rows, split_seed, device):
    episodes = np.array([r["episode"] for r in rows])
    unique = np.unique(episodes)
    rng = np.random.RandomState(split_seed)
    rng.shuffle(unique)
    train_episodes = set(unique[:int(0.7 * len(unique))])
    train = np.array([e in train_episodes for e in episodes])
    test = ~train
    y = np.array([r["cue"] for r in rows], dtype=np.float32)
    visible = np.array([r["visible"] for r in rows])
    known = np.array([r["known"] for r in rows])
    decision = np.array([r["decision"] for r in rows])
    phases = {"known": test & known, "visible": test & visible,
              "pure_memory": test & known & ~visible,
              "decision": test & known & ~visible & decision}
    result = {}
    test_indices = np.flatnonzero(test)
    local = {index: position for position, index in enumerate(test_indices)}
    for feature in FEATURES:
        x = np.stack([r[feature] for r in rows])
        train_mask = train & known
        prediction = (fit_probe(x[train_mask], y[train_mask], x[test], device) > 0.5).astype(int)
        result[feature] = {}
        for phase, mask in phases.items():
            indices = np.flatnonzero(mask)
            positions = np.array([local[i] for i in indices], dtype=int)
            result[feature][phase] = score(y[indices].astype(int), prediction[positions])
    return result, {"train_episodes": len(train_episodes),
                    "test_episodes": len(unique) - len(train_episodes),
                    "train_known_steps": int((train & known).sum())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--condition", choices=OVERLAYS, required=True)
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--split-seed", type=int, default=20260813)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    agent, config = load_agent(args.checkpoint, args.condition, args.device)
    rows, success = collect(agent, config, args.condition, args.episodes)
    probes, split = evaluate(rows, args.split_seed, args.device)
    grid_ba = probes["grid"]["pure_memory"]["balanced_accuracy"]
    belief_ba = max(probes[key]["pure_memory"]["balanced_accuracy"] or 0
                    for key in ("stoch", "deter", "feat"))
    result = {"checkpoint": str(pathlib.Path(args.checkpoint).resolve()),
              "condition": args.condition, "episodes": args.episodes,
              "evaluation_seeds": [0, args.episodes - 1], "success_rate": success,
              "split": split, "probes": probes,
              "acceptance": {"threshold_balanced_accuracy": 0.65,
                             "minimum_gain_over_grid": 0.10,
                             "best_belief_pure_memory_balanced_accuracy": belief_ba,
                             "grid_pure_memory_balanced_accuracy": grid_ba,
                             "passed": bool(grid_ba is not None and belief_ba >= 0.65
                                            and belief_ba - grid_ba >= 0.10)}}
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
