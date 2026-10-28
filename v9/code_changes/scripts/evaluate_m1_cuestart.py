"""Evaluate CueStart continuous-belief memory with episode-disjoint probes."""

import argparse
import hashlib
import json
import os
import pathlib
import tempfile
import sys

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score

REPO = pathlib.Path(os.environ.get("XDREAMER_REPO_ROOT", pathlib.Path(__file__).resolve().parent.parent))
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


def evaluation_seed_plan(episodes, seed_start=0, eval_seed_mode="pool"):
    """Reject accidental repeated seeds before collecting model trajectories."""
    if episodes < 2 or seed_start < 0:
        raise ValueError("Evaluation needs at least two episodes and nonnegative seeds")
    if eval_seed_mode not in ("pool", "direct"):
        raise ValueError("eval_seed_mode must be 'pool' or 'direct'")
    requested = list(range(seed_start, seed_start + episodes))
    actual = [s % M.TEST_POOL if eval_seed_mode == "pool" else s for s in requested]
    if len(set(actual)) != len(actual):
        raise ValueError("Repeated actual evaluation seeds: pool mode wraps after "
                         f"{M.TEST_POOL} episodes. Use <= {M.TEST_POOL} episodes or "
                         "explicit --eval-seed-mode direct with held-out seeds.")
    if any(M.TRAIN_SEED_START <= s < M.TRAIN_SEED_START + M.TRAIN_POOL for s in actual):
        raise ValueError("Requested evaluation seeds overlap the training seed pool")
    return requested, actual


def fixed_upper_action(god_state):
    """A fixed path chosen only from wall geometry, never cue/candidate identity."""
    return 0 if god_state["wall_ahead"] else 2


def grouped_split(rows, split_seed):
    """Keep every timestep/episode with the same actual seed in one partition."""
    groups = np.array([r["actual_seed"] for r in rows])
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("At least two distinct actual seeds are required for a probe split")
    rng = np.random.RandomState(split_seed)
    rng.shuffle(unique)
    cut = max(1, min(len(unique) - 1, int(0.7 * len(unique))))
    train_seeds, test_seeds = set(unique[:cut]), set(unique[cut:])
    train = np.array([g in train_seeds for g in groups])
    train_signatures = {r["initial_state_signature"] for r, keep in zip(rows, train) if keep}
    test_signatures = {r["initial_state_signature"] for r, keep in zip(rows, train) if not keep}
    meta = {"grouping_unit": "actual_environment_seed", "split_seed": int(split_seed),
            "train_actual_seeds": sorted(int(s) for s in train_seeds),
            "test_actual_seeds": sorted(int(s) for s in test_seeds),
            "actual_seed_overlap": len(train_seeds & test_seeds),
            "train_unique_initial_states": len(train_signatures),
            "test_unique_initial_states": len(test_signatures),
            "initial_state_overlap": len(train_signatures & test_signatures),
            "train_episodes": len({r["episode"] for r, keep in zip(rows, train) if keep}),
            "test_episodes": len({r["episode"] for r, keep in zip(rows, train) if not keep})}
    return train, ~train, meta


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
def collect(agent, config, condition, episodes, seed_start=0, eval_seed_mode="pool",
            coverage_controller="actor"):
    if coverage_controller not in ("actor", "fixed_upper"):
        raise ValueError("coverage_controller must be 'actor' or 'fixed_upper'")
    rows, successes = [], []
    emit = condition == "grounded"
    requested, _ = evaluation_seed_plan(episodes, seed_start, eval_seed_mode)
    for episode, requested_seed in enumerate(requested):
        env = M.MiniGrid("memoryS7_cuestart", mode="eval", seed=requested_seed,
                         max_steps=config.time_limit, emit_labels=emit,
                         eval_seed_mode=eval_seed_mode)
        obs, latent, action = env.reset(), None, None
        provenance = env.episode_provenance
        success = False
        for _ in range(config.time_limit + 1):
            batch = {k: np.asarray(v)[None] for k, v in obs.items()
                     if not k.startswith("log_")}
            model_data = agent._wm.preprocess(batch)
            embed = agent._wm.encoder(model_data)
            latent, _ = agent._wm.dynamics.obs_step(
                latent, action, embed, model_data["is_first"], sample=False)
            feat = agent._wm.dynamics.get_feat(latent)
            g = env.god_state()
            if coverage_controller == "actor":
                action = agent._task_behavior.actor(agent._wm.augment(feat)).mode()
                action_index = int(action.argmax(-1)[0].item())
            else:
                action_index = fixed_upper_action(g)
                action = torch.nn.functional.one_hot(
                    torch.tensor([action_index], device=config.device), 5).float()
            rows.append({"episode": episode, **provenance, "cue": int(g["cue_is_key"]),
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
        if (episode + 1) % 25 == 0:
            print(f"Collected {episode + 1}/{episodes} episodes", flush=True)
    return rows, float(np.mean(successes))


def evaluate(rows, split_seed, device):
    train, test, split = grouped_split(rows, split_seed)
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
    for feature_index, feature in enumerate(FEATURES):
        torch.manual_seed(split_seed + feature_index)
        x = np.stack([r[feature] for r in rows])
        train_mask = train & known
        prediction = (fit_probe(x[train_mask], y[train_mask], x[test], device) > 0.5).astype(int)
        result[feature] = {}
        for phase, mask in phases.items():
            indices = np.flatnonzero(mask)
            positions = np.array([local[i] for i in indices], dtype=int)
            result[feature][phase] = score(y[indices].astype(int), prediction[positions])
    split["train_known_steps"] = int((train & known).sum())
    return result, split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--condition", choices=OVERLAYS, required=True)
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--split-seed", type=int, default=20260813)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--eval-seed-mode", choices=("pool", "direct"), default="pool")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--coverage-controller", choices=("actor", "fixed_upper"), default="actor",
                        help="Actor trajectories or cue-independent fixed upper-branch coverage")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    evaluation_seed_plan(args.episodes, args.seed_start, args.eval_seed_mode)
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.split_seed)
    agent, config = load_agent(args.checkpoint, args.condition, args.device)
    rows, success = collect(agent, config, args.condition, args.episodes,
                            args.seed_start, args.eval_seed_mode, args.coverage_controller)
    probes, split = evaluate(rows, args.split_seed, args.device)
    grid_ba = probes["grid"]["pure_memory"]["balanced_accuracy"]
    belief_ba = max(probes[key]["pure_memory"]["balanced_accuracy"] or 0
                    for key in ("stoch", "deter", "feat"))
    result = {"checkpoint": str(pathlib.Path(args.checkpoint).resolve()),
              "checkpoint_sha256": hashlib.sha256(pathlib.Path(args.checkpoint).read_bytes()).hexdigest(),
              "command": sys.argv, "device": args.device,
              "time_limit": int(config.time_limit), "torch_threads": args.torch_threads,
              "condition": args.condition, "episodes": args.episodes,
              "evaluation_seeds": [args.seed_start, args.seed_start + args.episodes - 1],
              "eval_seed_mode": args.eval_seed_mode,
              "coverage_controller": args.coverage_controller,
              "coverage_controller_success_rate": success,
              "success_rate": success if args.coverage_controller == "actor" else None,
              "episode_provenance": [
                  {k: r[k] for k in ("episode", "requested_seed", "actual_seed",
                                     "initial_state_signature", "cue")}
                  for r in {r["episode"]: r for r in rows}.values()],
              "unique_actual_seeds": len({r["actual_seed"] for r in rows}),
              "unique_initial_states": len({r["initial_state_signature"] for r in rows}),
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
