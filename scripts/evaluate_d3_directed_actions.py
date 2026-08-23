"""Paired one-step D3 atom flips on correctly grounded policy states."""

import argparse
import json
import pathlib
import sys

import numpy as np
import torch


REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import ablate_sym


ACTION_NAMES = ("left", "right", "forward", "pickup", "toggle")
DIRECTED = {
    "t_ahead": (2, 1),
    "t_left": (0, 1),
    "t_right": (1, 1),
    "wall_ahead": (2, -1),
}


def bootstrap_mean(values, draws, seed):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return [None, None]
    rng = np.random.default_rng(seed)
    means = np.array([values[rng.integers(0, len(values), len(values))].mean() for _ in range(draws)])
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


@torch.no_grad()
def evaluate(checkpoint, device, episodes, seed_start, bootstrap):
    overlays = ["defaults", "minigrid", "minigrid_symbolic_nav"]
    agent, config = ablate_sym.load(checkpoint, device, overlays)
    wm, behavior = agent._wm, agent._task_behavior
    assert wm._sym_policy_input == "atoms"
    labels = list(wm._sym_head.labels)
    rows = {label: [] for label in labels}
    successes = 0
    for seed in range(seed_start, seed_start + episodes):
        env = ablate_sym.make_env("doorkey6x6", seed, config.time_limit)
        obs = env.reset()
        latent = action = None
        for step in range(config.time_limit + 1):
            obs_b = {k: np.array(v)[None] for k, v in obs.items() if not k.startswith("log_")}
            data = wm.preprocess(obs_b)
            embed = wm.encoder(data)
            latent, _ = wm.dynamics.obs_step(latent, action, embed, data["is_first"], sample=False)
            feat = wm.dynamics.get_feat(latent)
            atoms = wm._sym_head.atoms(feat).detach()
            base_dist = behavior.actor(atoms)
            base_probs = base_dist.probs[0].detach().cpu().numpy()
            base_mode = int(np.argmax(base_probs))
            truth = env.god_state()
            for index, label in enumerate(labels):
                predicted = bool(atoms[0, index].item() > 0)
                if predicted != bool(truth[label]):
                    continue
                edited = atoms.clone()
                edited[0, index] = -edited[0, index]
                changed_probs = behavior.actor(edited).probs[0].detach().cpu().numpy()
                item = {
                    "truth": int(predicted),
                    "mode_changed": int(base_mode != int(np.argmax(changed_probs))),
                    "tv": float(.5 * np.abs(changed_probs - base_probs).sum()),
                }
                if label in DIRECTED:
                    action_index, polarity = DIRECTED[label]
                    new_true = not predicted
                    expected_sign = polarity * (1 if new_true else -1)
                    item["signed_effect"] = float(expected_sign * (changed_probs[action_index] - base_probs[action_index]))
                rows[label].append(item)
            action = base_dist.mode()
            action_index = int(torch.argmax(action, dim=-1)[0].item())
            obs, reward, done, _ = env.step(action_index)
            if done:
                successes += int(reward > 0)
                break
        env.close()

    result = {}
    for index, (label, items) in enumerate(rows.items()):
        summary = {
            "n_correctly_grounded_states": len(items),
            "mode_switch_rate": float(np.mean([x["mode_changed"] for x in items])) if items else None,
            "mean_total_variation": float(np.mean([x["tv"] for x in items])) if items else None,
            "truth_counts": {str(value): sum(x["truth"] == value for x in items) for value in (0, 1)},
        }
        effects = [x["signed_effect"] for x in items if "signed_effect" in x]
        if effects:
            summary.update(
                mean_semantic_signed_effect=float(np.mean(effects)),
                semantic_signed_effect_ci95=bootstrap_mean(effects, bootstrap, 20260815 + index),
                semantic_alignment_rate=float(np.mean(np.asarray(effects) > 0)),
            )
        result[label] = summary
    return {
        "checkpoint": str(pathlib.Path(checkpoint).resolve()),
        "episodes": episodes,
        "evaluation_seeds": [seed_start, seed_start + episodes - 1],
        "normal_success_rate": successes / episodes,
        "selection": "states where predicted atom sign equals simulator truth",
        "directed_metric": "positive means flipped atom changed its named action probability in the edited semantic direction",
        "atoms": result,
    }


def self_test():
    assert bootstrap_mean([1, 1, 1], 100, 0) == [1.0, 1.0]
    assert DIRECTED["wall_ahead"] == (2, -1)
    print("PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(REPO / "models" / "doorkey" / "doorkey_experiment_3_d3" / "checkpoint.pt"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--out")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.out:
        parser.error("--out is required unless --self-test")
    result = evaluate(args.checkpoint, args.device, args.episodes, args.seed_start, args.bootstrap)
    pathlib.Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
