"""v9 controlled M1 check: fixed upper-branch coverage independent of hidden cue.

This standalone script leaves the code repository unchanged. The current main
M1 evaluator collects autonomous actor trajectories; the retained historical
JSON explicitly identifies a fixed-upper-branch controller instead. This makes
the coverage distinction explicit and records all actual seeds and start states.
"""
import argparse
import hashlib
import json
import pathlib
import sys
from collections import Counter

import numpy as np
import torch

REPO = pathlib.Path("C:/Users/super/Desktop/test/Xdreamer")
sys.path.insert(0, str(REPO))
from scripts.evaluate_m1_cuestart import load_agent, evaluate, evaluation_seed_plan
from envs.minigrid import MiniGrid


@torch.no_grad()
def collect_fixed_upper(agent, config, episodes, seed_start):
    requested, _ = evaluation_seed_plan(episodes, seed_start, "direct")
    rows, records = [], []
    for episode, seed in enumerate(requested):
        env = MiniGrid("memoryS7_cuestart", mode="eval", seed=seed,
                       max_steps=config.time_limit, emit_labels=True,
                       eval_seed_mode="direct")
        obs = env.reset()
        latent = action = None
        provenance = env.episode_provenance
        for step in range(config.time_limit):
            data = agent._wm.preprocess({k: np.asarray(v)[None] for k, v in obs.items()
                                         if not k.startswith("log_")})
            embed = agent._wm.encoder(data)
            latent, _ = agent._wm.dynamics.obs_step(latent, action, embed,
                                                    data["is_first"], sample=False)
            feat = agent._wm.dynamics.get_feat(latent)
            g = env.god_state()
            rows.append({"episode": episode, **provenance, "cue": int(g["cue_is_key"]),
                         "visible": bool(g["cue_visible"]), "known": bool(g["cue_known"]),
                         "decision": bool(g["key_reach"] or g["ball_reach"]),
                         "grid": np.asarray(obs["grid"], dtype=np.float32),
                         "stoch": latent["stoch"][0].reshape(-1).cpu().numpy(),
                         "deter": latent["deter"][0].cpu().numpy(),
                         "feat": feat[0].cpu().numpy()})
            # Only wall geometry determines this fixed path. Cue and candidate
            # identities never determine actions; they are labels for scoring.
            action_index = 0 if g["wall_ahead"] else 2
            action = torch.nn.functional.one_hot(
                torch.tensor([action_index], device=config.device), 5).float()
            obs, reward, done, _ = env.step(action_index)
            if done:
                break
        records.append({"episode": episode, **provenance, "cue": int(g["cue_is_key"]),
                        "steps": step + 1, "controller_success": bool(reward > 0)})
        env.close()
    return rows, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--split-seed", type=int, default=20260813)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.split_seed)
    agent, config = load_agent(args.checkpoint, "grounded", args.device)
    rows, records = collect_fixed_upper(agent, config, args.episodes, args.seed_start)
    probes, split = evaluate(rows, args.split_seed, args.device)
    result = {"checkpoint": str(pathlib.Path(args.checkpoint).resolve()),
              "checkpoint_sha256": hashlib.sha256(pathlib.Path(args.checkpoint).read_bytes()).hexdigest(),
              "command": sys.argv, "condition": "grounded", "device": args.device,
              "torch_threads": 1, "time_limit": config.time_limit,
              "episodes": args.episodes, "evaluation_seeds": [args.seed_start, args.seed_start + args.episodes - 1],
              "eval_seed_mode": "direct",
              "coverage_controller": "fixed upper branch; left at wall, forward otherwise; cue/object independent",
              "coverage_controller_success_rate": float(np.mean([r["controller_success"] for r in records])),
              "episode_provenance": records,
              "unique_actual_seeds": len({r["actual_seed"] for r in records}),
              "unique_initial_states": len({r["initial_state_signature"] for r in records}),
              "step_counts": dict(Counter(r["steps"] for r in records)),
              "probes": probes, "split": split}
    output = pathlib.Path(args.out)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "episode_provenance"}, indent=2))


if __name__ == "__main__":
    main()
