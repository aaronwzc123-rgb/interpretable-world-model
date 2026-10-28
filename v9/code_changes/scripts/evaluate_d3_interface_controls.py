"""Re-evaluate a fixed D3 actor under soft, sign-only and oracle inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(1)
    sys.path[:0] = [str(args.repo), str(args.repo / 'scripts')]
    import ablate_sym
    from envs.minigrid import MiniGrid

    checkpoint = args.repo / 'models/doorkey/doorkey_experiment_3_d3/checkpoint.pt'
    agent, config = ablate_sym.load(str(checkpoint), args.device,
                                   ['defaults', 'minigrid', 'minigrid_symbolic_nav'])
    wm, behavior = agent._wm, agent._task_behavior
    labels = list(wm._sym_head.labels)
    records = []
    began = time.time()
    with torch.no_grad():
        for condition in ('soft', 'sign_only', 'oracle_sign'):
            for seed in range(args.episodes):
                env = MiniGrid('doorkey6x6', mode='eval', seed=seed,
                               max_steps=config.time_limit, emit_labels=False,
                               eval_seed_mode='direct')
                obs = env.reset()
                provenance = env.episode_provenance
                latent = action = None
                for step in range(config.time_limit + 1):
                    batch = {k: np.asarray(v)[None] for k, v in obs.items()
                             if not k.startswith('log_')}
                    data = wm.preprocess(batch)
                    latent, _ = wm.dynamics.obs_step(latent, action, wm.encoder(data),
                                                     data['is_first'], sample=False)
                    atoms = wm._sym_head.atoms(wm.dynamics.get_feat(latent)).detach()
                    if condition == 'sign_only':
                        atoms = torch.where(atoms > 0, torch.ones_like(atoms), -torch.ones_like(atoms))
                    elif condition == 'oracle_sign':
                        truth = env.god_state()
                        atoms = torch.tensor([[1.0 if truth[label] else -1.0 for label in labels]],
                                             device=args.device)
                    action = behavior.actor(atoms).mode()
                    obs, reward, done, _ = env.step(int(action.argmax(-1)[0]))
                    if done:
                        records.append(dict(condition=condition, success=bool(reward > 0),
                                            steps=step + 1, **provenance))
                        break
                env.close()
            print(condition, sum(x['success'] for x in records if x['condition'] == condition),
                  '/', args.episodes, 'elapsed', round(time.time() - began, 1), flush=True)
    result = {
        'checkpoint': str(checkpoint.relative_to(args.repo)),
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'purpose': 'post-training input intervention; no policy retraining; oracle is privileged evaluation control',
        'conditions': {c: {'successes': sum(x['success'] for x in records if x['condition'] == c),
                           'episodes': args.episodes,
                           'mean_steps': float(np.mean([x['steps'] for x in records if x['condition'] == c]))}
                       for c in ('soft', 'sign_only', 'oracle_sign')},
        'device': args.device, 'torch_threads': 1, 'records': records,
    }
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
