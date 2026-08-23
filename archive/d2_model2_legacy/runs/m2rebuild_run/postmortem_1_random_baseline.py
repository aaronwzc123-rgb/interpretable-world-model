"""尸检1：随机策略基线。

在跟训练 eval 相同的测试 seed 池（M.MiniGrid mode="eval"，seed 从 0 起顺序取 TEST_POOL=200
里的号）上，用均匀随机策略（5 个合法动作等概率）跑 N 局，报 success rate / mean return。
纯环境交互，不碰任何模型/checkpoint/GPU。

用法：python postmortem_1_random_baseline.py --episodes 100
"""
import argparse
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).parent
sys.path.append(str(HERE))
import envs.minigrid as M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--task", default="doorkey6x6")
    ap.add_argument("--time_limit", type=int, default=360)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.RandomState(12345)
    env = M.MiniGrid(args.task, mode="eval", seed=args.seed,
                      max_steps=args.time_limit, render_obs=False, emit_labels=False)
    n_actions = env.action_space.n

    successes, returns, lengths = [], [], []
    for ep in range(args.episodes):
        obs = env.reset()
        ep_return, t, log_success = 0.0, 0, 0.0
        for t in range(args.time_limit + 1):
            a = int(rng.randint(n_actions))
            obs, r, done, _ = env.step(a)
            ep_return += float(r)
            if done:
                log_success = float(obs["log_success"])
                break
        successes.append(log_success)
        returns.append(ep_return)
        lengths.append(t + 1)
    env.close()

    successes = np.array(successes)
    returns = np.array(returns)
    lengths = np.array(lengths)
    print(f"random policy | {args.episodes} episodes | task={args.task} | test seed pool (mode=eval, start seed={args.seed})")
    print(f"  success rate : {successes.mean():.3f}  ({int(successes.sum())}/{args.episodes})")
    print(f"  mean return  : {returns.mean():.4f}  (std {returns.std():.4f}, min {returns.min():.4f}, max {returns.max():.4f})")
    print(f"  mean length  : {lengths.mean():.1f}")


if __name__ == "__main__":
    main()
