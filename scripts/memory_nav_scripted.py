"""
!! 诊断脚本：手写反应式控制器，只读 god_state 真值（cue_is_key + 自我中心导航谓词）。      !!
!! 这不是模型能力，是训练 Model3-on-MemoryS7 之前的零成本词表充分性自检。                 !!
!! 通关只证明"词表够用、god_state->谓词的算法没 bug"，不预示真模型会成——                !!
!! 真模型得自己从 belief 里学会读出 cue_is_key 并在长廊里记住它，这里没有这个难度。         !!
"""
import pathlib as _pathlib, sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))  # 仓库整理后脚本移入 scripts/，补回repo根目录到sys.path
import sys, pathlib
import numpy as np
REPO = pathlib.Path(__file__).parent
sys.path.insert(0, str(REPO))
import envs.minigrid as M
from memory_nav_god import god_memory_nav

LEFT, RIGHT, FWD = 0, 1, 2
N_EVAL = 20            # 与 belief-shaping MemoryS7 验收 notebook 同一批测试 seed，便于对比
TIME_LIMIT = 250        # 与 configs.yaml 的 minigrid_memory.time_limit 对齐


def scripted(g):
    """Privileged training/diagnostic controller using ``god_state`` predicates.

    Priority is target reach, unobstructed forward motion, then a target-directed
    turn at a wall. The learned evaluation policy never receives these fields.
    """
    reach, left, right = (g["key_reach"], g["key_left"], g["key_right"]) if g["cue_is_key"] \
        else (g["ball_reach"], g["ball_left"], g["ball_right"])
    if reach:
        return FWD                       # 踩进 success_pos，env 自动判终止
    if not g["wall_ahead"]:
        return FWD                       # 能走就先走，别在夹缝里瞎转
    if left:
        return LEFT
    if right:
        return RIGHT
    return LEFT                          # 兜底：撞墙且左右都判不出时，固定左转找方向


def run_episode(seed):
    env = M.MiniGrid("memoryS7", mode="eval", seed=seed, max_steps=TIME_LIMIT)
    env.reset()
    g = god_memory_nav(env)
    for t in range(1, TIME_LIMIT + 1):
        a = scripted(g)
        obs, r, done, info = env.step(a)
        g = god_memory_nav(env)
        if done:
            env.close()
            reason = "success" if r > 0 else ("hit_failure_pos" if g["at_fail"] else "timeout")
            return dict(seed=seed, success=bool(r > 0), steps=t, final_state=g, reason=reason)
    env.close()
    return dict(seed=seed, success=False, steps=TIME_LIMIT, final_state=g, reason="timeout_no_terminate")


def main():
    print("=" * 70)
    print(f"批量跑 scripted(g)，seeds 0..{N_EVAL-1}（测试池内），time_limit={TIME_LIMIT}，失败局照算不挑局")
    print("=" * 70)
    results = [run_episode(s) for s in range(N_EVAL)]
    succ = [r for r in results if r["success"]]
    fail = [r for r in results if not r["success"]]
    rate = len(succ) / N_EVAL

    for r in results:
        tag = "OK  " if r["success"] else "FAIL"
        print(f"seed={r['seed']:2d} {tag} steps={r['steps']:4d}  final={r['final_state']}")

    print()
    print(f"如实通关率: {len(succ)}/{N_EVAL} = {rate:.0%}  (含失败局，不挑局)")
    if succ:
        steps = [r["steps"] for r in succ]
        print(f"通关局步数: 平均 {np.mean(steps):.1f}  最大 {max(steps)}  (time_limit={TIME_LIMIT}，效率 = 平均步数/time_limit = {np.mean(steps)/TIME_LIMIT:.1%})")
    if fail:
        print(f"\n{len(fail)} 局失败，逐个打印终止时的真实状态（不是猜的）：")
        for r in fail:
            print(f"  seed={r['seed']} steps={r['steps']} reason={r['reason']}  final_state={r['final_state']}")


if __name__ == "__main__":
    main()
