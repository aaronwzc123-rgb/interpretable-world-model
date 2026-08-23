"""
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! 诊断脚本 —— 本文件里的一切字段都来自 env.god_state() / env 内部私有属性        !!
!! （god_state / grid / success_pos / failure_pos），全部是"上帝真值"。          !!
!! 这不是模型的感知能力，模型永远看不到这些字段。本脚本只用于回答一个问题：       !!
!! "如果一个策略能无损读到这些真值，这套谓词词表够不够、能不能高效解开 MemoryS7"  !!
!! 训好的 Model3-on-MemoryS7 依然要自己从 belief 里把这些东西编码/记住，          !!
!! 这里的通关率只是"必要条件"（词表不背锅），不代表真模型会成。                  !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""
import pathlib as _pathlib, sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))  # 仓库整理后脚本移入 scripts/，补回repo根目录到sys.path
import sys, pathlib
REPO = pathlib.Path(__file__).parent
sys.path.insert(0, str(REPO))
import envs.minigrid as M


def god_memory_nav(env_wrapper):
    """[上帝真值·诊断专用] env.god_state() 现在已经自带候选物体 target/other + 导航谓词
    （逻辑已搬进 envs/minigrid.py::_god_memory，这里不再重复计算，只补两个纯诊断字段：
    at_goal/at_fail，用来交叉校验 env 私有的 success_pos/failure_pos）。
    """
    g = env_wrapper.god_state()  # cue_is_key/cue_visible/cue_known/wall_ahead/t_*/target/other/agent_pos/agent_dir
    env = env_wrapper._env.unwrapped

    success_pos = tuple(int(v) for v in env.success_pos)
    failure_pos = tuple(int(v) for v in env.failure_pos)
    d_succ = abs(success_pos[0] - g["target"][0]) + abs(success_pos[1] - g["target"][1])
    d_fail = abs(failure_pos[0] - g["other"][0]) + abs(failure_pos[1] - g["other"][1])
    assert d_succ == 1, f"success_pos={success_pos} 应与匹配目标 target={g['target']} 相邻，实际曼哈顿距离={d_succ}"
    assert d_fail == 1, f"failure_pos={failure_pos} 应与非匹配物体 other={g['other']} 相邻，实际曼哈顿距离={d_fail}"

    g["success_pos"] = success_pos
    g["failure_pos"] = failure_pos
    g["at_goal"] = g["agent_pos"] == success_pos
    g["at_fail"] = g["agent_pos"] == failure_pos
    return g


def fmt(g):
    # 2026-07-28 字段名迁移（只改字段名）：t_ahead/t_left/t_right/t_reach 已被
    # envs/minigrid.py 的谓词重设计替换成 key_*/ball_* 两套，这里按 cue_is_key 选一套打印。
    reach, left, right = (g["key_reach"], g["key_left"], g["key_right"]) if g["cue_is_key"] \
        else (g["ball_reach"], g["ball_left"], g["ball_right"])
    return (f"pos={g['agent_pos']} dir={g['agent_dir']} "
            f"cue_is_key={int(g['cue_is_key'])} cue_vis={int(g['cue_visible'])} cue_known={int(g['cue_known'])} | "
            f"target={g['target']} other={g['other']} | "
            f"t_left={int(left)} t_right={int(right)} "
            f"t_reach={int(reach)} wall_ahead={int(g['wall_ahead'])} | "
            f"at_goal={int(g['at_goal'])} at_fail={int(g['at_fail'])}")


def main():
    print("=" * 70)
    print("[A] 原始 god_state() 字段一览（改前的现状，坐实缺什么）")
    print("=" * 70)
    env = M.MiniGrid("memoryS7", mode="eval", seed=0, max_steps=300)
    obs = env.reset()
    g0 = env.god_state()
    print("现有字段:", list(g0.keys()))
    print("t=0 值:", g0)
    env.close()

    print()
    print("=" * 70)
    print("[B] 候选物体扫描 + success_pos/failure_pos 交叉校验（跑 30 个测试 seed，全用断言，错了会直接报错崩溃）")
    print("=" * 70)
    ok = 0
    for seed in range(30):
        env = M.MiniGrid("memoryS7", mode="eval", seed=seed, max_steps=300)
        env.reset()
        g = god_memory_nav(env)
        print(f"seed={seed:2d}  cue_is_key={int(g['cue_is_key'])}  target={g['target']}  other={g['other']}  "
              f"success_pos={g['success_pos']}  failure_pos={g['failure_pos']}")
        env.close()
        ok += 1
    print(f"-> {ok}/30 局断言全过（没抛异常）：候选物体识别 + target 匹配 + success_pos 交叉校验全部正确。")

    print()
    print("=" * 70)
    print("[C] 抽查几局的逐步真值（policy=蠢策略：一直前进，撞墙就右转——仅用于让 agent 动起来抽查数字，不是第三步的正式控制器）")
    print("=" * 70)
    LEFT, RIGHT, FWD = 0, 1, 2
    for seed in [0, 1, 2]:
        env = M.MiniGrid("memoryS7", mode="eval", seed=seed, max_steps=300)
        env.reset()
        g = god_memory_nav(env)
        print(f"--- seed={seed} ---")
        print(f"  t=0  {fmt(g)}")
        for t in range(1, 16):
            a = RIGHT if g["wall_ahead"] else FWD
            obs, r, done, info = env.step(a)
            g = god_memory_nav(env)
            print(f"  t={t:<2} action={['left','right','forward'][a]:7s} {fmt(g)}")
            if done:
                print(f"  -> 局终止 done=True reward={r}")
                break
        env.close()


if __name__ == "__main__":
    main()
