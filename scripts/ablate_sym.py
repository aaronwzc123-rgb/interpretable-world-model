# -*- coding: utf-8 -*-
"""sym_run 消融 + 抽规则：策略到底多依赖逻辑原子？

对同一 checkpoint 跑两种 eval：
  normal  —— 原子照常喂策略（actor 输入 = concat(feat, atoms)）
  ablate  —— 把原子置零（concat(feat, 0)），feat 不变
通关率之差 = 策略对逻辑原子的依赖度（混合 concat 下 feat 冗余，可能依赖低）。
同时统计 eval 时原子 vs 上帝真值的符合率（确认原子在未见局仍=具名谓词），
并端到端抽出 `door_open :- ...` 规则。默认 CPU（不与 GPU 训练争显存）。

用法：python scripts/ablate_sym.py --checkpoint models/doorkey/doorkey_experiment_3_d3/checkpoint.pt --episodes 40
"""
import pathlib as _pathlib, sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))  # 仓库整理后脚本移入 scripts/，补回repo根目录到sys.path
import argparse
import json
import pathlib
import sys
import tempfile

import numpy as np
import torch

HERE = pathlib.Path(__file__).parent
REPO = HERE.parent   # 仓库整理后此脚本移入 scripts/，REPO 才是原来的 HERE（repo根目录）
sys.path.append(str(HERE))
import dreamer
import tools
import envs.wrappers as wrappers
import envs.minigrid as M
import probe_memory as P


def make_env(task, seed, tl):
    return M.MiniGrid(task, mode="eval", seed=seed, max_steps=tl,
                      render_obs=False, emit_labels=True)


def load(checkpoint, device, overlays):
    config = P.build_config(overlays)
    config.device = device
    config.num_actions = 5
    config.compile = False
    base = make_env("doorkey6x6", 0, config.time_limit)
    act_space = wrappers.OneHotAction(base).action_space
    obs_space = base.observation_space
    logger = tools.Logger(pathlib.Path(tempfile.mkdtemp()), 0)
    agent = dreamer.Dreamer(obs_space, act_space, config, logger, dataset=None).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    miss, unexp = agent.load_state_dict(ckpt["agent_state_dict"], strict=False)
    # Checkpoints predating 2026-08-01 do not contain these newly registered
    # SymbolicHead metadata buffers. They do not change learned weights; for a
    # non-adaptive D3 head the config value restores anneal_steps exactly.
    compatible_suffixes = (".anneal_steps", ".delta_value", ".target_acc")
    core_miss = [
        k for k in miss
        if "_shape_head" not in k
        and "_ndnf" not in k
        and not k.endswith(compatible_suffixes)
    ]
    assert not core_miss, f"核心权重缺失: {core_miss[:5]}"
    agent.requires_grad_(False)
    agent.eval()
    base.close()
    return agent, config


@torch.no_grad()
def run_eps(agent, config, task, episodes, ablate=None, seed0=1000):
    """ablate: None=正常；'all'=全原子置零；int i=只把第 i 个原子置零（逐谓词消融）。"""
    wm, beh = agent._wm, agent._task_behavior
    K, labels = wm.sym_dim, wm._sym_head.labels
    orig_atoms = wm._sym_head.atoms                       # 真原子（测量用，不受消融影响）
    if ablate == "all":
        wm._sym_head.atoms = lambda feat, delta=None: torch.zeros(
            *feat.shape[:-1], K, device=feat.device)
    elif isinstance(ablate, int):
        def _patch(feat, delta=None, _i=ablate):
            a = orig_atoms(feat, delta).clone()
            a[..., _i] = 0.0
            return a
        wm._sym_head.atoms = _patch
    succ, lengths = [], []
    hits = {l: [] for l in labels}
    for ep in range(episodes):
        env = make_env(task, seed=seed0 + ep, tl=config.time_limit)
        obs = env.reset()
        latent = action = None
        done_ok, t = False, 0
        for t in range(config.time_limit + 1):
            obs_b = {k: np.array(v)[None] for k, v in obs.items() if not k.startswith("log_")}
            data = wm.preprocess(obs_b)
            embed = wm.encoder(data)
            latent, _ = wm.dynamics.obs_step(latent, action, embed, data["is_first"], sample=False)
            feat = wm.dynamics.get_feat(latent)
            a_real = orig_atoms(feat)                     # (1,K)∈[-1,1]，测原子准确率
            g = env.god_state()
            for i, l in enumerate(labels):
                hits[l].append(int((a_real[0, i].item() > 0) == (bool(g[l]))))
            pol = wm.augment(feat)                        # 消融时此处原子=0
            action = beh.actor(pol).mode()
            a_idx = int(torch.argmax(action, dim=-1)[0].item())
            obs, r, done, _ = env.step(a_idx)
            if done:
                done_ok = bool(r > 0)
                break
        env.close()
        succ.append(done_ok)
        lengths.append(t + 1)
    wm._sym_head.atoms = orig_atoms
    return {"success": float(np.mean(succ)), "len": float(np.mean(lengths)),
            "atom_acc": {l: float(np.mean(v)) for l, v in hits.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(REPO / "models" / "doorkey" / "doorkey_experiment_3_d3" / "checkpoint.pt"))
    ap.add_argument("--task", default="doorkey6x6")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--nav", action="store_true",
                    help="用导航瓶颈配置 minigrid_symbolic_nav（否则用 minigrid_symbolic）")
    ap.add_argument("--per-atom", action="store_true", help="逐个原子消融（瓶颈下看策略靠哪些谓词）")
    ap.add_argument("--tag", default=None, help="规则快照文件名前缀")
    ap.add_argument("--out", default=None, help="JSON result path")
    args = ap.parse_args()

    overlays = ["defaults", "minigrid",
                "minigrid_symbolic_nav" if args.nav else "minigrid_symbolic"]
    agent, config = load(args.checkpoint, args.device, overlays)
    labels = agent._wm._sym_head.labels
    mode = agent._wm._sym_policy_input
    print(f"[load] {args.checkpoint} | device {args.device} | policy_input={mode} "
          f"| sym_dim {agent._wm.sym_dim}\n       labels {labels}")

    print(f"\n[eval normal]  {args.episodes} eps ...")
    normal = run_eps(agent, config, args.task, args.episodes, ablate=None)
    print(f"[eval ablate-all]  {args.episodes} eps (all atoms=0) ...")
    zero = run_eps(agent, config, args.task, args.episodes, ablate="all")

    print("\n" + "=" * 64)
    print(f"BOTTLENECK EVAL — policy_input={mode} (atoms=只看逻辑原子)")
    print("-" * 64)
    print(f"  success  normal {normal['success']:.3f}   all-atoms=0 {zero['success']:.3f}"
          f"   drop {normal['success']-zero['success']:+.3f}")
    print(f"  ep-len   normal {normal['len']:.1f}       all=0 {zero['len']:.1f}")
    print("-" * 64)
    print("ATOM accuracy at eval (real atoms vs god-state, unseen episodes)")
    for l, v in normal["atom_acc"].items():
        print(f"  {l:14} {v:.3f}")

    per = {}
    if args.per_atom:
        print("-" * 64)
        print("PER-ATOM ablation (zero one atom; success drop = 策略对该谓词的依赖)")
        for i, l in enumerate(labels):
            r = run_eps(agent, config, args.task, args.episodes, ablate=i)
            per[l] = r["success"]
            print(f"  zero {l:14} success {r['success']:.3f}  "
                  f"drop {normal['success']-r['success']:+.3f}")
    print("=" * 64)

    print("\nEnd-to-end extracted rules (sym_rules):")
    rules = agent._wm.sym_rules()
    for r in rules:
        print("  " + r)
    if not rules:
        print("  (none above threshold)")

    tag = args.tag or ("doorkey_experiment_3_d3" if args.nav else "sym_run1")
    out = HERE / "rules_snapshots" / f"{tag}_rules.txt"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"# {tag} 端到端符号规则 + 瓶颈消融\n# checkpoint {args.checkpoint}\n")
        f.write(f"# policy_input={mode}  labels={labels}\n")
        f.write(f"# success normal {normal['success']:.3f} / all-atoms=0 {zero['success']:.3f}\n")
        f.write(f"# atom_acc {normal['atom_acc']}\n")
        if per:
            f.write(f"# per-atom success {per}\n")
        f.write("\n" + "\n".join(rules) + "\n")
    print(f"\n-> saved {out}")
    json_out = pathlib.Path(args.out) if args.out else HERE / "results" / "d3_unified.json"
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps({
        "checkpoint": str(pathlib.Path(args.checkpoint).resolve()),
        "task": args.task,
        "episodes_per_condition": args.episodes,
        "evaluation_seed_start": 1000,
        "policy_input": mode,
        "normal": normal,
        "zero_all": zero,
        "per_atom_success": per,
        "rules": rules,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"-> saved {json_out}")


if __name__ == "__main__":
    main()
