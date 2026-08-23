"""
probe_memory.py — 记忆任务(MemoryS7)诊断：cue_is_key 到底编码在 belief 的哪一部分？

只读、不重训。加载论文保留的 M1 checkpoint，用训练好的策略滚 N 局，每步记录：
  belief：deter(h_t,256) / stoch(z_t,1024) / feat(1280)
  当前帧对照：grid(148)
  标签(上帝视角，只当 label)：cue_is_key / cue_visible / cue_known
然后分别用 4 处特征线性 probe cue_is_key（按 cue_known 掩码，按 episode 划分 train/test），
并单独统计"纯记忆子集"(cue_known=1 且 cue_visible=0：知道 cue 但当前看不见)的准确率。

判读：
  - deter/feat 在纯记忆子集上高、grid 在纯记忆子集上≈随机  → belief 把记忆存在循环态里（记忆证据成立）。
  - deter 也≈随机  → WM 根本没把 cue 编进 belief（任务太难/无压力），需先强化记忆压力。

用法：python scripts/probe_memory.py --checkpoint models/memory/memory_experiment_1_m1/grounded_seed0/checkpoint_084000.pt --episodes 150
"""
import pathlib as _pathlib, sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))  # 仓库整理后脚本移入 scripts/，补回repo根目录到sys.path
import argparse
import pathlib
import sys
import tempfile

import numpy as np
import ruamel.yaml as yaml
import torch

HERE = pathlib.Path(__file__).parent
REPO = HERE.parent   # 仓库整理后此脚本移入 scripts/，REPO 才是原来的 HERE（repo根目录）
sys.path.append(str(HERE))
import dreamer
import tools
import envs.wrappers as wrappers
import envs.minigrid as M


def recursive_update(base, update):
    for k, v in update.items():
        if isinstance(v, dict) and k in base:
            recursive_update(base[k], v)
        else:
            base[k] = v


def build_config(overlays):
    cfg = yaml.safe_load((REPO / "configs.yaml").read_text(encoding="utf-8"))
    defaults = {}
    for name in overlays:
        recursive_update(defaults, cfg[name])
    parser = argparse.ArgumentParser()
    for k, v in sorted(defaults.items(), key=lambda x: x[0]):
        t = tools.args_type(v)
        parser.add_argument(f"--{k}", type=t, default=t(v))
    return parser.parse_args([])


@torch.no_grad()
def rollout(wm, behavior, task, seed, time_limit):
    env = M.MiniGrid(task, mode="eval", seed=seed, max_steps=time_limit,
                     render_obs=False, emit_labels=False)
    obs = env.reset()
    latent, action = None, None
    rows, success, nsteps = [], False, 0
    for _ in range(time_limit + 1):
        obs_b = {k: np.array(v)[None] for k, v in obs.items() if not k.startswith("log_")}
        data = wm.preprocess(obs_b)
        embed = wm.encoder(data)
        latent, _ = wm.dynamics.obs_step(latent, action, embed, data["is_first"], sample=False)
        feat = wm.dynamics.get_feat(latent)
        action = behavior.actor(feat).mode()
        a_idx = int(torch.argmax(action, dim=-1)[0].item())

        g = env.god_state()
        rows.append({
            "deter": latent["deter"][0].cpu().numpy().astype(np.float32),         # (256,)
            "stoch": latent["stoch"][0].reshape(-1).cpu().numpy().astype(np.float32),  # (1024,)
            "feat":  feat[0].cpu().numpy().astype(np.float32),                     # (1280,)
            "grid":  obs["grid"].astype(np.float32),                              # (148,)
            "cue_is_key":  int(g["cue_is_key"]),
            "cue_visible": int(g["cue_visible"]),
            "cue_known":   int(g["cue_known"]),
        })
        obs, r, done, _ = env.step(a_idx)
        nsteps += 1
        if done:
            success = bool(r > 0)
            break
    env.close()
    return rows, success, nsteps


def fit_probe(Xtr, ytr, Xte, device, epochs=300, wd=1e-3):
    """简单 logistic 回归 probe（torch），返回在给定 X 上的预测概率。"""
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xtr_n = (Xtr - mu) / sd
    Xte_n = (Xte - mu) / sd
    xt = torch.tensor(Xtr_n, device=device)
    yt = torch.tensor(ytr, device=device, dtype=torch.float32)
    lin = torch.nn.Linear(Xtr.shape[1], 1).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=wd)
    lossf = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(lin(xt).squeeze(-1), yt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        p = torch.sigmoid(lin(torch.tensor(Xte_n, device=device)).squeeze(-1)).cpu().numpy()
    return p


def load_agent(checkpoint, device=None, verbose=True):
    """重建与（塑形后）checkpoint 结构一致的 agent 并加载权重。返回 (agent, config)。"""
    config = build_config(["defaults", "minigrid", "minigrid_memory", "minigrid_memory_shape"])
    config.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    config.num_actions = 5
    config.compile = False
    base = M.MiniGrid("memoryS7", mode="eval", seed=0, max_steps=config.time_limit,
                      render_obs=False, emit_labels=True)
    act_space = wrappers.OneHotAction(base).action_space
    obs_space = base.observation_space
    logger = tools.Logger(pathlib.Path(tempfile.mkdtemp()), 0)
    agent = dreamer.Dreamer(obs_space, act_space, config, logger, dataset=None).to(config.device)
    ckpt = torch.load(checkpoint, map_location=config.device, weights_only=False)
    miss, unexp = agent.load_state_dict(ckpt["agent_state_dict"], strict=False)
    core_miss = [k for k in miss if "_shape_head" not in k and "_ndnf" not in k]
    assert not core_miss, f"核心权重缺失（结构不匹配）: {core_miss[:5]}"
    if verbose:
        print(f"[load] {checkpoint} | missing {len(miss)} unexpected {len(unexp)} "
              f"(仅 _shape_head/_ndnf 头，不使用)")
    agent.requires_grad_(False)
    agent.eval()
    return agent, config


def collect(agent, config, task="memoryS7", episodes=150, verbose=True):
    """滚 N 局，返回每步 belief/grid/标签 的数组字典（+ success_rate）。"""
    wm, behavior = agent._wm, agent._task_behavior
    rows, successes = [], []
    for ep in range(episodes):
        r, succ, _ = rollout(wm, behavior, task, seed=ep, time_limit=config.time_limit)
        for x in r:
            x["episode"] = ep
        rows.extend(r)
        successes.append(succ)
        if verbose and (ep + 1) % 25 == 0:
            print(f"  ...{ep+1}/{episodes} 局，累计通关率 {np.mean(successes):.1%}")
    out = {k: np.array([r[k] for r in rows]) for k in
           ["deter", "stoch", "feat", "grid", "cue_is_key", "cue_visible", "cue_known", "episode"]}
    out["cue_is_key"] = out["cue_is_key"].astype(np.float32)
    out["cue_known"] = out["cue_known"].astype(bool)
    out["cue_visible"] = out["cue_visible"].astype(bool)
    out["success_rate"] = float(np.mean(successes))
    return out


FEATCOLS = [("deter", "deter · 记忆 h_t"), ("stoch", "stoch · z_t"),
            ("feat", "feat · deter+stoch"), ("grid", "grid · 当前帧(对照)")]


def _split(D, seed=0):
    """按 episode 划 train/test，只保留 cue_known 时刻。返回 (tr, te) 掩码。"""
    known, epa = D["cue_known"], D["episode"]
    uniq = np.unique(epa); rng = np.random.RandomState(seed); rng.shuffle(uniq)
    tr_ep = set(uniq[: int(0.7 * len(uniq))])
    intr = np.array([e in tr_ep for e in epa])
    return (intr & known), (~intr & known)


def decode_matrix(D, device=None, seed=0):
    """四处 latent × 三种视觉状态 的 cue_is_key 解码准确率表。返回 (res, meta)。"""
    dev = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    y, visible = D["cue_is_key"], D["cue_visible"]
    tr, te = _split(D, seed)
    te_idx = np.where(te)[0]; mem = ~visible[te_idx]; vis = visible[te_idx]
    res = {}
    for key, label in FEATCOLS:
        X = D[key]
        pred = (fit_probe(X[tr], y[tr], X[te], dev) > 0.5).astype(np.float32)
        yte = y[te]
        res[label] = {
            "test": float((pred == yte).mean()),
            "mem": float((pred[mem] == yte[mem]).mean()) if mem.any() else float("nan"),
            "vis": float((pred[vis] == yte[vis]).mean()) if vis.any() else float("nan"),
        }
    meta = {"n": int(len(y)), "known": int(D["cue_known"].sum()),
            "mem": int((D["cue_known"] & ~visible).sum()), "test": int(te.sum()),
            "base": float(max(y[te].mean(), 1 - y[te].mean())),
            "success_rate": float(D.get("success_rate", float("nan")))}
    return res, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(REPO / "models" / "memory" / "memory_experiment_1_m1" / "grounded_seed0" / "checkpoint_084000.pt"))
    ap.add_argument("--task", default="memoryS7")
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    agent, config = load_agent(args.checkpoint, args.device)
    dev = config.device
    print(f"[device] {dev} | checkpoint {args.checkpoint} | task {args.task}")
    D = collect(agent, config, args.task, args.episodes)
    deter, stoch, feat, grid = D["deter"], D["stoch"], D["feat"], D["grid"]
    y = D["cue_is_key"]
    known = D["cue_known"]
    visible = D["cue_visible"]
    epa = D["episode"]
    successes = [D["success_rate"]]

    print(f"\n总步数 {len(y)} | cue_known 时刻 {known.sum()} ({known.mean():.0%}) | "
          f"纯记忆(known&不可见) {int((known & ~visible).sum())} | 通关率(如实) {np.mean(successes):.1%}")
    print(f"cue=key 占比(known 内) {y[known].mean():.2f}（应≈0.5）")

    # 按 episode 划 train/test，避免同一局泄漏
    uniq = np.unique(epa)
    rng = np.random.RandomState(0)
    rng.shuffle(uniq)
    cut = int(0.7 * len(uniq))
    tr_ep, te_ep = set(uniq[:cut]), set(uniq[cut:])
    tr = np.array([e in tr_ep for e in epa]) & known          # 只在信息可知时刻监督
    te = np.array([e in te_ep for e in epa]) & known
    te_mem = te & ~visible                                     # 测试集里"纯记忆"子集
    te_vis = te & visible                                      # 对照：当前就能看见 cue
    print(f"train 样本 {tr.sum()} | test 样本 {te.sum()}（其中纯记忆 {te_mem.sum()} / 当前可见 {te_vis.sum()}）\n")

    feats = {"deter(记忆h_t,256)": deter, "stoch(z_t,1024)": stoch,
             "feat(deter+stoch,1280)": feat, "grid(当前帧,148)=防作弊对照": grid}
    print(f"{'特征':30} {'test-known':>11} {'纯记忆(known&不可见)':>20} {'当前可见':>10}")
    print("-" * 76)
    for name, X in feats.items():
        p = fit_probe(X[tr], y[tr], X[te], dev)          # 预测整个 test，再按子集切
        pred = (p > 0.5).astype(np.float32)
        yte = y[te]
        acc_all = (pred == yte).mean()
        # 纯记忆 / 可见 子集：需要在 te 索引内再切
        te_idx = np.where(te)[0]
        mem_local = ~visible[te_idx]
        vis_local = visible[te_idx]
        acc_mem = (pred[mem_local] == yte[mem_local]).mean() if mem_local.any() else float("nan")
        acc_vis = (pred[vis_local] == yte[vis_local]).mean() if vis_local.any() else float("nan")
        print(f"{name:30} {acc_all:>11.3f} {acc_mem:>20.3f} {acc_vis:>10.3f}")
    print("-" * 76)
    print("判读：纯记忆列 deter/feat 高、grid≈0.5 → 记忆存在循环态(belief=记忆成立)；"
          "deter 也≈0.5 → WM 没编码 cue。")


if __name__ == "__main__":
    main()
