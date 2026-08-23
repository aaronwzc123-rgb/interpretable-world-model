# -*- coding: utf-8 -*-
"""两阶段方案 · Stage A —— 从 mem_shape1（belief 已被塑形记住 cue_is_key）离线蒸馏全 v2 词表。

底座：models/memory/memory_experiment_3_m3/shared_world_model.pt（GroundedHead 塑形，deter probe 0.875 的
已验证先例）。用它的策略（sample 模式，训练池种子）跑 rollout，god_state 现算 v2 十原子
（不依赖任何 checkpoint 里存的旧标签），detach 的 belief feat 上蒸馏一个结构同 sym_head 的
N-DNF 头（feat -> 32 literal -> 8 conj -> 10 atom），MSE + mask_label='cue_known'（与 v1-v6
在线训练用的掩码规则一致，公平对照）。

产出：runs/memory_experiment_3_m3_reproduction/{distill_head_memoryS7.pt, distill_head_meta.json,
stageA_report.json, stageA_dataset.npz}，以及追加进 staged_log.md 的第1节。
"""
import pathlib as _pathlib, sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import argparse
import json
import pathlib
import sys
import tempfile
import time
import traceback

import numpy as np
import torch

HERE = pathlib.Path(__file__).parent
REPO = HERE.parent
sys.path.append(str(HERE))
import dreamer
import tools
import neuraldnf
import envs.wrappers as wrappers
import envs.minigrid as M
import probe_memory as PM

LABELS = ["cue_is_key", "wall_ahead", "key_ahead", "key_left", "key_right", "key_reach",
          "ball_ahead", "ball_left", "ball_right", "ball_reach"]  # 与 configs.yaml v2 sym_labels 顺序一致
CKPT_DEFAULT = REPO / "models" / "memory" / "memory_experiment_3_m3" / "shared_world_model.pt"
OUT_DIR = REPO / "runs" / "memory_experiment_3_m3_reproduction"
LOG_MD = OUT_DIR / "staged_log.md"


def load_mem_shape1(checkpoint, device):
    config = PM.build_config(["defaults", "minigrid", "minigrid_memory", "minigrid_memory_cuestart",
                              "minigrid_memory_shape"])
    config.device = device
    config.num_actions = 5
    config.compile = False
    base = M.MiniGrid("memoryS7_cuestart", mode="eval", seed=0, max_steps=config.time_limit, emit_labels=True)
    act_space = wrappers.OneHotAction(base).action_space
    obs_space = base.observation_space
    base.close()
    logger = tools.Logger(pathlib.Path(tempfile.mkdtemp()), 0)
    agent = dreamer.Dreamer(obs_space, act_space, config, logger, dataset=None).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    agent.load_state_dict(ckpt["agent_state_dict"])  # strict：确认结构与训练时一致
    agent.requires_grad_(False)
    agent.eval()
    return agent, config


@torch.no_grad()
def rollout_sample(wm, behavior, seed, time_limit):
    """actor.sample()（非确定性，增加状态多样性）+ 训练池种子（mode='train'）。
    god_state 现算 v2 十原子 + cue_visible/cue_known 掩码信息。"""
    env = M.MiniGrid("memoryS7_cuestart", mode="train", seed=seed, max_steps=time_limit, emit_labels=False)
    obs = env.reset()
    latent, action = None, None
    rows = []
    done, r, t = False, 0.0, 0
    for t in range(time_limit + 1):
        obs_b = {k: np.array(v)[None] for k, v in obs.items() if not k.startswith("log_")}
        data = wm.preprocess(obs_b)
        embed = wm.encoder(data)
        latent, _ = wm.dynamics.obs_step(latent, action, embed, data["is_first"])
        feat = wm.dynamics.get_feat(latent)
        action = behavior.actor(feat).sample()
        a_idx = int(torch.argmax(action, dim=-1)[0].item())
        g = env.god_state()
        rows.append((
            feat[0].cpu().numpy().astype(np.float32),
            np.array([1.0 if g[l] else 0.0 for l in LABELS], dtype=np.float32),
            bool(g["cue_known"]), bool(g["cue_visible"]),
            obs["grid"][:148].astype(np.float32),
        ))
        obs, r, done, _ = env.step(a_idx)
        if done:
            break
    env.close()
    return rows, bool(done and r > 0), t + 1


def collect(agent, episodes, time_limit, seed0, log_every=50):
    wm, behavior = agent._wm, agent._task_behavior
    feats, labels, cue_known, cue_visible, grids, ep_ids, lengths, successes = [], [], [], [], [], [], [], []
    t0 = time.time()
    for ep in range(episodes):
        rows, succ, n = rollout_sample(wm, behavior, seed0 + ep, time_limit)
        for f, y, ck, cv, gr in rows:
            feats.append(f); labels.append(y); cue_known.append(ck); cue_visible.append(cv)
            grids.append(gr); ep_ids.append(ep)
        successes.append(succ); lengths.append(n)
        if (ep + 1) % log_every == 0:
            print(f"  ...{ep + 1}/{episodes} 局 | sample策略期内success(仅参考) {np.mean(successes):.1%} | "
                  f"平均步数 {np.mean(lengths):.1f} | 累计样本 {len(feats)} | "
                  f"已用时 {time.time() - t0:.0f}s", flush=True)
    meta = dict(episodes=episodes, steps=len(feats), sample_success_rate=float(np.mean(successes)),
                avg_length=float(np.mean(lengths)), elapsed_sec=time.time() - t0)
    return (np.stack(feats), np.stack(labels), np.array(cue_known), np.array(cue_visible),
            np.stack(grids), np.array(ep_ids), meta)


def episode_split(ep_ids, seed=0, train_frac=0.7):
    uniq = np.unique(ep_ids)
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    cut = int(train_frac * len(uniq))
    tr_ep, te_ep = set(uniq[:cut].tolist()), set(uniq[cut:].tolist())
    tr = np.array([e in tr_ep for e in ep_ids])
    te = np.array([e in te_ep for e in ep_ids])
    return tr, te


def train_distill_head(feat, y01, cue_known, tr_mask, device, n_lit=32, n_conj=8, delta_max=4.0,
                        steps=8000, batch_size=256, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    head = neuraldnf.SymbolicHead(
        feat.shape[1], LABELS, n_lit=n_lit, n_conj=n_conj,
        delta_max=delta_max, anneal_steps=max(1, int(steps * 0.6)), mask_label="cue_known",
    ).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    Xtr = torch.tensor(feat[tr_mask], device=device).detach()
    Ytr = torch.tensor(y01[tr_mask], device=device)
    Ktr = torch.tensor(cue_known[tr_mask].astype(np.float32), device=device)
    n = Xtr.shape[0]
    losses = []
    t0 = time.time()
    for step in range(steps):
        idx = torch.randint(0, n, (min(batch_size, n),), device=device)
        xb = Xtr[idx]
        data_b = {f"label_{l}": Ytr[idx, i].unsqueeze(-1) for i, l in enumerate(LABELS)}
        data_b["label_cue_known"] = Ktr[idx].unsqueeze(-1)
        loss, _mets = head.masked_loss(xb, data_b)
        opt.zero_grad(); loss.backward(); opt.step()
        head.updates += 1
        losses.append(float(loss.item()))
        if (step + 1) % 1000 == 0:
            print(f"    distill step {step + 1}/{steps} loss={np.mean(losses[-200:]):.4f} "
                  f"delta={head.current_delta():.2f} 用时{time.time() - t0:.0f}s", flush=True)
    head.updates.fill_(head.anneal_steps)  # 存盘前钉死硬阈值
    head.eval()
    return head


def eval_distill_head(head, feat, y01, cue_known, cue_visible, te_mask, device):
    with torch.no_grad():
        Xte = torch.tensor(feat[te_mask], device=device)
        atoms = head.atoms(Xte).cpu().numpy()
    pred = (atoms > 0).astype(np.float32)
    yte = y01[te_mask]
    known_te = cue_known[te_mask]
    rows = []
    for i, l in enumerate(LABELS):
        m = known_te.astype(bool)
        y, p = yte[m, i], pred[m, i]
        pos, neg = int((y > 0.5).sum()), int((y <= 0.5).sum())
        acc = float((p == y).mean()) if len(y) else float("nan")
        base = float(max(y.mean(), 1 - y.mean())) if len(y) else float("nan")
        rows.append(dict(label=l, acc=acc, base=base, gain=acc - base, pos=pos, neg=neg,
                          reliable=pos >= 50 and neg >= 50))
    # cue_is_key 纯记忆步（cue_known & ~cue_visible）
    mem_mask = te_mask & cue_known.astype(bool) & (~cue_visible.astype(bool))
    if mem_mask.sum() > 0:
        with torch.no_grad():
            atoms_mem = head.atoms(torch.tensor(feat[mem_mask], device=device)).cpu().numpy()
        y_mem = y01[mem_mask, 0]
        pred_mem = (atoms_mem[:, 0] > 0).astype(np.float32)
        cue_mem = dict(n=int(mem_mask.sum()), acc=float((pred_mem == y_mem).mean()),
                        base=float(max(y_mem.mean(), 1 - y_mem.mean())),
                        pred_pos_rate=float(pred_mem.mean()))
        cue_mem["gain"] = cue_mem["acc"] - cue_mem["base"]
    else:
        cue_mem = None
    return rows, cue_mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(CKPT_DEFAULT))
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--distill-steps", type=int, default=8000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed0", type=int, default=0)
    args = ap.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[device] {device} | checkpoint {args.checkpoint}", flush=True)

    agent, config = load_mem_shape1(args.checkpoint, device)
    print(f"[rollout] {args.episodes} 局 | policy=actor.sample() | "
          f"seed pool=train ({M.TRAIN_SEED_START}..+{M.TRAIN_POOL}) | time_limit={config.time_limit}", flush=True)
    feat, y01, cue_known, cue_visible, grids, ep_ids, meta = collect(
        agent, args.episodes, config.time_limit, args.seed0)
    print(f"[collect done] 总步数 {meta['steps']} | 局数 {meta['episodes']} | "
          f"采样期内success(仅参考) {meta['sample_success_rate']:.1%} | "
          f"平均步数 {meta['avg_length']:.1f} | 用时 {meta['elapsed_sec']:.0f}s", flush=True)

    np.savez_compressed(OUT_DIR / "stageA_dataset.npz", feat=feat, labels=y01,
                         cue_known=cue_known, cue_visible=cue_visible, episode=ep_ids,
                         label_names=np.array(LABELS))

    tr, te = episode_split(ep_ids)
    print(f"[split] train步数 {int(tr.sum())} | test步数 {int(te.sum())}（按episode 70/30）", flush=True)

    head = train_distill_head(feat, y01, cue_known, tr, device, steps=args.distill_steps)
    torch.save(head.state_dict(), OUT_DIR / "distill_head_memoryS7.pt")
    meta_json = dict(in_dim=feat.shape[1], labels=LABELS, n_lit=32, n_conj=8, delta_max=4.0,
                      anneal_steps=int(head.anneal_steps.item()), mask_label="cue_known")
    with open(OUT_DIR / "distill_head_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_json, f, ensure_ascii=False, indent=2)

    rows, cue_mem = eval_distill_head(head, feat, y01, cue_known, cue_visible, te, device)

    # 当前帧对照探针（纯记忆步，grid 148维）
    mem_mask = te & cue_known.astype(bool) & (~cue_visible.astype(bool))
    grid_probe = None
    if mem_mask.sum() > 20 and (tr & cue_known.astype(bool)).sum() > 20:
        tr_probe = tr & cue_known.astype(bool)
        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        p_grid = PM.fit_probe(grids[tr_probe], y01[tr_probe, 0], grids[mem_mask], dev)
        y_mem = y01[mem_mask, 0]
        pred_grid = (p_grid > 0.5).astype(np.float32)
        grid_probe = dict(n=int(mem_mask.sum()), acc=float((pred_grid == y_mem).mean()),
                           base=float(max(y_mem.mean(), 1 - y_mem.mean())))
        grid_probe["gain"] = grid_probe["acc"] - grid_probe["base"]

    report = dict(meta=meta, rows=rows, cue_pure_memory=cue_mem, grid_control_probe=grid_probe,
                  config=dict(episodes=args.episodes, distill_steps=args.distill_steps,
                              policy="actor.sample()", seed_pool="train", train_test_split="70/30 by episode"))
    with open(OUT_DIR / "stageA_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 84, flush=True)
    print(f"{'谓词':<12}{'test-acc':>10}{'base':>8}{'gain':>8}{'pos':>8}{'neg':>8}", flush=True)
    print("-" * 84, flush=True)
    for r in rows:
        flag = "" if r["reliable"] else "  <50 样本不足"
        print(f"{r['label']:<12}{r['acc']:>10.3f}{r['base']:>8.3f}{r['gain']:>+8.3f}"
              f"{r['pos']:>8d}{r['neg']:>8d}{flag}", flush=True)
    print("=" * 84, flush=True)
    if cue_mem:
        print(f"\ncue_is_key 纯记忆步(cue_known&~cue_visible): n={cue_mem['n']} "
              f"acc={cue_mem['acc']:.3f} base={cue_mem['base']:.3f} gain={cue_mem['gain']:+.3f} "
              f"pred_pos_rate={cue_mem['pred_pos_rate']:.1%}", flush=True)
    if grid_probe:
        print(f"当前帧对照探针(同一批纯记忆步): n={grid_probe['n']} acc={grid_probe['acc']:.3f} "
              f"base={grid_probe['base']:.3f} gain={grid_probe['gain']:+.3f}", flush=True)

    gate_pass = cue_mem is not None and cue_mem["acc"] >= 0.85
    gate_note = (f"闸门通过：cue纯记忆步精度 {cue_mem['acc']:.1%} >= 85%" if gate_pass else
                 f"闸门未通过：cue纯记忆步精度 {cue_mem['acc']:.1%} < 85%（如实记录，仍继续Stage B）"
                 if cue_mem else "闸门无法判定：纯记忆步样本不足")
    print(f"\n[闸门] {gate_note}", flush=True)

    with LOG_MD.open("a", encoding="utf-8") as f:
        f.write("## 1. Stage A：离线蒸馏（全 v2 词表，10 原子）\n\n")
        f.write(f"底座：`{args.checkpoint}`（mem_shape1，belief 已被塑形记住 cue_is_key）。"
                f"rollout {meta['episodes']} 局（actor.sample()，训练池种子），"
                f"总步数 {meta['steps']}，蒸馏 {args.distill_steps} 步。\n\n")
        f.write("| 谓词 | test-acc | base | gain | pos | neg | 备注 |\n|---|---|---|---|---|---|---|\n")
        for r in rows:
            flag = "" if r["reliable"] else "样本不足<50"
            f.write(f"| {r['label']} | {r['acc']:.3f} | {r['base']:.3f} | {r['gain']:+.3f} | "
                    f"{r['pos']} | {r['neg']} | {flag} |\n")
        f.write("\n")
        if cue_mem:
            f.write(f"**cue_is_key 纯记忆步**（cue_known & ~cue_visible，n={cue_mem['n']}）："
                    f"acc={cue_mem['acc']:.1%}, base={cue_mem['base']:.1%}, "
                    f"gain={cue_mem['gain']:+.1%}, 预测正类比例={cue_mem['pred_pos_rate']:.1%}\n\n")
        if grid_probe:
            f.write(f"**当前帧对照探针**（同一批纯记忆步）：acc={grid_probe['acc']:.1%}, "
                    f"base={grid_probe['base']:.1%}, gain={grid_probe['gain']:+.1%}\n\n")
        f.write(f"**闸门判读**：{gate_note}\n\n---\n\n")

    print(f"\n-> Stage A 完成，产出已存 {OUT_DIR}，已追加进 {LOG_MD}", flush=True)
    return gate_pass


if __name__ == "__main__":
    try:
        ok = main()
        sys.exit(0)
    except Exception:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUT_DIR / "stageA_error.log", "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        print("!! Stage A 异常，完整堆栈已存 stageA_error.log", flush=True)
        raise
