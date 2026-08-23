# -*- coding: utf-8 -*-
"""两阶段方案 · Stage C —— 评估电池（通关率、终点-cue关联、消融、条件性翻转实验）。

Stage B 产出的 agent 结构：共享记忆世界模型（148维grid，无回流）+ 冻结蒸馏头（10原子）+
新训 actor/critic（sym_policy_input='atoms'）。rollout 力学与最终 M3 验收协议
相同（无 AtomRegisterWrapper，纯 obs_step 循环）。
"""
import pathlib as _pathlib, sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import argparse
import json
import pathlib
import sys
import tempfile
from collections import Counter

import numpy as np
import torch

HERE = pathlib.Path(__file__).parent
REPO = HERE.parent
sys.path.append(str(HERE))
import dreamer
import tools
import envs.minigrid as M
import probe_memory as PM
from staged_stage_b_train import build_config_b

OUT_DIR = REPO / "runs" / "memory_experiment_3_m3_reproduction"
FINAL_MODEL_DIR = REPO / "models" / "memory" / "memory_experiment_3_m3"
LOG_MD = OUT_DIR / "staged_log.md"
LABELS = ["cue_is_key", "wall_ahead", "key_ahead", "key_left", "key_right", "key_reach",
          "ball_ahead", "ball_left", "ball_right", "ball_reach"]
ACTION_NAMES = ["left", "right", "forward", "pickup", "toggle"]


def load(checkpoint, device, distill_meta_path=None, logdir=None):
    config = build_config_b(logdir or OUT_DIR, 84000, device)
    config.num_actions = 5
    base = M.MiniGrid("memoryS7_cuestart", mode="eval", seed=0, max_steps=config.time_limit, emit_labels=True)
    import envs.wrappers as wrappers
    act_space = wrappers.OneHotAction(base).action_space
    obs_space = base.observation_space
    base.close()
    logger = tools.Logger(pathlib.Path(tempfile.mkdtemp()), 0)
    agent = dreamer.Dreamer(obs_space, act_space, config, logger, dataset=None).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    miss, unexp = agent.load_state_dict(ckpt["agent_state_dict"], strict=False)
    # 兼容 anneal_steps 从普通属性改为 buffer 之前保存的旧 checkpoint：那些 state_dict
    # 里本就没有 anneal_steps 这个 key，load_state_dict 自然报 missing。用蒸馏头
    # meta 里的真值手动补上（和 bootstrap_fresh 构造蒸馏头时的逻辑一致）。
    legacy_anneal_miss = [k for k in miss if k.endswith("_sym_head.anneal_steps")]
    if legacy_anneal_miss:
        meta_path = distill_meta_path or (OUT_DIR / "distill_head_meta.json")
        with open(meta_path, encoding="utf-8") as f:
            true_anneal_steps = json.load(f)["anneal_steps"]
        for k in legacy_anneal_miss:
            head = agent
            for attr in k.split(".")[:-1]:
                head = getattr(head, attr)
            head.anneal_steps.fill_(true_anneal_steps)
            print(f"[load] 旧checkpoint兼容：手动恢复 {k} = {true_anneal_steps}")
        miss = [k for k in miss if k not in legacy_anneal_miss]
    assert not miss and not unexp, f"权重加载不完整: missing={miss[:5]} unexpected={unexp[:5]}"
    print(f"[load] {checkpoint} | missing {len(miss)} unexpected {len(unexp)}")
    agent.requires_grad_(False)
    agent.eval()
    return agent, config


@torch.no_grad()
def rollout(wm, behavior, seed, tl, episode_id=None, want_actions=False,
            flip_idx=None, flip_start=None, force_register=None):
    """无回流架构：atoms 只读不回流进 obs，但 flip 实验需要"篡改 actor 看到的 atoms"——
    通过 force_register 参数在 augment 前替换 atoms 实现（等价于寄存器篡改，只是没有真的
    寄存器，直接在算 policy 输入这一步替换）。"""
    env = M.MiniGrid("memoryS7_cuestart", mode="eval", seed=seed, max_steps=tl, render_obs=False, emit_labels=False)
    obs = env.reset()
    latent, action = None, None
    rows, succ, t = [], False, 0
    acts = []
    for t in range(tl + 1):
        obs_b = {k: np.array(v)[None] for k, v in obs.items() if not k.startswith("log_")}
        data = wm.preprocess(obs_b)
        embed = wm.encoder(data)
        latent, _ = wm.dynamics.obs_step(latent, action, embed, data["is_first"], sample=False)
        feat = wm.dynamics.get_feat(latent)
        atoms_t = wm._sym_head.atoms(feat).detach()
        pol_atoms = atoms_t.clone()
        if flip_idx is not None and flip_start is not None and t >= flip_start:
            pol_atoms[0, flip_idx] = -pol_atoms[0, flip_idx]
        if force_register == "zero_all":
            pol_atoms = torch.zeros_like(pol_atoms)
        elif force_register == "zero_cue":
            pol_atoms[0, 0] = 0.0
        elif isinstance(force_register, int):  # zero one specific atom index
            pol_atoms[0, force_register] = 0.0
        action = behavior.actor(pol_atoms).mode()
        a_idx = int(torch.argmax(action, dim=-1)[0].item())
        if want_actions:
            acts.append(a_idx)
        g = env.god_state()
        rows.append(dict(atoms=atoms_t[0].cpu().numpy(), cue_is_key=int(g["cue_is_key"]),
                          cue_known=bool(g["cue_known"]), cue_visible=bool(g["cue_visible"]),
                          episode=episode_id))
        obs, r, done, info = env.step(a_idx)
        if done:
            succ = bool(r > 0)
            final_g = env.god_state()
            break
    else:
        final_g = env.god_state()
    env.close()
    return rows, dict(success=succ, steps=t + 1, final_pos=final_g["agent_pos"],
                       key_pos=final_g["key_pos"], ball_pos=final_g["ball_pos"],
                       cue_is_key=final_g["cue_is_key"], truncated="discount" in info, actions=acts)


def report(mask, atom, y, name):
    n = int(mask.sum())
    if n == 0:
        print(f"{name:34}{'n=0':>8}")
        return None
    yy = y[mask]
    pred = (atom[mask] > 0).astype(np.float32)
    acc = float((pred == yy).mean())
    base = float(max(yy.mean(), 1 - yy.mean()))
    print(f"{name:34}{n:>8}{base:>11.1%}{acc:>11.1%}{acc-base:>+9.1%}")
    return dict(n=n, base=base, acc=acc, gain=acc - base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(FINAL_MODEL_DIR / "actor_seed0_400demo.pt"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--rundir", default=str(OUT_DIR), help="存放该checkpoint的run目录（config.logdir用）")
    ap.add_argument("--distill-meta", default=str(FINAL_MODEL_DIR / "frozen_symbolic_head_meta.json"), help="冻结符号头元数据")
    ap.add_argument("--out-log", default=None, help="结果追加到哪个md文件，默认=rundir/staged_log.md")
    ap.add_argument("--n-eval", type=int, default=20)
    ap.add_argument("--n-abl", type=int, default=40)
    ap.add_argument("--n-flip", type=int, default=20)
    ap.add_argument("--flip-threshold", type=float, default=0.5,
                     help="通关率达到这个值才做翻转实验")
    args = ap.parse_args()

    rundir = pathlib.Path(args.rundir)
    out_log = pathlib.Path(args.out_log) if args.out_log else rundir / "staged_log.md"
    distill_meta = pathlib.Path(args.distill_meta) if args.distill_meta else rundir / "distill_head_meta.json"

    agent, config = load(args.checkpoint, args.device, distill_meta_path=distill_meta, logdir=rundir)
    wm, behavior = agent._wm, agent._task_behavior
    labels = wm._sym_head.labels
    print(f"labels={labels}  policy_input={wm._sym_policy_input}  "
          f"actor_in_dim={behavior.actor.layers[0].in_features}  encoder_mlp={wm.encoder.mlp_shapes}")

    md = []
    md.append("## 2. Stage C：评估电池\n")

    # ---- [1] 通关率 ----
    N_EVAL = args.n_eval
    succs, lens = [], []
    for s in range(N_EVAL):
        _, out = rollout(wm, behavior, s, config.time_limit)
        succs.append(out["success"]); lens.append(out["steps"])
    rate = float(np.mean(succs))
    print(f"\n[1] 如实通关率 {N_EVAL} 局 (seeds 0..{N_EVAL-1})")
    print(f"success {sum(succs)}/{N_EVAL} = {rate:.1%}   平均步数 {np.mean(lens):.1f}  (time_limit={config.time_limit})")
    md.append(f"### 通关率\n**success {sum(succs)}/{N_EVAL} = {rate:.1%}**，平均步数 {np.mean(lens):.1f}"
               f"（time_limit={config.time_limit}）。对照：早期失败变体最好 25%（且与cue无关），"
               f"scripted controller 上限 100%。\n")

    print(f"\n[1b] 5局动作序列（检查坍缩/自旋锁）")
    seq_lines = []
    for s in range(5):
        _, out = rollout(wm, behavior, s, config.time_limit, want_actions=True)
        cnt = Counter(ACTION_NAMES[a] for a in out["actions"])
        line = f"  seed={s} steps={len(out['actions'])} success={out['success']} 动作: {dict(cnt)}"
        print(line); seq_lines.append(line)
    md.append("动作序列抽查（seed 0-4）：\n```\n" + "\n".join(seq_lines) + "\n```\n")

    # ---- [2] cue精度 + 纯记忆步 ----
    N_PROBE = 150
    all_rows = []
    for s in range(N_PROBE):
        rows, _ = rollout(wm, behavior, s, config.time_limit, episode_id=s)
        all_rows.extend(rows)
    atom_cue = np.array([r["atoms"][0] for r in all_rows])
    y = np.array([r["cue_is_key"] for r in all_rows], dtype=np.float32)
    known = np.array([r["cue_known"] for r in all_rows], dtype=bool)
    visible = np.array([r["cue_visible"] for r in all_rows], dtype=bool)
    mem_mask = known & ~visible
    print(f"\n[2] cue_is_key 精度表（{N_PROBE}局测试池，总步数{len(all_rows)}）")
    print(f"{'条件':34}{'样本数':>8}{'base rate':>11}{'原子精度':>11}{'gain':>9}")
    res_all = report(known, atom_cue, y, "[A] 全部测试步")
    res_mem = report(mem_mask, atom_cue, y, "[B] 纯记忆步")
    pred_pos_rate = float((atom_cue[mem_mask] > 0).mean()) if mem_mask.sum() else float("nan")
    print(f"\n[5] sym_head纯记忆步预测'key'比例 = {pred_pos_rate:.1%}")
    md.append(f"### cue 精度 + sym_head 输出分布\n"
               f"全部测试步：n={res_all['n'] if res_all else 0}, base={res_all['base']:.1%}, "
               f"acc={res_all['acc']:.1%}, gain={res_all['gain']:+.1%}\n\n"
               f"纯记忆步：n={res_mem['n'] if res_mem else 0}, base={res_mem['base']:.1%}, "
               f"acc={res_mem['acc']:.1%}, gain={res_mem['gain']:+.1%}\n\n"
               f"纯记忆步预测正类('key')比例 = {pred_pos_rate:.1%}（对照早期失败变体的恒定崩溃）\n")

    # ---- [3] 消融 ----
    N_ABL = args.n_abl
    print(f"\n[3] 消融（{N_ABL}局，seed 1000-{1000+N_ABL-1}）")
    abl_lines = []
    abl_rates = {}
    for mode in [None, "zero_all", "zero_cue"]:
        succs_m = []
        for i in range(N_ABL):
            _, out = rollout(wm, behavior, 1000 + i, config.time_limit, force_register=mode)
            succs_m.append(out["success"])
        name = mode or "normal"
        abl_rates[name] = float(np.mean(succs_m))
        line = f"  {name:10} success = {sum(succs_m)}/{N_ABL} = {np.mean(succs_m):.1%}"
        print(line); abl_lines.append(line)
    print(f"\n[3b] 逐原子清零（20局/原子，seed 2000-2019）")
    for i, l in enumerate(LABELS):
        succs_i = []
        for j in range(20):
            _, out = rollout(wm, behavior, 2000 + j, config.time_limit, force_register=i)
            succs_i.append(out["success"])
        line = f"  zero[{l}]  success = {sum(succs_i)}/20 = {np.mean(succs_i):.1%}"
        print(line); abl_lines.append(line)
    md.append("### 消融\n```\n" + "\n".join(abl_lines) + "\n```\n")

    # ---- [4] 终点-cue关联 ----
    print(f"\n[4] 终点与真实cue关联 —— 100局(seed 1000-1099)")
    by_cue = {0: {"key": 0, "ball": 0, "neither": 0}, 1: {"key": 0, "ball": 0, "neither": 0}}
    n_timeout = 0
    for i in range(100):
        _, out = rollout(wm, behavior, 1000 + i, config.time_limit)
        if out["truncated"]:
            n_timeout += 1; continue
        fx, fy = out["final_pos"]; kx, ky = out["key_pos"]; bx, by_ = out["ball_pos"]
        dk = abs(fx - kx) + abs(fy - ky); db = abs(fx - bx) + abs(fy - by_)
        cue = out["cue_is_key"]
        if dk < db: by_cue[cue]["key"] += 1
        elif db < dk: by_cue[cue]["ball"] += 1
        else: by_cue[cue]["neither"] += 1
    print(f"  超时局数 = {n_timeout}/100")
    for cv in [0, 1]:
        d = by_cue[cv]; tot = sum(d.values())
        print(f"  真实cue_is_key={cv} (n={tot}): 终点更近key={d['key']} 更近ball={d['ball']} 持平={d['neither']}")
    md.append(f"### 终点-cue 关联（100局，超时{n_timeout}局）\n"
               f"- 真实cue_is_key=0 (n={sum(by_cue[0].values())}): "
               f"终点更近key={by_cue[0]['key']} 更近ball={by_cue[0]['ball']} 持平={by_cue[0]['neither']}\n"
               f"- 真实cue_is_key=1 (n={sum(by_cue[1].values())}): "
               f"终点更近key={by_cue[1]['key']} 更近ball={by_cue[1]['ball']} 持平={by_cue[1]['neither']}\n")

    # ---- [7] 翻转实验（通关率>=flip_threshold才做）----
    flip_done = False
    flip_redirect_rate = None
    ctrl_success_rate = None
    N_FLIP = args.n_flip
    if rate >= args.flip_threshold:
        print(f"\n[7] 翻转实验（通关率{rate:.1%}>={args.flip_threshold:.0%}，执行，{N_FLIP}局/时机点）")
        flip_done = True

        def scan_and_flip(seed, flip_idx):
            rows, _ = rollout(wm, behavior, seed, config.time_limit, episode_id=seed)
            hide_step = next((i for i, r in enumerate(rows) if r["cue_known"] and not r["cue_visible"]), None)
            if hide_step is None:
                return None, None
            n = len(rows)
            early, mid, late = int(n * 0.2) + hide_step, hide_step + max(1, (n - hide_step) // 2), max(hide_step, n - 3)
            return hide_step, dict(early=early, mid=mid, late=late)

        timing_results = {"early": [], "mid": [], "late": []}
        ctrl_results = []
        for i in range(N_FLIP):
            hide_step, points = scan_and_flip(1000 + i, 0)
            if hide_step is None:
                continue
            for tname, tstep in points.items():
                _, flip_out = rollout(wm, behavior, 1000 + i, config.time_limit, flip_idx=0, flip_start=tstep)
                timing_results[tname].append(flip_out)
            _, ctrl_out = rollout(wm, behavior, 1000 + i, config.time_limit, flip_idx=1, flip_start=hide_step)
            ctrl_results.append(ctrl_out)

        flip_lines = []
        redirect_rates = []
        for tname, results in timing_results.items():
            n_flip_to_wrong = sum(1 for r in results if not r["success"])  # success=选中真实cue对应门
            redirect_rates.append(n_flip_to_wrong / len(results) if results else float("nan"))
            line = f"  主实验[{tname}翻cue] 有效局={len(results)}: 仍选真实cue(success)={sum(r['success'] for r in results)}  走向翻转后错误门={n_flip_to_wrong}"
            print(line); flip_lines.append(line)
        flip_redirect_rate = float(np.mean(redirect_rates)) if redirect_rates else None
        ctrl_success_rate = float(np.mean([r["success"] for r in ctrl_results])) if ctrl_results else None
        ctrl_line = f"  特异性对照[翻wall_ahead] 有效局={len(ctrl_results)}: success={sum(r['success'] for r in ctrl_results)}"
        print(ctrl_line); flip_lines.append(ctrl_line)
        md.append("### 翻转实验\n```\n" + "\n".join(flip_lines) + "\n```\n")
    else:
        print(f"\n[7] 通关率{rate:.1%}<{args.flip_threshold:.0%}，按纪律跳过翻转实验")
        md.append(f"### 翻转实验\n通关率{rate:.1%} < {args.flip_threshold:.0%}，按纪律跳过，如实说明：通关率不足，"
                   f"翻转实验结果不可解读。\n")

    with out_log.open("a", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n---\n\n")

    summary = dict(success_rate=rate, avg_steps=float(np.mean(lens)),
          gate_b_all=res_all, gate_b_mem=res_mem, pred_pos_rate_mem=pred_pos_rate,
          by_cue=by_cue, n_timeout=n_timeout, flip_done=flip_done, ablation=abl_rates,
          flip_redirect_rate=flip_redirect_rate, ctrl_success_rate=ctrl_success_rate)
    print("\n[JSON_SUMMARY]", summary)
    print(f"\n-> Stage C 完成，已追加进 {out_log}")
    return summary


if __name__ == "__main__":
    main()
