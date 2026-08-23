"""
distill_belief.py — 阶段1②：从 belief 里读出谓词 + 防作弊对照。

核心问题：belief 是不是真的"记住了看不见的隐藏状态"(尤其 has_key —— 钥匙拿手上时画面里看不到)？
做法(对照《验收规范》§3/§5)：
  对每个谓词训两个线性 probe：
    主 probe   : 输入 = belief(feat=stoch+deter)         —— 用了历史记忆
    对照 probe : 输入 = grid(当前帧符号观察, 148维)        —— 只看当下，无记忆
  并排比 base rate(瞎猜常见类)。判读：
    主 probe ≫ 对照 probe ≈ base rate  →  belief 真从历史推断了隐藏态 ✅
  按 episode 划分 train/test(防止同一局的步泄漏)，报 test 精度，诚实。

可选 --ndnf：在最有信息量的若干 stoch 单元上跑真正的 N-DNF(neuraldnf.fit_dnf)，抽出 `:-` 逻辑规则。

用法：
  .\.venv\Scripts\python.exe distill_belief.py
  .\.venv\Scripts\python.exe distill_belief.py --ndnf
"""
import argparse
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupShuffleSplit

HERE = pathlib.Path(__file__).parent
BINARY_LABELS = ["has_key", "door_locked", "door_open", "carrying"]


def probe(X, y, groups, seed=0):
    """按 episode 分 train/test 的线性 probe，返回 test 精度。"""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    tr, te = next(gss.split(X, y, groups))
    if len(np.unique(y[tr])) < 2:           # 标签在训练集是常数 → 无法学
        return float("nan"), tr, te
    clf = make_pipeline(StandardScaler(with_mean=True),
                        LogisticRegression(max_iter=2000, C=1.0))
    clf.fit(X[tr], y[tr])
    return clf.score(X[te], y[te]), tr, te


def base_rate(y, te):
    p = y[te].mean()
    return max(p, 1 - p)


# MiniGrid OBJECT_TO_IDX: door=4, key=5。grid 通道0 = obj/10。
REL_VIS = {"has_key": 5, "carrying": 5, "door_locked": 4, "door_open": 4}


def visible_mask(grid, obj_idx):
    """每步：该物体是否出现在当前 7×7 视野里。"""
    img = grid[:, :147].reshape(-1, 7, 7, 3)
    obj = np.round(img[..., 0] * 10).astype(int)
    return (obj == obj_idx).any(axis=(1, 2))


def restricted_analysis(d):
    """只取『相关物体不在当前视野』的步，重做对照——直击位置泄漏假说。"""
    feat, grid, groups = d["feat"], d["grid"], d["episode"]
    print("\n===== 受限 probe：只取『相关物体不在当前视野』的步 =====")
    print(f"{'谓词':<14}{'#隐藏步':>8}{'base':>8}{'对照(当前帧)':>14}{'主(belief)':>12}   判读")
    for lab, obj_idx in REL_VIS.items():
        mask = ~visible_mask(grid, obj_idx)
        y = d[lab].astype(int)
        yy, gg = y[mask], groups[mask]
        if mask.sum() < 40 or len(np.unique(yy)) < 2 or len(np.unique(gg)) < 4:
            print(f"{lab:<14}{int(mask.sum()):>8}   (隐藏步太少/标签单一，跳过)")
            continue
        am, tr, te = probe(feat[mask], yy, gg)
        ac, _, _ = probe(grid[mask], yy, gg)
        base = base_rate(yy, te)
        if not np.isnan(am) and am - max(ac, base) > 0.1:
            verdict = "✅ belief 记忆(对照≈base)"
        elif ac - base > 0.1:
            verdict = "仍被当前帧决定(位置泄漏)"
        else:
            verdict = "belief 也没编码"
        print(f"{lab:<14}{int(mask.sum()):>8}{base:>8.2%}{ac:>14.2%}{am:>12.2%}   {verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default=str(HERE / "results" / "belief_doorkey6x6" / "trajectory.npz"))
    ap.add_argument("--ndnf", action="store_true", help="额外跑真正的 N-DNF 抽逻辑规则")
    ap.add_argument("--ndnf_label", default="has_key")
    ap.add_argument("--ndnf_topk", type=int, default=12, help="选最有信息量的 K 个 stoch 单元喂 N-DNF")
    args = ap.parse_args()

    d = np.load(args.traj)
    feat = d["feat"]                      # belief：stoch+deter (1280)
    grid = d["grid"]                      # 当前帧观察 (148)
    groups = d["episode"]
    n = len(feat)
    print(f"样本数 {n}  | feat {feat.shape[1]}  grid {grid.shape[1]}  | "
          f"通关率 {d['success_per_ep'].mean():.1%} ({d['success_per_ep'].sum()}/{len(d['success_per_ep'])})\n")

    # ── 主表：每个谓词，主 probe(belief) vs 对照 probe(当前帧) vs base rate ──
    print(f"{'谓词':<14}{'base':>7}{'对照(当前帧)':>14}{'主(belief)':>12}{'gain':>8}   判读")
    rows = []
    for lab in BINARY_LABELS:
        y = d[lab].astype(int)
        acc_main, tr, te = probe(feat, y, groups, seed=0)
        acc_ctrl, _, _ = probe(grid, y, groups, seed=0)
        base = base_rate(y, te)
        gain = acc_main - base
        # 判读：主明显高于对照且对照≈base → belief 真推断了隐藏态
        verdict = ""
        if not np.isnan(acc_main):
            if acc_main - max(acc_ctrl, base) > 0.1:
                verdict = "✅ belief 推断出来的"
            elif acc_ctrl - base > 0.1:
                verdict = "（当前帧本就可见）"
            else:
                verdict = "（belief 没怎么编码）"
        rows.append((lab, base, acc_ctrl, acc_main, gain, verdict))
        print(f"{lab:<14}{base:>7.2%}{acc_ctrl:>14.2%}{acc_main:>12.2%}{gain:>+8.2%}   {verdict}")

    # ── 防作弊柱状图（以 has_key 为主角）──
    fig, ax = plt.subplots(figsize=(6, 4))
    labs = [r[0] for r in rows]
    x = np.arange(len(labs))
    w = 0.27
    ax.bar(x - w, [r[1] for r in rows], w, label="base rate", color="#bbb")
    ax.bar(x,     [r[2] for r in rows], w, label="control (current frame)", color="#e8a")
    ax.bar(x + w, [r[3] for r in rows], w, label="main (belief)", color="#5a8")
    ax.set_xticks(x); ax.set_xticklabels(labs)
    ax.set_ylabel("test accuracy"); ax.set_ylim(0, 1.05)
    ax.set_title("Anti-cheat: belief inference vs current-frame leakage")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = pathlib.Path(args.traj).parent / "anticheat_bars.png"
    plt.savefig(out, dpi=140); plt.close()
    print(f"\n→ 对照柱状图已存 {out}")
    any_memory = any((r[3] - max(r[2], r[1])) > 0.1 for r in rows if not np.isnan(r[3]))
    if any_memory:
        print("判读：存在『主 probe ≫ 对照(当前帧)≈base』的谓词 → belief 确从历史推断了隐藏态 ✅")
    else:
        print("判读：所有谓词的『对照(当前帧)』都和主 probe 一样高 → 当前帧已足够决定这些事实。")
        print("      => 本任务对这些谓词其实是『可观察』的，无法证明 belief 在做记忆推断。")
        print("      => 要干净地证明 belief=记忆，需要有感知混淆/记忆需求的环境(见文档建议)。")

    # ── 受限 probe 诊断：位置泄漏假说 ──
    restricted_analysis(d)

    # ── 可选：真正的 N-DNF 抽逻辑规则 ──
    if args.ndnf:
        run_ndnf(d, args.ndnf_label, args.ndnf_topk)


def run_ndnf(d, label, topk):
    import neuraldnf
    print(f"\n===== N-DNF 蒸馏：从 stoch 谓词推 {label} =====")
    stoch = d["stoch"].reshape(len(d["stoch"]), -1)     # (N, 1024) one-hot {0,1}
    y = d[label].astype(int)
    # 选最有信息量的 topk 个 stoch 单元(按与标签的|相关|)
    yc = y - y.mean()
    corr = np.abs((stoch - stoch.mean(0)).T @ yc) / (len(y) + 1e-6)
    sel = np.argsort(-corr)[:topk]
    names = [f"z{u//32}_{u%32}" for u in sel]            # z<组>_<类>
    X = np.where(stoch[:, sel] > 0.5, 1.0, -1.0).astype(np.float32)   # → ±1
    Y = np.where(y > 0.5, 1.0, -1.0).astype(np.float32)[:, None]
    net, acc = neuraldnf.fit_dnf(X, Y, names, [label], n_conj=4, steps=4000, seed=0)
    print(f"N-DNF 拟合精度(train all): {acc:.2%}  （选了 {topk} 个 stoch 单元）")
    rules = net.extract_rules(names, [label])
    if rules:
        print("抽出的逻辑规则：")
        for r in rules:
            print("   ", r)
    else:
        print("（阈值化后没抽出规则——可调 topk / n_conj / steps）")


if __name__ == "__main__":
    main()
