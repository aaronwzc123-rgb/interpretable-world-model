"""尸检2：逐谓词接地精度 vs base rate 表。

acc 列：直接读 metrics.jsonl 里已经记录的 ndnf_ground_acc_<label>（训练批次上的真实精度，
不重新跑模型）。
base 列：定义与 example/distill_belief.py 的 base_rate() 一致 —— max(p, 1-p)，p=该谓词
在数据里的真值均值。这里的"数据"取全部已存盘的训练局（logdir/m2rebuild_s0/train_eps/*.npz
的 label_<name> 字段），纯读磁盘，不跑环境、不碰 GPU、不碰模型。
gain = acc - base，按 gain 降序排（跟验收规范的表格约定一致）。

用法：python postmortem_2_grounding_vs_baserate.py
"""
import glob
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent
LABELS = ["has_key", "door_locked", "door_open", "carrying",
          "t_ahead", "t_left", "t_right", "t_reach", "wall_ahead"]


def base_rate(p):
    return max(p, 1 - p)


def main():
    # ---- acc: 最近若干条 metrics.jsonl 训练行的均值（delta 已封顶稳定段，最能代表"学到的样子"）----
    rows = [json.loads(l) for l in open(HERE / "logdir/m2rebuild_s0/metrics.jsonl")]
    train_rows = [r for r in rows if "ndnf_ground_acc_has_key" in r]
    tail = train_rows[-5:]                      # delta=4.0 早已封顶稳定的最后 5 个记录点
    acc = {l: float(np.mean([r[f"ndnf_ground_acc_{l}"] for r in tail])) for l in LABELS}
    acc_final_step = tail[-1]["step"]

    # ---- base rate: 全部已存盘训练局的真值均值（纯读磁盘）----
    files = sorted(glob.glob(str(HERE / "logdir/m2rebuild_s0/train_eps/*.npz")))
    sums = {l: 0.0 for l in LABELS}
    counts = {l: 0 for l in LABELS}
    for f in files:
        d = np.load(f)
        for l in LABELS:
            key = f"label_{l}"
            if key in d:
                v = d[key].ravel()
                sums[l] += float(v.sum())
                counts[l] += len(v)
    base = {l: base_rate(sums[l] / counts[l]) for l in LABELS}
    p_true = {l: sums[l] / counts[l] for l in LABELS}

    print(f"数据来源：acc = metrics.jsonl 最后 5 个训练记录点均值（到 step {acc_final_step}，"
          f"delta 早已封顶）；base = {len(files)} 个已存盘训练局（约 {sum(counts.values())//len(LABELS)} 步/谓词）"
          f"的真值 base rate = max(p,1-p)。")
    print()
    header = f"{'谓词':<14}{'p(真=1)':>9}{'base rate':>11}{'acc(N-DNF)':>12}{'gain':>9}   判读"
    print(header)
    print("-" * len(header))
    rows_out = []
    for l in LABELS:
        gain = acc[l] - base[l]
        if gain > 0.15:
            verdict = "✅ belief 真学到了（明显超过瞎猜）"
        elif gain > 0.03:
            verdict = "弱信号（略超瞎猜）"
        else:
            verdict = "✗ 跟瞎猜没区别（belief 没学到/或该谓词本就极不平衡）"
        rows_out.append((l, p_true[l], base[l], acc[l], gain, verdict))
    rows_out.sort(key=lambda r: -r[4])
    for l, p, b, a, g, v in rows_out:
        print(f"{l:<14}{p:>9.3f}{b:>11.3f}{a:>12.3f}{g:>+9.3f}   {v}")


if __name__ == "__main__":
    main()
