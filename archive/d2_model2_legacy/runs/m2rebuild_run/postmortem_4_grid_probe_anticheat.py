"""尸检4（防作弊对照）：当前帧观测(grid, 148维)能不能直接线性判出 9 个谓词？

复用 example/distill_belief.py 的既定规范：按 episode 分 train/test（GroupShuffleSplit，
防止同一局内步与步之间泄漏），在 grid（148维符号观测，模型编码器实际吃的输入）上训逻辑回归
probe，跟 base rate（该谓词在 test 集里的多数类占比）对比。

要钉死的问题：postmortem_2 里 has_key/carrying/door_locked 等"持久性"谓词精度跟 base rate
持平甚至更差，前提假设是"这些谓词当前帧不可观测、必须靠 belief 记忆"——但这个假设本身没验证过。
如果 grid 本身就能判出来（比如钥匙画在 agent 自己格子里，参考 model1/r2dreamer 那边的教训），
那 belief 学不好就不能全赖"记忆"，而是循环本身的噪声连可观测量都没学到，指控更重。

数据来源：logdir/m2rebuild_s0/train_eps/*.npz（跟 postmortem_2 同一批已存盘训练局），
纯读磁盘 + sklearn 逻辑回归，不碰 GPU/dreamer 代码。

用法：python postmortem_4_grid_probe_anticheat.py
"""
import glob
import pathlib

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = pathlib.Path(__file__).parent
LABELS = ["has_key", "door_locked", "door_open", "carrying",
          "t_ahead", "t_left", "t_right", "t_reach", "wall_ahead"]


def base_rate(y, te):
    p = y[te].mean()
    return max(p, 1 - p)


def probe(X, y, groups, seed=0):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    tr, te = next(gss.split(X, y, groups))
    if len(np.unique(y[tr])) < 2:
        return float("nan"), te
    clf = make_pipeline(StandardScaler(with_mean=True), LogisticRegression(max_iter=2000, C=1.0))
    clf.fit(X[tr], y[tr])
    return clf.score(X[te], y[te]), te


def main():
    files = sorted(glob.glob(str(HERE / "logdir/m2rebuild_s0/train_eps/*.npz")))
    print(f"读取 {len(files)} 个已存盘训练局...")

    grids, labels_arr, groups = [], {l: [] for l in LABELS}, []
    for ep_idx, f in enumerate(files):
        d = np.load(f)
        n = len(d["grid"])
        grids.append(d["grid"])
        groups.append(np.full(n, ep_idx))
        for l in LABELS:
            labels_arr[l].append(d[f"label_{l}"].ravel())
    grid = np.concatenate(grids, axis=0).astype(np.float32)
    groups = np.concatenate(groups, axis=0)
    print(f"总步数 {len(grid)}  grid 维度 {grid.shape[1]}  局数 {len(files)}")

    print()
    header = f"{'谓词':<14}{'base rate':>11}{'grid probe acc':>16}{'gain':>9}   判读（当前帧是否可观测）"
    print(header)
    print("-" * len(header))
    rows = []
    for l in LABELS:
        y = np.concatenate(labels_arr[l], axis=0).astype(int)
        acc, te = probe(grid, y, groups, seed=0)
        base = base_rate(y, te)
        gain = acc - base if not np.isnan(acc) else float("nan")
        if np.isnan(acc):
            verdict = "（test 集标签单一，跳过）"
        elif gain > 0.10:
            verdict = "★ 当前帧本就可判（可观测，不需要记忆）"
        else:
            verdict = "当前帧判不出（跟 base rate 一样瞎猜 → 真需要记忆/belief）"
        rows.append((l, base, acc, gain, verdict))
        print(f"{l:<14}{base:>11.3f}{acc:>16.3f}{gain:>+9.3f}   {verdict}")

    print("\n对照 postmortem_2 的 belief(N-DNF) 精度，判断链条上钉死的那颗钉子：")
    print("  - 若某谓词这里★（当前帧可判）且 postmortem_2 里 belief 也学到了 → 不能证明是belief在记忆，只是抄近路")
    print("  - 若某谓词这里★（当前帧可判）但 postmortem_2 里 belief 没学到/更差 → 循环本身在污染可观测量的学习，指控更重")
    print("  - 若某谓词这里不可判 且 postmortem_2 里 belief 也没学到 → 干净的'需要记忆但没学会'（原假设成立）")


if __name__ == "__main__":
    main()
