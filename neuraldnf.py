"""Minimal but real Neural DNF (semi-symbolic conj + disj, pix2rule/DNF-MT style)
with delta annealing and threshold-to-logic rule extraction."""
import collections
import warnings
import torch, torch.nn as nn, numpy as np


class SemiSymbolic(nn.Module):
    def __init__(self, in_dim, out_dim, kind):
        super().__init__()
        self.kind = kind
        self.w = nn.Parameter(torch.randn(out_dim, in_dim) * 0.1)

    def forward(self, x, delta):
        absw = self.w.abs()
        maxw = absw.max(dim=1).values            # (out,)
        sumabs = absw.sum(dim=1)                  # (out,)
        s = x @ self.w.t()                        # (B,out)
        if self.kind == "conj":
            bias = maxw - sumabs                  # AND: true only if all required match
        else:
            bias = sumabs - maxw                  # OR: true if any matches
        return torch.tanh(delta * (s + bias))


class NeuralDNF(nn.Module):
    def __init__(self, in_dim, n_conj, out_dim):
        super().__init__()
        self.conj = SemiSymbolic(in_dim, n_conj, "conj")
        self.disj = SemiSymbolic(n_conj, out_dim, "disj")

    def forward(self, x, delta):
        return self.disj(self.conj(x, delta), delta)

    def extract_rules(self, in_names, out_names, w_thr=0.5):
        """Read thresholded weights as logic rules."""
        cw = self.conj.w.detach().cpu().numpy()
        dw = self.disj.w.detach().cpu().numpy()
        conj_defs = {}
        for j in range(cw.shape[0]):
            thr = w_thr * np.abs(cw[j]).max() if np.abs(cw[j]).max() > 1e-6 else 1e9
            lits = []
            for i in range(cw.shape[1]):
                if cw[j, i] > thr:
                    lits.append(in_names[i])
                elif cw[j, i] < -thr:
                    lits.append("not " + in_names[i])
            if lits:
                conj_defs[j] = lits
        rules = []
        for k in range(dw.shape[0]):
            thr = w_thr * np.abs(dw[k]).max() if np.abs(dw[k]).max() > 1e-6 else 1e9
            negative = [j for j in range(dw.shape[1]) if dw[k, j] < -thr]
            if negative:
                warnings.warn(
                    f"Output {out_names[k]!r} has {len(negative)} significant negative "
                    "disjunction weight(s); exported rules are partial because this "
                    "rule format cannot represent them faithfully.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            bodies = [j for j in range(dw.shape[1]) if dw[k, j] > thr and j in conj_defs]
            for j in bodies:
                rules.append(f"{out_names[k]} :- " + ", ".join(conj_defs[j]) + ".")
        return rules


class NDNFHead(nn.Module):
    """训练期挂载的可微 N-DNF 谓词头（**纯读不塑形**）。

    - 输入 = RSSM stoch one-hot 展平（[0,1] → {-1,+1}）；输出 = 各谓词 tanh 值∈[-1,1]。
    - 由 WorldModel 用 **detached** 的 belief 喂入 + 独立优化器训练：梯度只更新本头，
      绝不回传世界模型 → 对基线通关率零风险（对应用户选定的「纯读不塑形」路线）。
    - delta 随更新步退火（0.5→delta_max），末期近似硬阈值，可 `rules()` 抽成 `:-` 规则。
    """

    def __init__(self, in_dim, labels, n_conj=8, delta_max=4.0, anneal_steps=20000):
        super().__init__()
        self.labels = list(labels)
        self.net = NeuralDNF(in_dim, n_conj, len(self.labels))
        self.delta_max = float(delta_max)
        # anneal_steps 注册为 buffer（而非普通属性）：普通属性不会被 state_dict 捕获，
        # 若某处代码在"用与训练时不同的 anneal_steps 重新构造本类，再 load_state_dict"
        # 这种模式下复用checkpoint（例如把一个训练好的头搬进另一个用不同config构造的
        # 容器模块里），普通属性会静默保留"重新构造时"的错误值，导致 current_delta()
        # 算出错误的（通常偏软）delta——2026-08-01 在两阶段蒸馏方案里实测过这个问题：
        # reload 后 delta 从训练时钉死的 4.0 变成 1.06，策略从75%通关率跌到0%，见
        # docs/m3_memoryS7_staged_distill_result 相关记录。注册成 buffer 后
        # load_state_dict 会自动正确恢复它，无需在每个调用点手动补丁。
        self.register_buffer("anneal_steps", torch.tensor(int(anneal_steps), dtype=torch.long))
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))

    def current_delta(self):
        frac = float(self.updates.item()) / max(1, int(self.anneal_steps.item()))
        return 0.5 + (self.delta_max - 0.5) * min(1.0, frac / 0.6)

    def forward(self, stoch, delta=None):
        # stoch: (..., in_dim) in [0,1]。映射到 {-1,+1} 后展平前导维喂 NeuralDNF。
        lead = stoch.shape[:-1]
        x = (2.0 * stoch - 1.0).reshape(-1, stoch.shape[-1])
        d = self.current_delta() if delta is None else delta
        out = self.net(x, d)                                # (N, L) in [-1,1]
        return out.reshape(*lead, len(self.labels))

    def loss(self, stoch, label_dict):
        """返回 (per_step_loss[...], out[...,L], acc_dict)。label_dict[l]: (...,) 值∈{0,1}。"""
        out = self.forward(stoch)                           # (...,L)
        y = torch.stack([2.0 * label_dict[l] - 1.0 for l in self.labels], -1)  # {-1,+1}
        loss = ((out - y) ** 2).mean(-1)                    # (...)
        with torch.no_grad():
            acc = {l: ((out[..., i] > 0) == (y[..., i] > 0)).float().mean().item()
                   for i, l in enumerate(self.labels)}
        return loss, out, acc

    def rules(self, in_names):
        return self.net.extract_rules(in_names, self.labels)


class GroundedHead(nn.Module):
    """接地谓词头（**塑形 belief**）：读 belief(feat) 预测谓词，loss 回传世界模型。

    与 NDNFHead（纯读不塑形/detach）相反：本头参数在 model_opt 内、输入 **不 detach**，
    其 loss 作为一项加进世界模型总损失，强迫 RSSM 把该谓词编码进 belief 并跨时保留。
    带监督掩码（如 cue_known）：只在"信息可知"时刻监督/统计，避免用无信息时刻污染。
    用于 MemoryS7 这类"重构会丢弃的记忆谓词"——纯读读不出来时，必须塑形。
    """

    def __init__(self, in_dim, labels, hidden=256, layers=2, mask_label=""):
        super().__init__()
        self.labels = list(labels)
        self.mask_label = mask_label or ""
        mods, d = [], in_dim
        for _ in range(max(1, layers)):
            mods += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        mods += [nn.Linear(d, len(self.labels))]
        self.net = nn.Sequential(*mods)

    def masked_loss(self, feat, data):
        """返回 (标量 loss, mets)。feat: (B,T,in_dim) 不 detach；data 含 label_<l>/label_<mask>。"""
        logits = self.net(feat)                                             # (B,T,L)
        y = torch.stack([data[f"label_{l}"].squeeze(-1) for l in self.labels], -1)  # (B,T,L)
        bce = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none").mean(-1)  # (B,T)
        mk = f"label_{self.mask_label}"
        if self.mask_label and mk in data:
            m = data[mk].squeeze(-1)                                        # (B,T)∈{0,1}
            denom = m.sum().clamp_min(1.0)
            loss = (bce * m).sum() / denom                                  # 只在可知时刻
        else:
            m, denom, loss = None, None, bce.mean()
        mets = {}
        with torch.no_grad():
            for i, l in enumerate(self.labels):
                corr = ((logits[..., i] > 0) == (y[..., i] > 0.5)).float()
                mets[f"shape_acc_{l}"] = (
                    (corr * m).sum() / denom).item() if m is not None else corr.mean().item()
            if m is not None:
                mets["shape_mask_frac"] = m.mean().item()
        return loss, mets


class SymbolicHead(nn.Module):
    """训练期活在 RSSM 前向里的 N-DNF 谓词层（**塑形 belief + 前馈进策略**）。

    feat →(感知层 Linear+tanh) L 个 literal ∈[-1,1] → NeuralDNF(conj/disj) → K 个**具名原子**∈[-1,1]。
    - 原子被监督到具名谓词（door_open/has_key/…），loss 加进世界模型总损失 → 梯度回传 RSSM，
      强迫 belief 编码这些逻辑原子（与 GroundedHead 同向，但核心是可端到端抽规则的半符号 N-DNF）。
    - 原子（detach）由 ImagBehavior 拼进 actor/critic 输入 → 策略**透过逻辑决策**（混合：连续 feat 仍在）。
    - delta 随更新步退火，末期近硬阈值；`rules()` 端到端抽出 `door_open :- p3, not p7.` 规则。
    """

    def __init__(self, in_dim, labels, n_lit=32, n_conj=8, delta_max=4.0,
                 anneal_steps=40000, mask_label="", loss_type="mse",
                 adaptive_delta=False, adaptive_gate_label=None,
                 adaptive_delay=2000, adaptive_window=50, adaptive_check_every=200,
                 adaptive_tol=0.02):
        super().__init__()
        self.labels = list(labels)
        self.K = len(self.labels)
        self.mask_label = mask_label or ""
        self.loss_type = str(loss_type).lower()
        if self.loss_type not in {"mse", "bce"}:
            raise ValueError(f"Unsupported SymbolicHead loss_type: {loss_type!r}")
        self.perceive = nn.Linear(in_dim, n_lit)            # feat -> literal 预激活
        self.dnf = NeuralDNF(n_lit, n_conj, self.K)
        self.delta_max = float(delta_max)
        # anneal_steps 注册为 buffer（不是普通属性）：见 NDNFHead 同名字段上的注释——
        # 普通属性在"用不同config重新构造本类再load_state_dict"（例如两阶段蒸馏方案
        # 把训练好的头搬进新构造的容器模块）这种模式下不会被正确恢复，导致 delta 算错。
        self.register_buffer("anneal_steps", torch.tensor(int(anneal_steps), dtype=torch.long))
        self.lit_names = [f"p{i}" for i in range(n_lit)]
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))

        # ── v6 自适应 delta（移植 Kitty neural-dnf 2.0.0 的 Monitoring 调度器机制，
        #    不 import 该库——它的 delta 封顶 1.0，我们的语义是 0.5→delta_max）──────
        # 门控规则：初始延迟 adaptive_delay 次更新不调；此后每 adaptive_check_every 次
        # 更新检查一次"最近 adaptive_window 次更新的 gate_label 精度滑窗均值"，
        # ≥ 上次记录的目标精度-容差 → 通过：delta 按 adaptive_rate 棘轮式上调一档，
        # 滑窗均值记为新目标；不通过 → delta 原地不动，下个检查节点再试。
        # adaptive_rate 不是随手取的常数，是从"沿用现有固定退火公式的整体速度"反推：
        # 固定公式里 delta 在 frac=0.6（即 0.6*anneal_steps 次更新）时封顶到 delta_max，
        # 减去初始延迟后按 check_every 换算出"若每次检查都通过"总共要走多少个检查节点，
        # 再据此反解出每节点的增长倍率——保证"一路顺利通过"时与固定退火同速封顶，
        # 但只要中途卡住，delta 就原地等待，不强行推进（这正是 v6 要验证的核心机制）。
        self.adaptive_delta = bool(adaptive_delta)
        self.adaptive_gate_label = adaptive_gate_label or self.labels[0]
        self.adaptive_delay = int(adaptive_delay)
        self.adaptive_window = int(adaptive_window)
        self.adaptive_check_every = int(adaptive_check_every)
        self.adaptive_tol = float(adaptive_tol)
        full_speed_updates = max(1, 0.6 * int(self.anneal_steps.item()) - self.adaptive_delay)
        n_checks = max(1, full_speed_updates / self.adaptive_check_every)
        self.adaptive_rate = (self.delta_max / 0.5) ** (1.0 / n_checks)
        self.register_buffer("delta_value", torch.tensor(0.5))
        self.register_buffer("target_acc", torch.tensor(-1.0))  # -1 = 还没建立基线目标
        self._acc_window = collections.deque(maxlen=self.adaptive_window)
        # metrics.jsonl 里的 sym_gate_pass 等字段会被 dreamer.py 的日志聚合逻辑在每个
        # log_every 窗口内取均值（多次门控事件被平均成一个小数），拿不到逐次的真实决策。
        # 这里额外存一份不聚合、不丢失的完整决策历史，供早报"门控通过/拒绝的完整决策记录"
        # 使用；dreamer.py::main() 会在每个 checkpoint 周期把它整份 dump 成 json（见那边注释）。
        self.gate_log = []

    def current_delta(self):
        if self.adaptive_delta:
            return float(self.delta_value.item())
        frac = float(self.updates.item()) / max(1, int(self.anneal_steps.item()))
        return 0.5 + (self.delta_max - 0.5) * min(1.0, frac / 0.6)

    def _adaptive_step(self, gate_acc_this_batch):
        """每次 masked_loss 调用末尾跑一次；只在检查节点上真正判定门控，其余时刻只是
        把本批次精度记进滑窗。返回值直接并入 masked_loss 的 mets，供写进 metrics.jsonl。

        已知的门控局限（实施时分析清楚、刻意保留，不是遗漏）：这是一个"容忍小幅退步"
        的门（window_acc ≥ 上次目标−tol），不是"要求真正提升"的门。如果 gate_label 精度
        长期完全持平（尤其是已经退化成常量预测、精度锁死在 base rate 附近——v1-v5 的
        cue_is_key 正是这种情况），持平的读数每次都会精确等于"上次目标"，永远满足
        "≥ 上次目标−tol"，delta 依然会一路棘轮硬化，不会被这道门拦住。这不是本次移植的
        bug，是"容忍退步"这条门控规则本身的数学性质，按用户给的四条规则字面实现。
        早报里必须如实检查、报告这一情况是否发生（若发生，是本实验一个独立于三条判读
        分支之外的新发现，见 v6_log.md）。
        """
        self._acc_window.append(float(gate_acc_this_batch))
        upd = int(self.updates.item())
        out = {"sym_gate_checked": 0.0}
        if upd < self.adaptive_delay:
            return out                                  # 初始延迟期，不检查也不调
        if upd % self.adaptive_check_every != 0:
            return out                                  # 未到检查节点
        if len(self._acc_window) < self.adaptive_window:
            return out                                  # 刚过延迟期，滑窗数据还不够
        window_acc = sum(self._acc_window) / len(self._acc_window)
        out["sym_gate_checked"] = 1.0
        out["sym_gate_window_acc"] = window_acc
        if self.target_acc.item() < 0:
            # 首次检查：只建立基线目标，还没有"上次目标"可比较，不算通过也不算拒绝
            self.target_acc.fill_(window_acc)
            out["sym_gate_pass"] = -1.0
            out["sym_gate_target_acc"] = window_acc
            self.gate_log.append(dict(updates=upd, window_acc=window_acc,
                                       target_acc=window_acc, pass_=-1,
                                       delta_after=self.current_delta()))
            return out
        target = float(self.target_acc.item())
        passed = window_acc >= target - self.adaptive_tol
        out["sym_gate_pass"] = 1.0 if passed else 0.0
        out["sym_gate_target_acc"] = target
        if passed:
            new_delta = min(self.delta_max, float(self.delta_value.item()) * self.adaptive_rate)
            self.delta_value.fill_(new_delta)
            self.target_acc.fill_(window_acc)
        self.gate_log.append(dict(updates=upd, window_acc=window_acc, target_acc=target,
                                   pass_=int(passed), delta_after=self.current_delta()))
        return out

    def atoms(self, feat, delta=None):
        """feat: (...,in_dim) → K 个具名原子 (...,K) ∈[-1,1]。"""
        d = self.current_delta() if delta is None else delta
        lit = torch.tanh(d * self.perceive(feat))           # (...,L) ∈[-1,1] literals
        return self.dnf(lit, d)                             # (...,K) ∈[-1,1] atoms

    def masked_loss(self, feat, data):
        """监督原子到具名谓词。feat 不 detach → 梯度塑形 belief。返回 (标量 loss, mets)。"""
        out = self.atoms(feat)                              # (B,T,K)
        y = torch.stack([2.0 * data[f"label_{l}"].squeeze(-1) - 1.0
                         for l in self.labels], -1)          # {-1,+1}
        if self.loss_type == "mse":
            per_step = ((out - y) ** 2).mean(-1)            # (B,T)
        else:
            # The N-DNF output is already tanh-bounded, so map it to a
            # probability rather than incorrectly treating it as a logit.
            prob = ((out + 1.0) * 0.5).clamp(1e-6, 1.0 - 1e-6)
            target = (y + 1.0) * 0.5
            per_step = nn.functional.binary_cross_entropy(
                prob, target, reduction="none"
            ).mean(-1)
        mk = f"label_{self.mask_label}"
        if self.mask_label and mk in data:
            m = data[mk].squeeze(-1); denom = m.sum().clamp_min(1.0)
            loss = (per_step * m).sum() / denom
        else:
            m, denom, loss = None, None, per_step.mean()
        mets = {}
        gate_acc = None
        with torch.no_grad():
            for i, l in enumerate(self.labels):
                corr = ((out[..., i] > 0) == (y[..., i] > 0)).float()
                acc = ((corr * m).sum() / denom).item() if m is not None else corr.mean().item()
                mets[f"sym_acc_{l}"] = acc
                if l == self.adaptive_gate_label:
                    gate_acc = acc
            if m is not None:
                mets["sym_mask_frac"] = m.mean().item()
        if self.adaptive_delta:
            assert gate_acc is not None, self.adaptive_gate_label
            mets.update(self._adaptive_step(gate_acc))
        mets["sym_delta"] = self.current_delta()
        return loss, mets

    def rules(self):
        """端到端把当前权重阈值化成 `谓词 :- lit, not lit.` 规则（读学到的 literal p_i）。"""
        return self.dnf.extract_rules(self.lit_names, self.labels)


def fit_dnf(X, Y, in_names, out_names, n_conj=6, steps=5000, lr=0.03,
            delta_max=4.0, reg=1e-4, seed=0, verbose=False):
    """X: (N,in) in {-1,+1}; Y: (N,out) in {-1,+1}. Returns best-accuracy NeuralDNF."""
    import copy
    torch.manual_seed(seed)
    X = torch.tensor(X, dtype=torch.float32); Y = torch.tensor(Y, dtype=torch.float32)
    net = NeuralDNF(X.shape[1], n_conj, Y.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    best_acc, best_state = -1.0, None
    for s in range(steps):
        frac = s / steps
        delta = 0.5 + (delta_max - 0.5) * min(1.0, frac / 0.6)        # 0.5 -> delta_max by 60%
        for g in opt.param_groups:                                   # lr decay last 30%
            g["lr"] = lr * (0.15 if frac > 0.7 else 1.0)
        out = net(X, delta)
        loss = ((out - Y) ** 2).mean() + reg * sum(p.abs().mean() for p in net.parameters())
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            acc = ((net(X, delta_max) > 0) == (Y > 0)).float().mean().item()
        if frac > 0.4 and acc > best_acc:
            best_acc, best_state = acc, copy.deepcopy(net.state_dict())
        if verbose and s % 1000 == 0:
            print(f"   dnf step {s} loss={loss.item():.3f} acc={acc:.3f} delta={delta:.2f}")
    if best_state is not None:
        net.load_state_dict(best_state)
    net.delta_eval = delta_max
    return net, best_acc


if __name__ == "__main__":
    # self-test: learn  y = (a AND NOT b) OR (c AND d)  over all 16 assignments
    import itertools
    names = ["a", "b", "c", "d"]
    rows = list(itertools.product([1, -1], repeat=4))   # +1=True,-1=False
    X = np.array(rows, dtype=np.float32)
    def target(a, b, c, d):
        A = (a > 0) and (b < 0)
        B = (c > 0) and (d > 0)
        return 1.0 if (A or B) else -1.0
    Y = np.array([[target(*r)] for r in rows], dtype=np.float32)
    net, ba = fit_dnf(X, Y, names, ["y"], n_conj=4, steps=5000, verbose=True)
    acc = ((net(torch.tensor(X), net.delta_eval) > 0) == (torch.tensor(Y) > 0)).float().mean().item()
    print("final acc:", acc)
    print("extracted rules:")
    for r in net.extract_rules(names, ["y"]):
        print("  ", r)
