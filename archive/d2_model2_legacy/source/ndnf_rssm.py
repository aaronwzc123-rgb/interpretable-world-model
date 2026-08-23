"""
ndnf_rssm.py — Model 2: N-DNF 谓词 RSSM (r2dreamer 的 drop-in 替换)

放置位置: r2dreamer/ndnf_rssm.py (与 rssm.py 并列)

设计 (+prior 架构, 对应 methodology Phase 2):
  - belief state = N_PRED 个谓词, 每个是 K=2 的 categorical (true/false)
    → Dreamer 的 OneHotDist 采样 / unimix / kl_loss 机制原样可用
  - deter ≡ Prior N-DNF 的原始输出 y_prior (B, N_PRED)
    → dreamer.py 里 `_, prior_logit = rssm.prior(post_deter)` 天然成立
  - KL(post_logit ‖ prior_logit) = Dreamer 自带的 dyn/rep loss
    = +prior 架构需要的 prior-post 一致性, 免费获得

  数据流 (与 methodology 邮件一致):
    m_t(±1, 从 stoch one-hot) ⊕ a_t(±1) → PriorNDNF → y_prior ("deter")
    tanh(y_prior) ⊕ enc(embed) → PosteriorNDNF → y_post
    y_post → 2类 logits → 采样 → stoch (下一步的 m)

  与原 RSSM 的接口完全一致 (9 个方法), dreamer.py 主体不用改;
  额外提供 grounding_loss / aux_losses / delta 控制 / 规则提取。

前置: uv pip install -e .\\neural-dnf --no-deps
"""

import re

import torch
from torch import distributions as torchd
from torch import nn

import distributions as dists
from neural_dnf import NeuralDNF

# ── 谓词布局 (与 predicate_module.py 保持一致) ────────────
# 前 4 个是状态谓词，后 5 个是导航/方向谓词（当前子目标相对 agent
# 在前方/该左转/该右转/已到位，以及正前方是否有墙）——见 envs/minigrid.py
# 的 _nav_preds，参照 example/ 里另一实现的谓词设计（缺方向信息是
# 之前策略只能靠采样撞运气、argmax 贪心必失败的一个重要原因）。
GROUNDED_PREDICATES = ["has_key", "door_locked", "door_open", "carrying",
                       "t_ahead", "t_left", "t_right", "t_reach", "wall_ahead"]
N_GROUNDED = len(GROUNDED_PREDICATES)
# MiniGrid 动作裁成 5 个 (envs/minigrid.py 的 VALID_ACTIONS = [0,1,2,3,5])，
# 去掉 drop 和硬编码 no-op 的 done。
ACTION_NAMES = ["turn_left", "turn_right", "forward", "pickup", "toggle"]

# logits 平滑截断幅度: N-DNF 原始输出可达 ±(6×n_conj), 直接当 logit 会让
# KL 爆炸; 用 s·tanh(y/s) 平滑限幅, 保号保梯度
LOGIT_SCALE = 5.0


def _rpad(x, n):
    """右侧补维度 (等价 tools.rpad, 内联避免依赖)。"""
    return x.reshape(*x.shape, *([1] * n))


class NDNFRSSM(nn.Module):
    """N-DNF 谓词 belief 的 RSSM, 接口与 rssm.RSSM 一致。"""

    def __init__(self, config, embed_size, act_dim):
        super().__init__()
        self._device = torch.device(config.device)
        self._unimix_ratio = float(config.unimix_ratio)
        self._act_dim = act_dim

        # 谓词配置 (config.ndnf 新增块, 见补丁说明)
        nc = config.ndnf
        self._n_pred = int(nc.n_pred)          # 建议 16 (4 接地 + 12 自由)
        self._n_enc = int(nc.n_enc)            # 观测编码维度, 建议 16
        n_conj_prior = int(nc.n_conj_prior)    # 建议 64
        n_conj_post = int(nc.n_conj_post)      # 建议 48
        initial_delta = float(nc.initial_delta)  # 0.1

        assert self._n_pred > N_GROUNDED, "自由谓词数量必须 > 0"

        # ── 与原 RSSM 对齐的形状约定 ──
        # stoch: (B, N_PRED, 2);  deter: (B, N_PRED)
        self._stoch = self._n_pred
        self._discrete = 2
        self._deter = self._n_pred
        self.flat_stoch = self._n_pred * 2
        self.feat_size = self.flat_stoch + self._deter   # 3 × N_PRED

        # ── Prior N-DNF: m(±1) ⊕ a(±1) → y_prior ──
        self.prior_ndnf = NeuralDNF(
            n_in=self._n_pred + act_dim,
            n_conjunctions=n_conj_prior,
            n_out=self._n_pred,
            delta=initial_delta,
            weight_init_type="normal",
        )
        # ── Posterior N-DNF: tanh(y_prior) ⊕ enc(o) → y_post ──
        self.posterior_ndnf = NeuralDNF(
            n_in=self._n_pred + self._n_enc,
            n_conjunctions=n_conj_post,
            n_out=self._n_pred,
            delta=initial_delta,
            weight_init_type="normal",
        )
        # ── 观测投影: embed → [-1,1]^N_ENC ──
        self._obs_proj = nn.Sequential(
            nn.Linear(embed_size, 128), nn.SiLU(),
            nn.Linear(128, self._n_enc), nn.Tanh(),
        )

        # delta 退火内部状态 (由 dreamer.py 补丁每个 train step 调 step_delta)
        self._delta_cfg = dict(
            delay=int(nc.delta_delay),        # 建议 20000 (train steps)
            steps=int(nc.delta_steps),        # 建议 500
            rate=float(nc.delta_rate),        # 建议 1.1
        )
        self._delta_counter = 0

    # ══════════ 与原 RSSM 一致的 9 个接口 ══════════

    def initial(self, batch_size):
        stoch = torch.zeros(batch_size, self._n_pred, 2,
                            dtype=torch.float32, device=self._device)
        deter = torch.zeros(batch_size, self._n_pred,
                            dtype=torch.float32, device=self._device)
        return stoch, deter

    def observe(self, embed, action, initial, reset):
        L = action.shape[1]
        stoch, deter = initial
        stochs, deters, logits = [], [], []
        for i in range(L):
            stoch, deter, logit = self.obs_step(
                stoch, deter, action[:, i], embed[:, i], reset[:, i])
            stochs.append(stoch)
            deters.append(deter)
            logits.append(logit)
        return (torch.stack(stochs, 1), torch.stack(deters, 1),
                torch.stack(logits, 1))

    def obs_step(self, stoch, deter, prev_action, embed, reset):
        # episode 开头清零 (清零 stoch → bipolar 全 0 = "未知")
        stoch = torch.where(_rpad(reset, stoch.dim() - reset.dim()),
                            torch.zeros_like(stoch), stoch)
        deter = torch.where(_rpad(reset, deter.dim() - reset.dim()),
                            torch.zeros_like(deter), deter)
        prev_action = torch.where(
            _rpad(reset, prev_action.dim() - reset.dim()),
            torch.zeros_like(prev_action), prev_action)

        # 1) Prior N-DNF 转移: y_prior 即新的 "deter"
        deter = self._prior_raw(stoch, prev_action)
        # 2) Posterior N-DNF 观测修正
        enc = self._obs_proj(embed)                        # (B, N_ENC)
        y_post = self.posterior_ndnf(
            torch.cat([torch.tanh(deter), enc], -1))       # (B, P)
        logit = self._to_logit(y_post)                     # (B, P, 2)
        # 3) 直通采样 → one-hot 谓词 (下一步 N-DNF 的 ±1 输入来源)
        stoch = self.get_dist(logit).rsample()
        return stoch, deter, logit

    def img_step(self, stoch, deter, prev_action):
        deter = self._prior_raw(stoch, prev_action)
        stoch, _ = self.prior(deter)
        return stoch, deter

    def prior(self, deter):
        """deter 就是 y_prior, 转 logits 采样即可 (无额外网络)。"""
        logit = self._to_logit(deter)
        stoch = self.get_dist(logit).rsample()
        return stoch, logit

    def imagine_with_action(self, stoch, deter, actions):
        L = actions.shape[1]
        stochs, deters = [], []
        for i in range(L):
            stoch, deter = self.img_step(stoch, deter, actions[:, i])
            stochs.append(stoch)
            deters.append(deter)
        return torch.stack(stochs, 1), torch.stack(deters, 1)

    def get_feat(self, stoch, deter):
        stoch = stoch.reshape(*stoch.shape[:-2], self.flat_stoch)
        # deter 是无界 logit, tanh 限幅后进 heads
        return torch.cat([stoch, torch.tanh(deter)], -1)

    def get_dist(self, logit):
        return torchd.independent.Independent(
            dists.OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1)

    def kl_loss(self, post_logit, prior_logit, free):
        kld = dists.kl
        rep_loss = kld(post_logit, prior_logit.detach()).sum(-1)
        dyn_loss = kld(post_logit.detach(), prior_logit).sum(-1)
        rep_loss = torch.clip(rep_loss, min=free)
        dyn_loss = torch.clip(dyn_loss, min=free)
        return dyn_loss, rep_loss

    # ══════════ 内部 ══════════

    @staticmethod
    def _bipolar(stoch):
        """one-hot (…,P,2) → ±1 (…,P); 全零(初始) → 0。"""
        return stoch[..., 0] - stoch[..., 1]

    def _prior_raw(self, stoch, prev_action):
        m = self._bipolar(stoch)                            # (B, P)
        a = prev_action * 2.0 - 1.0                         # one-hot → ±1
        return self.prior_ndnf(torch.cat([m, a], -1))       # (B, P)

    @staticmethod
    def _to_logit(y):
        """N-DNF 原始输出 → 2类 logits, 平滑限幅防 KL 爆炸。"""
        y = LOGIT_SCALE * torch.tanh(y / LOGIT_SCALE)
        return torch.stack([y, -y], -1)                     # (…, P, 2)

    # ══════════ Model 2 专属: 接地/离散化/delta/规则 ══════════

    def grounding_loss(self, post_logit, gt_preds, mask=None):
        """
        接地谓词辅助监督。
        post_logit: (B, T, P, 2); gt_preds: (B, T, N_GROUNDED) ∈ {-1,+1}
        mask: (B, T) 可选 (is_first 之类不用 mask, 每步都有效)
        """
        # logit_true - logit_false = 2·y → 直接做二分类 logit
        y = (post_logit[..., :N_GROUNDED, 0]
             - post_logit[..., :N_GROUNDED, 1])
        target = (gt_preds + 1) / 2
        bce = nn.functional.binary_cross_entropy_with_logits(
            y, target, reduction="none").mean(-1)           # (B, T)
        if mask is not None:
            return (bce * mask).sum() / mask.sum().clamp(min=1)
        return bce.mean()

    def aux_losses(self):
        """N-DNF 权重离散化正则 (推向 {-6,0,6})。输入已由采样保证 ±1。"""
        def disj_l1(ndnf):
            p = torch.cat([w.view(-1)
                           for w in ndnf.disjunctions.parameters()])
            return torch.abs(p * (6 - torch.abs(p))).mean()
        return (disj_l1(self.prior_ndnf)
                + disj_l1(self.posterior_ndnf)) / 2

    def step_delta(self):
        """每个 world-model train step 调一次 (dreamer.py 补丁)。"""
        self._delta_counter += 1
        c = self._delta_cfg
        if self._delta_counter <= c["delay"]:
            return self.get_delta()
        if (self._delta_counter - c["delay"]) % c["steps"] == 0:
            new = min(self.get_delta() * c["rate"], 4.0)
            self.set_delta(new)
        return self.get_delta()

    def set_delta(self, v):
        self.prior_ndnf.set_delta_val(v)
        self.posterior_ndnf.set_delta_val(v)

    def get_delta(self):
        return self.prior_ndnf.get_delta_val()[0]

    # ── 规则提取 (delta=1 + prune 之后语义才可靠) ──

    def _pred_name(self, i, nxt=False):
        base = (GROUNDED_PREDICATES[i] if i < N_GROUNDED
                else f"free_{i - N_GROUNDED}")
        return f"next_{base}" if nxt else base

    def extract_rules(self, which="prior"):
        from neural_dnf.post_training import extract_asp_rules
        assert which in ("prior", "posterior")
        ndnf = self.prior_ndnf if which == "prior" else self.posterior_ndnf

        in_names = {i: self._pred_name(i) for i in range(self._n_pred)}
        if which == "prior":
            for i in range(self._act_dim):
                nm = (ACTION_NAMES[i] if i < len(ACTION_NAMES)
                      else f"action_{i}")
                in_names[self._n_pred + i] = f"act_{nm}"
        else:
            for i in range(self._n_enc):
                in_names[self._n_pred + i] = f"obs_{i}"
        out_names = {i: self._pred_name(i, nxt=(which == "prior"))
                     for i in range(self._n_pred)}

        rules = extract_asp_rules(ndnf.state_dict())
        readable = []
        for r in rules:
            for idx, name in in_names.items():
                r = re.sub(rf"\ba_{idx}\b", name, r)
            for idx, name in out_names.items():
                r = re.sub(rf"\bdisj_{idx}\b", name, r)
            readable.append(r)
        return readable

    # ── 信念干预 (Phase 3) ──

    def intervene(self, stoch, predicate, value):
        """把 stoch 里指定谓词的 one-hot 改为 value。返回新 tensor。"""
        if predicate in GROUNDED_PREDICATES:
            idx = GROUNDED_PREDICATES.index(predicate)
        elif predicate.startswith("free_"):
            idx = N_GROUNDED + int(predicate.split("_")[1])
        else:
            raise ValueError(f"未知谓词: {predicate}")
        s = stoch.clone()
        s[..., idx, 0] = 1.0 if value else 0.0
        s[..., idx, 1] = 0.0 if value else 1.0
        return s
