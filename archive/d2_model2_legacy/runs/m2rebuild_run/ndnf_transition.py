"""N-DNF transition (图3)：用 K 维谓词向量 m_t 替换标准 RSSM 的连续 belief。

    prior 通路（想象，无观测）:
        m_t(±1) ⊕ a_t(±1) --NeuralDNF--> y_prior            ("deter")
    posterior 通路（filtering，用观测）:
        tanh(y_prior) ⊕ enc(embed) --NeuralDNF--> y_post --> 2类logit --> 采样(straight-through) --> m_{t+1}

接口对齐 networks.RSSM 的方法集合（initial/observe/img_step/imagine_with_action/
get_feat/get_dist/kl_loss/obs_step），WorldModel/ImagBehavior 不需要任何改动就能
把 self.dynamics 换成本类。

delta 调度数字（initial=0.1, delay=500, 每 100 步 ×1.1, cap=4.0）字面继承自
model2 的 m2fix_exp1_delta_full/m2fix_exp3_optsplit 已验证配置（见 m2rebuild_log.md），
本文件独立实现，不 import 旧 model2 代码。

采样用 straight-through：直接复用 tools.OneHotDist.sample()，它本来就实现了
hard_sample.detach() + probs - probs.detach()，梯度天然能穿过离散采样，
不需要额外的 ST 技巧。
"""
import torch
from torch import nn
from torch import distributions as torchd

import tools
from neuraldnf import NeuralDNF

# 9 个接地谓词 —— 与 example/envs/minigrid.py 的 LABELS_DOORKEY_NAV 顺序一致
# （god_state() 产出的 label_* 键就是按这个词表来的，直接复用，不重写）。
GROUNDED_PREDICATES = ["has_key", "door_locked", "door_open", "carrying",
                       "t_ahead", "t_left", "t_right", "t_reach", "wall_ahead"]
N_GROUNDED = len(GROUNDED_PREDICATES)

# N-DNF 原始输出无界，直接当 2 类 logit 会在 δ 退火到大值后让 KL 爆炸；
# 用 s·tanh(y/s) 平滑限幅，保号保梯度（做法与 model2/ndnf_rssm.py 一致，
# 因为这是数值稳定性必需，不是"训练基建"，属于新 transition 自身的一部分）。
LOGIT_SCALE = 5.0


class NDNFTransition(nn.Module):
    def __init__(self, n_pred, n_conj, n_enc, embed_size, act_dim, unimix_ratio,
                 delta_initial, delta_delay, delta_steps, delta_rate, delta_cap,
                 device):
        super().__init__()
        assert n_pred > N_GROUNDED, "自由谓词数量必须 > 0"
        self.n_pred = int(n_pred)
        self._act_dim = int(act_dim)
        self._unimix_ratio = float(unimix_ratio)
        self._device = torch.device(device)

        self.prior_ndnf = NeuralDNF(self.n_pred + self._act_dim, n_conj, self.n_pred)
        self.posterior_ndnf = NeuralDNF(self.n_pred + n_enc, n_conj, self.n_pred)
        self.obs_proj = nn.Sequential(
            nn.Linear(embed_size, 128), nn.SiLU(),
            nn.Linear(128, n_enc), nn.Tanh(),
        )

        self._delta_cfg = dict(delay=int(delta_delay), steps=int(delta_steps),
                                rate=float(delta_rate), cap=float(delta_cap))
        self.register_buffer("_delta", torch.tensor(float(delta_initial)))
        self.register_buffer("_counter", torch.zeros((), dtype=torch.long))

    # ══════════ delta 调度 ══════════

    def step_delta(self):
        """每个 world-model train step 调一次。返回调用后的当前 delta（供 metrics 记录）。"""
        self._counter += 1
        c = self._delta_cfg
        ctr = int(self._counter.item())
        if ctr > c["delay"] and (ctr - c["delay"]) % c["steps"] == 0:
            new = min(float(self._delta.item()) * c["rate"], c["cap"])
            self._delta.fill_(new)
        return float(self._delta.item())

    # ══════════ 与 networks.RSSM 一致的接口 ══════════

    def initial(self, batch_size):
        return dict(
            logit=torch.zeros(batch_size, self.n_pred, 2, device=self._device),
            stoch=torch.zeros(batch_size, self.n_pred, 2, device=self._device),
            deter=torch.zeros(batch_size, self.n_pred, device=self._device),
        )

    def observe(self, embed, action, is_first, state=None):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        embed, action, is_first = swap(embed), swap(action), swap(is_first)
        post, prior = tools.static_scan(
            lambda prev_state, prev_act, embed, is_first: self.obs_step(
                prev_state[0], prev_act, embed, is_first
            ),
            (action, embed, is_first),
            (state, state),
        )
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    def imagine_with_action(self, action, state):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        action = swap(action)
        prior = tools.static_scan(self.img_step, [action], state)
        prior = prior[0]
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        if prev_state is None or torch.sum(is_first) == len(is_first):
            prev_state = self.initial(len(is_first))
            prev_action = torch.zeros((len(is_first), self._act_dim), device=self._device)
        elif torch.sum(is_first) > 0:
            is_first_a = is_first[:, None]
            prev_action = prev_action * (1.0 - is_first_a)
            init_state = self.initial(len(is_first))
            for key, val in prev_state.items():
                isf = is_first.reshape(is_first.shape + (1,) * (len(val.shape) - len(is_first.shape)))
                prev_state[key] = val * (1.0 - isf) + init_state[key] * isf

        prior = self.img_step(prev_state, prev_action, sample=sample)
        enc = self.obs_proj(embed)                                      # (B, n_enc)
        y_post = self.posterior_ndnf(
            torch.cat([torch.tanh(prior["deter"]), enc], -1), self._delta.item())
        logit = self._to_logit(y_post)
        dist = self.get_dist({"logit": logit})
        stoch = dist.sample() if sample else dist.mode()
        post = {"stoch": stoch, "deter": prior["deter"], "logit": logit}
        return post, prior

    def img_step(self, prev_state, prev_action, sample=True):
        m = self._bipolar(prev_state["stoch"])                          # (B, P)
        a = prev_action * 2.0 - 1.0                                     # one-hot -> ±1
        deter = self.prior_ndnf(torch.cat([m, a], -1), self._delta.item())  # (B, P) y_prior
        logit = self._to_logit(deter)
        dist = self.get_dist({"logit": logit})
        stoch = dist.sample() if sample else dist.mode()
        return {"stoch": stoch, "deter": deter, "logit": logit}

    def get_feat(self, state):
        stoch = state["stoch"]
        shape = list(stoch.shape[:-2]) + [self.n_pred * 2]
        stoch = stoch.reshape(shape)
        return torch.cat([stoch, torch.tanh(state["deter"])], -1)

    def get_dist(self, state, dtype=None):
        logit = state["logit"]
        return torchd.independent.Independent(
            tools.OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1
        )

    def kl_loss(self, post, prior, free, dyn_scale, rep_scale):
        kld = torchd.kl.kl_divergence
        dist = lambda x: self.get_dist(x)
        sg = lambda x: {k: v.detach() for k, v in x.items()}

        rep_loss = value = kld(dist(post), dist(sg(prior)))
        dyn_loss = kld(dist(sg(post)), dist(prior))
        rep_loss = torch.clip(rep_loss, min=free)
        dyn_loss = torch.clip(dyn_loss, min=free)
        loss = dyn_scale * dyn_loss + rep_scale * rep_loss
        return loss, value, dyn_loss, rep_loss

    # ══════════ 内部 ══════════

    @staticmethod
    def _bipolar(stoch):
        """one-hot (...,P,2) -> ±1 (...,P)；初始全零 -> 0（"未知"）。"""
        return stoch[..., 0] - stoch[..., 1]

    @staticmethod
    def _to_logit(y):
        y = LOGIT_SCALE * torch.tanh(y / LOGIT_SCALE)
        return torch.stack([y, -y], -1)                                 # (..., P, 2)

    # ══════════ 谓词接地（仅监督，真值不喂模型）══════════

    def grounding_loss(self, post_logit, data):
        """9 个接地谓词的 BCE 监督。post_logit: (B,T,P,2)；data 含 label_<name> ∈ {0,1}。
        返回 (标量 loss, mets)。真值只用来算 loss，从不作为模型输入。"""
        y = post_logit[..., :N_GROUNDED, 0] - post_logit[..., :N_GROUNDED, 1]   # (B,T,N_GROUNDED)
        target = torch.stack(
            [data[f"label_{l}"].squeeze(-1) for l in GROUNDED_PREDICATES], -1
        )                                                                       # (B,T,N_GROUNDED) ∈ {0,1}
        bce = nn.functional.binary_cross_entropy_with_logits(y, target, reduction="none")
        loss = bce.mean()
        mets = {"ndnf_ground_loss": float(loss.detach())}
        with torch.no_grad():
            for i, l in enumerate(GROUNDED_PREDICATES):
                mets[f"ndnf_ground_acc_{l}"] = ((y[..., i] > 0) == (target[..., i] > 0.5)).float().mean().item()
        return loss, mets
