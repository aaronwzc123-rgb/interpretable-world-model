import copy
import torch
from torch import nn

import networks
import tools
import neuraldnf

to_np = lambda x: x.detach().cpu().numpy()


class RewardEMA:
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95], device=device)

    def __call__(self, x, ema_vals):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        # this should be in-place operation
        ema_vals[:] = self.alpha * x_quantile + (1 - self.alpha) * ema_vals
        scale = torch.clip(ema_vals[1] - ema_vals[0], min=1.0)
        offset = ema_vals[0]
        return offset.detach(), scale.detach()


class WorldModel(nn.Module):
    def __init__(self, obs_space, act_space, step, config):
        super(WorldModel, self).__init__()
        self._step = step
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        self.encoder = networks.MultiEncoder(shapes, **config.encoder)
        self.embed_size = self.encoder.outdim
        # v5 全量原子回流：只有 sym_enabled 且 sym_recur_atoms 时，img_step 才有原子输入端；
        # 其余情况 atom_dim=0，RSSM 内部逻辑与改动前完全一致（见 networks.py::RSSM.img_step）
        recur_atom_dim = (
            len(config.sym_labels)
            if getattr(config, "sym_enabled", False) and getattr(config, "sym_recur_atoms", False)
            else 0
        )
        if recur_atom_dim > 0:
            # _imagine() 直接把 augment(feat) 的输出喂给 img_step 的原子输入端（维度=recur_atom_dim
            # =len(sym_labels)），这只在 sym_policy_input='atoms'（纯原子瓶颈）时维度对得上；
            # 'concat' 模式下 augment() 输出是 feat+atoms 拼接，维度会错——目前只支持 atoms 模式，
            # 提前断言，不静默喂错维度的数据。
            assert getattr(config, "sym_policy_input", "concat") == "atoms", (
                "sym_recur_atoms=True 目前只支持 sym_policy_input='atoms'（见 _imagine() 的实现假设）"
            )
        self.dynamics = networks.RSSM(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_rec_depth,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_min_std,
            config.unimix_ratio,
            config.initial,
            config.num_actions,
            self.embed_size,
            config.device,
            atom_dim=recur_atom_dim,
        )
        self.heads = nn.ModuleDict()
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.heads["decoder"] = networks.MultiDecoder(
            feat_size, shapes, **config.decoder
        )
        self.heads["reward"] = networks.MLP(
            feat_size,
            (255,) if config.reward_head["dist"] == "symlog_disc" else (),
            config.reward_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head["dist"],
            outscale=config.reward_head["outscale"],
            device=config.device,
            name="Reward",
        )
        self.heads["cont"] = networks.MLP(
            feat_size,
            (),
            config.cont_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist="binary",
            outscale=config.cont_head["outscale"],
            device=config.device,
            name="Cont",
        )
        for name in config.grad_heads:
            assert name in self.heads, name
        # ── 接地谓词头（塑形 belief）──────────────────────────────────────
        # 关键：在 self._model_opt 之前创建，故本头参数进世界模型的 Adam；
        # 输入 feat 不 detach、loss 加进 model_loss → 梯度回传 WM，强迫其编码该谓词。
        # 用于 MemoryS7 这类"重构会丢弃的记忆谓词"（纯读不塑形读不出→必须塑形）。
        self._shape_head = None
        if getattr(config, "shape_enabled", False):
            self._shape_head = neuraldnf.GroundedHead(
                feat_size,
                config.shape_labels,
                hidden=getattr(config, "shape_hidden", 256),
                layers=getattr(config, "shape_layers", 2),
                mask_label=getattr(config, "shape_mask_label", ""),
            )
            self._shape_scale = float(getattr(config, "shape_scale", 1.0))
            print(
                f"Grounded shaping head enabled: {feat_size}->{len(self._shape_head.labels)} "
                f"labels={self._shape_head.labels} mask={self._shape_head.mask_label or 'none'} "
                f"scale={self._shape_scale} (grad FLOWS into WM — shapes belief)"
            )
        # ── 符号 belief 层（N-DNF 原子塑形 belief + 前馈进策略）──────────────
        # 同样在 model_opt 之前创建（参数进 WM Adam）、feat 不 detach、loss 进总损失 → 梯度回传
        # RSSM 强迫 belief 编码逻辑原子；原子（detach）由 ImagBehavior 拼进 actor/critic 输入，
        # 令策略「透过逻辑」决策（混合：连续 feat 仍保留，保住重构/规划容量）。
        self._sym_head = None
        self.sym_dim = 0
        if getattr(config, "sym_enabled", False):
            self._sym_head = neuraldnf.SymbolicHead(
                feat_size,
                config.sym_labels,
                n_lit=getattr(config, "sym_n_lit", 32),
                n_conj=getattr(config, "sym_conj", 8),
                delta_max=getattr(config, "sym_delta_max", 4.0),
                anneal_steps=getattr(config, "sym_anneal", 40000),
                mask_label=getattr(config, "sym_mask_label", ""),
                loss_type=getattr(config, "sym_loss", "mse"),
                adaptive_delta=getattr(config, "sym_adaptive_delta", False),
                adaptive_gate_label=getattr(config, "sym_adaptive_gate_label", None),
                adaptive_delay=getattr(config, "sym_adaptive_delay", 2000),
                adaptive_window=getattr(config, "sym_adaptive_window", 50),
                adaptive_check_every=getattr(config, "sym_adaptive_check_every", 200),
                adaptive_tol=getattr(config, "sym_adaptive_tol", 0.02),
            )
            self.sym_dim = self._sym_head.K
            self._sym_scale = float(getattr(config, "sym_scale", 1.0))
            # 策略输入模式：'concat'=拼在 feat 后（混合，保留连续通道）；
            #             'atoms'=只喂原子（符号瓶颈，策略仅透过逻辑决策）。
            self._sym_policy_input = getattr(config, "sym_policy_input", "concat")
            print(
                f"Symbolic belief head enabled: feat{feat_size}->{self._sym_head.K} atoms "
                f"labels={self._sym_head.labels} n_lit={len(self._sym_head.lit_names)} "
                f"mask={self._sym_head.mask_label or 'none'} loss={self._sym_head.loss_type} "
                f"scale={self._sym_scale} "
                f"(atoms shape WM AND feed policy)"
            )
            if self._sym_head.adaptive_delta:
                print(
                    f"  adaptive delta ON: gate_label={self._sym_head.adaptive_gate_label} "
                    f"delay={self._sym_head.adaptive_delay} window={self._sym_head.adaptive_window} "
                    f"check_every={self._sym_head.adaptive_check_every} "
                    f"tol={self._sym_head.adaptive_tol} rate={self._sym_head.adaptive_rate:.5f}"
                )
        self._model_opt = tools.Optimizer(
            "model",
            self.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )
        print(
            f"Optimizer model_opt has {sum(param.numel() for param in self.parameters())} variables."
        )
        # other losses are scaled by 1.0.
        self._scales = dict(
            reward=config.reward_head["loss_scale"],
            cont=config.cont_head["loss_scale"],
        )
        # ── 训练期 N-DNF 谓词头（纯读不塑形）──────────────────────────────
        # 关键：在 self._model_opt 之后创建，故本头参数不进世界模型的 Adam；
        # 用独立优化器 + detached belief 训练，梯度绝不回传 WM（对基线零风险）。
        self._ndnf = None
        if getattr(config, "ndnf_enabled", False):
            stoch_dim = config.dyn_stoch * config.dyn_discrete
            self._ndnf = neuraldnf.NDNFHead(
                stoch_dim,
                config.ndnf_labels,
                n_conj=config.ndnf_conj,
                anneal_steps=config.ndnf_anneal,
            )
            self._ndnf_opt = tools.Optimizer(
                "ndnf",
                self._ndnf.parameters(),
                config.ndnf_lr,
                config.opt_eps,
                config.grad_clip,
                0.0,
                opt=config.opt,
                use_amp=self._use_amp,
            )
            self._ndnf_names = [
                f"z{u // config.dyn_discrete}_{u % config.dyn_discrete}"
                for u in range(stoch_dim)
            ]
            # 监督/统计掩码：只在 label_<mask> 为真的时刻算 loss/acc（记忆任务用 cue_known）。
            # 空串=不掩码（DoorKey 行为不变）。
            self._ndnf_mask_key = getattr(config, "ndnf_mask_label", "") or ""
            print(
                f"N-DNF head enabled: {stoch_dim}->{len(self._ndnf.labels)} "
                f"labels={self._ndnf.labels} mask={self._ndnf_mask_key or 'none'} "
                f"(reads DETACHED belief, zero-risk to WM)"
            )

    def _train(self, data):
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
        data = self.preprocess(data)

        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                embed = self.encoder(data)
                post, prior = self.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
                kl_free = self._config.kl_free
                dyn_scale = self._config.dyn_scale
                rep_scale = self._config.rep_scale
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                assert kl_loss.shape == embed.shape[:2], kl_loss.shape
                preds = {}
                for name, head in self.heads.items():
                    grad_head = name in self._config.grad_heads
                    feat = self.dynamics.get_feat(post)
                    feat = feat if grad_head else feat.detach()
                    pred = head(feat)
                    if type(pred) is dict:
                        preds.update(pred)
                    else:
                        preds[name] = pred
                losses = {}
                for name, pred in preds.items():
                    loss = -pred.log_prob(data[name])
                    assert loss.shape == embed.shape[:2], (name, loss.shape)
                    losses[name] = loss
                scaled = {
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                }
                model_loss = sum(scaled.values()) + kl_loss
                total_loss = torch.mean(model_loss)
                shape_mets = None
                if self._shape_head is not None:
                    # feat 不 detach → 梯度回传 RSSM/encoder，塑形 belief
                    feat_shape = self.dynamics.get_feat(post)
                    shape_loss, shape_mets = self._shape_head.masked_loss(feat_shape, data)
                    total_loss = total_loss + self._shape_scale * shape_loss
                sym_mets = None
                if self._sym_head is not None:
                    # feat 不 detach → 梯度塑形 belief，令逻辑原子编码进 RSSM
                    feat_sym = self.dynamics.get_feat(post)
                    sym_loss, sym_mets = self._sym_head.masked_loss(feat_sym, data)
                    total_loss = total_loss + self._sym_scale * sym_loss
            metrics = self._model_opt(total_loss, self.parameters())
        if shape_mets is not None:
            metrics.update(shape_mets)
        if sym_mets is not None:
            metrics.update(sym_mets)
        if self._sym_head is not None:
            self._sym_head.updates += 1

        # Store scalar metrics to avoid keeping (batch,time) arrays until the next log step.
        metrics.update(
            {f"{name}_loss": to_np(torch.mean(loss)) for name, loss in losses.items()}
        )
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(torch.mean(dyn_loss))
        metrics["rep_loss"] = to_np(torch.mean(rep_loss))
        metrics["kl"] = to_np(torch.mean(kl_value))
        with torch.cuda.amp.autocast(self._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(post).entropy())
            )
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
            )
        post = {k: v.detach() for k, v in post.items()}
        if self._ndnf is not None:
            metrics.update(self._train_ndnf(post, data))
        return post, context, metrics

    def _train_ndnf(self, post, data):
        """纯读不塑形：输入是已 detach 的 belief，独立优化器只更新 N-DNF 头。"""
        stoch = post["stoch"]                             # (B,T,S,K)，已 detach
        stoch = stoch.reshape(stoch.shape[0], stoch.shape[1], -1)  # (B,T,S*K)
        labels = {l: data[f"label_{l}"].squeeze(-1) for l in self._ndnf.labels}
        # 掩码：只在"信息可知"时刻监督/统计（记忆任务=cue_known）；无掩码则全时刻等权。
        mask = None
        mk = f"label_{self._ndnf_mask_key}"
        if self._ndnf_mask_key and mk in data:
            mask = data[mk].squeeze(-1)                   # (B,T) ∈ {0,1}
        with tools.RequiresGrad(self._ndnf):
            with torch.cuda.amp.autocast(self._use_amp):
                loss_t, out, _acc = self._ndnf.loss(stoch, labels)  # loss_t (B,T), out (B,T,L)
                if mask is not None:
                    m = mask.to(loss_t.dtype)
                    denom = m.sum().clamp_min(1.0)
                    loss = (loss_t * m).sum() / denom
                else:
                    loss = torch.mean(loss_t)
            mets = self._ndnf_opt(loss, self._ndnf.parameters())
        self._ndnf.updates += 1
        mets["ndnf_delta"] = self._ndnf.current_delta()
        # 掩码下重算 acc（只统计信息可知的时刻）
        with torch.no_grad():
            for i, l in enumerate(self._ndnf.labels):
                correct = ((out[..., i] > 0) == (labels[l] > 0.5)).float()
                if mask is not None:
                    mets[f"ndnf_acc_{l}"] = (correct * m).sum().item() / denom.item()
                else:
                    mets[f"ndnf_acc_{l}"] = correct.mean().item()
        if mask is not None:
            mets["ndnf_mask_frac"] = mask.mean().item()   # 信息可知时刻占比
        return mets

    def ndnf_rules(self):
        """把当前 N-DNF 头阈值化成 `谓词 :- ...` 规则（供日志/展示）。"""
        if self._ndnf is None:
            return []
        return self._ndnf.rules(self._ndnf_names)

    def augment(self, feat):
        """actor/critic 输入变换。原子恒 detach（不让策略梯度污染塑形）。
        - 'concat'：concat(feat, atoms) —— 混合，保留连续通道。
        - 'atoms' ：仅 atoms —— 符号瓶颈，策略只透过逻辑决策。
        - 'feat'  ：仅连续 belief —— 在线head诊断时隔离策略反馈。
        未启用符号层时原样返回 feat → 对基线零改动。"""
        if self._sym_head is None:
            return feat
        if self._sym_policy_input == "feat":
            return feat
        atoms = self._sym_head.atoms(feat).detach()
        if self._sym_policy_input == "atoms":
            return atoms
        return torch.cat([feat, atoms], -1)

    def sym_rules(self):
        """端到端把符号 belief 层抽成 `谓词 :- ...` 规则（供日志/展示）。"""
        return self._sym_head.rules() if self._sym_head is not None else []

    # this function is called during both rollout and training
    def preprocess(self, obs):
        obs = {
            k: torch.tensor(v, device=self._config.device, dtype=torch.float32)
            for k, v in obs.items()
        }
        obs["image"] = obs["image"] / 255.0
        if "discount" in obs:
            obs["discount"] *= self._config.discount
            # (batch_size, batch_length) -> (batch_size, batch_length, 1)
            obs["discount"] = obs["discount"].unsqueeze(-1)
        # 'is_first' is necesarry to initialize hidden state at training
        assert "is_first" in obs
        # 'is_terminal' is necesarry to train cont_head
        assert "is_terminal" in obs
        obs["cont"] = (1.0 - obs["is_terminal"]).unsqueeze(-1)
        return obs

    def video_pred(self, data):
        data = self.preprocess(data)
        embed = self.encoder(data)

        states, _ = self.dynamics.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        recon = self.heads["decoder"](self.dynamics.get_feat(states))["image"].mode()[
            :6
        ]
        reward_post = self.heads["reward"](self.dynamics.get_feat(states)).mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine_with_action(data["action"][:6, 5:], init)
        openl = self.heads["decoder"](self.dynamics.get_feat(prior))["image"].mode()
        reward_prior = self.heads["reward"](self.dynamics.get_feat(prior)).mode()
        # observed image is given until 5 steps
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["image"][:6]
        model = model
        error = (model - truth + 1.0) / 2.0

        return torch.cat([truth, model, error], 2)


class ImagBehavior(nn.Module):
    def __init__(self, config, world_model):
        super(ImagBehavior, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self._world_model = world_model
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        # 策略/价值网络输入维度随符号模式变化：
        #   concat → feat+原子；atoms(瓶颈) → 仅原子；feat/无符号层 → feat。
        sym_dim = getattr(world_model, "sym_dim", 0)
        policy_input = getattr(config, "sym_policy_input", "concat")
        if sym_dim and policy_input == "atoms":
            feat_size = sym_dim
        elif sym_dim and policy_input == "concat":
            feat_size += sym_dim
        self.actor = networks.MLP(
            feat_size,
            (config.num_actions,),
            config.actor["layers"],
            config.units,
            config.act,
            config.norm,
            config.actor["dist"],
            config.actor["std"],
            config.actor["min_std"],
            config.actor["max_std"],
            absmax=1.0,
            temp=config.actor["temp"],
            unimix_ratio=config.actor["unimix_ratio"],
            outscale=config.actor["outscale"],
            name="Actor",
        )
        self.value = networks.MLP(
            feat_size,
            (255,) if config.critic["dist"] == "symlog_disc" else (),
            config.critic["layers"],
            config.units,
            config.act,
            config.norm,
            config.critic["dist"],
            outscale=config.critic["outscale"],
            device=config.device,
            name="Value",
        )
        if config.critic["slow_target"]:
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._actor_opt = tools.Optimizer(
            "actor",
            self.actor.parameters(),
            config.actor["lr"],
            config.actor["eps"],
            config.actor["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer actor_opt has {sum(param.numel() for param in self.actor.parameters())} variables."
        )
        self._value_opt = tools.Optimizer(
            "value",
            self.value.parameters(),
            config.critic["lr"],
            config.critic["eps"],
            config.critic["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer value_opt has {sum(param.numel() for param in self.value.parameters())} variables."
        )
        if self._config.reward_EMA:
            # register ema_vals to nn.Module for enabling torch.save and torch.load
            self.register_buffer(
                "ema_vals", torch.zeros((2,), device=self._config.device)
            )
            self.reward_ema = RewardEMA(device=self._config.device)

    def _train(
        self,
        start,
        objective,
    ):
        self._update_slow_target()
        metrics = {}

        with tools.RequiresGrad(self.actor):
            with torch.cuda.amp.autocast(self._use_amp):
                imag_feat, imag_state, imag_action = self._imagine(
                    start, self.actor, self._config.imag_horizon
                )
                reward = objective(imag_feat, imag_state, imag_action)
                actor_ent = self.actor(self._world_model.augment(imag_feat)).entropy()
                state_ent = self._world_model.dynamics.get_dist(imag_state).entropy()
                # this target is not scaled by ema or sym_log.
                target, weights, base = self._compute_target(
                    imag_feat, imag_state, reward
                )
                actor_loss, mets = self._compute_actor_loss(
                    imag_feat,
                    imag_action,
                    target,
                    weights,
                    base,
                )
                actor_loss -= self._config.actor["entropy"] * actor_ent[:-1, ..., None]
                actor_loss = torch.mean(actor_loss)
                metrics.update(mets)
                value_input = imag_feat

        with tools.RequiresGrad(self.value):
            with torch.cuda.amp.autocast(self._use_amp):
                value = self.value(self._world_model.augment(value_input[:-1]).detach())
                target = torch.stack(target, dim=1)
                # (time, batch, 1), (time, batch, 1) -> (time, batch)
                value_loss = -value.log_prob(target.detach())
                slow_target = self._slow_value(
                    self._world_model.augment(value_input[:-1]).detach())
                if self._config.critic["slow_target"]:
                    value_loss -= value.log_prob(slow_target.mode().detach())
                # (time, batch, 1), (time, batch, 1) -> (1,)
                value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])

        metrics.update(tools.tensorstats(value.mode(), "value"))
        metrics.update(tools.tensorstats(target, "target"))
        metrics.update(tools.tensorstats(reward, "imag_reward"))
        if self._config.actor["dist"] in ["onehot"]:
            metrics.update(
                tools.tensorstats(
                    torch.argmax(imag_action, dim=-1).float(), "imag_action"
                )
            )
        else:
            metrics.update(tools.tensorstats(imag_action, "imag_action"))
        metrics["actor_entropy"] = to_np(torch.mean(actor_ent))
        with tools.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            metrics.update(self._value_opt(value_loss, self.value.parameters()))
        return imag_feat, imag_state, imag_action, weights, metrics

    def _imagine(self, start, policy, horizon):
        dynamics = self._world_model.dynamics
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}

        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            inp = self._world_model.augment(feat).detach()
            action = policy(inp).sample()
            # v5 全量原子回流：sym_policy_input='atoms' 时 inp 就是本步 atoms(t)，直接喂给
            # img_step 的原子输入端，让想象轨迹里的记忆和真实环境路径一样逐步保持
            # （atom_dim=0 时 RSSM 忽略这个参数，行为与改动前完全一致）
            succ = dynamics.img_step(state, action, atoms=inp)
            return succ, feat, action

        succ, feats, actions = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None)
        )
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}

        return feats, states, actions

    def _compute_target(self, imag_feat, imag_state, reward):
        if "cont" in self._world_model.heads:
            inp = self._world_model.dynamics.get_feat(imag_state)
            discount = self._config.discount * self._world_model.heads["cont"](inp).mean
        else:
            discount = self._config.discount * torch.ones_like(reward)
        value = self.value(self._world_model.augment(imag_feat)).mode()
        target = tools.lambda_return(
            reward[1:],
            value[:-1],
            discount[1:],
            bootstrap=value[-1],
            lambda_=self._config.discount_lambda,
            axis=0,
        )
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()
        return target, weights, value[:-1]

    def _compute_actor_loss(
        self,
        imag_feat,
        imag_action,
        target,
        weights,
        base,
    ):
        metrics = {}
        inp = self._world_model.augment(imag_feat).detach()
        policy = self.actor(inp)
        # Q-val for actor is not transformed using symlog
        target = torch.stack(target, dim=1)
        if self._config.reward_EMA:
            offset, scale = self.reward_ema(target, self.ema_vals)
            normed_target = (target - offset) / scale
            normed_base = (base - offset) / scale
            adv = normed_target - normed_base
            metrics.update(tools.tensorstats(normed_target, "normed_target"))
            metrics["EMA_005"] = to_np(self.ema_vals[0])
            metrics["EMA_095"] = to_np(self.ema_vals[1])
        else:
            adv = target - base

        if self._config.imag_gradient == "dynamics":
            actor_target = adv
        elif self._config.imag_gradient == "reinforce":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(
                    self._world_model.augment(imag_feat[:-1])).mode()).detach()
            )
        elif self._config.imag_gradient == "both":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(
                    self._world_model.augment(imag_feat[:-1])).mode()).detach()
            )
            mix = self._config.imag_gradient_mix
            actor_target = mix * target + (1 - mix) * actor_target
            metrics["imag_gradient_mix"] = mix
        else:
            raise NotImplementedError(self._config.imag_gradient)
        actor_loss = -weights[:-1] * actor_target
        return actor_loss, metrics

    def _update_slow_target(self):
        if self._config.critic["slow_target"]:
            if self._updates % self._config.critic["slow_target_update"] == 0:
                mix = self._config.critic["slow_target_fraction"]
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1
