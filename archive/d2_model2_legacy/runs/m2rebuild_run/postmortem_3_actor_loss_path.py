"""尸检3：actor 损失数值路径解剖。只读 latest.pt，不训练、不碰优化器。

加载 checkpoint 后，从一段真实 eval rollout 里取一个真实 belief 状态做想象起点，跑一次
ImagBehavior._imagine + _compute_target + _compute_actor_loss（跟训练时调用的是同一份代码，
只是不调用 optimizer.step()），逐环节打印：
  imag_reward 分布 → value 估计分布 → advantage(target-base，EMA 归一化前后各一份)分布
  → actor_loss.backward() 之后，actor 网络 与 dynamics.prior_ndnf/posterior_ndnf 的梯度范数

用来分辨：
  a) advantage≈0（想象里到处是零回报，奖励饥饿）
  b) advantage 不小但梯度穿不过硬化的 prior（结构性问题）
  c) 原子/belief 输入本身饱和到少数几种模式（方差探针）

关键结构性事实（跑之前就能从 config 读出来，不用等数字）：本任务 imag_gradient="reinforce"
（继承自 minigrid 配置，m2rebuild 没有覆盖），REINFORCE 估计器的 actor_loss 只通过
`policy.log_prob(action)` 回传到 actor 网络本身，天然不经过 dynamics（跟 imag_gradient="dynamics"
模式不同）。所以"梯度穿不过硬化 prior"(b) 在当前配置下无论如何都不会是 actor 侧的直接死因——
但下面仍然实测验证这一点，而不是只靠读代码推断。

用法：python postmortem_3_actor_loss_path.py --checkpoint logdir/m2rebuild_s0/latest.pt
"""
import argparse
import pathlib
import sys
import tempfile

import numpy as np
import ruamel.yaml as yaml
import torch

HERE = pathlib.Path(__file__).parent
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
    cfg = yaml.safe_load((HERE / "configs.yaml").read_text(encoding="utf-8"))
    defaults = {}
    for name in overlays:
        recursive_update(defaults, cfg[name])
    import argparse as _argparse
    parser = _argparse.ArgumentParser()
    for k, v in sorted(defaults.items(), key=lambda x: x[0]):
        t = tools.args_type(v)
        parser.add_argument(f"--{k}", type=t, default=t(v))
    return parser.parse_args([])


def tstat(x, name):
    x = x.detach().cpu().numpy().ravel()
    print(f"  {name:<22} mean {x.mean():+.5f}  std {x.std():.5f}  min {x.min():+.5f}  max {x.max():+.5f}")


def grad_norm(module):
    sq = 0.0
    n = 0
    for p in module.parameters():
        if p.grad is not None:
            sq += float(p.grad.detach().pow(2).sum())
            n += p.numel()
    return sq ** 0.5, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(HERE / "logdir/m2rebuild_s0/latest.pt"))
    ap.add_argument("--task", default="doorkey6x6")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup_steps", type=int, default=48,
                     help="先用真实 checkpoint 策略跑这么多步，取最后的 belief 做想象起点")
    ap.add_argument("--n_parallel", type=int, default=256,
                     help="把起点复制成多少条并行想象轨迹（想象本身仍是随机采样，给点统计量）")
    args = ap.parse_args()

    config = build_config(["defaults", "minigrid", "m2rebuild"])
    config.device = "cpu"
    config.num_actions = 5
    config.compile = False

    base_env = M.MiniGrid(args.task, mode="eval", seed=args.seed,
                           max_steps=config.time_limit, render_obs=False, emit_labels=True)
    act_space = wrappers.OneHotAction(base_env).action_space
    obs_space = base_env.observation_space
    logger = tools.Logger(pathlib.Path(tempfile.mkdtemp()), 0)
    agent = dreamer.Dreamer(obs_space, act_space, config, logger, dataset=None).to("cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    agent.load_state_dict(ckpt["agent_state_dict"])

    print(f"[结构性事实] config.imag_gradient = {config.imag_gradient!r} "
          f"(不是 'dynamics' → actor 的 REINFORCE 梯度按设计就不经过 dynamics)")
    print(f"[已加载] {args.checkpoint}")

    wm, beh = agent._wm, agent._task_behavior

    # ---- 用真实 checkpoint 策略跑 warmup_steps 步，取一个真实（非零初始）belief 起点 ----
    obs = base_env.reset()
    latent = action = None
    with torch.no_grad():
        for _ in range(args.warmup_steps):
            obs_b = {k: np.array(v)[None] for k, v in obs.items() if not k.startswith("log_")}
            data = wm.preprocess(obs_b)
            embed = wm.encoder(data)
            latent, _ = wm.dynamics.obs_step(latent, action, embed, data["is_first"], sample=False)
            feat = wm.dynamics.get_feat(latent)
            action = beh.actor(wm.augment(feat)).sample()
            a_idx = int(torch.argmax(action, dim=-1)[0].item())
            obs, r, done, _ = base_env.step(a_idx)
            if done:
                obs = base_env.reset()
                latent = action = None
    base_env.close()
    print(f"[想象起点] 真实 rollout {args.warmup_steps} 步后的 belief（非零初始态）")

    # ---- 复制成 n_parallel 条起点，跑一次跟训练时同样的 _imagine/_compute_target/_compute_actor_loss ----
    start = {k: v[None].expand(args.n_parallel, *v.shape).reshape(args.n_parallel, *v.shape[1:])
             for k, v in latent.items()}
    # _imagine 期望 start 形状是 (B,T,...)，内部会 flatten 成 (B*T,...)；这里给 T=1。
    start = {k: v[:, None] for k, v in start.items()}

    agent.requires_grad_(True)
    horizon = config.imag_horizon
    imag_feat, imag_state, imag_action = beh._imagine(start, beh.actor, horizon)
    reward = wm.heads["reward"](wm.dynamics.get_feat(imag_state)).mode()
    actor_ent = beh.actor(wm.augment(imag_feat)).entropy()
    target, weights, base = beh._compute_target(imag_feat, imag_state, reward)
    actor_loss, mets = beh._compute_actor_loss(imag_feat, imag_action, target, weights, base)
    actor_loss = actor_loss - config.actor["entropy"] * actor_ent[:-1, ..., None]
    actor_loss = torch.mean(actor_loss)

    target_stacked = torch.stack(target, dim=1)
    raw_adv = target_stacked - base                      # EMA 归一化之前的原始 advantage

    print("\n===== 想象轨迹数值分布（跨 %d 条并行轨迹 × %d 步 horizon）=====" % (args.n_parallel, horizon))
    tstat(reward, "imag_reward")
    tstat(target_stacked, "target (lambda-return)")
    tstat(base, "value baseline")
    tstat(raw_adv, "advantage RAW (target-base，归一化前)")
    if "normed_target" in mets:
        print(f"  {'normed_target':<22} mean {mets['normed_target_mean']:+.5f}  std {mets['normed_target_std']:.5f}"
              if "normed_target_mean" in mets else f"  normed_target mets: {mets.get('normed_target', 'n/a')}")
    print(f"  EMA_005={mets.get('EMA_005', float('nan')):.5f}  EMA_095={mets.get('EMA_095', float('nan')):.5f}"
          f"  (reward_ema 的 5%/95% 分位，training 里持续更新到现在的状态)")
    print(f"  actor_entropy mean = {float(actor_ent.mean()):.4f}  (ln(5)={np.log(5):.4f} 是最大熵)")
    print(f"  actor_loss (scalar, 未反传前) = {float(actor_loss):.6f}")

    # ---- 反传，看梯度落在哪 ----
    for p in agent.parameters():
        if p.grad is not None:
            p.grad = None
    actor_loss.backward()
    ga, na = grad_norm(beh.actor)
    gp, npp = grad_norm(wm.dynamics.prior_ndnf)
    gpo, npo = grad_norm(wm.dynamics.posterior_ndnf)
    gob, nob = grad_norm(wm.dynamics.obs_proj)
    print("\n===== actor_loss.backward() 后的梯度范数 =====")
    print(f"  actor 网络            grad_norm={ga:.6f}  ({na} 参数)")
    print(f"  dynamics.prior_ndnf   grad_norm={gp:.6f}  ({npp} 参数)  <- 应为 0（reinforce 不经过 dynamics）")
    print(f"  dynamics.posterior_ndnf grad_norm={gpo:.6f}  ({npo} 参数)  <- 应为 0（同上）")
    print(f"  dynamics.obs_proj     grad_norm={gob:.6f}  ({nob} 参数)  <- 应为 0（同上）")


if __name__ == "__main__":
    main()
