# -*- coding: utf-8 -*-
"""两阶段方案 · Stage B —— 冻结世界模型（mem_shape1）+ 冻结 Stage A 蒸馏头，只训全新 actor/critic。

复用 m2b_train_ac.py 验证过的"冻结训策略"范式（替换 Dreamer._train 跳过 wm._train，
其余 ImagBehavior._train 不变）。唯一新增：正式训练前用 scripted controller 播种 N 局
成功示范进 replay buffer，给价值学习一个非零起点。

    环境保留 v3 全部修复：显式使用 memoryS7_cuestart 保证 cue 可见起步、
    actor.entropy=1e-2、v2 谓词（无泄漏）。sym_policy_input='atoms'。

可续跑：中途崩溃/被杀，下次直接跑同一命令会从 latest.pt 续训（不会重新播种/重新引导）。
"""
import pathlib as _pathlib, sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import argparse
import functools
import json
import pathlib
import sys
import time
import traceback
import types

import numpy as np
import torch

HERE = pathlib.Path(__file__).parent
REPO = HERE.parent
sys.path.append(str(HERE))
import dreamer
import tools
import neuraldnf
import envs.minigrid as M
from parallel import Damy
import probe_memory as PM
from memory_nav_scripted import scripted

OUT_DIR = REPO / "runs" / "memory_experiment_3_m3_reproduction"
MEM_SHAPE1_CKPT = REPO / "models" / "memory" / "memory_experiment_3_m3" / "shared_world_model.pt"
DISTILL_HEAD = OUT_DIR / "distill_head_memoryS7.pt"
DISTILL_META = OUT_DIR / "distill_head_meta.json"


def build_config_b(logdir, steps, device):
    config = PM.build_config(["defaults", "minigrid", "minigrid_memory", "minigrid_memory_cuestart",
                               "minigrid_memory_symbolic_nav_v2",
                               "minigrid_memory_symbolic_nav_v2_entropy"])
    config.logdir = str(logdir)
    config.device = device
    config.steps = steps
    config.compile = False
    return config


def bootstrap_fresh(config, obs_space, act_space, logger, wm_ckpt, distill_head_path,
                     distill_meta_path, device):
    agent = dreamer.Dreamer(obs_space, act_space, config, logger, dataset=None).to(device)

    ckpt = torch.load(wm_ckpt, map_location=device, weights_only=False)
    wm_state = {k[len("_wm."):]: v for k, v in ckpt["agent_state_dict"].items()
                if k.startswith("_wm.") and not k.startswith("_wm._shape_head")}
    miss, unexp = agent._wm.load_state_dict(wm_state, strict=False)
    print(f"[bootstrap] 灌入 mem_shape1 WM 权重：missing={len(miss)} unexpected={len(unexp)}", flush=True)
    assert not unexp, f"意外的unexpected keys（结构不匹配）: {unexp[:10]}"
    core_miss = [k for k in miss if "_sym_head" not in k]
    assert not core_miss, f"核心WM权重缺失: {core_miss[:10]}"
    print("  (missing 仅 _sym_head.* —— 预期内：mem_shape1 本无此头，随机初始化占位，下一步替换；"
          "mem_shape1 的 _shape_head.* 已主动丢弃，新agent不需要它)", flush=True)

    with open(distill_meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    distill_head = neuraldnf.SymbolicHead(
        meta["in_dim"], meta["labels"], n_lit=meta["n_lit"], n_conj=meta["n_conj"],
        delta_max=meta["delta_max"], anneal_steps=meta["anneal_steps"],
        mask_label=meta.get("mask_label", ""),
    )
    _sd = torch.load(distill_head_path, map_location=device)
    _miss, _unexp = distill_head.load_state_dict(_sd, strict=False)
    if _miss == ["anneal_steps"] and not _unexp:
        # distill_head_memoryS7.pt 是 anneal_steps buffer 修复之前存的旧文件，
        # 那时 anneal_steps 还是普通属性，state_dict 里本来就没有这个key。
        # 用meta里的真值手动补上（构造时已经传了 anneal_steps=meta["anneal_steps"]，这里只是
        # 确保 load_state_dict 不会因为它是"missing"就报错——真值本来就已经在构造参数里了）。
        distill_head.anneal_steps.fill_(meta["anneal_steps"])
        print(f"[bootstrap] 旧distill_head文件兼容：手动恢复 anneal_steps={meta['anneal_steps']}", flush=True)
    elif _miss or _unexp:
        raise RuntimeError(f"蒸馏头state_dict差异异常，需要人工检查: missing={_miss} unexpected={_unexp}")
    distill_head.updates.fill_(distill_head.anneal_steps)
    distill_head.to(device)

    assert list(agent._wm._sym_head.labels) == meta["labels"], "蒸馏头词表与config的sym_labels不一致"
    agent._wm._sym_head = distill_head
    agent._wm.sym_dim = distill_head.K
    agent._wm._sym_policy_input = "atoms"
    agent.requires_grad_(False)
    print(f"[bootstrap] 已挂上冻结的Stage A蒸馏头（labels={distill_head.labels}），"
          f"世界模型+蒸馏头 requires_grad 全False", flush=True)
    return agent


def bootstrap_resume(config, obs_space, act_space, logger, device, ckpt_path):
    agent = dreamer.Dreamer(obs_space, act_space, config, logger, dataset=None).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    agent.load_state_dict(ckpt["agent_state_dict"])
    tools.recursively_load_optim_state_dict(agent, ckpt["optims_state_dict"])
    agent.requires_grad_(False)
    return agent


def make_frozen_train(agent):
    def _train(self, data):
        metrics = {}
        with torch.no_grad():
            data_p = self._wm.preprocess(data)
            embed = self._wm.encoder(data_p)
            post, _prior = self._wm.dynamics.observe(embed, data_p["action"], data_p["is_first"])
            post = {k: v.detach() for k, v in post.items()}
        start = post
        reward = lambda f, s, a: self._wm.heads["reward"](self._wm.dynamics.get_feat(s)).mode()
        metrics.update(self._task_behavior._train(start, reward)[-1])
        for name, value in metrics.items():
            if name not in self._metrics:
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)
    return types.MethodType(_train, agent)


def collect_demo_episode(config, seed, time_limit):
    """手写scripted controller的一整局，手动拼transition，格式与tools.simulate内部
    的add_to_cache完全一致，可以直接被tools.save_episodes/load_episodes读取。"""
    import envs.wrappers as wrappers
    # emit_labels=True：必须与 dreamer.make_env() 对这份 sym_enabled=True 的 config 产出的
    # 真实训练/prefill episode 同一套 obs schema（含 label_* 字段），否则 sample_episodes 把
    # 播种局和真实局混进同一个训练窗口时，字段集不一致会直接 KeyError 崩溃。
    base = M.MiniGrid("memoryS7_cuestart", mode="train", seed=seed, max_steps=time_limit, emit_labels=True)
    env = wrappers.OneHotAction(base)
    env = wrappers.TimeLimit(env, time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)

    cache = {}
    obs = env.reset()
    t = {k: tools.convert(v) for k, v in obs.items()}
    t["reward"] = 0.0
    t["discount"] = 1.0
    tools.add_to_cache(cache, env.id, t)
    done, r = False, 0.0
    cue_is_key = bool(env.god_state()["cue_is_key"])  # 读一次即可，全局固定，reset后才有效
    for step in range(time_limit + 1):
        g = env.god_state()
        a_idx = scripted(g)
        onehot = np.zeros(5, dtype=np.float32)
        onehot[a_idx] = 1.0
        obs, r, done, info = env.step({"action": onehot})
        transition = {k: tools.convert(v) for k, v in obs.items()}
        transition["action"] = onehot
        # logprob：真实agent(prefill随机策略/训练中的actor)的transition都带这个字段
        # （见Dreamer._policy()的policy_output），scripted controller是确定性选择，
        # 按“概率1的选择”记 ln(1)=0，只为凑齐 schema，不参与任何梯度/损失计算
        transition["logprob"] = np.array(0.0, dtype=np.float32)
        transition["reward"] = r
        transition["discount"] = info.get("discount", np.array(1 - float(done)))
        tools.add_to_cache(cache, env.id, transition)
        if done:
            break
    env_id = env.id
    env.close()
    success = bool(r > 0)
    return cache.get(env_id), success, env_id, cue_is_key


def seed_demonstrations(config, traindir, n_target, seed_start=20000, max_attempts=None):
    """跑 scripted controller 直到收集到 n_target 局成功示范，写进 traindir。
    seed_start 特意避开 Stage A 蒸馏用过的训练池种子起点（20..300+20），减少重叠但仍在训练池
    （TRAIN_SEED_START..+TRAIN_POOL）内、与最终评估用的测试池（0..199, mode=eval）完全隔离。
    """
    max_attempts = max_attempts or int(n_target * 1.5) + 50
    n_success, n_key, n_ball, attempts = 0, 0, 0, 0
    t0 = time.time()
    while n_success < n_target and attempts < max_attempts:
        seed = M.TRAIN_SEED_START + seed_start + attempts
        attempts += 1
        episode, success, env_id, cue_is_key = collect_demo_episode(config, seed, config.time_limit)
        if success:
            tools.save_episodes(traindir, {env_id: episode})
            n_success += 1
            n_key += int(cue_is_key); n_ball += int(not cue_is_key)
            if n_success % 50 == 0:
                print(f"  ...播种 {n_success}/{n_target} 局成功示范（已尝试{attempts}局，"
                      f"用时{time.time()-t0:.0f}s）", flush=True)
    return dict(n_success=n_success, n_attempts=attempts, n_key=n_key, n_ball=n_ball,
                elapsed_sec=time.time() - t0)


def seed_demonstrations_balanced(config, traindir, n_per_cue, seed_start=20000, max_attempts=None):
    """跟 seed_demonstrations 一样，但显式控制 cue=key / cue=ball 各存满 n_per_cue 局才停——
    某一类已经存够之后，同类型的新成功局不再写入replay（避免继续拉偏平衡），只统计尝试数。
    """
    max_attempts = max_attempts or int(n_per_cue * 2 * 1.5) + 100
    n_key, n_ball, attempts = 0, 0, 0
    t0 = time.time()
    while (n_key < n_per_cue or n_ball < n_per_cue) and attempts < max_attempts:
        seed = M.TRAIN_SEED_START + seed_start + attempts
        attempts += 1
        episode, success, env_id, cue_is_key = collect_demo_episode(config, seed, config.time_limit)
        if not success:
            continue
        if cue_is_key and n_key >= n_per_cue:
            continue
        if (not cue_is_key) and n_ball >= n_per_cue:
            continue
        tools.save_episodes(traindir, {env_id: episode})
        if cue_is_key:
            n_key += 1
        else:
            n_ball += 1
        if (n_key + n_ball) % 50 == 0:
            print(f"  ...播种 key={n_key}/{n_per_cue} ball={n_ball}/{n_per_cue}（已尝试{attempts}局，"
                  f"用时{time.time()-t0:.0f}s）", flush=True)
    return dict(n_success=n_key + n_ball, n_attempts=attempts, n_key=n_key, n_ball=n_ball,
                elapsed_sec=time.time() - t0)


@torch.no_grad()
def final_eval(agent, config, n=20):
    wm, behavior = agent._wm, agent._task_behavior
    succ, lengths = [], []
    for seed in range(n):
        env = M.MiniGrid("memoryS7_cuestart", mode="eval", seed=seed, max_steps=config.time_limit, emit_labels=False)
        obs = env.reset()
        latent, action = None, None
        done, r, t = False, 0.0, 0
        for t in range(config.time_limit + 1):
            obs_b = {k: np.array(v)[None] for k, v in obs.items() if not k.startswith("log_")}
            data = wm.preprocess(obs_b)
            embed = wm.encoder(data)
            latent, _ = wm.dynamics.obs_step(latent, action, embed, data["is_first"], sample=False)
            feat = wm.dynamics.get_feat(latent)
            pol = wm.augment(feat)
            action = behavior.actor(pol).mode()
            a_idx = int(torch.argmax(action, dim=-1)[0].item())
            obs, r, done, _ = env.step(a_idx)
            if done:
                break
        env.close()
        succ.append(bool(done and r > 0)); lengths.append(t + 1)
    succ_lengths = [l for l, s in zip(lengths, succ) if s]
    return dict(n=n, n_success=int(sum(succ)), success_rate=float(np.mean(succ)),
                avg_steps_success_only=float(np.mean(succ_lengths)) if succ_lengths else float("nan"),
                avg_steps_all=float(np.mean(lengths)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm-checkpoint", default=str(MEM_SHAPE1_CKPT))
    ap.add_argument("--distill-head", default=str(DISTILL_HEAD))
    ap.add_argument("--distill-meta", default=str(DISTILL_META))
    ap.add_argument("--steps", type=int, default=84000)
    ap.add_argument("--n-demo", type=int, default=400)
    ap.add_argument("--n-demo-per-cue", type=int, default=None,
                     help="设置后改用平衡播种：cue=key/cue=ball各存满这么多局才停，忽略--n-demo")
    ap.add_argument("--seed", type=int, default=None, help="覆盖config.seed（控制actor/critic随机初始化等）")
    ap.add_argument("--device", default=None)
    ap.add_argument("--logdir", default=None, help="覆盖输出目录（冒烟测试用隔离目录，不污染正式run）")
    ap.add_argument("--eval-every", type=int, default=None, help="覆盖config.eval_every（调试用）")
    ap.add_argument("--prefill", type=int, default=None, help="覆盖config.prefill（调试用）")
    args = ap.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    logdir = pathlib.Path(args.logdir) if args.logdir else OUT_DIR
    logdir.mkdir(parents=True, exist_ok=True)

    config = build_config_b(logdir, args.steps, device)
    if args.eval_every is not None:
        config.eval_every = args.eval_every
    if args.prefill is not None:
        config.prefill = args.prefill
    if args.seed is not None:
        config.seed = args.seed
    tools.set_seed_everywhere(config.seed)
    config.traindir = logdir / "train_eps"
    config.evaldir = logdir / "eval_eps"
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    step0 = dreamer.count_steps(config.traindir)
    logger = tools.Logger(logdir, config.action_repeat * step0)

    resume_ckpt = logdir / "latest.pt"
    demo_stats_path = logdir / "demo_seeding_stats.json"
    is_resume = resume_ckpt.exists()

    if not is_resume:
        if args.n_demo_per_cue is not None:
            print(f"[demo-seeding] 用 scripted controller 平衡播种：cue=key/cue=ball 各"
                  f"{args.n_demo_per_cue}局", flush=True)
            demo_stats = seed_demonstrations_balanced(config, config.traindir, args.n_demo_per_cue)
        else:
            print(f"[demo-seeding] 用 scripted controller 播种 {args.n_demo} 局成功示范", flush=True)
            demo_stats = seed_demonstrations(config, config.traindir, args.n_demo)
        print(f"[demo-seeding done] 成功{demo_stats['n_success']}局/尝试{demo_stats['n_attempts']}局 "
              f"| cue=key:{demo_stats['n_key']} cue=ball:{demo_stats['n_ball']} "
              f"| 用时{demo_stats['elapsed_sec']:.0f}s", flush=True)
        with open(demo_stats_path, "w", encoding="utf-8") as f:
            json.dump(demo_stats, f, ensure_ascii=False, indent=2)
    elif demo_stats_path.exists():
        print(f"[resume] 复用已有播种统计 {demo_stats_path}", flush=True)

    train_eps = tools.load_episodes(config.traindir, limit=config.dataset_size)
    eval_eps = tools.load_episodes(config.evaldir, limit=1)
    make = lambda mode, id: dreamer.make_env(config, mode, id)
    train_envs = [Damy(make("train", i)) for i in range(config.envs)]
    eval_envs = [Damy(make("eval", i)) for i in range(config.envs)]
    acts = train_envs[0].action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
    obs_space = train_envs[0].observation_space
    act_space = train_envs[0].action_space

    state = None
    if is_resume:
        print(f"[resume] 发现已有 checkpoint {resume_ckpt}，从此续跑（不重新播种/引导）", flush=True)
        agent = bootstrap_resume(config, obs_space, act_space, logger, device, resume_ckpt)
    else:
        print("[bootstrap] 首次启动：mem_shape1 WM + Stage A 蒸馏头 + 全新 actor/critic", flush=True)
        assert pathlib.Path(args.distill_head).exists(), f"找不到Stage A蒸馏头 {args.distill_head}"
        agent = bootstrap_fresh(config, obs_space, act_space, logger, args.wm_checkpoint,
                                 args.distill_head, args.distill_meta, device)
        prefill = max(0, config.prefill - dreamer.count_steps(config.traindir))
        if prefill > 0:
            print(f"[prefill] 随机策略 {prefill} 步（在已播种示范数据基础上继续填充replay）", flush=True)
            random_actor = tools.OneHotDist(torch.zeros(config.num_actions).repeat(config.envs, 1))

            def random_agent(o, d, s):
                action = random_actor.sample()
                logprob = random_actor.log_prob(action)
                return {"action": action, "logprob": logprob}, None

            state = tools.simulate(random_agent, train_envs, train_eps, config.traindir, logger,
                                    limit=config.dataset_size, steps=prefill)
            logger.step += prefill * config.action_repeat

    agent._train = make_frozen_train(agent)
    agent.requires_grad_(False)
    agent._dataset = dreamer.make_dataset(train_eps, config)

    eval_length_near_cap_history = []
    t_start = time.time()
    stop_reason = "steps_reached"
    try:
        while agent._step < config.steps:
            logger.write()
            if config.eval_episode_num > 0:
                eval_policy = functools.partial(agent, training=False)
                tools.simulate(eval_policy, eval_envs, eval_eps, config.evaldir, logger,
                                is_eval=True, episodes=config.eval_episode_num)
                eval_len = getattr(logger, "last_eval_length", None)
                if eval_len is not None:
                    near_cap = eval_len >= 0.9 * config.time_limit
                    eval_length_near_cap_history.append(near_cap)
                    collapsing = len(eval_length_near_cap_history) >= 2 and all(
                        eval_length_near_cap_history[-2:])
                    if collapsing:
                        print(f"[WARN] eval_length={eval_len:.1f} 连续2次逼近time_limit="
                              f"{config.time_limit}——策略可能正在坍缩", flush=True)
                    logger.scalar("policy_collapse_warning", 1.0 if collapsing else 0.0)
                    logger.write(step=logger.step)
            train_steps = tools.training_chunk(agent._step, config.steps, config.eval_every)
            state = tools.simulate(agent, train_envs, train_eps, config.traindir, logger,
                                    limit=config.dataset_size, steps=train_steps, state=state)
            items_to_save = {"agent_state_dict": agent.state_dict(),
                              "optims_state_dict": tools.recursively_collect_optim_state_dict(agent)}
            tools.atomic_torch_save(items_to_save, logdir / "latest.pt")
            tools.atomic_torch_save(items_to_save, logdir / f"ckpt_{agent._step:07d}.pt")
            elapsed = (time.time() - t_start) / 60
            print(f"[checkpoint] step={agent._step} 已用时 {elapsed:.1f}min，已存 latest.pt", flush=True)
    except Exception:
        with open(logdir / "stageB_error.log", "a", encoding="utf-8") as f:
            f.write(f"\n===== {time.ctime()} =====\n")
            f.write(traceback.format_exc())
        print("!! Stage B 训练循环异常，完整堆栈已存 stageB_error.log", flush=True)
        try:
            tools.atomic_torch_save(
                {"agent_state_dict": agent.state_dict(),
                 "optims_state_dict": tools.recursively_collect_optim_state_dict(agent)},
                logdir / "latest_on_error.pt",
            )
        except Exception:
            pass
        stop_reason = "exception"
        with open(logdir / "stageB_stop_reason.txt", "w", encoding="utf-8") as f:
            f.write(f"exception at step={agent._step}\n")
        raise
    finally:
        for env in train_envs + eval_envs:
            try:
                env.close()
            except Exception:
                pass

    with open(logdir / "stageB_stop_reason.txt", "w", encoding="utf-8") as f:
        f.write(f"{stop_reason} at step={agent._step}, elapsed={(time.time() - t_start) / 60:.1f}min\n")

    print("[final-eval] 训练循环结束，跑最终20局确定性评估（seeds 0..19）", flush=True)
    final = final_eval(agent, config, n=20)
    with open(logdir / "stageB_final_eval.json", "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
    print(f"[final-eval done] success {final['success_rate']:.1%} ({final['n_success']}/{final['n']}) "
          f"avg_steps(通关局) {final['avg_steps_success_only']:.1f}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUT_DIR / "stageB_error.log", "a", encoding="utf-8") as f:
            f.write(f"\n===== {time.ctime()} (top-level) =====\n")
            f.write(traceback.format_exc())
        print("!! Stage B 顶层异常，完整堆栈已存 stageB_error.log", flush=True)
        raise
