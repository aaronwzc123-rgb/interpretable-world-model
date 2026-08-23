# m2rebuild 实验日志

目标：以 `example/` 复刻成功的 Dreamer 代码为底座，唯一改动把 RSSM 换成 N-DNF transition（图3），
其余训练基建原样继承，单变量实验。详见任务说明与确认记录（本文件只记设计决定/结果/判读）。

## 2026-07-18 · 侦察 + 底座确认

【实锤】当前无训练在跑（`Get-Process python` 为空），8GB 显存空闲。

【实锤】`example/` 是 dreamerv3-torch 旧版风格代码（非 r2dreamer），三个优化器
（`_model_opt` lr=1e-4 / `_actor_opt` lr=3e-5 / `_value_opt` lr=3e-5，
[configs.yaml:50-52](../example/configs.yaml#L50)）本来就是分开的，跟 model2 已修的
`m2fix_exp3_optsplit_s0` 数字一致，互相印证。9 谓词词表 + `god_state()` + 5 动作空间
已在 `envs/minigrid.py` 现成实现，直接复用。

【推断】delta 调度（initial=0.1, delay=500, 每100步×1.1, cap=4.0）不存在于 example 自己的
N-DNF 头里（那边用的是比例退火公式，见 `neuraldnf.py:74`），而是精确匹配 model2 已验证的
`m2fix_exp1_delta_full_s0`/`m2fix_exp3_optsplit_s0` 配置。已与用户确认：字面沿用这组数字，
独立实现（不 import model2），选它是为了跟 m2fix 系列可比。

按 500+100*n 步、0.1*1.1^n≥4.0 反推 n≈39 → 500+3900=4400 步封顶，与判死线基准
"~4400 步内爬到 4.0" 吻合，交叉验证了调度实现正确。

## 设计决定

1. **不修改 `example/`**：新建同级目录 `m2rebuild/`，复制训练所需文件（dreamer.py/models.py/
   networks.py/tools.py/neuraldnf.py/parallel.py/exploration.py/configs.yaml/train.py +
   envs/ 除 setup_scripts），逐字节复制后立刻生成 `baseline_diff.txt`（diff -q + sha256 双重
   校验，19 个文件全部 MATCH）作为"底座=example 未改动"的证据，再在复制品上动刀。
   复用 `example/.venv`（外部路径调用 `python.exe`，不重装依赖，已验证 torch 2.8.0+cu126 +
   cuda 可用）。

2. **`ndnf_transition.py`（新文件）**：`NDNFTransition` 类，接口对齐 `networks.RSSM` 的
   `initial/observe/img_step/imagine_with_action/get_feat/get_dist/kl_loss/obs_step`，
   `WorldModel`/`ImagBehavior` 零改动即可换掉 `self.dynamics`。
   - belief = K=16 谓词（9 接地 has_key/door_locked/door_open/carrying/t_ahead/t_left/
     t_right/t_reach/wall_ahead + 7 自由），每个 2 类 categorical，`state` 用
     `{"logit","stoch","deter"}` 字典，跟标准 RSSM 的 discrete 分支同构 → `tools.OneHotDist`/
     `kl_loss` 完全复用，不用改。
   - prior: `NeuralDNF(bipolar(m_t) ⊕ bipolar(a_t))` → y_prior（当 deter）。
   - posterior: `NeuralDNF(tanh(y_prior) ⊕ obs_proj(embed))` → y_post → `s·tanh(y/s)`
     限幅成 2 类 logit（数值稳定性必需，防 delta 退火到大值后 KL 爆炸；沿用 model2 同样的
     trick，属于新 transition 自身设计，不算"训练基建"）。
   - **straight-through 采样**：直接调用 `dist.sample()`（`tools.OneHotDist.sample()` 本来
     就是 `hard.detach() + probs - probs.detach()`），不需要额外写 ST 逻辑——复用已有机制。
   - delta 调度：`step_delta()`，字面数字见上，独立实现。
   - `grounding_loss(post_logit, data)`：9 接地谓词 BCE，只监督 posterior 的前 9 个 logit，
     真值来自 `data["label_<name>"]`（env 的 `god_state()` 产出），真值不进模型输入。

3. **`models.py`（复制品，非 example 原件）改动**：
   - `WorldModel.__init__`：`config.dyn_ndnf=True` 时用 `NDNFTransition` 换掉
     `networks.RSSM`（assert `dyn_deter==dyn_stoch`，因为 NDNFTransition 的 deter 维度
     就是谓词数 K，复用 `feat_size = dyn_stoch*dyn_discrete + dyn_deter` 现成公式，
     不用改 heads/actor/critic 的输入维度计算）。
   - `_train`：在总 loss 里加 `ndnf_ground_scale * grounding_loss`（`post["logit"]`
     不 detach，梯度回传 posterior_ndnf/prior_ndnf/obs_proj，强迫谓词编码这 9 个真值）；
     每个 train step 调一次 `dynamics.step_delta()`，记录 `metrics["ndnf_delta"]`。
   - 诊断预埋点（应用户要求 b）：`_model_opt(...)` 调用时传 `probe={"ndnf_prior":...,
     "ndnf_post":..., "ndnf_obsproj":...}`，让 `tools.Optimizer.__call__` 在 unscale
     之后、`opt.step()` 之前顺手读三个子模块的梯度范数，存进 `metrics["gradnorm_*"]`——
     每个 train step 都有，不只是冒烟测试，为 150k 判死线随时可查"梯度是否还在流动"。

4. **`tools.py`（复制品）改动**：`Optimizer.__call__` 加一个可选 `probe: dict[str, Module]`
   参数，`None` 时行为完全不变（零风险，向后兼容其余不带 probe 的调用点）。

5. **`configs.yaml`（复制品）新增 `m2rebuild` overlay**：`steps=300000, train_ratio=32,
   time_limit=360`（对齐 model2 m2fix 系列），`dyn_ndnf=true, dyn_stoch=dyn_deter=16,
   dyn_discrete=2, ndnf_trans_conj=12, ndnf_trans_enc=16`，delta 五个参数，
   `ndnf_ground_scale=1.0`。用法：`--configs minigrid m2rebuild`。

6. **`dreamer.py`（复制品）改动**：`make_env` 的 `emit_labels` 判断加一项
   `or getattr(config, "dyn_ndnf", False)`（只有这一行），让 env 在这个新架构下也吐
   `label_*` 键供 grounding_loss 用。

## 冒烟测试（2026-07-18）

命令：`dreamer.py --configs minigrid m2rebuild debug --steps 2000 --prefill 200
--eval_episode_num 2 --envs 2 --logdir logdir/m2rebuild_smoke_s0`

【实锤】主循环按 `eval_every`（未在命令行覆盖，取 minigrid 默认 1e4）成块推进，
`while agent._step < config.steps + config.eval_every` 实际跑到了约 20058 步才退出
（比预期的 2000 步多，这是 example 底座 main() 的既有粒度行为，不是新代码的 bug）——
相当于白捡了一段~20k步的早期数据，一并记录判读。exit code 0，全程无报错、无 NaN。

【实锤】梯度探针（`tools.py` 的 probe 机制，读的是 metrics.jsonl 原始精度，非控制台四舍五入
到 1 位小数的显示值）：

| step | gradnorm_ndnf_prior | gradnorm_ndnf_post | gradnorm_ndnf_obsproj | ndnf_delta(窗口均值) | kl |
|---|---|---|---|---|---|
| 200 | 3.198069 | 1.272167 | 0.011810 | 0.1000 | 0.0045 |
| 5200 | 0.030095 | 0.079188 | 0.000336 | 0.1040 | 0.0106 |
| 10200 | 0.014945 | 0.097878 | 0.000458 | 0.1906 | 0.1388 |
| 15200 | 0.041459 | 0.155700 | 0.002013 | 0.4087 | 0.6943 |

**梯度穿过离散采样一路传回 prior_ndnf 这件事，通过了**：`gradnorm_ndnf_prior` 在全部
4 个记录点都非零（控制台显示的 "0.0" 是打印四舍五入到 1 位小数的假象，metrics.jsonl 里的
真实精度从未归零）。这是应用户要求预埋的 150k 判死线诊断点，现在证明这条通路从训练一开始
就是通的，以后如果学习信号消失，可以直接查这个数字是否真的归零来区分"梯度死了"还是
"梯度活着但学不出东西"。

（注：ndnf_delta/gradnorm 在 jsonl 里记的是每次 `log_every` 窗口内所有 train() 调用的
**均值**，不是瞬时值——`dreamer.py` 的 `__call__` 对 `self._metrics` 做 `np.mean` 再写
日志。所以 delta 均值爬升到 0.4 不代表瞬时 delta=0.4，是窗口内 delta 从更小值爬升的平均，
方向和量级与调度设计一致。）

【实锤】9 个接地谓词准确率随训练上升（例：t_ahead 0.3→0.3→0.5→0.8，t_reach 0.0→0.1→1.0→1.0，
carrying 0.4→0.7→0.9→0.9，has_key 0.4→0.7→0.9→0.9）——grounding_loss 在真实塑形 belief，
不是摆设。

【实锤】**这一版没有 BFS/demo 热启动**，但训练滚动中已经自发出现 `log_success=1.0`
的完整通关局（step 2044/4320/5686/9942/13712/15664/15738，共 7 次，在约 20000 步内），
`actor_entropy` 仍处于 1.3-1.6（还在探索，未收敛）。样本太少、太早，不能当成任何结论，
但确认了链路端到端能产生真实通关信号，值得记录。

**冒烟结论：通过**。可以启动正式 300k 步训练。

## 正式训练（2026-07-18，`logdir/m2rebuild_s0`，300000 步，实际跑到 312456 步正常退出）

命令：`dreamer.py --configs minigrid m2rebuild --logdir logdir/m2rebuild_s0`（无 debug 覆盖，
batch_size=16/batch_length=64 默认值）。exit code 0，全程无 Traceback/NaN，`latest.pt` 正常存盘
（112.8MB）。用户在启动后未做任何中途改动，符合"不许自行加改动"的纪律。

### delta 调度：按 update_count（梯度更新计数）算，不是按 env 步数

【实锤】windowed-mean `ndnf_delta` 在 env step 147500 首次达到 4.0 封顶。当时
`update_count≈9319*(147500/297500)≈4626`，与设计时反推的 "delay(500)+39次×100≈4400 次
梯度更新封顶" 吻合（train_ratio=32 下，update_count 相对 env step 有固定比例，二者不是 1:1，
之前汇报里"~4400 步"指的是梯度更新计数，不是环境步数——这里用真实数据把口径钉死，
以后不要混淆）。**调度机制本身验证正确**。

### 判死线判读：满足死线条件，训练失败【实锤级证据】

`eval_return` / `log_success`（每 10000 步、20 局评估）完整轨迹：

| env step | eval_return | log_success | 备注 |
|---|---|---|---|
| 2500-32500 | 0.05→0.12 | 0（偶发未落在评估窗口） | delta 仍软(0.10→0.15)，早期弱信号 |
| 42500 | 0.129 | **1.0** | 20局评估里出现真实通关 |
| 52500 | 0.112 | **1.0** | 同上 |
| 62500-102500 | 0.06→0.07 | 0 | delta 软→中(0.36→1.19)，return 缓慢下滑 |
| 112500 | 0.103 | **1.0** | 最后一次评估通关 |
| 122500-172500 | 0.114→0.022 | 0 | delta 硬化中(2.17→4.00)，**return 单调下滑** |
| **152500** | **0.028** | **0** | **150k 判死线检查点：< 0.05 ✓，且是下滑趋势，不是"还没起来"** |
| 182500-302500（12个连续检查点，共240局评估） | **恒为 0.000** | **恒为 0** | delta 已封顶(4.0)，**彻底且持续崩溃，直到跑完** |

150k 判死线的两个条件（score<0.05 且无上升趋势）**都满足，而且比最低门槛更极端**：不是"停滞在
低位"，而是从 delta 硬化开始（约 step 130k）单调滑向 0，并在 182500 步之后的后半程（130000+步、
240 个评估局）**精确归零、无一次反弹**，直到训练结束。

### 排除"表征死了"——这是"策略端接不住"的失败模式，不是"belief 编码坏了"

【实锤】用梯度探针 + 接地准确率交叉验证，把"死"具体定位到了哪一环：

- **接地谓词准确率全程健康**（9 谓词均值 0.40→0.79→0.80，从训练早期爬升后一直稳定在 ~0.78-00.80，
  哪怕在 eval 彻底归零的后半程也没有掉），说明 posterior belief **仍然在正确编码 9 个真值谓词**，
  没有塌缩成常数或噪声。
- **`gradnorm_ndnf_prior`/`gradnorm_ndnf_post` 在 delta 硬化后不但没消失，反而显著变大**
  （硬化前 ~0.01-0.1，硬化后稳定在 2.5-3.3 / 0.5-0.8），排除了"delta 越大 tanh 越饱和梯度消失"
  的假设——世界模型侧的梯度通路自始至终是通的，而且硬化后更活跃。
- **但 `actor_grad_norm` 全程钉在 ~0.0001-0.0005，300k 步里从未真正大起来过**（对照
  `value_grad_norm` 有正常的 12→0.02 的收敛曲线）。
- **`actor_entropy` 从 step 2500 到约 step 240000（全程 80%）死死焊在 ~1.60（≈ln(5)，5 动作下的
  最大熵)**，直到最后 20% 才缓慢降到 1.15——但这时 eval 早已崩溃了 6 万步以上，降熵降得太晚、
  跟真实学习脱钩。

**结论（实锤）**：这与旧 model2（未修复版）诊断出的失败签名**同一类型**——belief/表征是活的、
在学、能编码真值，但 actor 从未获得足够强的学习信号去利用它（advantage/梯度始终接近零，熵焊在
上限），策略实质上还是接近随机初始化的产物。区别在于：旧版是"表征塑形不够 + 训练基建有 bug"
的混合问题；这一版**训练基建已确认健康**（三优化器、delta 调度、无 BFS 依赖、批次/学习率均继承
成功版且冒烟验证过梯度能穿过离散采样），**唯一变量是 N-DNF transition 本身**——所以这次的失败
可以干净地归因到**架构本身**：N-DNF transition 产出的 belief（更确切说，是它在 delta 硬化后的
行为/决策边界）没能给 actor-critic 提供想象里可用的、可微分的、幅度够大的学习信号，尤其是当
delta 退火把它推向近二值逻辑之后，policy 训练反而**加速崩溃**而不是加速收敛。

**结论文本（供归档）**：「N-DNF-only transition 在本实验设置（K=16, n_conj=12, DoorKey-6x6,
无 BFS 热启动, 3 独立优化器, delta 0.1→4.0 按 500/100/1.1 调度）下不可训练【实锤级证据】——
world model 侧（重构/接地/KL）健康且梯度持续增长，actor 侧学习信号在全程 300k 步内从未突破
接近零的量级，eval 表现在 delta 完全硬化后（约 150k 步起）从零星成功单调崩溃至完全归零并
维持到训练结束（150k+ 步、240 个评估局无一次成功）。」

**下一步**：按任务纪律，死线已触发，**停止**，不自行开跑混合方案（连续 GRU 旁路）实验，
等待用户决定。

## 三个零成本尸检（2026-07-18，不训练、不花 GPU）

用户明确否决了"现在上混合方案"和"先跑 seed 复核"两个选项，要求先做三个几十分钟内能完成、
不花训练时间的分析，脚本见 `postmortem_1_random_baseline.py` / `postmortem_2_grounding_vs_baserate.py`
/ `postmortem_3_actor_loss_path.py`（只读 checkpoint/日志/磁盘，不修改任何已有文件）。

### 尸检1：随机策略基线（100 局，同一 eval 测试 seed 池）

【实锤】均匀随机策略（5 动作等概率）：**success rate 24.0%（24/100），mean return 0.0953**
（std 0.189，max 0.82）。

**标定结果**：训练曲线里"看起来有信号"的 eval_return 0.10-0.13（steps 12500-122500）**跟随机
策略的 0.095 在同一量级，统计上分不出来**——之前汇报里说这段是"弱但真实的学习信号"，
现在看更准确的说法是"跟不采取任何学习相比，看不出明显提升"。

### 尸检2：逐谓词接地精度 vs base rate（纯读磁盘，939 个已存训练局 + metrics.jsonl 尾段均值）

【实锤，且更正了之前的误判】之前"9 谓词均值 acc 稳定在 0.78-0.80，belief 表征健康"的说法
**没有对比 base rate，是误导性的**——按验收规范补上 base rate 之后：

| 谓词 | p(真=1) | base rate | acc(N-DNF) | gain | 判读 |
|---|---|---|---|---|---|
| wall_ahead | 0.578 | 0.578 | 0.942 | **+0.364** | ✅ 真学到了 |
| t_ahead | 0.272 | 0.728 | 0.778 | +0.050 | 弱信号 |
| door_open | 0.201 | 0.799 | 0.811 | +0.011 | ✗ 跟瞎猜没区别 |
| t_reach | 0.044 | 0.956 | 0.956 | -0.000 | ✗ |
| t_left | 0.190 | 0.810 | 0.792 | -0.018 | ✗ |
| t_right | 0.172 | 0.828 | 0.794 | -0.034 | ✗ |
| has_key | 0.805 | 0.805 | 0.753 | -0.051 | ✗ 比瞎猜还差 |
| carrying | 0.805 | 0.805 | 0.751 | -0.054 | ✗ 比瞎猜还差 |
| door_locked | 0.724 | 0.724 | 0.645 | -0.079 | ✗ 比瞎猜还差 |

**9 个谓词里只有 1 个（wall_ahead）真正超过 base rate，且 wall_ahead 恰好是唯一"当前帧几何
即可判断、不需要记忆"的谓词**（正前方是不是墙，从当下观察直接可推，不需要 belief 跨时保留
信息）。其余 8 个——尤其 has_key/carrying/door_locked 这三个"需要记住物体已消失在视野外仍
持有/已开锁"的持久性谓词——**精度低于或约等于 base rate，有 3 个甚至比瞎猜还差**。

### 尸检3：actor 损失数值路径解剖（加载 latest.pt，真实 rollout 48 步取想象起点，跑一次
`_imagine`/`_compute_target`/`_compute_actor_loss`，不调用 optimizer）

【实锤】结构性事实（读 config 即可确认）：本次训练 `imag_gradient='reinforce'`（继承自
`minigrid` 配置，`m2rebuild` 没覆盖），REINFORCE 的 actor_loss 主项按设计不经过 dynamics。

数值结果（256 条并行想象轨迹 × 15 步 horizon）：
- `imag_reward`：mean +0.00039，std 0.00100，max 0.0103 —— **想象窗口里几乎处处是零回报**。
- `target`(λ-return) ≈ `value baseline` ≈ **0.082**（两者几乎相等），因为 reward≈0 时
  λ-return 退化成纯 bootstrap value。
- **advantage RAW = target-base：mean +0.00036，std 0.00168** —— 就是噪声量级，不是一个
  能驱动策略学习的信号。
- `actor_entropy` = 1.116（ln(5)=1.609 是上限，尚未完全焊死但也远未收敛/达到有效探索-利用平衡）。
- `actor_loss.backward()` 后：actor 网络 grad_norm=0.00042（跟训练日志里全程 ~0.0001-0.0005
  的 `actor_grad_norm` 完全对得上）；`dynamics.prior_ndnf` grad_norm=0.00146（非零但极小——
  来自 actor_ent 那一项对未 detach 的 `imag_feat` 的依赖，是一条很小的旁路，不是 REINFORCE
  score-function 主项，量级上跟训练时世界模型损失单独喂给 prior_ndnf 的梯度(~2.5-3.3)比可
  忽略不计）；posterior_ndnf/obs_proj grad_norm=0（确认无路径）。

**结论**：不是"梯度穿不过硬化的 prior"（结构上 REINFORCE 本来就不走这条路，实测那条旁路
也没被堵，只是量级本来就微不足道）——是 **advantage 全程≈0，想象里到处是零回报，奖励饥饿**：
15 步想象 horizon 太短，从随机起点几乎摸不到 DoorKey 6x6 的终局奖励，λ-return 退化成常数
bootstrap value，advantage 淹没在噪声里，actor 拿不到任何方向感。

### 三尸检合并判读（对照用户预先定好的分岔）

- 尸检1 → **不是**"0.10-0.13 显著高于随机"，是"就是随机水平"。
- 尸检3 → **是**"advantage 全程≈0，奖励饥饿"，**不是**"梯度穿不过硬化 prior"（结构上和数值
  上都排除了）。
- 尸检2 → **不完全是**用户预设的"belief 接近瞎猜"（那是二选一的第三分支），而是一个更细的
  中间结果：9 个谓词里 8 个在 base rate 附近或以下（其中 3 个比瞎猜还差），只有 1 个（且是
  唯一不需要记忆、当帧几何直接可判的那个）真正学到——**更接近"belief 没能学会任何需要跨时
  记忆的谓词，只学会了不需要记忆的那个"**，比"全盘瞎猜"更具体，也比"表征健康"更负面。

这三条合起来指向用户预设的第二条路径（"N-DNF transition 在稀疏奖励下无法自举策略学习"，
标准 RSSM 同条件能自举，可作干净对照），但尸检2 同时说明问题不止在 actor 侧——**世界模型
自己在有直接密集 BCE 监督(每步 9 个标签)的最有利条件下，仍然学不会 8/9 个需要记忆的谓词**，
这对"降 delta 封顶"或"自适应门控 delta"这类只改 delta 调度的便宜方案的预期效果是个坏消息：
delta 从来不是这里的瓶颈（硬化后 prior_ndnf 梯度反而更大，不是更小），瓶颈更像是 conj/disj
容量或优化动力学本身接不住"记住消失物体的状态"这类需要跨时保持信息的谓词。

已如实记录三份数据，具体如何分岔（下调 delta 封顶重训 / 直接升级到"稀疏奖励下不可自举"结论
交给导师定夺是否上混合方案）留给用户判断，本轮不自行开跑任何后续实验。

## 尸检4（防作弊对照）：当前帧观测本身能不能判出这 9 个谓词？——补上尸检2 缺的前提

用户指出尸检2 的"持久性谓词需要记忆"这个前提本身没验证过，并点名 model1/r2dreamer 那边
"has_key 其实当前帧可观测（钥匙画在 agent 自格）"的先例，要求补一个防作弊对照：在原始 grid
（148维当前帧观测，模型编码器实际吃的输入）上直接训逻辑回归 probe（复用
`example/distill_belief.py` 的 probe/base_rate 规范：按 episode 分 train/test 防泄漏），
脚本 `postmortem_4_grid_probe_anticheat.py`，纯读磁盘 + sklearn，同一批 939 个训练局。

【实锤，钉死了那颗钉子——而且钉出的是更严重的那个结论】

| 谓词 | base rate | grid probe acc | gain | 当前帧可观测？ |
|---|---|---|---|---|
| wall_ahead | 0.570 | 1.000 | +0.430 | ★ 可观测 |
| t_ahead | 0.727 | 0.988 | +0.261 | ★ 可观测 |
| has_key | 0.812 | **1.000** | +0.188 | ★ 可观测（钥匙画在 agent 自格，跟 model1 那边一样） |
| carrying | 0.812 | **1.000** | +0.188 | ★ 可观测 |
| t_left | 0.813 | 0.952 | +0.139 | ★ 可观测 |
| door_locked | 0.710 | 0.884 | +0.174 | ★ 可观测 |
| t_right | 0.827 | 0.940 | +0.114 | ★ 可观测 |
| door_open | 0.782 | 0.911 | +0.129 | ★ 可观测 |
| t_reach | 0.955 | 0.999 | +0.044(绝对) | 看似"增益小"，但 base 已 95.5%，probe 把剩余 4.5%
  的错误几乎消灭到 0.1%——错误率降了约45倍，本质上也是可观测，不是"增益小=不可观测"，是类别
  极不平衡把绝对 gain 压小了 |

**9 个谓词，全部 9 个，仅凭当前帧（无需任何跨时记忆）就能被一个线性 probe 判到 88%-100% 精度。**
"has_key/carrying/door_locked 这类需要记忆"的前提**不成立**——这套观测编码（`grid`：7×7×3
符号栅格 + 朝向）跟 model1/r2dreamer 防作弊对照发现的问题是**同一类**：物体状态/朝向信息
本来就编码在当前帧里，不需要 belief 跨时保留。

**这把结论从"干净"版推向"更重"版**：不是"N-DNF 循环无法承载记忆"（因为这个任务本来就不
需要记忆），而是**"N-DNF 循环（prior/posterior 的 conj/disj 结构 + obs_proj 的 16 维瓶颈 +
delta 退火）连一个线性 probe 都能在无记忆条件下轻松判出的可观测量，都没学好"**——8/9 谓词
在 belief 里跟 base rate 打平或更差，而这些量本身对一个纯前馈线性分类器来说毫无难度。真正
需要归因的不是"记忆能力"，是 N-DNF transition 本身（prior⊕posterior 的合取析取结构、
`n_conj=12`、`n_enc=16` 观测投影瓶颈、或 delta 调度跟这套结构的相互作用）为什么连这么简单的
监督信号都学不进去。

**结论文本更新（供归档，替换/补充前一版）**：「N-DNF-only transition 在本实验设置下不可
训练【实锤级证据】，且根因比"稀疏奖励下无法自举策略学习"更基础：9 个接地谓词在这套观测编码
下全部（或近乎全部）可由当前帧线性可分，不需要跨时记忆；但 N-DNF transition 的 belief 仅
学会了其中语义上最平凡的 1 个（wall_ahead，也是唯一多多少少不涉及"物体/朝向"复杂符号解析
的一个），其余 8 个（含高度类别失衡但线性可分的 has_key/carrying/door_locked/door_open/
t_left/t_right/t_reach）精度停留在 base rate 附近甚至更差。这说明失败发生在 N-DNF
prior/posterior 循环结构本身学习普通监督信号的能力上，而不是"记忆"这个更高层的能力——
后者的失败甚至无从谈起，因为任务本身没有要求记忆。」

**下一步**：三尸检+防作弊对照已经把链条钉死到了 N-DNF transition 结构本身（不是记忆、
不是 delta 硬化、不是奖励稀疏单独的锅——虽然奖励稀疏确实也独立成立，但即使有干净的监督
信号，circuit 本身也学不进去），比最初设想的任何一条分岔都更负面。混合 GRU 旁路能不能救，
本质上要看"救的是 actor 侧信号(稀疏奖励)"还是"救的是 belief 学习本身(N-DNF 学不进简单监督
信号)"——如果是后者，加 GRU 旁路更像是绕过问题而不是解决它，价值判断留给用户/导师。
不自行开跑任何后续实验。
