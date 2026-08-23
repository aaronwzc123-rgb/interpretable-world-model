# Model 2（符号瓶颈）实现细节与两版对比

本文档面向 ML/代码初学者，术语第一次出现会用括号简单解释。所有结论标注三种口吻：

- **【事实】** = 我真的读到了对应的代码/config，会给出文件路径+行号。
- **【推断】** = 从事实合理推出的，但没有直接的代码/实验证据。
- **【假设】** = 还没验证的猜测，需要单变量复训才能确认，不代表结论。

两个版本：
- **成功版**：`example/logs/symnav_bottleneck1/latest.pt`，DoorKey-6x6 通关率 100%，代码在 `example/`（旧版 `dreamerv3-torch` 代码库）。
- **失败版**：`model2/r2dreamer/logdir/ndnf_dreamer_demo_anneal_s0/latest.pt`，300k 步训练全程 `eval_score` 基本为 0，代码在 `model2/r2dreamer/`（较新的 `r2dreamer` 代码库）。

**先说最重要的一句话【事实】**：这两版**不是同一份代码改了几个参数**，而是**两套独立实现的"N-DNF 符号瓶颈"设计**，架构上有本质区别（细节见 B 部分）。这意味着"为什么一个成一个败"不能简单归因于某一个超参数，很可能是架构选择本身的差异。

---

## 术语速查（先看这个，后面遇到生词回来查）

| 术语 | 大白话解释 |
|---|---|
| belief / RSSM | Dreamer 世界模型里"记住了什么"的那个隐藏状态（类似循环神经网络的隐状态），RSSM 是产生它的那个模块（Recurrent State-Space Model） |
| feat | belief 拼起来给下游用的那个向量（= stoch 展平 + deter 拼在一起） |
| stoch / deter | belief 的两部分：deter 是"确定性"的循环记忆（类似 GRU 隐状态），stoch 是"随机采样"的一小块分类隐变量，每步都重新采样 |
| N-DNF | Neural Disjunctive Normal Form，一种"可微的逻辑电路"：用神经网络的权重去逼近一组"与/或"逻辑规则，训练完可以把权重读成人能看懂的 `:-` 规则（类似 Prolog 语法：`A :- B, C.` 读作"如果 B 且 C，则 A"） |
| 合取层 / conj (conjunction) | 逻辑里的"与"（AND）——所有条件都满足才为真 |
| 析取层 / disj (disjunction) | 逻辑里的"或"（OR）——任一条件满足就为真 |
| logit | 还没经过 sigmoid/softmax 压缩的"原始打分"，可以是任意实数，正负表示倾向 |
| delta 退火 | N-DNF 内部有个温度参数 delta，delta 小的时候网络输出很"软"（像模糊逻辑），delta 大的时候输出趋近于硬 0/1（像真正的布尔逻辑）。"退火"就是训练过程中把 delta 从小慢慢调大 |
| BCE / MSE | 两种常见损失函数：BCE（二分类交叉熵）专门给"是/否"标签用；MSE（均方误差）是差值平方，什么标签都能用但对二分类不是最优 |
| detach | PyTorch 里"切断梯度"的操作，用了 detach 之后，这个值虽然还能用来算别的东西，但反向传播不会顺着它往回传梯度 |
| imagination rollout（想象轨迹） | Dreamer 训练策略（actor）时不在真实环境里走，而是用世界模型自己"脑补"（想象）出一串轨迹来训练，这个脑补的过程就叫 imagination rollout |

---

## A. 成功版架构精讲（`example/`，checkpoint = `symnav_bottleneck1`）

### A.1 符号头（`_sym_head`）长什么样

**类定义位置【事实】**：`example/neuraldnf.py:143-200`（`class SymbolicHead`），底层用到的 `NeuralDNF`/`SemiSymbolic` 定义在同文件 `L6-54`。

数据流（文字版流程图）：

```
belief feat (1280维，= stoch展平1024 + deter 256)
      │  Linear(1280→48)          <- neuraldnf.py:159  self.perceive
      ▼
raw literal 预激活 (48维)
      │  tanh(delta × raw)         <- neuraldnf.py:173  torch.tanh(d * self.perceive(feat))
      ▼
literals p0..p47 ∈ [-1,1]  （"48个中间概念"，没有名字，纯靠训练学出来）
      │  合取层 conj: Linear-like(48→12) + tanh(delta×(s+bias))   <- neuraldnf.py:12-21, 27
      ▼
conj_out (12维) ∈ [-1,1]
      │  析取层 disj: Linear-like(12→9) + tanh(delta×(s+bias))
      ▼
atoms (9维) ∈ [-1,1]  = has_key / door_locked / door_open / carrying /
                         t_ahead / t_left / t_right / t_reach / wall_ahead
```

关键代码（`neuraldnf.py:170-174`）：
```python
def atoms(self, feat, delta=None):
    d = self.current_delta() if delta is None else delta
    lit = torch.tanh(d * self.perceive(feat))           # (...,L) ∈[-1,1] literals
    return self.dnf(lit, d)                             # (...,K) ∈[-1,1] atoms
```
一句话：先把 1280 维 belief 压缩成 48 个"中间概念"，再用一个与层+一个或层把这 48 个中间概念组合成 9 个具名谓词。

**"与/或"是怎么实现的（`SemiSymbolic`，`neuraldnf.py:6-21`）**：
```python
absw = self.w.abs(); maxw = absw.max(dim=1).values; sumabs = absw.sum(dim=1)
s = x @ self.w.t()
bias = maxw - sumabs if self.kind == "conj" else sumabs - maxw
return torch.tanh(delta * (s + bias))
```
【事实】这里**没有对权重做任何截断/裁剪**（不是"权重约束"），"与"还是"或"的语义完全来自这一行 bias 的算法——AND 用 `maxw - sumabs`（必须所有相关项都为正才能让 s+bias>0），OR 用 `sumabs - maxw`（任一项为正就够）。这是 pix2rule / DNF-MT 论文里的标准技巧。

### A.2 9 个谓词的监督标签怎么来的（`env.god_state()`）

文件：`example/envs/minigrid.py`，方法 `_god_doorkey`（`L181-249`）。

**4 个状态谓词**【事实】：
- `has_key`（`L202`）：`carrying is not None and carrying.type == "key"`，即"手上正拿着钥匙"
- `door_locked`/`door_open`（`L192-195`）：遍历地图格子找到门对象，直接读它的 `is_locked`/`is_open` 属性
- `carrying`（`L238`）：手上有没有拿任何东西（不分是钥匙还是别的）

**5 个导航谓词**【事实】，做法是"以自我为中心的几何分解"，**不是 BFS**（这点很重要，见 B 部分对比）：

第一步，先决定"当前阶段的目标点"（`L207-214`）：
```python
if not has_key:
    target = key_pos              # 没钥匙 -> 目标=钥匙
elif not door_open:
    target = door_pos             # 有钥匙但门没开 -> 目标=门
elif door_pos is not None and ax <= door_pos[0]:
    target = (door_pos[0]+1, door_pos[1])   # 门开了但还没穿过去 -> 目标=门的东侧入口格
else:
    target = goal_pos              # 已经进了房间 -> 目标=终点
```
第二步，把"目标相对我在哪"投影到"前/左/右/到位"（`L216-228`）：把 agent 朝向拆成前向量 `fwd` 和右向量 `rgt`，算目标方向在这两个轴上的分量 `fc`（前向分量）、`rc`（右向分量），再用简单的符号/大小比较判断 `t_ahead`（目标在正前方一带）、`t_left`/`t_right`（目标偏左/偏右）、`t_reach`（目标就在正前方紧邻一格，可以直接执行动作了）。
`wall_ahead`（`L229-232`）：正前方那一格是不是墙。

一句话：这套谓词的核心思想是"当前该干嘛的目标点是哪"（钥匙→门→终点，按阶段切换），然后用向量几何算目标在自己的前/左/右方向，不需要路径搜索。

### A.3 符号损失怎么加进训练

**损失函数**【事实】：`neuraldnf.py:176-196`，`masked_loss` 方法，用的是 **MSE**（不是 BCE）：
```python
out = self.atoms(feat)                              # (B,T,K) ∈[-1,1]
y = torch.stack([2.0*data[f"label_{l}"].squeeze(-1)-1.0 for l in self.labels], -1)  # {-1,+1}
se = ((out - y) ** 2).mean(-1)                       # (B,T)
```
把 0/1 标签映射成 -1/+1，跟网络输出（同样在 [-1,1] 里）做均方误差。

**加进总损失的位置**【事实】：`example/models.py:236-241`
```python
if self._sym_head is not None:
    feat_sym = self.dynamics.get_feat(post)          # 注意：没有 .detach()
    sym_loss, sym_mets = self._sym_head.masked_loss(feat_sym, data)
    total_loss = total_loss + self._sym_scale * sym_loss   # sym_scale = 2.0
```
**梯度是否回传世界模型**：从这里能直接看出来——`feat_sym` 没有调用 `.detach()`，所以 `sym_loss` 反向传播时，梯度会一路传回 `get_feat` 用到的 `post`（RSSM 的输出），再传回 RSSM 本身。加上 `_sym_head` 是在 `self._model_opt` **创建之前**就实例化的（`models.py:113` vs `L136` 创建 `_model_opt`，注释在 `L91-92/L110-112` 写得很明白），所以符号头自己的权重（perceive/conj/disj）也在世界模型的同一个 Adam 优化器里，跟重构/奖励损失一起被优化——这就是"塑形 belief"的字面意思：符号头的监督信号会真的改变 RSSM 学到的隐状态。

### A.4 actor 输入切换机制（`sym_policy_input`）

**替换发生的函数**【事实】：`example/models.py:318-328`，`WorldModel.augment(feat)`：
```python
def augment(self, feat):
    if self._sym_head is None:
        return feat
    atoms = self._sym_head.atoms(feat).detach()      # 注意这里 detach 了
    if self._sym_policy_input == "atoms":
        return atoms                                  # 瓶颈：只给 9 个原子
    return torch.cat([feat, atoms], -1)               # 混合模式
```
**这里的 detach 很关键**：`atoms` 在喂给策略之前被 detach 了，意味着策略（actor/critic）的训练梯度**不会**通过这条路径反过来污染符号头——符号头只被 A.3 里那条 `sym_loss` 塑形，不会被"为了让策略表现更好"这个目标反向拉扯。

**imagination rollout 路径上是否也生效（关键问题）**：**是的，确认生效**。证据：
- 真实/评估时的策略调用：`ablate_sym.py:87` `pol = wm.augment(feat); action = beh.actor(pol).mode()`
- 训练时"脑补"轨迹的每一步：`models.py:525`（`ImagBehavior._imagine` 内部的 `step` 函数）：
  ```python
  feat = dynamics.get_feat(state)
  inp = self._world_model.augment(feat).detach()
  action = policy(inp).sample()
  ```
- 训练时算 actor loss/critic loss 时也调用了同一个函数：`models.py:469`（`actor_ent = self.actor(self._world_model.augment(imag_feat)).entropy()`）、`models.py:489/494`（critic 的输入同样过 `augment`）。

一句话：**真实推理、训练时的想象轨迹、actor/critic 损失计算，三处全部通过同一个 `augment()` 函数**，是名副其实的"一处开关，全局生效"，不存在"训练时偷看 feat、推理时才切原子"这种偷懒的可能。

顺带一提，`ImagBehavior.__init__`（`models.py:386-392`）在**构建网络结构**这一步就已经把 actor/critic 的输入维度设成了 9（而不是 1289），也就是说"瓶颈"不只是运行时的选择，连神经网络的输入层大小都是按 9 维建的，物理上没有多余的输入口子留给连续 feat。

### A.5 训练期 vs 推理期的差别

**前向计算公式本身没有变化**——训练和推理用的是同一份 `atoms()` 代码，唯一变化的是 `delta`（温度参数）：
- `current_delta()`（`neuraldnf.py:166-168`）：`delta = 0.5 + (delta_max-0.5) * min(1, frac/0.6)`，其中 `frac = updates/anneal_steps`。`sym_anneal=40000`（配置项），意味着更新次数达到 `0.6×40000=24000` 次之后，delta 就封顶在 `sym_delta_max=4.0`，不再变化。
- 【事实，实测】我读取了这份 checkpoint 里 `_wm._sym_head.updates` 这个计数器，值是 **70097**，远超 24000 ⇒ 保存这份 checkpoint 时 `delta` 已经稳定在 **4.0**（比较接近硬阈值的"陡"tanh）。也就是说训练到后期和推理时用的 delta 其实是**同一个已经封顶的值**，不存在"训练软、推理硬"的切换——只是训练早期（updates<24000）delta 比较小、比较"软"。

**规则抽取（`sym_rules()`/`extract_rules`）是另外一回事**——它不改变网络的实际前向计算，只是"读一遍权重、离散化成人能看的规则"，供人看/记录用：
```python
thr = w_thr * np.abs(cw[j]).max()          # w_thr = 0.5，即每行最大权重的一半
for i in range(cw.shape[1]):
    if cw[j, i] > thr: lits.append(in_names[i])
    elif cw[j, i] < -thr: lits.append("not " + in_names[i])
```
（`neuraldnf.py:39-45`）一句话：这一步是把连续权重"一刀切"成"这个字面量算不算数"的二元判断，只在你主动调用规则提取时发生，不影响 checkpoint 实际怎么决策——**打印出来的 `:-` 规则是对网络的一个近似读出，不是网络真正在执行的代码**，这个区别值得跟导师说清楚。

### A.6 关键超参数一览表

来源：`example/configs.yaml`（`defaults` + `minigrid` + `minigrid_symbolic_nav` 三层叠加）。

| 超参数 | 值 | 出处 |
|---|---|---|
| 任务 | DoorKey-6x6 | `configs.yaml:207` (`minigrid.task`) |
| 总步数 | 5e5 | `configs.yaml:208` |
| train_ratio | 512 | `configs.yaml:213` |
| batch_size / batch_length | 16 / 64 | `configs.yaml:91-92` (defaults) |
| 世界模型学习率 (model_lr) | 1e-4 | `configs.yaml:95` (defaults) |
| actor / critic 学习率 | 3e-5 / 3e-5 | `configs.yaml:50, 52` (defaults) |
| actor 熵系数 (entropy) | 3e-4，**不退火**，全程固定 | `configs.yaml:50` |
| dyn_deter / dyn_stoch / dyn_discrete | 256 / 32 / 32 | `configs.yaml:228-230`（deter/units 被 minigrid 覆盖为256）, `configs.yaml:35-36`（stoch/discrete 用 defaults 默认值） |
| sym_n_lit（中间literal数） | 48 | `configs.yaml:300` |
| sym_conj（合取子数） | 12 | `configs.yaml:301` |
| sym_scale（符号损失权重） | 2.0 | `configs.yaml:299` |
| sym_anneal（delta退火步数） | 40000 | `configs.yaml:302` |
| sym_delta_max | 4.0（默认值，未在 nav 覆盖里重设） | `configs.yaml:87` (defaults `sym_delta_max: 4.0`) |
| imag_horizon（想象轨迹长度） | 15 | `configs.yaml:104` (defaults) |
| 实测已训练更新步数 | 70097（从 checkpoint 的 `_sym_head.updates` 读出） | 【事实，直接读 checkpoint】 |

---

## B. 两版 diff

### B.1 训练配置 diff（只列有差异的项）

| 配置项 | 成功版 (`example/`) | 失败版 (`model2/r2dreamer/ndnf_dreamer_demo_anneal_s0`) |
|---|---|---|
| 代码库 | 旧版 `dreamerv3-torch`（`example/dreamer.py` 等） | 较新的 `r2dreamer`（`model2/r2dreamer/dreamer.py` 等） |
| 总步数 | 5e5（`configs.yaml:208`） | 300000（`.hydra/config.yaml: env.steps`） |
| train_ratio | 512（`configs.yaml:213`） | 32（`overrides.yaml: env.train_ratio=32`） |
| 世界模型学习率 | 1e-4（`configs.yaml:95`） | 4e-05（`.hydra/config.yaml: model.lr`），**且 actor/critic 和世界模型共用同一个学习率**（见 B.2） |
| 优化器数量/种类 | 3 个独立 Adam：`model_opt`(1e-4) / `actor_opt`(3e-5) / `value_opt`(3e-5)（`example/models.py:136-145`, `models.py:426-444`；checkpoint 里 `optims_state_dict` 也确认是这 3 个键） | 1 个 LaProp 优化器覆盖**所有**参数（rssm+actor+critic+encoder+decoder），单一 lr=4e-05（`model2/r2dreamer/dreamer.py:159-164`） |
| actor 熵系数 | 固定 3e-4，全程不变（`configs.yaml:50`；`example/models.py:482` 直接用 `config.actor["entropy"]`，没有退火逻辑） | 初始 3e-4，但线性退火到 5% (即 1.5e-5)，8000 次更新内退完（`dreamer.py:30-31, 325-327`；`.hydra/config.yaml: model.act_entropy=0.0003`，退火比例是代码里的默认值不在 config 里） |
| delta 退火调度 | 从 0.5 线性升到 4.0，60% 进度（24000/40000 次更新）内升完，之后封顶（`sym_anneal=40000`；checkpoint 实测已到 4.0） | `initial_delta=0.1`，`delta_delay=1,000,000,000`（十亿！）——300k 步训练全程都不会触发退火（`.hydra/config.yaml: model.rssm.ndnf.delta_delay`）。见 `ndnf_rssm.py:238-247` `step_delta()`：`if self._delta_counter <= c["delay"]: return self.get_delta()`（不变） |
| 符号/接地损失函数 | MSE（`neuraldnf.py:181`：`(out-y)**2`） | BCE（`ndnf_rssm.py:223-224`：`binary_cross_entropy_with_logits`） |
| 符号/接地损失权重 | sym_scale=2.0 | loss_scales.ground=1.0（`.hydra/config.yaml: model.loss_scales.ground`）——注意两边损失函数不同，权重数字不能直接比大小 |
| 离散化正则 | 无 | 有：`ndnf_aux` 额外把 disj 层权重推向 {-6,0,6}（`ndnf_rssm.py:229-236`），权重 0.001（`.hydra/config.yaml: model.loss_scales.ndnf_aux`） |
| 合取子数量 (n_conj) | 12（策略瓶颈层） | prior 64 / posterior 48（`.hydra/config.yaml: model.rssm.ndnf.n_conj_prior/n_conj_post`）——容量比成功版大得多 |
| BFS 演示轨迹热启动 (demo warm-start) | 无此机制，环境也不提供 BFS 解算器 | 有：前 20000 步用 BFS 解出的动作覆盖 agent 的真实动作，往 replay buffer 注入必胜轨迹（`.hydra/config.yaml: +trainer.demo_steps=20000`；`trainer.py:169-172`；`envs/minigrid.py:23-58` 的 `_bfs`） |
| 导航谓词算法 | 以自我为中心的几何分解（前向/右向量点积，见 A.2） | BFS 搜索最短路径，取路径第一步方向（`envs/minigrid.py:111-129`） |
| rep_loss（表征学习目标） | 固定用重构（decoder 预测 grid 观测） | 本次跑的是 `rep_loss=dreamer`（同样是重构，`.hydra/config.yaml: model.rep_loss=dreamer`；`overrides.yaml` 也确认）——**这一项跟成功版其实一致**，可以排除"表征目标类型不同"这个候选原因 |
| 动作空间 | 5 个动作（`VALID_ACTIONS=[0,1,2,3,5]`） | 同样 5 个动作（`envs/minigrid.py:13`）——**无差异**，两边通关率理论上可比 |

### B.2 代码/架构层面的差异（比 config 更根本）

这是本文档最重要的部分——两版对"符号瓶颈"的实现思路完全不同：

1. **belief 的本体不同**
   - 成功版【事实】：RSSM 是标准、完全未改动的 Dreamer 结构（GRU 循环 + 32×32 分类隐变量），`SymbolicHead` 只是**架在它上面的一个额外读出层**——`feat`（1280维）先正常算出来，`_sym_head` 再拿 `feat` 去算 9 个原子。世界模型的"记忆"载体本身不是逻辑谓词。
   - 失败版【事实】：`NDNFRSSM`（`model2/r2dreamer/ndnf_rssm.py`）**直接把 RSSM 整个换掉**——belief 的 `deter` 字段本身就是 N-DNF 的原始输出（`ndnf_rssm.py:9,146`：`deter = self._prior_raw(stoch, prev_action)`），没有另外的连续 GRU 状态。也就是说失败版**没有"多出来的连续通道"可以选择切不切断**——它从设计上就是纯符号瓶颈，成功版的 `sym_policy_input='concat'` 这种"混合模式"选项在失败版里根本不存在。

2. **belief 是否被反复采样（这一条我认为是最值得怀疑的差异）**
   - 成功版【事实】：`SymbolicHead.atoms()` 是 `feat` 的**确定性函数**（一路 tanh，没有采样），随机性只存在于底层 RSSM 自带的、Dreamer 标准设计里的 32×32 stoch 采样，且这部分代码完全没有改动过，是被广泛验证过的成熟设计。
   - 失败版【事实】：`obs_step()`（`ndnf_rssm.py:135-154`）每一步都要 `self.get_dist(logit).rsample()` 采样出这一步的谓词 one-hot，而这个采样结果（`_bipolar(stoch)`，`ndnf_rssm.py:196-198,200-203`）**直接成为下一步 Prior N-DNF 的输入**——即整条时间序列的"记忆传递"完全靠一条离散采样的链条，没有类似 GRU 那种平滑、确定性的隐状态兜底。
   - 【推断】这意味着失败版的世界模型在时间上的梯度/信号传递天生比成功版噪声大得多、也更难优化——这是一个结构性的、非超参数能简单调好的差异。

3. **训练时到底谁在被优化、按什么速度**
   - 成功版：3 个优化器分层——世界模型（含符号头）用较快的 1e-4，策略/价值用较慢的 3e-5，这是 Dreamer 系工作里常见的稳定性技巧（策略不要追得比世界模型快）。
   - 失败版：单一优化器、单一学习率 4e-05，世界模型（含 NDNF prior/posterior）和策略/价值网络**混在一起以同一个节奏更新**。

4. **符号损失的监督方式不同**：MSE vs BCE（见 B.1），且失败版多了一个"离散化正则" `ndnf_aux` 去推权重到 {-6,0,6}，成功版完全没有这类正则。

5. **BFS 热启动 + 导航谓词算法**：失败版引入了一整套 BFS 基础设施（求解器 + demo 轨迹注入 + 用 BFS 结果定义导航谓词），成功版没有任何 BFS，靠纯几何算导航谓词——这是两边代码规模和复杂度上的明显差异，BFS 方案理论上更"正确"（真实最短路），但也多了一套可能出 bug 或行为不稳定的逻辑。

### B.3 【假设】失败版为什么训不起来——候选原因列表（按可能性从高到低排序，全部待验证）

> 以下全部是**假设**，不是结论。验证需要控制单一变量分别重跑，不在本次任务范围内。

1. **【假设，可能性最高】delta 退火从未触发，逻辑层整场训练都停留在"很软"的状态。**
   依据：B.1 里 `delta_delay=1,000,000,000` 远超 300k 步总训练量，`step_delta()`（`ndnf_rssm.py:238-247`）在这种配置下永远走"不变"分支，`delta` 卡在初始值 0.1。相比之下成功版实测在训练中段（约 24000 次更新）就已经把 delta 升到 4.0。delta 太小意味着 `tanh(delta×x)` 几乎是线性、"和稀泥"的，"与/或"语义（依赖 `bias=maxw-sumabs` 这套设计要在 delta 较大时才能真正体现"非此即彼"的判断）可能根本没有机会体现出来，网络也许在一个过于模糊的解空间里打转，难以收敛出清晰的策略依据。

2. **【假设，可能性较高】belief 完全由离散采样链条构成，缺少平滑的确定性记忆通道，训练信号噪声大、难收敛。**
   依据：B.2 第2条。这是两版最本质的架构差异，如果这是主因，那**不是调参能解决的**，需要重新设计（比如给 NDNFRSSM 加一个并行的连续通道，或减少每步采样的随机性）。

3. **【假设，中等可能性】单一学习率+单一优化器，世界模型和策略/价值互相"抢跑"，缺少成功版那种"世界模型先跑稳、策略慢慢跟上"的分层节奏。**
   依据：B.2 第3条。这类不稳定性在 Dreamer 类工作的历史踩坑记录里比较常见（本项目自己的训练日志也观察到过 entropy 长期封顶、advantage 很小等现象，虽然那是另一个更早的失败 run，但同属"策略学不动"的症状）。

4. **【假设，中等可能性】BFS 热启动窗口太短，一停止注入网络就崩。**
   依据【事实，本次实测】：我翻了同目录下另一次相关训练（`model2/r2dreamer/logdir/ndnf_dreamer_demo_s0/metrics.jsonl`，同样 `demo_steps=20000`）的 `episode/score` 曲线，按 2 万步分桶取均值：`step 0-20000` 均值 **0.960**（这一段几乎肯定被 BFS 热启动直接注入的动作主导，不代表策略自己学会了），**`step 20000-160000` 均值几乎全是 0**（热启动一停，成绩立刻掉回接近 0，且维持了将近 14 万步），`180000` 步之后才开始出现一些零星、幅度不大的回升（0.03～0.12 之间波动，从未接近热启动期间的水平）。这跟"热启动喂了一段必胜轨迹，但训练自己没能把这批数据转化成稳定策略"的方向一致（【推断】），比"训练一度真正学会又崩掉"更准确的描述是"热启动效果几乎没有延续下来"。注意：这条实测来自 `demo_s0` 这次训练，不是本文档主角 `demo_anneal_s0`，两者 `demo_steps` 配置相同，仅供参考，不直接等同。

5. **【假设，可能性较低但值得一提】BCE + 更大的合取子容量（64/48 vs 12）可能让符号头本身更难训（更容易过拟合到嘈杂的中间态，或者优化面更复杂），不是不可能，但比起前两条证据更弱。**

**明确排除的候选原因**：`rep_loss` 类型不是原因——两版这次跑的都是 `rep_loss=dreamer`（重构目标），配置一致（B.1）。动作空间也不是原因——两边都是 5 个动作。

---

## C. 一页速答卡

**Q1. 符号头的损失是怎么和世界模型损失合在一起的？**
成功版：`sym_loss`（谓词预测的 MSE）乘上权重 `sym_scale=2.0` 后直接加到 `total_loss` 上，一起交给唯一的世界模型 Adam 优化器（`models.py:241-242`）。因为喂给符号头的 `feat` 没有 `.detach()`，这条损失的梯度会一路传回 RSSM，逼着它把这些谓词编码进 belief（A.3）。

**Q2. 怎么保证策略真的只看原子、没有偷看连续 belief？**
`sym_policy_input='atoms'` 时，`augment(feat)` 只返回 9 维原子（且这 9 维在喂给策略前做了 `.detach()`），而且 actor/critic 网络的输入层从建网络那一刻起就只有 9 个输入口——不是运行时选择性忽略,是物理上没有多余通道。真实评估、训练时的想象轨迹、actor/critic 损失计算三处都统一走这个函数（A.4），我在代码里逐一确认过调用点。

**Q3. t_ahead 这些导航谓词的真值是怎么算的？这算不算把任务答案喂给了模型？**
用"当前阶段目标点"（没钥匙→钥匙，有钥匙未开门→门，开门后→终点）加上以自我为中心的向量几何算出方向关系（A.2）。这些真值只在训练时当**监督标签**，模型实际吃到的是"从 belief 预测出来的原子"，不是这个真值本身（这点在 `augment()`/`atoms()` 的实现里能确认：策略拿到的是网络的输出，不是 `god_state()` 的返回值）。是否"喂了任务结构"这件事本身是真的——`rules_snapshot`（已抽出的规则）显示网络学到的规则确实用到了这套谓词的任务语义，这是"做符号瓶颈让人能看懂"必须付出的代价，不是作弊，但确实意味着词表本身包含了设计者对任务的理解。

**Q4. 两个版本差在哪，为什么一个成一个败？**
两版是**两套独立实现**：成功版是"在正常 RSSM 上加一个符号读出层"，失败版是"把整个 RSSM 换成 N-DNF"（B.2）。config 层面最扎眼的差异是失败版的 delta 退火实际从未触发、全程停留在很软的状态（B.1、B.3 假设1），架构层面最大的差异是失败版的 belief 完全靠离散采样链条传递、没有平滑的确定性记忆通道（B.2第2条、B.3假设2）。这些目前都是**假设**，排序在 B.3，需要分别控制变量重跑才能坐实。

**Q5. 原子清零后通关率归零，这个实验排除了哪些别的解释？**
这个实验（把 9 个原子全部置零，其余不变，重新跑 20 局）排除了"策略其实在偷偷利用某些没被发现的旁路信息通关"的可能——因为策略的输入层物理上就只有这 9 个数字，把它们全清零之后网络还能拿到的信息只剩 0 向量，通关率应声跌到 0，说明这 9 个数字（而不是训练过程留下的其它副作用，比如网络权重恰好记住了地图套路）确实是策略做决策的**唯一**信息来源。它不能排除的是"这 9 个谓词具体是怎么被策略使用的"（那需要逐个原子消融才能回答，另见验收 notebook 里的对照区）。
