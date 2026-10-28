# v13 新证据索引

本目录以 2026-09-05 的 checkpoint 重评估和历史记录审计为主，2026-09-06 补做最终代码一致性复验。JSON 保留执行时元数据、原 checkpoint 路径及散列。所有模型均为保留的既有 checkpoint，没有训练更新。

| 文件 | 论文用途和样本单位 |
|---|---|
| `d3_episode_clustered.json` | 100 个实际 seed 的局部干预；保存逐状态作用，使用完整 episode bootstrap。 |
| `d3_bootstrap_verification.json` | 从上面的 state_records 精确复算四个 5000-draw 区间。 |
| `d3_interface_controls.json` | 同一 D3 actor，100 seeds × soft/sign-only/oracle 三条件；成功 100/13/100。 |
| `m1_grounded_seed{0,1}_fixed_upper_direct200.json` | 主确认：cue 无关固定上分支覆盖；每模型 200 direct seeds，probe 140/60 划分。 |
| `m1_grounded_seed{0,1}_direct200.json` | 自主 actor 覆盖补充；不能与固定路径逐状态分数当作完全匹配比较。 |
| `m1_packaged_cli_verification.json` | 最终统一 CLI 重复两次 200 局固定上分支评估，验证与已存指标一致；不增加独立样本。 |
| `m1_device_check.json` | 小规模 CPU/GPU actor 路径检查；排除先前协议差异被误归因于设备。 |
| `m2_seed{0,1}_direct200_with_probe.json` | 200 direct seeds 的在线 head 与同 checkpoint 离线 probe；在线 1000 个 pure-memory steps，离线测试 300 步。 |
| `check_m2_checkpoint_distinctness.json` | 不同 checkpoint 的参数和后验 features 确实不同；相同离线汇总分数不表示重复载入模型。 |
| `m3_actor{0,1}_unique200_historical.json` | 对旧 500 nominal records 按实际 seed 去重，得到 200 条、四种完整初始配置；不是新 rollout。 |
| `m3_actor{0,1}_configuration_grounding.json` | 每 actor 四种配置 × 七条件 = 28 次新执行，含 cue 真值、原始预测、供策略使用的值、终点及时间。 |
| `m3_replay_validation.json` | 两 actor × 四种配置 × 五条件 = 40 次第二 seed 复演；40/40完整新轨迹检查一致，2000/2000历史终点/时间比较一致。 |
| `m3_revision_summary.json`、`.md` | M3 汇总，区分历史 seed 权重与新配置检查；核心诊断及复演合计 96 次执行。 |
| `v12_evaluation_audit.json` | 历史 v12 分母纠错和终点重算记录。 |

Memory 的不同实际 seeds 仍只覆盖四种完整配置；probe 两侧均含这四种配置。所有时间步分数都是各自收集协议下的描述性指标，不能据此宣称未见配置泛化。M3 原始 cue 符号全程正确的结论专指 **cue-negation** 轨迹，不能扩展到 wall-edit 轨迹。

对应原代码与补跑命令见 [代码复现说明](../code_changes/CODE_CHANGES.md)。D1、D2、历史 M1、R1 等保留证据仍在原 `Xdreamer/results/`、`archive/` 和 `v8/INST0062-24081539.zip` 所列位置，路径也见论文附录。本目录不复制大型 checkpoint；复现 rollout 需要原仓库及模型。`.log` 与辅助 `.py` 保留执行过程，最终 PDF 只引用已经完成并核验的结果。
