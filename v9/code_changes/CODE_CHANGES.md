# v9 评估代码修订与复现

本目录包含最终的 11 个完整源文件和统一补丁 `v9-evaluation-fixes.patch`。补丁基于 Xdreamer 的 `7805619c4cb7b4ab51945dc2396351085c431b46`。`manifest.json` 给出各文件 SHA-256。没有训练更新，没有修改已有 checkpoint 或历史结果文件。

## 修改内容

| 文件 | 修改及用途 |
|---|---|
| `envs/minigrid.py` | 增加显式 `pool` / `direct` 评估种子模式；记录 requested seed、actual seed 和包含地图、智能体姿态、任务及携带物的初始状态签名。默认历史池和训练随机序列保持原协议；元数据不进入模型观测。 |
| `scripts/evaluate_m1_cuestart.py` | 拒绝重复实际种子和训练池重叠；按实际种子划分 probe 数据，报告初始配置重叠；区分自主 actor 与 cue/物体身份无关的固定上分支控制器；记录 checkpoint hash、命令和覆盖协议。 |
| `scripts/evaluate_m2_head.py` | 使用同一套种子核查，并可在同一 checkpoint、同一批轨迹上增加离线 probe，避免用另一模型的 M1 结果推断 M2 的记忆存储。 |
| `scripts/evaluate_d3_directed_actions.py` | 保存逐状态效应和 episode ID；置信区间按完整 episode bootstrap，估计量仍为合格状态的合并均值；保留旧逐状态重采样值供审计对照。 |
| `scripts/evaluate_d3_interface_controls.py` | 新增 soft、符号化为 ±1、oracle 符号三种推理输入控制；逐 seed 记录结果。oracle 仅是使用模拟器真值的评估对照。 |
| `scripts/staged_stage_c_eval.py` | 明确当前输出取反与固定值替换的差别；记录干预起点和每个干预步的原始/输入 atom；按准确时间上限终止；旧评估入口也将超时与语义重定向分开。 |
| `scripts/evaluate_m3_corrected_flip.py` | 实际种子去重；完整初始配置盘点；可只运行每配置代表 seed；增加 grounding 轨迹、zero controls、checkpoint hash，以及描述性汇总时关闭抽样区间。 |
| `scripts/audit_m3_existing_results.py` | 验证重复记录一致后，对历史 500 条结果按实际种子去重为 200 条，并列出四种完整初始配置及其权重。此脚本不生成新的模型 rollout。 |
| 三个 `tests/test_*.py` | 2 项 episode bootstrap、8 项种子/划分/覆盖协议、6 项 M3 超时与干预语义测试。另运行仓库原有的 12 项回归测试。 |

`direct` 只消除 seed 取模重复，不能把四种初始配置变成 200 种不同环境。probe 的实际种子划分无重叠，但四种配置在两侧仍有重叠。固定上分支覆盖与 M2 的 privileged 正确路线覆盖是不同协议，结果不能作为完全匹配的模型对照。

## 已完成的验证

- 在 `v9/build/code_validation_shadow` 从上述 commit 提取的代码副本上，`git apply --no-index --check` 和实际应用均通过。原 `test` 仓库未被此次打包过程写入。
- 全套 **28/28 测试通过**，0 failures、0 errors，测试耗时 4.543 秒；见 `../build/code_validation_tests.log` 和 `validation.json`。固定上分支路径测试的预期动作序列修正为实际六步 `[forward, forward, forward, forward, left, forward]`。
- 打包后的 M1 CLI 分别对两个 checkpoint 重跑 200 局固定上分支覆盖，与已保存的 standalone 评估在全部 probe 分数、split、实际 seed 和控制器成功率上完全一致。见 `../evidence/m1_packaged_cli_verification.json`。这 400 次是实现一致性复验，不增加独立样本。
- D3 保存的四个区间从逐状态记录重新计算，与 **5000 次**完整 episode bootstrap 逐浮点数完全一致。随机种子为 `20260815 + atom 在标签列表中的索引`：t_ahead=20260819，t_left=20260820，t_right=20260821，wall_ahead=20260823。见 `../evidence/d3_bootstrap_verification.json`。
- M3 的第二 seed 复演在 40/40 个配对条件上重现完整 grounding 轨迹、终点和时长；2000/2000 个历史终点/时长比较一致。这支持已观察四种配置的确定性复演，不扩充配置覆盖。

## 环境

实际验证环境为 Windows、Python 3.11.15、PyTorch 2.8.0+cu126、CUDA 12.6；GPU 为 NVIDIA GeForce RTX 4060 Laptop GPU。M1、M2、D3 新评估使用 CPU，PyTorch 线程数为 1；保存的 M3 配置检查与第二 seed 复演使用 `cuda:0`。

关键包：NumPy 1.26.4、gym 0.22.0、gymnasium 0.29.1、MiniGrid 2.3.1、ruamel.yaml 0.17.4、scikit-learn 1.9.0、SciPy 1.17.1、matplotlib 3.7.5、einops 0.7.0。完整实际安装清单见 `environment-observed.json`，不是新安装或网络解析得到的环境。仓库 `requirements.txt` 未改变。评估继续使用现有 `.venv` 即可；新机器仍须准备相应依赖和原 checkpoint。

## 如何使用代码包

最终文件在 `code_changes/envs`、`code_changes/scripts`、`code_changes/tests` 中按仓库相对路径排列。原 `test` 仓库现已同步最终 11 个文件，并在实际仓库再次通过 28 项测试。不要对它重复应用完整补丁。同步前已有差异文件的备份在 `../build/pre_final_sync/`，同步清单见 `repository_sync.json`。

对于基于上述 commit、未修改这些文件的 checkout，先在该 checkout 根目录检查再应用：

```powershell
git apply --check C:/Users/super/Desktop/UCL/dissertation/v9/code_changes/v9-evaluation-fixes.patch
git apply C:/Users/super/Desktop/UCL/dissertation/v9/code_changes/v9-evaluation-fixes.patch
```

对于已包含本次早期修订的工作副本，可以将目录中这 11 个完整文件逐个复制到对应位置。不要覆盖其他文件。本目录的 `manifest.json` 是最终文件清单；修改后的模型、训练产物和新依赖不在该清单中。

## 主要结果复现命令

以下从应用最终文件后的 Xdreamer 根目录运行。输出放到单独的 `evidence/reproduced`，保留论文所引用的原始新评估记录。`$Repo`、`$V9` 可替换为其他机器上的绝对路径。

```powershell
$Repo = 'C:/Users/super/Desktop/test/Xdreamer'
$V9 = 'C:/Users/super/Desktop/UCL/dissertation/v9'
$Py = Join-Path $Repo '.venv/Scripts/python.exe'
$Out = Join-Path $V9 'evidence/reproduced'
New-Item -ItemType Directory -Force -Path $Out | Out-Null
Set-Location -LiteralPath $Repo
$env:PYTHONDONTWRITEBYTECODE = '1'
& $Py -B -m unittest discover -s tests -v
```

D3 完整 episode 区间和三种输入控制：

```powershell
& $Py -B scripts/evaluate_d3_directed_actions.py --checkpoint models/doorkey/doorkey_experiment_3_d3/checkpoint.pt --episodes 100 --seed-start 0 --bootstrap 5000 --device cpu --torch-threads 1 --out "$Out/d3_episode_clustered.json"
& $Py -B scripts/evaluate_d3_interface_controls.py --repo $Repo --episodes 100 --device cpu --out "$Out/d3_interface_controls.json"
& $Py -B "$V9/scripts/verify_d3_saved_bootstrap.py" --repo $Repo --source "$V9/evidence/d3_episode_clustered.json" --out "$Out/d3_bootstrap_verification.json"
```

M1 主确认结果为固定上分支覆盖，`actor` 是另存的访问分布补充。两者使用各自轨迹拟合离线 probe：

```powershell
foreach ($Seed in 0, 1) {
    & $Py -B scripts/evaluate_m1_cuestart.py --checkpoint "models/memory/memory_experiment_1_m1/grounded_seed$Seed/checkpoint_084000.pt" --condition grounded --episodes 200 --seed-start 1000 --eval-seed-mode direct --split-seed 20260813 --coverage-controller fixed_upper --torch-threads 1 --device cpu --out "$Out/m1_grounded_seed${Seed}_fixed_upper_direct200.json"
    & $Py -B scripts/evaluate_m1_cuestart.py --checkpoint "models/memory/memory_experiment_1_m1/grounded_seed$Seed/checkpoint_084000.pt" --condition grounded --episodes 200 --seed-start 1000 --eval-seed-mode direct --split-seed 20260813 --coverage-controller actor --torch-threads 1 --device cpu --out "$Out/m1_grounded_seed${Seed}_direct200.json"
}
```

M2 online head 与同 checkpoint 离线 probe：

```powershell
foreach ($Seed in 0, 1) {
    & $Py -B scripts/evaluate_m2_head.py --checkpoint "models/memory/memory_experiment_2_m2/seed$Seed/checkpoint_084000.pt" --episodes 200 --seed-start 1000 --eval-seed-mode direct --offline-probe --split-seed 20260813 --torch-threads 1 --device cpu --out "$Out/m2_seed${Seed}_direct200_with_probe.json"
}
```

M3 历史去重和新配置检查必须分开执行、分开报告。每个 actor 的 `--representative-configs` 先盘点 200 个实际 seed，仅对四个代表初始配置分别执行 baseline、三种 cue 时点、wall-mid、zero-all、zero-cue 共 28 次 rollout。不是每条件新执行 200 次：

```powershell
foreach ($Seed in 0, 1) {
    & $Py -B scripts/audit_m3_existing_results.py --source "results/memory_experiment_3_m3/m3_seed${Seed}_corrected_500.json" --out "$Out/m3_actor${Seed}_unique200_historical.json"
    & $Py -B scripts/evaluate_m3_corrected_flip.py --checkpoint "models/memory/memory_experiment_3_m3/actor_seed${Seed}_400demo.pt" --rundir "$Out/m3_logs" --distill-meta models/memory/memory_experiment_3_m3/frozen_symbolic_head_meta.json --episodes 200 --seed-start 0 --seed-mode direct --representative-configs --zero-controls --bootstrap 0 --torch-threads 1 --device cuda:0 --out "$Out/m3_actor${Seed}_configuration_grounding.json"
}
```

`../evidence/validate_m3_replay.py` 和 `../evidence/check_m2_checkpoint_distinctness.py` 是额外复演/模型差异检查的原始脚本，保留了本机的绝对路径。前者读取原始 v9 JSON、使用原代码仓库和 CUDA；后者使用 CPU。需要在其他机器重复这两项辅助检查时，应先修改其中 `REPO` 和输出路径，避免覆盖本版证据。主要评估 CLI 均使用上面的显式参数。

M1 固定上分支原始 JSON 由 `../evidence/recheck_m1_fixed_upper.py` 生成。最终统一 CLI 与其数值完全一致，但 JSON 元数据字段略有差别；重现时比较记录的指标、种子及划分，不要求文件 SHA-256 相同。GPU/CPU 或库版本改变时也不应预设逐浮点数相同。
