# Model 2 对照实验：use_ndnf=False（换回原版 GRU RSSM），rep_loss 留在默认值 r2dreamer 不动。
# 只改 use_ndnf 这一个变量（相对失败的 ndnf_soft_s0 run），看 vanilla RSSM 配
# Barlow-Twins 自监督（r2dreamer）能不能训出来：
#   - 能 → 证实问题是"N-DNF 的 48 维窄瓶颈配不上 r2dreamer"，不是 r2dreamer 本身、
#     也不是 model2 代码库里非 RSSM 部分（trainer/buffer/minigrid.py 新增 key）有 bug。
#   - 不能 → r2dreamer 本身在这个环境/配置下有问题，需要重新怀疑方向。
#
# 用法：
#   cd C:\Users\super\Desktop\test\model2\r2dreamer
#   .\runs\minigrid_vanilla.ps1                   # 完整 300000 步
#   $env:STEPS=5000; .\runs\minigrid_vanilla.ps1   # 冒烟测试

$RepoRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $RepoRoot "..\.venv\Scripts\python.exe"

$Seed = 0
if (-not $env:STEPS) { $env:STEPS = "300000" }
$RunName = "noNdnf_r2dreamer_s$Seed"

& $VenvPython (Join-Path $RepoRoot "train.py") `
    "env=minigrid" `
    "env.task=minigrid_DoorKey-6x6-v0" `
    "env.steps=$($env:STEPS)" `
    "env.train_ratio=32" `
    "model=size12M" `
    "model.compile=False" `
    "model.use_ndnf=False" `
    "seed=$Seed" `
    "logdir=./logdir/$RunName" `
    "+run_name=$RunName"
