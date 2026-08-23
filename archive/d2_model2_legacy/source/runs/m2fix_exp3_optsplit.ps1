# 实验3：优化器分层（单变量：world model / actor / critic 各用独立 LaProp 优化器 + 学习率）
# 基于实验2的配置（demo_steps=0，实验1/2 已确认 delta 修复和去热启动都不是主因，
# 继续沿用更简单的 no-demo 基线）：
#   1. dreamer.py: 单一 LaProp 优化器 -> 拆成 _model_optimizer/_actor_optimizer/_value_optimizer
#      （代码已改，仍是 LaProp，不换算法，只拆分参数组+学习率，避免混入第二个变量）
#   2. configs/model/_base_.yaml: lr=4e-5 (单一) -> model_lr=1e-4/actor_lr=3e-5/value_lr=3e-5
#      （直接对齐成功版 example/configs.yaml 的 model_lr/actor.lr/critic.lr）
# 其余（delta_delay=500+clamp 4.0、rep_loss=dreamer、demo_steps=0、BCE grounding_loss、
# n_conj_prior/post=64/48、agc/beta/warmup）保持不变。
#
# 用法：
#   冒烟测试: $env:STEPS=5000; .\runs\m2fix_exp3_optsplit.ps1
#   完整跑:   .\runs\m2fix_exp3_optsplit.ps1

$RepoRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $RepoRoot "..\.venv\Scripts\python.exe"

$Seed = 0
if (-not $env:STEPS) { $env:STEPS = "300000" }
$RunName = "m2fix_exp3_optsplit_s$Seed"
if ($env:STEPS -ne "300000") { $RunName = "m2fix_exp3_optsplit_smoke_s$Seed" }

& $VenvPython (Join-Path $RepoRoot "train.py") `
    "env=minigrid" `
    "env.task=minigrid_DoorKey-6x6-v0" `
    "env.steps=$($env:STEPS)" `
    "env.train_ratio=32" `
    "model=size12M" `
    "model.compile=False" `
    "model.rep_loss=dreamer" `
    "+trainer.demo_steps=0" `
    "model.rssm.ndnf.delta_delay=500" `
    "seed=$Seed" `
    "logdir=./logdir/$RunName" `
    "+run_name=$RunName"
