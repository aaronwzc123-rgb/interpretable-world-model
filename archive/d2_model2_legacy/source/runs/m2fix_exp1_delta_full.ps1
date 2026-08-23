# 实验1：delta 退火修复（单变量：delta 调度）
# 相对失败版基线 (ndnf_dreamer_demo_anneal_s0) 只改两处，算同一个变量：
#   1. model.rssm.ndnf.delta_delay: 1e9 -> 500 （退火实际启动）
#   2. ndnf_rssm.py:245 clamp 上限 1.0 -> 4.0 （对齐成功版 sym_delta_max=4.0，代码已改）
# 其余配置（rep_loss=dreamer, demo_steps=20000, 单优化器, BCE grounding_loss,
# n_conj_prior/post=64/48）保持不变，避免混入其它候选病因。
#
# 用法：
#   cd C:\Users\super\Desktop\test\model2\r2dreamer
#   .\runs\m2fix_exp1_delta_full.ps1

$RepoRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $RepoRoot "..\.venv\Scripts\python.exe"

$Seed = 0
$RunName = "m2fix_exp1_delta_full_s$Seed"

& $VenvPython (Join-Path $RepoRoot "train.py") `
    "env=minigrid" `
    "env.task=minigrid_DoorKey-6x6-v0" `
    "env.steps=300000" `
    "env.train_ratio=32" `
    "model=size12M" `
    "model.compile=False" `
    "model.rep_loss=dreamer" `
    "+trainer.demo_steps=20000" `
    "model.rssm.ndnf.delta_delay=500" `
    "seed=$Seed" `
    "logdir=./logdir/$RunName" `
    "+run_name=$RunName"
