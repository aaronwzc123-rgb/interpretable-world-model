# 实验2：去掉 BFS 热启动（单变量：demo_steps）
# 在实验1（delta_delay=500 + clamp 4.0，已确认 delta 退火机制本身工作正常但
# 不是唯一病因）配置基础上，只改 demo_steps: 20000 -> 0。
# 成功版没用热启动也训成了；热启动可能让策略学会"等喂答案"，一旦停止接管就
# 学不动。其余配置（rep_loss=dreamer、单优化器、BCE grounding_loss、
# n_conj_prior/post=64/48、delta_delay=500 + clamp 4.0）保持不变。
#
# 用法：
#   cd C:\Users\super\Desktop\test\model2\r2dreamer
#   .\runs\m2fix_exp2_nodemo.ps1

$RepoRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $RepoRoot "..\.venv\Scripts\python.exe"

$Seed = 0
$RunName = "m2fix_exp2_nodemo_s$Seed"

& $VenvPython (Join-Path $RepoRoot "train.py") `
    "env=minigrid" `
    "env.task=minigrid_DoorKey-6x6-v0" `
    "env.steps=300000" `
    "env.train_ratio=32" `
    "model=size12M" `
    "model.compile=False" `
    "model.rep_loss=dreamer" `
    "+trainer.demo_steps=0" `
    "model.rssm.ndnf.delta_delay=500" `
    "seed=$Seed" `
    "logdir=./logdir/$RunName" `
    "+run_name=$RunName"
