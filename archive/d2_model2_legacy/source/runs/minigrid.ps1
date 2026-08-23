# Model 2 目标：让 use_ndnf=true 训出能用的策略（不是做干净单变量对照，是"尽量成功"）。
# 第一轮组合（rep_loss=dreamer + demo 暖启动 + grounding_loss 修复）跑出来的结果：
# 训练期成功率在 demo 暖启动结束后一度到 83.6%（steps ~10800-60500），之后在剩下 80%
# 的训练步数里又崩回到 ~7%（接近随机基线），action_entropy 也从低点 0.17-0.36 回升到
# 0.8-1.2。原因：策略变好后 imagine 出的 return 趋于一致，advantage 自然缩小，固定不衰减
# 的 act_entropy=3e-4 熵奖励系数就重新压过（已缩小的）advantage，把收敛好的策略往回拉。
# 这一轮新增：
#   4. dreamer.py 里 act_entropy 现在随训练线性衰减（默认 8000 次更新内衰减到 5%），
#      防止收敛后被熵奖励拉垮
#   5. trainer.py 现在每次 eval（默认每 10000 步）顺带存一次 checkpoint，不再只有跑完
#      整个 300000 步才存一次——这样中途出问题也能保留进度、也能事后探查中间状态
#
# 用法：
#   cd C:\Users\super\Desktop\test\model2\r2dreamer
#   .\runs\minigrid.ps1                        # 完整 300000 步
#   $env:STEPS=5000; .\runs\minigrid.ps1        # 冒烟测试

$RepoRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $RepoRoot "..\.venv\Scripts\python.exe"

$Seed = 0
if (-not $env:STEPS) { $env:STEPS = "300000" }
if (-not $env:DEMO_STEPS) { $env:DEMO_STEPS = "20000" }
$RunName = "ndnf_dreamer_demo_anneal_s$Seed"

& $VenvPython (Join-Path $RepoRoot "train.py") `
    "env=minigrid" `
    "env.task=minigrid_DoorKey-6x6-v0" `
    "env.steps=$($env:STEPS)" `
    "env.train_ratio=32" `
    "model=size12M" `
    "model.compile=False" `
    "model.rep_loss=dreamer" `
    "+trainer.demo_steps=$($env:DEMO_STEPS)" `
    "seed=$Seed" `
    "logdir=./logdir/$RunName" `
    "+run_name=$RunName"
