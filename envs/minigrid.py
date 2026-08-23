"""
MiniGrid 适配器 —— 把 gymnasium 版 MiniGrid-DoorKey 包成 dreamerv3-torch 期望的接口。

要点（对齐本项目的纪律）：
- 动作裁剪到 5 个（去掉 drop/done），与 v1.5 的 env_wrapper 一致。
- 观察给两路：
    grid  (148,) float32  —— 喂给模型（符号化 7x7x3 归一化 + 方向），走 MLP encoder。
    image (H,W,3) uint8   —— 仅供 dreamer 的日志/视频用，模型不看（cnn_keys='$^' 忽略它）。
- 训练/测试 seed 严格隔离：train 用 seed>=10000，eval 用测试池 0~199（模型从不在训练里见）。
- obs 字典必须含 is_first/is_last/is_terminal（dreamer 约定），log_success 走 log_ 前缀（不喂模型）。
"""
import gym                      # 老版 gym(0.22)，dreamer 的 wrappers 用它的 spaces
import numpy as np
import gymnasium               # MiniGrid 基于 gymnasium
import minigrid                # noqa: F401  —— import 即注册 MiniGrid-* 环境

# MiniGrid 原始动作: 0=left 1=right 2=forward 3=pickup 4=drop 5=toggle 6=done
VALID_ACTIONS = [0, 1, 2, 3, 5]      # 去掉 drop(4)、done(6)

TRAIN_SEED_START = 10_000            # 训练图 seed 起点
TRAIN_POOL       = 100_000           # 训练图池大小（每局随机取一张，保证多样性）
TEST_POOL        = 200               # 测试图 seed 0~199

ENV_IDS = {
    "doorkey6x6":   "MiniGrid-DoorKey-6x6-v0",
    "doorkey8x8":   "MiniGrid-DoorKey-8x8-v0",
    "doorkey6x6_rel":  "MiniGrid-DoorKey-6x6-v0",
    "doorkey8x8_rel":  "MiniGrid-DoorKey-8x8-v0",
    "doorkey16x16_rel": "MiniGrid-DoorKey-16x16-v0",
    "unlockpickup": "MiniGrid-UnlockPickup-v0",
    # 记忆任务：起始房间给一个 cue（Key/Ball），走过长廊后到岔口，选与 cue 相同的物体。
    # cue 离开起始房间后当前帧看不见 → 必须靠 belief 记住，正是证明 belief=记忆的干净场景。
    "memoryS7":     "MiniGrid-MemoryS7-v0",
    # 显式实验变体：与官方 MemoryS7 相同，但 reset 后固定从 cue 列起步。
    # 不再全局 monkey-patch MemoryEnv，避免历史 memoryS7 实验被静默改变。
    "memoryS7_cuestart": "MiniGrid-MemoryS7-v0",
    "memoryS9":     "MiniGrid-MemoryS9-v0",
    "memoryS11":    "MiniGrid-MemoryS11-v0",
    "memoryS13":    "MiniGrid-MemoryS13-v0",
}
MEMORY_TASKS = {"memoryS7", "memoryS7_cuestart", "memoryS9", "memoryS11", "memoryS13"}
CUE_START_TASKS = {"memoryS7_cuestart"}
DOORKEY_REL_TASKS = {"doorkey6x6_rel", "doorkey8x8_rel", "doorkey16x16_rel"}


class MiniGrid:
    metadata = {}

    # god_state 里对应布尔谓词（作 N-DNF/probe 标签，绝不喂模型）
    LABELS_DOORKEY = ("has_key", "door_locked", "door_open", "carrying")
    # 导航完备词表（符号瓶颈用）：状态谓词 + egocentric 关系谓词。
    #   target 按阶段选：没钥匙→key；有钥匙但门没开→door；否则→goal。
    #   t_ahead/left/right = target 在 agent 前/左/右（agent 自身朝向坐标系）；
    #   t_reach = target 正前方相邻一格（可 pickup/toggle/前进的时刻）；wall_ahead = 正前方是墙。
    #   这套让「只看谓词」的反应式策略也能导航（转向 target→前进→到位执行），供阶段2 瓶颈证明。
    LABELS_DOORKEY_NAV = ("has_key", "door_locked", "door_open", "carrying",
                          "t_ahead", "t_left", "t_right", "t_reach", "wall_ahead")
    # Model3 relational：只描述独立世界事实，不再由 oracle 先选 target 再输出统一 t_*。
    # door_between_agent_goal 是拓扑关系：agent 尚未穿过阻挡 goal 的门。
    LABELS_DOORKEY_REL = (
        "has_key", "door_locked", "door_open", "carrying",
        "door_between_agent_goal", "agent_on_door", "wall_ahead",
        "key_ahead", "key_left", "key_right", "key_reach",
        "door_ahead", "door_left", "door_right", "door_reach",
        "goal_ahead", "goal_left", "goal_right", "goal_reach",
    )
    # 记忆任务：cue_is_key 是需要记住的谓词；cue_visible 是"当前帧是否还能看到 cue"；
    #   cue_known 是"本局到当前步 cue 是否曾进入过视野"（监督/统计掩码——只有信息可知时，
    #   要求 belief 记住 cue_is_key 才有意义，避免用"还没见过"的无信息时刻稀释指标）。
    LABELS_MEMORY = ("cue_is_key", "cue_visible", "cue_known")
    # 记忆任务·导航完备词表 v2（符号瓶颈用，2026-07-27 谓词重设计修复版）：
    #   v1 的 bug（已修复，原文件备份在 backup/）：t_ahead/t_left/t_right/t_reach 相对
    #   "cue_is_key 挑出的 target" 计算，导致这 4 个"导航"谓词的真值定义本身就嵌入了 cue 答案——
    #   策略只要跟着箭头走就能赢，根本不需要读 cue_is_key。v1 训练里 cue_is_key 退化成常数、
    #   清零它对通关率无影响，根因就是这个标签泄漏，不是记忆学不会。
    #   v2 改成两套独立、对称、不依赖 cue_is_key 的导航谓词，分别相对 key 候选物和 ball 候选物
    #   计算（key_ahead/left/right/reach 与 ball_ahead/left/right/reach），策略必须自己用
    #   cue_is_key 决定跟哪一套，才能完成任务——cue_is_key 才是真正必要的信息。
    #   wall_ahead 保留（本来就与 cue 无关）。egocentric 分解算法与 v1/DoorKey 完全一致，
    #   只是从"算一次(对 target)"变成"独立算两次(对 key_pos 和 ball_pos)"。
    LABELS_MEMORY_NAV = ("cue_is_key", "cue_known", "wall_ahead",
                         "key_ahead", "key_left", "key_right", "key_reach",
                         "ball_ahead", "ball_left", "ball_right", "ball_reach")

    def __init__(self, task, mode="train", seed=0, max_steps=300, tile_size=8,
                 render_obs=False, emit_labels=False):
        assert task in ENV_IDS, f"未知 task={task}，可选 {list(ENV_IDS)}"
        self._task = task
        self._is_memory = task in MEMORY_TASKS
        self._cue_start = task in CUE_START_TASKS
        self._doorkey_rel = task in DOORKEY_REL_TASKS
        self._cue_seen = False        # 记忆任务：本局 cue 是否曾进入视野（cue_known 用）
        # 谓词标签随任务族切换（DoorKey vs Memory），供 N-DNF/probe 监督用。
        self.LABELS = (self.LABELS_MEMORY_NAV if self._is_memory else
                       self.LABELS_DOORKEY_REL if self._doorkey_rel else
                       self.LABELS_DOORKEY_NAV)
        self._env = gymnasium.make(
            ENV_IDS[task], max_steps=int(max_steps), render_mode="rgb_array"
        )
        self._mode = "train" if "train" in mode else "eval"
        self._rng = np.random.RandomState(seed)        # 训练图随机采样用
        self._eval_seed = seed % TEST_POOL             # eval 起始测试 seed（各 env 错开）
        self._tile = tile_size
        # render_obs=False：obs["image"] 只是 1x1x3 占位。该图训练本就没喂模型
        #   （encoder cnn_keys='$^' 忽略、video_pred_log=false 不做视频），每步全图渲染纯属白耗时。
        #   需要真图时（如 eval 走迷宫动画）用 env.render()，与此无关。
        self._render_obs = bool(render_obs)
        # emit_labels=True：把 god_state 4 个布尔谓词作为 label_* 键塞进 obs，
        #   供训练期 N-DNF 头监督（encoder mlp_keys='grid' 不匹配 label_*，模型看不到）。
        self._emit_labels = bool(emit_labels)
        self.reward_range = [0.0, 1.0]
        # 探一次 render 拿到 image 形状（用固定 seed，不消耗 eval 计数）
        self._env.reset(seed=0)
        self._img_shape = self._frame().shape

    # ── spaces ────────────────────────────────────────────────
    @property
    def observation_space(self):
        spaces = {
            "grid":        gym.spaces.Box(0.0, 1.0, (148,), np.float32),
            "image":       gym.spaces.Box(0, 255, self._img_shape, np.uint8),
            "is_first":    gym.spaces.Box(0, 1, (1,), np.uint8),
            "is_last":     gym.spaces.Box(0, 1, (1,), np.uint8),
            "is_terminal": gym.spaces.Box(0, 1, (1,), np.uint8),
            "log_success": gym.spaces.Box(-np.inf, np.inf, (1,), np.float32),
        }
        if self._emit_labels:
            for l in self.LABELS:
                spaces[f"label_{l}"] = gym.spaces.Box(0.0, 1.0, (1,), np.float32)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        space = gym.spaces.Discrete(len(VALID_ACTIONS))
        space.discrete = True
        return space

    # ── helpers ───────────────────────────────────────────────
    def _frame(self):
        if not self._render_obs:
            return np.zeros((1, 1, 3), np.uint8)      # 占位，省掉每步全图渲染
        return self._env.unwrapped.get_frame(
            tile_size=self._tile, agent_pov=False
        ).astype(np.uint8)

    def _make_obs(self, raw, is_first, is_last, is_terminal, success):
        img = raw["image"].astype(np.float32)          # (7,7,3) 符号编码
        img[:, :, 0] /= 10.0                            # object type 最大10
        img[:, :, 1] /= 5.0                             # color 最大5
        img[:, :, 2] /= 2.0                             # state 最大2
        direction = np.float32(raw["direction"] / 3.0)  # 0~3 → 0~1
        grid = np.concatenate([img.reshape(-1), [direction]]).astype(np.float32)  # (148,)
        obs = {
            "grid":        grid,
            "image":       self._frame(),
            "is_first":    is_first,
            "is_last":     is_last,
            "is_terminal": is_terminal,
            "log_success": np.float32(success),
        }
        if self._emit_labels:
            g = self.god_state()                        # 当前步真值（作标签，不喂模型）
            for l in self.LABELS:
                obs[f"label_{l}"] = np.array([1.0 if g[l] else 0.0], np.float32)
        return obs

    def _next_seed(self):
        if self._mode == "train":
            return int(TRAIN_SEED_START + self._rng.randint(TRAIN_POOL))
        seed = int(self._eval_seed % TEST_POOL)        # eval：在测试池里循环
        self._eval_seed += 1
        return seed

    # ── gym 老接口（4 元组）────────────────────────────────────
    def reset(self):
        raw, _ = self._env.reset(seed=self._next_seed())
        if self._cue_start:
            base = self._env.unwrapped
            base.agent_pos = np.array((1, base.height // 2))
            base.agent_dir = 0
            raw = base.gen_obs()  # 位置改变后必须重建首帧局部观测
        self._cue_seen = False       # 新的一局，清空"曾见过 cue"（须在建 obs 前）
        return self._make_obs(raw, True, False, False, 0.0)

    def step(self, action):
        real = VALID_ACTIONS[int(action)]
        raw, reward, terminated, truncated, _ = self._env.step(real)
        done = bool(terminated or truncated)
        success = 1.0 if (terminated and reward > 0) else 0.0
        obs = self._make_obs(raw, False, done, bool(terminated), success)
        info = {}
        if truncated and not terminated:               # 截断不是真终止 → discount=1
            info["discount"] = np.float32(1.0)
        return obs, np.float32(reward), done, info

    def god_state(self):
        """上帝视角真值 —— 仅用作 belief 探针/N-DNF 的标签，绝不喂给模型。"""
        return self._god_memory() if self._is_memory else self._god_doorkey()

    def _god_memory(self):
        """记忆任务真值 v2（2026-07-27 谓词重设计修复版，原 v1 实现见 backup/）：
        cue_is_key（要记住的谓词）+ cue_visible（当前帧能否看到 cue 的掩码）+ wall_ahead +
        两套独立、对称的自我中心导航原子——key_ahead/left/right/reach（相对 key 候选物）、
        ball_ahead/left/right/reach（相对 ball 候选物）。两套原子的真值只取决于候选物的
        物理位置，与 cue_is_key 完全无关（v1 的 bug：用 cue_is_key 先挑出唯一的 target 再算
        导航原子，导致"导航"原子里偷偷编码了 cue 答案）。target/other 仍在返回值里保留，
        供 memory_nav_god.py 等既有诊断脚本的 success_pos/failure_pos 交叉校验使用，但不再
        参与任何导航原子的真值计算。"""
        env = self._env.unwrapped
        cue_x, cue_y = 1, env.height // 2 - 1        # cue 固定位置（_gen_grid 里放置）
        cue = env.grid.get(cue_x, cue_y)             # 全程不被拾取，随时可读真值
        cue_is_key = bool(cue is not None and cue.type == "key")
        try:
            cue_visible = bool(env.agent_sees(cue_x, cue_y))
        except Exception:
            cue_visible = False
        self._cue_seen = self._cue_seen or cue_visible   # 一旦见过就置真，直到本局结束
        ax, ay = int(env.agent_pos[0]), int(env.agent_pos[1])
        d = int(env.agent_dir)

        # 扫 grid 找走廊尽头两个候选物体（排除 cue 格）——按物体自身类型分类，不依赖 cue_is_key
        candidates = []
        for x in range(env.grid.width):
            for y in range(env.grid.height):
                if (x, y) == (cue_x, cue_y):
                    continue
                c = env.grid.get(x, y)
                if c is not None and c.type in ("key", "ball"):
                    candidates.append((x, y, c.type))
        assert len(candidates) == 2, f"MemoryS7 期望 2 个候选物体，实际 {candidates}"
        key_cands = [c[:2] for c in candidates if c[2] == "key"]
        ball_cands = [c[:2] for c in candidates if c[2] == "ball"]
        assert len(key_cands) == 1 and len(ball_cands) == 1, \
            f"应恰好一个 key 候选物、一个 ball 候选物，实际 {candidates}"
        key_pos, ball_pos = key_cands[0], ball_cands[0]
        # target/other：纯诊断标注（"cue 类型匹配的那个候选物"），不参与导航原子真值计算
        target = key_pos if cue_is_key else ball_pos
        other = ball_pos if cue_is_key else key_pos

        # egocentric 分解：与 _god_doorkey 完全相同的算法，对 key_pos/ball_pos 各独立算一次
        DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]      # 0右 1下 2左 3上
        fwd = DIRS[d]
        rgt = (-fwd[1], fwd[0])

        def _ego(pos):
            px, py = pos
            dx, dy = px - ax, py - ay
            fc = dx * fwd[0] + dy * fwd[1]
            rc = dx * rgt[0] + dy * rgt[1]
            reach = (fc == 1 and rc == 0)              # 已验证可达（10/10 scripted 局各触发一次），不放宽
            ahead = (fc > 0 and abs(fc) >= abs(rc))
            left = (rc < 0 and abs(rc) > abs(fc))
            right = (rc > 0 and abs(rc) > abs(fc))
            return ahead, left, right, reach

        key_ahead, key_left, key_right, key_reach = _ego(key_pos)
        ball_ahead, ball_left, ball_right, ball_reach = _ego(ball_pos)

        fx, fy = ax + fwd[0], ay + fwd[1]
        fcell = env.grid.get(fx, fy) if (0 <= fx < env.grid.width and 0 <= fy < env.grid.height) else None
        wall_ahead = bool(fcell is not None and fcell.type == "wall")

        return {
            "cue_is_key":  cue_is_key,
            "cue_visible": cue_visible,
            "cue_known":   bool(self._cue_seen),
            "wall_ahead":  wall_ahead,
            "key_ahead":   key_ahead, "key_left": key_left,
            "key_right":   key_right, "key_reach": key_reach,
            "ball_ahead":  ball_ahead, "ball_left": ball_left,
            "ball_right":  ball_right, "ball_reach": ball_reach,
            "agent_pos":   (ax, ay),
            "agent_dir":   d,
            "target":      target,
            "other":       other,
            "key_pos":     key_pos,
            "ball_pos":    ball_pos,
        }

    def _god_doorkey(self):
        """DoorKey 任务真值 + egocentric 关系谓词（供符号导航词表）。"""
        env = self._env.unwrapped
        carrying = env.carrying
        door_locked = door_open = False
        door_pos = key_pos = goal_pos = None
        for x in range(env.grid.width):
            for y in range(env.grid.height):
                c = env.grid.get(x, y)
                if c is None:
                    continue
                if c.type == "door":
                    door_locked = bool(getattr(c, "is_locked", False))
                    door_open = bool(getattr(c, "is_open", False))
                    door_pos = (int(x), int(y))
                elif c.type == "key":
                    key_pos = (int(x), int(y))
                elif c.type == "goal":
                    goal_pos = (int(x), int(y))
        ax, ay = int(env.agent_pos[0]), int(env.agent_pos[1])
        d = int(env.agent_dir)
        has_key = bool(carrying is not None and carrying.type == "key")

        # 阶段目标（把门当强制路径点，避免撞分隔墙的局部最优）：
        #   没钥匙→key；有钥匙门没开→door（走到门前 toggle）；
        #   门开但还没进 goal 房间→门东侧一格 E（拉着穿过门口，别斜奔 goal 撞墙）；进了房间→goal。
        if not has_key:
            target = key_pos
        elif not door_open:
            target = door_pos
        elif door_pos is not None and ax <= door_pos[0]:
            target = (door_pos[0] + 1, door_pos[1])        # goal 房间入口
        else:
            target = goal_pos

        # egocentric 分解：fwd=朝向向量，right=朝向右转 90°
        DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]      # 0右 1下 2左 3上
        fwd = DIRS[d]
        rgt = (-fwd[1], fwd[0])
        def _ego(pos):
            if pos is None:
                return False, False, False, False
            dx, dy = pos[0] - ax, pos[1] - ay
            fc = dx * fwd[0] + dy * fwd[1]             # 前向分量
            rc = dx * rgt[0] + dy * rgt[1]             # 右向分量（左为负）
            return (fc > 0 and abs(fc) >= abs(rc),
                    rc < 0 and abs(rc) > abs(fc),
                    rc > 0 and abs(rc) > abs(fc),
                    fc == 1 and rc == 0)
        t_ahead, t_left, t_right, t_reach = _ego(target)
        key_ahead, key_left, key_right, key_reach = _ego(key_pos)
        door_ahead, door_left, door_right, door_reach = _ego(door_pos)
        goal_ahead, goal_left, goal_right, goal_reach = _ego(goal_pos)
        # 正前方是否是墙（避免撞墙 / 提示需要绕行或开门）
        fx, fy = ax + fwd[0], ay + fwd[1]
        fcell = env.grid.get(fx, fy) if (0 <= fx < env.grid.width and 0 <= fy < env.grid.height) else None
        wall_ahead = bool(fcell is not None and fcell.type == "wall")
        agent_cell = env.grid.get(ax, ay)

        return {
            "has_key":     has_key,
            "door_locked": door_locked,
            "door_open":   door_open,
            "carrying":    bool(carrying is not None),
            "t_ahead":     t_ahead,
            "t_left":      t_left,
            "t_right":     t_right,
            "t_reach":     t_reach,
            "wall_ahead":  wall_ahead,
            "door_between_agent_goal": bool(
                door_pos is not None and goal_pos is not None
                and ax <= door_pos[0] < goal_pos[0]),
            "agent_on_door": bool(agent_cell is not None and agent_cell.type == "door"),
            "key_ahead": key_ahead, "key_left": key_left,
            "key_right": key_right, "key_reach": key_reach,
            "door_ahead": door_ahead, "door_left": door_left,
            "door_right": door_right, "door_reach": door_reach,
            "goal_ahead": goal_ahead, "goal_left": goal_left,
            "goal_right": goal_right, "goal_reach": goal_reach,
            "agent_pos":   (ax, ay),
            "agent_dir":   d,
            "door_pos":    door_pos,
            "key_pos":     key_pos,
            "goal_pos":    goal_pos,
        }

    def render(self):
        return self._env.render()

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass
