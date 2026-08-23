import gymnasium as gym
import numpy as np
import minigrid
from minigrid.wrappers import ImgObsWrapper
from collections import deque


# ── BFS 求解器 (demo 数据注入用) + 导航谓词 ─────────────────────────
DIR_VEC = [(1, 0), (0, 1), (-1, 0), (0, -1)]
# MiniGrid 原生动作只用这 5 个: left, right, forward, pickup, toggle。
# drop(4) 这个任务用不到, done(6) 是 minigrid_env.py 里硬编码的 no-op
# (elif action == self.actions.done: pass) —— 全部舍弃, 避免策略 argmax 收敛到无效动作上。
VALID_ACTIONS = [0, 1, 2, 3, 5]


def _walkable(grid, x, y):
    if x < 0 or y < 0 or x >= grid.width or y >= grid.height:
        return False
    obj = grid.get(x, y)
    return obj is None or obj.can_overlap()


def _bfs(grid, start, target_pos, stand_on):
    """BFS 到 "面朝 target" (stand_on=False) 或 "站上 target" (True)。
    返回动作列表 (MiniGrid 原生: 0左转 1右转 2前进), 不可达返回 None。"""
    sx, sy, sd = start
    tx, ty = target_pos

    def hit(x, y, d):
        if stand_on:
            return (x, y) == (tx, ty)
        dx, dy = DIR_VEC[d]
        return (x + dx, y + dy) == (tx, ty)

    if hit(sx, sy, sd):
        return []
    visited = {(sx, sy, sd)}
    queue = deque([((sx, sy, sd), [])])
    while queue:
        (x, y, d), path = queue.popleft()
        for a in (0, 1, 2):
            if a == 0:
                nx, ny, nd = x, y, (d - 1) % 4
            elif a == 1:
                nx, ny, nd = x, y, (d + 1) % 4
            else:
                dx, dy = DIR_VEC[d]
                nx, ny, nd = x + dx, y + dy, d
                if not _walkable(grid, nx, ny):
                    continue
            if (nx, ny, nd) in visited:
                continue
            npath = path + [a]
            if hit(nx, ny, nd):
                return npath
            visited.add((nx, ny, nd))
            queue.append(((nx, ny, nd), npath))
    return None


def _current_target(uw):
    """当前子目标: 没钥匙→钥匙; 有钥匙且门未开→门; 门已开→终点。
    返回 (target_pos 或 None, stand_on, door_obj 或 None)。"""
    from minigrid.core.world_object import Key, Door, Goal
    grid = uw.grid

    def find(cls):
        for i, o in enumerate(grid.grid):
            if o is not None and isinstance(o, cls):
                return (i % grid.width, i // grid.width), o
        return None, None

    has_key = uw.carrying is not None and isinstance(uw.carrying, Key)
    key_pos, _ = find(Key)
    door_pos, door_obj = find(Door)
    goal_pos, _ = find(Goal)

    if not has_key and key_pos is not None:
        return key_pos, False, door_obj
    if door_obj is not None and not door_obj.is_open:
        return door_pos, False, door_obj
    if goal_pos is not None:
        return goal_pos, True, door_obj
    return None, False, door_obj


def _solver_action(uw, rng, epsilon=0.2):
    """闭环求解: 返回 MiniGrid 原生动作 (限制在 VALID_ACTIONS 内)。epsilon 概率随机。"""
    from minigrid.core.world_object import Key

    if rng.random() < epsilon:
        return VALID_ACTIONS[int(rng.integers(0, len(VALID_ACTIONS)))]

    state = (int(uw.agent_pos[0]), int(uw.agent_pos[1]), int(uw.agent_dir))
    has_key = uw.carrying is not None and isinstance(uw.carrying, Key)
    target_pos, stand_on, door_obj = _current_target(uw)
    if target_pos is None:
        return 2  # 找不到目标: 前进兜底

    path = _bfs(uw.grid, state, target_pos, stand_on)
    if path:
        return path[0]
    # 已到位 (path == [])：按阶段执行
    if not has_key:
        return 3  # 面朝钥匙 → pickup
    if door_obj is not None and not door_obj.is_open:
        return 5  # 面朝门 → toggle
    return 2  # 目标是终点 → 前进踩上去


def _nav_preds(uw):
    """5 个导航谓词 (bool): t_ahead, t_left, t_right, t_reach, wall_ahead。
    复用 _current_target + _bfs 的路径规划结果，不重新设计导航几何。"""
    state = (int(uw.agent_pos[0]), int(uw.agent_pos[1]), int(uw.agent_dir))
    grid = uw.grid
    target_pos, stand_on, _ = _current_target(uw)
    t_ahead = t_left = t_right = t_reach = False
    if target_pos is not None:
        path = _bfs(grid, state, target_pos, stand_on)
        if path == []:
            t_reach = True
        elif path:
            step0 = path[0]
            t_left = step0 == 0
            t_right = step0 == 1
            t_ahead = step0 == 2
    dx, dy = DIR_VEC[state[2]]
    wall_ahead = not _walkable(grid, state[0] + dx, state[1] + dy)
    return t_ahead, t_left, t_right, t_reach, wall_ahead


class MiniGrid(gym.Env):
    """MiniGrid 的 7x7x3 符号观测 -> 展平成向量走 MLP(不用 CNN)。

    Model 2 改动:
      - 动作空间裁成 5 个 (VALID_ACTIONS)，去掉 drop/done——done 是 minigrid
        原生的硬编码 no-op，避免策略 argmax 收敛到无效动作上。
      - obs 新增 gt_preds: 9 个接地谓词 GT (±1) [has_key, door_locked,
        door_open, carrying, t_ahead, t_left, t_right, t_reach, wall_ahead]，
        供 NDNFRSSM 接地损失。后 5 个是导航/方向谓词 (当前子目标相对 agent
        在前方/该左转/该右转/已到位, 以及正前方是否有墙)。
      - obs 新增 demo_action: BFS 求解器建议动作 (one-hot，维度=len(VALID_ACTIONS))，
        供 trainer 在 demo 预热期覆盖 agent 动作 (稀疏奖励下往 buffer 注入成功轨迹)。
      两个新 key 都不会进 encoder (mlp_keys='symbolic' 精确匹配)。
    """

    metadata = {}

    def __init__(self, task, size=None, seed=0):
        env_id = task if task.startswith('MiniGrid-') else f'MiniGrid-{task}'
        env = gym.make(env_id)
        env = ImgObsWrapper(env)
        self._env = env
        self._dim = int(np.prod(env.observation_space.shape))  # 147
        self.reward_range = [-np.inf, np.inf]
        self._seed = seed
        self._seeded = False
        self._rng = np.random.default_rng(seed + 777)

    @property
    def observation_space(self):
        return gym.spaces.Dict({
            'symbolic': gym.spaces.Box(-np.inf, np.inf, (self._dim,), dtype=np.float32),
            'gt_preds': gym.spaces.Box(-1.0, 1.0, (9,), dtype=np.float32),
            'demo_action': gym.spaces.Box(0.0, 1.0, (len(VALID_ACTIONS),), dtype=np.float32),
        })

    @property
    def action_space(self):
        return gym.spaces.Discrete(len(VALID_ACTIONS))

    def _vec(self, raw):
        return (raw.astype(np.float32) / 10.0).reshape(-1)

    def _gt_preds(self):
        from minigrid.core.world_object import Key, Door
        uw = self._env.unwrapped
        door = next((o for o in uw.grid.grid
                     if o is not None and isinstance(o, Door)), None)
        t_ahead, t_left, t_right, t_reach, wall_ahead = _nav_preds(uw)
        vals = [
            uw.carrying is not None and isinstance(uw.carrying, Key),
            door.is_locked if door is not None else False,
            door.is_open if door is not None else False,
            uw.carrying is not None,
            t_ahead, t_left, t_right, t_reach, wall_ahead,
        ]
        return np.array([1.0 if v else -1.0 for v in vals], dtype=np.float32)

    def _demo_action(self):
        a_native = _solver_action(self._env.unwrapped, self._rng)
        a_idx = VALID_ACTIONS.index(a_native)
        onehot = np.zeros(len(VALID_ACTIONS), dtype=np.float32)
        onehot[a_idx] = 1.0
        return onehot

    def _obs(self, raw, is_first, is_last, is_terminal):
        return {'symbolic': self._vec(raw), 'gt_preds': self._gt_preds(),
                'demo_action': self._demo_action(),
                'is_first': is_first, 'is_last': is_last,
                'is_terminal': is_terminal}

    def reset(self):
        if not self._seeded:
            raw, _ = self._env.reset(seed=self._seed)
            self._seeded = True
        else:
            raw, _ = self._env.reset()
        return self._obs(raw, True, False, False)

    def step(self, action):
        native_action = VALID_ACTIONS[int(action)]
        raw, reward, terminated, truncated, info = self._env.step(native_action)
        done = bool(terminated or truncated)
        return (self._obs(raw, False, done, bool(terminated)),
                np.float32(reward), done, info)

    def render(self):
        return self._env.render()
