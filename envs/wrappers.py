import datetime
import gym
import numpy as np
import uuid


class TimeLimit(gym.Wrapper):
    def __init__(self, env, duration):
        super().__init__(env)
        self._duration = duration
        self._step = None

    def step(self, action):
        assert self._step is not None, "Must reset environment."
        obs, reward, done, info = self.env.step(action)
        self._step += 1
        if self._step >= self._duration:
            done = True
            if "discount" not in info:
                info["discount"] = np.array(1.0).astype(np.float32)
            self._step = None
        return obs, reward, done, info

    def reset(self):
        self._step = 0
        return self.env.reset()


class AtomRegisterWrapper(gym.Wrapper):
    """M3 全量原子回流架构（v5，2026-07-30）：把上一步 sym_head 算出的原子 atoms(t-1) 拼进
    `grid` 观测，供 encoder 在下一步读取——真实环境路径的"显式符号记忆寄存器"。

    时序：episode 开头 `reset()` 清零寄存器；`step()`/`reset()` 返回的 obs["grid"] 里拼的是
    **当前寄存器值**（即上一次 `set_register()` 写入的值，也就是 atoms(t-1)），不是本步算出的
    新值——新值要等 `Dreamer._policy()` 算完 atoms(t) 后，由 `tools.simulate()` 调用
    `set_register()` 写入，供**下一次** `step()`/`reset()` 使用，满足
    `input(t) = o(t) ⊕ atoms(t-1)` 的规格（见 m3_memoryS7_v5_log.md 第零步说明）。
    """

    def __init__(self, env, atom_dim):
        super().__init__(env)
        self._atom_dim = int(atom_dim)
        self._register = np.zeros(self._atom_dim, dtype=np.float32)
        base_space = env.observation_space.spaces["grid"]
        new_shape = (base_space.shape[0] + self._atom_dim,)
        spaces = dict(env.observation_space.spaces)
        # 边界仅供形状占位（grid 本身是[0,1]，原子是[-1,1]，训练读 preprocess() 时不做 clip，
        # 用 [-1,1] 覆盖两者即可，不影响正确性）
        spaces["grid"] = gym.spaces.Box(-1.0, 1.0, new_shape, np.float32)
        self.observation_space = gym.spaces.Dict(spaces)

    def set_register(self, atoms):
        """外部（tools.simulate()）在算出 atoms(t) 后调用，写入寄存器供下一步 obs 使用。"""
        self._register = np.asarray(atoms, dtype=np.float32).reshape(self._atom_dim)

    def _augment(self, obs):
        obs = dict(obs)
        obs["grid"] = np.concatenate([obs["grid"], self._register]).astype(np.float32)
        return obs

    def reset(self):
        self._register = np.zeros(self._atom_dim, dtype=np.float32)  # episode 开头清零
        return self._augment(self.env.reset())

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._augment(obs), reward, done, info


class NormalizeActions(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._mask = np.logical_and(
            np.isfinite(env.action_space.low), np.isfinite(env.action_space.high)
        )
        self._low = np.where(self._mask, env.action_space.low, -1)
        self._high = np.where(self._mask, env.action_space.high, 1)
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)

    def step(self, action):
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._mask, original, action)
        return self.env.step(original)


class OneHotAction(gym.Wrapper):
    def __init__(self, env):
        assert isinstance(env.action_space, gym.spaces.Discrete)
        super().__init__(env)
        self._random = np.random.RandomState()
        shape = (self.env.action_space.n,)
        space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        space.discrete = True
        self.action_space = space

    def step(self, action):
        index = np.argmax(action).astype(int)
        reference = np.zeros_like(action)
        reference[index] = 1
        if not np.allclose(reference, action):
            raise ValueError(f"Invalid one-hot action:\n{action}")
        return self.env.step(index)

    def reset(self):
        return self.env.reset()

    def _sample_action(self):
        actions = self.env.action_space.n
        index = self._random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference


class RewardObs(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        spaces = self.env.observation_space.spaces
        if "obs_reward" not in spaces:
            spaces["obs_reward"] = gym.spaces.Box(
                -np.inf, np.inf, shape=(1,), dtype=np.float32
            )
        self.observation_space = gym.spaces.Dict(spaces)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if "obs_reward" not in obs:
            obs["obs_reward"] = np.array([reward], dtype=np.float32)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        if "obs_reward" not in obs:
            obs["obs_reward"] = np.array([0.0], dtype=np.float32)
        return obs


class SelectAction(gym.Wrapper):
    def __init__(self, env, key):
        super().__init__(env)
        self._key = key

    def step(self, action):
        return self.env.step(action[self._key])


class UUID(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.id = f"{timestamp}-{str(uuid.uuid4().hex)}"

    def reset(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.id = f"{timestamp}-{str(uuid.uuid4().hex)}"
        return self.env.reset()
