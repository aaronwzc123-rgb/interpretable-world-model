"""Local interactive MiniGrid + Dreamer checkpoint explorer.

Run with the project environment:
    .\.venv\Scripts\python.exe interactive_demo.py
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import pathlib
import sys
import tempfile
import threading
import types
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import ruamel.yaml as yaml
import torch
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import dreamer
import tools
import envs.minigrid as minigrid_env
import envs.wrappers as wrappers


ACTIONS = [
    {"id": 0, "name": "左转", "key": "A / ←", "english": "left"},
    {"id": 1, "name": "右转", "key": "D / →", "english": "right"},
    {"id": 2, "name": "前进", "key": "W / ↑", "english": "forward"},
    {"id": 3, "name": "拾取", "key": "E", "english": "pickup"},
    {"id": 4, "name": "开关门", "key": "Space", "english": "toggle"},
]

MAP_NAMES = {
    "doorkey6x6": "DoorKey · 6×6",
    "doorkey8x8": "DoorKey · 8×8",
    "doorkey6x6_rel": "DoorKey Relational · 6×6",
    "doorkey8x8_rel": "DoorKey Relational · 8×8",
    "doorkey16x16_rel": "DoorKey Relational · 16×16",
    "unlockpickup": "UnlockPickup",
    "memoryS7": "Memory · S7",
    "memoryS7_cuestart": "Memory · S7 Cue-start",
    "memoryS9": "Memory · S9",
    "memoryS11": "Memory · S11",
    "memoryS13": "Memory · S13",
}


def recursive_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            recursive_update(base[key], value)
        else:
            base[key] = value


def build_config(overlays, device):
    source = yaml.safe_load((ROOT / "configs.yaml").read_text(encoding="utf-8"))
    values = {}
    for name in overlays:
        recursive_update(values, source[name])
    parser = argparse.ArgumentParser(add_help=False)
    for key, value in sorted(values.items()):
        kind = tools.args_type(value)
        parser.add_argument(f"--{key}", type=kind, default=kind(value))
    config = parser.parse_args([])
    config.device = device
    config.num_actions = len(ACTIONS)
    config.compile = False
    return config


def checkpoint_paths():
    paths = {
        path.resolve()
        for path in (ROOT / "models").rglob("*.pt")
        if path.name not in {"frozen_symbolic_head.pt", "shared_world_model.pt"}
    }
    return sorted(
        paths,
        key=lambda p: (
            0 if "models" in p.parts else 1 if p.name == "latest.pt" else 2,
            p.as_posix().lower(),
        ),
    )


def model_catalog():
    result = []
    for index, path in enumerate(checkpoint_paths()):
        rel = path.relative_to(ROOT).as_posix()
        lowered = rel.lower()
        if "doorkey" in lowered or "m2b_distilled" in lowered or "model3_rel" in lowered:
            task = "DoorKey"
        else:
            task = "Memory"
        group = "论文模型与保留检查点"
        result.append({
            "id": str(index),
            "path": rel,
            "name": f"{path.parent.name} / {path.name}",
            "group": group,
            "task": task,
            "latest": path.name == "latest.pt",
        })
    return result


def _tensor_shape(state, suffix):
    for key, value in state.items():
        if key.endswith(suffix):
            return tuple(value.shape)
    return None


def infer_overlays(state):
    if any("dynamics.prior_ndnf" in key for key in state):
        raise ValueError("此归档 checkpoint 依赖已移除的 m2rebuild 运行源码，不能由当前 Xdreamer 推理。")
    sym_shape = _tensor_shape(state, "_wm._sym_head.dnf.disj.w")
    enc_shape = _tensor_shape(state, "_wm.encoder._mlp.layers.Encoder_linear0.weight")
    actor_shape = None
    for key, value in state.items():
        if key.startswith("_task_behavior.actor") and key.endswith("weight") and len(value.shape) == 2:
            actor_shape = tuple(value.shape)
            break
    has_shape = any(key.startswith("_wm._shape_head") for key in state)
    if has_shape:
        return ["defaults", "minigrid", "minigrid_memory", "minigrid_memory_shape"]
    if not sym_shape:
        return ["defaults", "minigrid"]
    atoms = sym_shape[0]
    if atoms == 19:
        return ["defaults", "minigrid", "minigrid_symbolic_nav_rel"]
    if atoms == 9:
        return ["defaults", "minigrid", "minigrid_symbolic_nav"]
    if atoms == 6:
        return ["defaults", "minigrid", "minigrid_memory", "minigrid_memory_symbolic_nav"]
    if atoms != 10:
        raise ValueError(f"无法识别该符号模型的原子维度：{atoms}")
    overlays = [
        "defaults", "minigrid", "minigrid_memory", "minigrid_memory_cuestart",
        "minigrid_memory_symbolic_nav_v2", "minigrid_memory_symbolic_nav_v2_entropy",
    ]
    if actor_shape and actor_shape[1] == 1280:
        overlays.append("minigrid_memory_symbolic_nav_v2_bce")
    if enc_shape and enc_shape[1] == 158:
        overlays.append("minigrid_memory_symbolic_nav_v2_recur")
        if any(key.endswith("_sym_head.delta_value") for key in state):
            overlays.append("minigrid_memory_symbolic_nav_v2_adaptive_delta")
    return overlays


def json_value(value):
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    return value


class Session:
    def __init__(self, task, seed, view, checkpoint):
        self.task = task
        self.seed = int(seed) % minigrid_env.TEST_POOL
        self.view = view
        self.checkpoint = checkpoint
        self.step_count = 0
        self.reward = 0.0
        self.total_reward = 0.0
        self.done = False
        self.latent = None
        self.previous_action = None
        self.current_atoms = None

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=True)
        state = checkpoint_data.get("agent_state_dict")
        if not isinstance(state, dict):
            raise ValueError("所选文件不是完整 Dreamer agent checkpoint。")
        self.config = build_config(infer_overlays(state), device)

        self.base = minigrid_env.MiniGrid(
            task, mode="eval", seed=self.seed, max_steps=self.config.time_limit,
            render_obs=False, emit_labels=False,
        )
        self.env = self.base
        if getattr(self.config, "sym_recur_atoms", False):
            self.env = wrappers.AtomRegisterWrapper(self.base, len(self.config.sym_labels))
        action_space = wrappers.OneHotAction(self.env).action_space
        logger = types.SimpleNamespace(step=0)
        self.agent = dreamer.Dreamer(
            self.env.observation_space, action_space, self.config, logger, dataset=None
        ).to(device)
        missing, unexpected = self.agent.load_state_dict(state, strict=False)
        bad_missing = [key for key in missing if not key.endswith(("anneal_steps", "delta_value", "target_acc"))]
        if bad_missing or unexpected:
            raise ValueError(
                "checkpoint 与当前模型结构不兼容："
                f"missing={bad_missing[:3]}, unexpected={unexpected[:3]}"
            )
        self.agent.requires_grad_(False)
        self.agent.eval()
        self.obs = self.env.reset()
        self._infer()

    def close(self):
        try:
            self.base.close()
        except Exception:
            pass

    @torch.no_grad()
    def _infer(self):
        batch = {
            key: np.asarray(value)[None]
            for key, value in self.obs.items()
            if not key.startswith("log_")
        }
        wm = self.agent._wm
        data = wm.preprocess(batch)
        embed = wm.encoder(data)
        self.latent, _ = wm.dynamics.obs_step(
            self.latent, self.previous_action, embed, data["is_first"], sample=False
        )
        feat = wm.dynamics.get_feat(self.latent)
        actor = self.agent._task_behavior.actor(wm.augment(feat))
        probabilities = actor.probs.detach().float().cpu().numpy().reshape(-1)
        probabilities = probabilities / probabilities.sum()
        self.probabilities = probabilities.tolist()
        self.recommended = int(np.argmax(probabilities))
        self.feat = feat[0].detach().float().cpu().numpy()

        head = getattr(wm, "_sym_head", None)
        self.atom_values = {}
        if head is not None:
            atoms = head.atoms(feat)[0].detach().float().cpu().numpy()
            self.current_atoms = atoms
            self.atom_values = {name: float(value) for name, value in zip(head.labels, atoms)}
        shape_head = getattr(wm, "_shape_head", None)
        self.grounded_values = {}
        if shape_head is not None:
            values = torch.sigmoid(shape_head.net(feat))[0].detach().float().cpu().numpy()
            self.grounded_values = {name: float(value) for name, value in zip(shape_head.labels, values)}

    def step(self, action):
        if self.done:
            raise ValueError("本回合已经结束，请重新开始。")
        action = int(action)
        if action not in range(len(ACTIONS)):
            raise ValueError("未知动作。")
        if hasattr(self.env, "set_register") and self.current_atoms is not None:
            self.env.set_register(self.current_atoms)
        self.previous_action = torch.nn.functional.one_hot(
            torch.tensor([action], device=self.config.device), len(ACTIONS)
        ).float()
        self.obs, reward, self.done, _ = self.env.step(action)
        self.reward = float(reward)
        self.total_reward += self.reward
        self.step_count += 1
        self._infer()
        return self.state()

    def image(self):
        frame = self.base._env.unwrapped.get_frame(
            tile_size=32, agent_pov=self.view == "partial"
        ).astype(np.uint8)
        output = io.BytesIO()
        Image.fromarray(frame).save(output, format="PNG", optimize=True)
        return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")

    def belief(self):
        deter = self.latent["deter"][0].detach().float().cpu().numpy()
        stoch = self.latent["stoch"][0].detach().float().cpu().numpy()
        if stoch.ndim == 2:
            stoch_index = np.argmax(stoch, axis=-1)
            stoch_confidence = np.max(stoch, axis=-1)
        else:
            stoch_index = stoch.reshape(-1)
            stoch_confidence = np.ones_like(stoch_index, dtype=np.float32)
        strongest = np.argsort(np.abs(deter))[-16:][::-1]
        return {
            "deter_dim": int(deter.size),
            "stoch_shape": list(stoch.shape),
            "feat_dim": int(self.feat.size),
            "deter_mean": float(deter.mean()),
            "deter_std": float(deter.std()),
            "deter_norm": float(np.linalg.norm(deter)),
            "strongest": [{"index": int(i), "value": float(deter[i])} for i in strongest],
            "deter": np.round(deter, 5).tolist(),
            "stoch_index": stoch_index.astype(int).tolist(),
            "stoch_confidence": np.round(stoch_confidence, 5).tolist(),
            "atoms": self.atom_values,
            "grounded": self.grounded_values,
        }

    def state(self):
        god = json_value(self.base.god_state())
        return {
            "task": self.task,
            "seed": self.seed,
            "view": self.view,
            "model_view": "partial_7x7",
            "step": self.step_count,
            "reward": self.reward,
            "total_reward": self.total_reward,
            "done": self.done,
            "success": bool(self.done and self.reward > 0),
            "image": self.image(),
            "actions": [dict(action, probability=self.probabilities[action["id"]]) for action in ACTIONS],
            "recommended": self.recommended,
            "belief": self.belief(),
            "god_state": god,
            "device": self.config.device,
            "checkpoint": self.checkpoint.relative_to(ROOT).as_posix(),
        }


CATALOG = model_catalog()
MODEL_BY_ID = {item["id"]: ROOT / item["path"] for item in CATALOG}
SESSION = None
LOCK = threading.RLock()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, format, *args):
        pass

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self.send_response(302)
            self.send_header("Location", "/interactive_demo.html")
            self.end_headers()
            return
        if self.path == "/api/catalog":
            maps = [{"id": key, "name": MAP_NAMES.get(key, key)} for key in minigrid_env.ENV_IDS]
            self.send_json({"maps": maps, "models": CATALOG, "actions": ACTIONS})
            return
        super().do_GET()

    def do_POST(self):
        global SESSION
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            with LOCK:
                if self.path == "/api/start":
                    task = payload.get("task")
                    model_id = str(payload.get("model", ""))
                    view = payload.get("view", "god")
                    if task not in minigrid_env.ENV_IDS:
                        raise ValueError("请选择有效地图。")
                    if model_id not in MODEL_BY_ID:
                        raise ValueError("请选择有效模型。")
                    if view not in {"god", "partial"}:
                        raise ValueError("请选择有效视角。")
                    old = SESSION
                    SESSION = None
                    if old:
                        old.close()
                    SESSION = Session(task, int(payload.get("seed", 0)), view, MODEL_BY_ID[model_id])
                    self.send_json(SESSION.state())
                    return
                if SESSION is None:
                    raise ValueError("请先开始一个回合。")
                if self.path == "/api/step":
                    self.send_json(SESSION.step(payload.get("action")))
                    return
                if self.path == "/api/view":
                    view = payload.get("view")
                    if view not in {"god", "partial"}:
                        raise ValueError("请选择有效视角。")
                    SESSION.view = view
                    self.send_json(SESSION.state())
                    return
            self.send_json({"error": "Not found"}, 404)
        except Exception as error:
            self.send_json({"error": str(error)}, 400)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/interactive_demo.html"
    print(f"MiniGrid Dreamer Lab: {url}", flush=True)
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if SESSION:
            SESSION.close()
        server.server_close()


if __name__ == "__main__":
    main()

