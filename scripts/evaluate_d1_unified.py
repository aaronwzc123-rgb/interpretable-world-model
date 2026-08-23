"""Unified D1 evaluation on one frozen 1280-d DoorKey baseline checkpoint.

The split is by complete episode. Feature selection, scaling, and model fitting use
training episodes only; every reported score below is on held-out test episodes.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import neuraldnf


LABELS = [
    "has_key", "door_locked", "door_open", "carrying",
    "t_ahead", "t_left", "t_right", "t_reach", "wall_ahead",
]


def scores(y, pred):
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "base_rate": float(max(y.mean(), 1.0 - y.mean())),
        "positive_rate": float(np.mean(pred)),
        "n": int(len(y)),
    }


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--top-per-label", type=int, default=6)
    parser.add_argument("--conjunctions", type=int, default=18)
    parser.add_argument("--steps", type=int, default=4000)
    args = parser.parse_args()

    data = np.load(args.traj)
    episodes = np.unique(data["episode"])
    if len(episodes) < 20:
        raise ValueError("D1 requires at least 20 complete episodes")
    rng = np.random.default_rng(args.seed)
    shuffled = rng.permutation(episodes)
    n_train = int(0.70 * len(shuffled))
    n_val = int(0.15 * len(shuffled))
    train_eps = shuffled[:n_train]
    val_eps = shuffled[n_train:n_train + n_val]
    test_eps = shuffled[n_train + n_val:]
    train = np.isin(data["episode"], train_eps)
    test = np.isin(data["episode"], test_eps)

    y_train = np.stack([data[label][train] for label in LABELS], axis=1).astype(int)
    y_test = np.stack([data[label][test] for label in LABELS], axis=1).astype(int)
    feat_train = data["feat"][train].astype(np.float32)
    feat_test = data["feat"][test].astype(np.float32)

    linear = {}
    for index, label in enumerate(LABELS):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, C=1.0, random_state=args.seed),
        )
        model.fit(feat_train, y_train[:, index])
        linear[label] = scores(y_test[:, index], model.predict(feat_test))

    stoch_train = data["stoch"][train].reshape(train.sum(), -1).astype(np.float32)
    stoch_test = data["stoch"][test].reshape(test.sum(), -1).astype(np.float32)
    selected = set()
    centered_x = stoch_train - stoch_train.mean(0, keepdims=True)
    for index in range(len(LABELS)):
        centered_y = y_train[:, index] - y_train[:, index].mean()
        corr = np.abs(centered_x.T @ centered_y) / max(1, len(centered_y))
        selected.update(np.argsort(-corr)[:args.top_per_label].tolist())
    selected = np.array(sorted(selected), dtype=int)
    names = [f"z{unit // 32}_{unit % 32}" for unit in selected]
    x_train = np.where(stoch_train[:, selected] > 0.5, 1.0, -1.0).astype(np.float32)
    x_test = np.where(stoch_test[:, selected] > 0.5, 1.0, -1.0).astype(np.float32)
    target_train = np.where(y_train > 0, 1.0, -1.0).astype(np.float32)

    net, train_accuracy = neuraldnf.fit_dnf(
        x_train, target_train, names, LABELS,
        n_conj=args.conjunctions, steps=args.steps, seed=args.seed, verbose=True,
    )
    with torch.no_grad():
        xt = torch.tensor(x_test)
        soft_pred = (net(xt, 1.0).cpu().numpy() > 0).astype(int)
        hard_pred = (net(xt, 4.0).cpu().numpy() > 0).astype(int)
    ndnf = {
        label: {
            "soft_delta_1": scores(y_test[:, index], soft_pred[:, index]),
            "hard_delta_4": scores(y_test[:, index], hard_pred[:, index]),
        }
        for index, label in enumerate(LABELS)
    }

    result = {
        "protocol": {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_sha256": sha256(args.checkpoint),
            "trajectory": str(Path(args.traj).resolve()),
            "seed": args.seed,
            "episode_split": {
                "train": train_eps.tolist(), "validation": val_eps.tolist(),
                "test": test_eps.tolist(),
            },
            "feature_dimensions": {"feat": int(feat_train.shape[1]), "stoch": int(stoch_train.shape[1])},
            "ndnf_selected_stoch_units": selected.tolist(),
            "ndnf_train_accuracy": float(train_accuracy),
        },
        "task": {
            "success_rate": float(data["success_per_ep"].mean()),
            "successes": int(data["success_per_ep"].sum()),
            "episodes": int(len(data["success_per_ep"])),
            "mean_length": float(data["length_per_ep"].mean()),
        },
        "linear_full_belief": linear,
        "ndnf_selected_stoch": ndnf,
        "rules": net.extract_rules(names, LABELS),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
