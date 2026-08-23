"""Re-evaluate staged M3 cue flips without counting timeouts as redirection."""

import argparse
import json
import pathlib
import sys

import numpy as np


REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import staged_stage_c_eval as stage_c


def outcome(result):
    if result["truncated"]:
        return "timeout"
    fx, fy = result["final_pos"]
    key_distance = abs(fx - result["key_pos"][0]) + abs(fy - result["key_pos"][1])
    ball_distance = abs(fx - result["ball_pos"][0]) + abs(fy - result["ball_pos"][1])
    if key_distance == ball_distance:
        return "neither"
    chosen_key = key_distance < ball_distance
    return "target" if chosen_key == bool(result["cue_is_key"]) else "other"


def summarize(base, changed, bootstrap, seed):
    labels = ("target", "other", "timeout", "neither")
    base_out = np.array([x["outcome"] for x in base])
    changed_out = np.array([x["outcome"] for x in changed])
    n = len(base)
    transition = {
        f"{a}_to_{b}": int(np.sum((base_out == a) & (changed_out == b)))
        for a in labels for b in labels
    }
    base_other = base_out == "other"
    changed_other = changed_out == "other"
    paired_gain = changed_other.astype(float) - base_other.astype(float)
    rng = np.random.default_rng(seed)
    means = np.array([
        paired_gain[rng.integers(0, n, n)].mean() for _ in range(bootstrap)
    ]) if n and bootstrap else np.array([paired_gain.mean() if n else np.nan])
    target_mask = base_out == "target"
    target_n = int(target_mask.sum())
    target_to_other = int(np.sum(target_mask & (changed_out == "other")))
    target_to_timeout = int(np.sum(target_mask & (changed_out == "timeout")))
    return {
        "n": n,
        "base": {name: int(np.sum(base_out == name)) for name in labels},
        "changed": {name: int(np.sum(changed_out == name)) for name in labels},
        "other_rate_gain": float(paired_gain.mean()) if n else None,
        "other_rate_gain_ci95": [float(np.quantile(means, .025)), float(np.quantile(means, .975))],
        "baseline_target_n": target_n,
        "baseline_target_to_other_rate": target_to_other / target_n if target_n else None,
        "baseline_target_to_timeout_rate": target_to_timeout / target_n if target_n else None,
        "transitions": transition,
    }


def evaluate(checkpoint, rundir, distill_meta, device, episodes, seed_start, bootstrap):
    agent, config = stage_c.load(checkpoint, device, distill_meta_path=distill_meta, logdir=rundir)
    wm, behavior = agent._wm, agent._task_behavior
    assert wm._sym_policy_input == "atoms"
    records = []
    for seed in range(seed_start, seed_start + episodes):
        rows, base_result = stage_c.rollout(wm, behavior, seed, config.time_limit, episode_id=seed)
        hide = next((i for i, row in enumerate(rows) if row["cue_known"] and not row["cue_visible"]), None)
        if hide is None:
            continue
        last = max(hide, len(rows) - 1)
        points = {"early": hide, "mid": hide + (last - hide) // 2, "late": last}
        record = {
            "seed": seed,
            "cue_is_key": int(base_result["cue_is_key"]),
            "baseline": {"outcome": outcome(base_result), "steps": base_result["steps"]},
            "flip_steps": points,
        }
        for timing, step in points.items():
            _, result = stage_c.rollout(
                wm, behavior, seed, config.time_limit, flip_idx=0, flip_start=step
            )
            record[f"cue_{timing}"] = {"outcome": outcome(result), "steps": result["steps"]}
        _, result = stage_c.rollout(
            wm, behavior, seed, config.time_limit, flip_idx=1, flip_start=points["mid"]
        )
        record["wall_mid"] = {"outcome": outcome(result), "steps": result["steps"]}
        records.append(record)

    base = [x["baseline"] for x in records]
    comparisons = {
        name: summarize(base, [x[name] for x in records], bootstrap, 20260815 + i)
        for i, name in enumerate(("cue_early", "cue_mid", "cue_late", "wall_mid"))
    }
    primary = comparisons["cue_mid"]
    control = comparisons["wall_mid"]
    supported = bool(
        primary["other_rate_gain"] >= .20
        and primary["other_rate_gain_ci95"][0] > 0
        and primary["baseline_target_to_other_rate"] is not None
        and primary["baseline_target_to_other_rate"] >= .50
        and primary["baseline_target_to_other_rate"] > primary["baseline_target_to_timeout_rate"]
        and primary["other_rate_gain"] >= control["other_rate_gain"] + .10
    )
    return {
        "checkpoint": str(pathlib.Path(checkpoint).resolve()),
        "episodes_requested": episodes,
        "episodes_valid": len(records),
        "evaluation_seeds": [seed_start, seed_start + episodes - 1],
        "outcome_definition": "target/edited-target(other)/timeout/neither; timeout is never redirection",
        "primary_condition": "cue_mid",
        "predeclared_support_gate": {
            "other_gain_min": .20,
            "paired_ci_lower_above_zero": True,
            "baseline_target_to_other_min": .50,
            "target_to_other_exceeds_target_to_timeout": True,
            "cue_gain_exceeds_wall_gain_by": .10,
            "passed": supported,
        },
        "comparisons": comparisons,
        "records": records,
    }


def self_test():
    target = {"truncated": False, "final_pos": (1, 1), "key_pos": (1, 2), "ball_pos": (5, 5), "cue_is_key": 1}
    other = dict(target, final_pos=(5, 4))
    timeout = dict(target, truncated=True)
    assert outcome(target) == "target"
    assert outcome(other) == "other"
    assert outcome(timeout) == "timeout"
    summary = summarize(
        [{"outcome": "target"}, {"outcome": "target"}],
        [{"outcome": "other"}, {"outcome": "timeout"}], 100, 0,
    )
    assert summary["baseline_target_to_other_rate"] == .5
    assert summary["baseline_target_to_timeout_rate"] == .5
    print("PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--rundir")
    parser.add_argument("--distill-meta")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--out")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    missing = [name for name in ("checkpoint", "rundir", "distill_meta", "out") if not getattr(args, name)]
    if missing:
        parser.error("required unless --self-test: " + ", ".join(missing))
    result = evaluate(
        args.checkpoint, args.rundir, args.distill_meta, args.device,
        args.episodes, args.seed_start, args.bootstrap,
    )
    pathlib.Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "valid": result["episodes_valid"],
        "gate": result["predeclared_support_gate"],
        "comparisons": result["comparisons"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
