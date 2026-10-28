"""Re-evaluate staged M3 cue flips without counting timeouts as redirection."""

import argparse
import hashlib
import time
from collections import Counter
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
        "other_rate_gain_ci95": [float(np.quantile(means, .025)), float(np.quantile(means, .975))] if n and bootstrap else None,
        "baseline_target_n": target_n,
        "baseline_target_to_other_rate": target_to_other / target_n if target_n else None,
        "zero_redirection_upper95_one_sided": 1 - .05 ** (1 / target_n)
            if target_n and target_to_other == 0 and bootstrap else None,
        "baseline_target_to_timeout_rate": target_to_timeout / target_n if target_n else None,
        "transitions": transition,
    }


def assert_unique_actual_seeds(provenance):
    actual = [p["actual_seed"] for p in provenance]
    if len(set(actual)) != len(actual):
        repeated = {s: n for s, n in Counter(actual).items() if n > 1}
        raise ValueError(f"Repeated actual evaluation seeds: {repeated}. Use direct mode or reduce the cohort.")


def seed_inventory(seeds, seed_mode, time_limit):
    result = []
    for seed in seeds:
        env = stage_c.M.MiniGrid("memoryS7_cuestart", mode="eval", seed=seed,
                                max_steps=time_limit, eval_seed_mode=seed_mode)
        try:
            env.reset()
            result.append(env.episode_provenance)
        finally:
            env.close()
    assert_unique_actual_seeds(result)
    return result


def result_record(result):
    trace = result["intervention_trace"]
    signs = [r["predicted_cue_sign"] for r in trace]
    return {
        "outcome": outcome(result), "steps": result["steps"],
        "truncated": result["truncated"],
        "initial_state_signature": result["initial_state_signature"],
        "intervention_operation": result["intervention_operation"],
        "grounding_at_intervention": trace[0] if trace else None,
        "intervened_steps": len(trace),
        "cue_correct_steps": sum(r["cue_sign_correct"] for r in trace),
        "cue_prediction_sign_switches": sum(a != b for a, b in zip(signs, signs[1:])),
        "intervention_trace": trace,
    }


def cohort_comparisons(records, names, bootstrap):
    result = {}
    for i, name in enumerate(names):
        selected = [r for r in records if r.get(name) is not None]
        result[name] = summarize([r["baseline"] for r in selected],
                                 [r[name] for r in selected], bootstrap, 20260815 + i)
    return result


def evaluate(checkpoint, rundir, distill_meta, device, episodes, seed_start, bootstrap,
             seed_mode="direct", representative_configs=False, zero_controls=False):
    agent, config = stage_c.load(checkpoint, device, distill_meta_path=distill_meta, logdir=rundir)
    wm, behavior = agent._wm, agent._task_behavior
    assert wm._sym_policy_input == "atoms"
    inventory = seed_inventory(range(seed_start, seed_start + episodes), seed_mode, config.time_limit)
    selected = inventory
    if representative_configs:
        by_signature = {}
        for entry in inventory:
            by_signature.setdefault(entry["initial_state_signature"], entry)
        selected = list(by_signature.values())
    records, skipped = [], []
    names = ["cue_early", "cue_mid", "cue_late", "wall_mid"]
    if zero_controls:
        names += ["zero_all", "zero_cue"]
    started = time.perf_counter()
    for episode_i, provenance in enumerate(selected):
        seed = provenance["requested_seed"]
        rows, base_result = stage_c.rollout(wm, behavior, seed, config.time_limit,
                                            episode_id=seed, eval_seed_mode=seed_mode)
        hide = next((i for i, row in enumerate(rows) if row["cue_known"] and not row["cue_visible"]), None)
        if hide is None:
            skipped.append(dict(**provenance, reason="cue_never_hidden_after_known"))
            continue
        last = max(hide, len(rows) - 1)
        points = {"early": hide, "mid": hide + (last - hide) // 2, "late": last}
        record = {
            "seed": seed, **provenance,
            "cue_is_key": int(base_result["cue_is_key"]),
            "baseline": result_record(base_result), "flip_steps": points,
        }
        for timing, step in points.items():
            _, result = stage_c.rollout(wm, behavior, seed, config.time_limit,
                flip_idx=0, flip_start=step, eval_seed_mode=seed_mode)
            record[f"cue_{timing}"] = result_record(result)
        _, result = stage_c.rollout(wm, behavior, seed, config.time_limit,
            flip_idx=1, flip_start=points["mid"], eval_seed_mode=seed_mode)
        record["wall_mid"] = result_record(result)
        if zero_controls:
            for mode in ("zero_all", "zero_cue"):
                _, result = stage_c.rollout(wm, behavior, seed, config.time_limit,
                    force_register=mode, eval_seed_mode=seed_mode)
                record[mode] = result_record(result)
        records.append(record)
        print(f"[progress] {episode_i + 1}/{len(selected)} seed={seed} "
              f"baseline={record['baseline']['outcome']} elapsed={time.perf_counter()-started:.1f}s", flush=True)

    if representative_configs:
        bootstrap = 0  # finite configuration census: descriptive counts, no sampling intervals
    comparisons = cohort_comparisons(records, names, bootstrap)
    grounding_comparisons = {}
    grounding_coverage = {}
    for name in ("cue_early", "cue_mid", "cue_late"):
        correct = [r for r in records if r[name]["grounding_at_intervention"] and
                   r[name]["grounding_at_intervention"]["cue_sign_correct"]]
        grounding_comparisons[name] = cohort_comparisons(correct, [name], bootstrap)[name]
        grounding_coverage[name] = {
            "evaluated": len(records), "grounding_correct_at_start": len(correct),
            "baseline_target_grounding_correct": sum(r["baseline"]["outcome"] == "target" for r in correct),
            "continuous_cue_correct": sum(r[name]["cue_correct_steps"] == r[name]["intervened_steps"]
                                           and r[name]["intervened_steps"] > 0 for r in records),
            "trajectories_with_cue_prediction_sign_switch": sum(r[name]["cue_prediction_sign_switches"] > 0
                                                                  for r in records),
        }
    primary, control = comparisons["cue_mid"], comparisons["wall_mid"]
    supported = bool(primary["n"] and primary["other_rate_gain"] >= .20
        and primary["other_rate_gain_ci95"] is not None
        and primary["other_rate_gain_ci95"][0] > 0
        and primary["baseline_target_to_other_rate"] is not None
        and primary["baseline_target_to_other_rate"] >= .50
        and primary["baseline_target_to_other_rate"] > primary["baseline_target_to_timeout_rate"]
        and primary["other_rate_gain"] >= control["other_rate_gain"] + .10)
    signatures = Counter(p["initial_state_signature"] for p in inventory)
    return {
        "schema_version": 2,
        "checkpoint": str(pathlib.Path(checkpoint).resolve()),
        "checkpoint_sha256": hashlib.sha256(pathlib.Path(checkpoint).read_bytes()).hexdigest(),
        "distill_meta": str(pathlib.Path(distill_meta).resolve()),
        "device": device, "time_limit": config.time_limit,
        "bootstrap_resamples": bootstrap,
        "inferential_intervals_enabled": bool(bootstrap) and not representative_configs,
        "episodes_requested": episodes, "episodes_evaluated": len(selected),
        "episodes_valid": len(records), "skipped": skipped,
        "evaluation_seeds": [seed_start, seed_start + episodes - 1],
        "eval_seed_mode": seed_mode, "unique_actual_seeds": len(inventory),
        "unique_initial_state_signatures": len(signatures),
        "initial_state_multiplicities": dict(signatures), "seed_inventory": inventory,
        "representative_config_diagnostic": representative_configs,
        "sampling_statement": ("One newly executed rollout per initial configuration and condition; "
            "unweighted descriptive diagnostic, NOT a fresh run for each seed or an independent sample expansion."
            if representative_configs else "One newly executed rollout per actual seed and condition. "
            "Distinct seeds need not produce distinct initial configurations; report multiplicities."),
        "intervention_operation": "negate_current_output at every step from the selected onset; "
            "not a fixed counterfactual value and not a truth-labelled flip",
        "outcome_definition": "target/other/timeout/neither; timeout is never redirection",
        "zero_controls_in_this_cohort": zero_controls,
        "primary_condition": "cue_mid",
        "predeclared_support_gate": {
            "other_gain_min": .20, "paired_ci_lower_above_zero": True,
            "baseline_target_to_other_min": .50,
            "target_to_other_exceeds_target_to_timeout": True,
            "cue_gain_exceeds_wall_gain_by": .10, "passed": supported,
        },
        "comparisons": comparisons, "grounding_correct_comparisons": grounding_comparisons,
        "grounding_coverage": grounding_coverage,
        "elapsed_seconds": time.perf_counter() - started,
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
    parser.add_argument("--seed-mode", choices=("pool", "direct"), default="direct")
    parser.add_argument("--representative-configs", action="store_true",
                        help="execute only one seed per initial state; descriptive diagnostic only")
    parser.add_argument("--zero-controls", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--out")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    missing = [name for name in ("checkpoint", "rundir", "distill_meta", "out") if not getattr(args, name)]
    if missing:
        parser.error("required unless --self-test: " + ", ".join(missing))
    import torch
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    result = evaluate(
        args.checkpoint, args.rundir, args.distill_meta, args.device,
        args.episodes, args.seed_start, args.bootstrap, args.seed_mode,
        args.representative_configs, args.zero_controls,
    )
    pathlib.Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "valid": result["episodes_valid"],
        "gate": result["predeclared_support_gate"],
        "comparisons": result["comparisons"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
