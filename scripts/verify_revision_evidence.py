"""Verify retained revision bytes and recompute key counts without model execution."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(repo, check_models=False):
    revision = repo / "v9"
    evidence = revision / "evidence"
    checks = []

    def check(name, operation):
        try:
            detail = operation()
            checks.append({"name": name, "passed": True, "detail": detail})
        except (AssertionError, KeyError, OSError, ValueError, TypeError) as exc:
            checks.append({"name": name, "passed": False, "detail": str(exc)})

    def retained_bytes():
        entries = read_json(revision / "manifest.json")["files"]
        for entry in entries:
            path = (revision / entry["path"]).resolve()
            assert path.is_relative_to(revision.resolve()), entry["path"]
            assert path.stat().st_size == entry["bytes"], entry["path"]
            assert digest(path) == entry["sha256"], entry["path"]
        return {"files": len(entries), "includes_execution_logs": True}

    def canonical_code():
        entries = read_json(revision / "code_changes/manifest.json")["files"]
        for entry in entries:
            assert digest(repo / entry["path"]) == entry["sha256"], entry["path"]
        return {"files_matching_retained_evaluation_code": len(entries)}

    model_entries = []
    for line in (repo / "models/SHA256SUMS.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            expected, name = line.split(None, 1)
            model_entries.append((expected, name.strip().lstrip("*")))

    def checkpoint_references():
        known = {expected for expected, _ in model_entries}
        references = 0
        for path in evidence.glob("*.json"):
            record = read_json(path)
            if "checkpoint_sha256" in record:
                values = record["checkpoint_sha256"]
                if isinstance(values, str):
                    values = [values]
                assert isinstance(values, list) and values, path.name
                assert all(value in known for value in values), path.name
                references += len(values)
        return {"references_to_retained_model_digests": references,
                "actual_model_bytes_checked": check_models}

    def model_bytes():
        for expected, name in model_entries:
            assert digest(repo / name) == expected, name
        return {"model_files": len(model_entries)}

    def direct_seed_splits():
        files = list(evidence.glob("m1_grounded_seed*_direct200.json"))
        files += list(evidence.glob("m2_seed*_direct200_with_probe.json"))
        assert len(files) == 6, "Expected four M1 and two M2 direct evaluations"
        for path in files:
            data = read_json(path)
            actual = [row["actual_seed"] for row in data["episode_provenance"]]
            assert actual == list(range(1000, 1200)), path.name
            split = data.get("split", data.get("offline_probe_split"))
            train, test = set(split["train_actual_seeds"]), set(split["test_actual_seeds"])
            assert len(train) == 140 and len(test) == 60, path.name
            assert not train & test and train | test == set(actual), path.name
            assert split["initial_state_overlap"] == 4, path.name
        return {"evaluations": len(files), "actual_seed_overlap": 0,
                "configuration_overlap_per_split": 4}

    def d3_inputs():
        data = read_json(evidence / "d3_interface_controls.json")
        records = data["records"]
        assert len(records) == 300
        signatures = defaultdict(set)
        counts = {}
        for condition in ("soft", "sign_only", "oracle_sign"):
            rows = [row for row in records if row["condition"] == condition]
            assert len(rows) == 100
            assert {row["actual_seed"] for row in rows} == set(range(100))
            counts[condition] = sum(row["success"] for row in rows)
            assert counts[condition] == data["conditions"][condition]["successes"]
            for row in rows:
                signatures[row["actual_seed"]].add(row["initial_state_signature"])
        assert all(len(values) == 1 for values in signatures.values())
        assert counts == {"soft": 100, "sign_only": 13, "oracle_sign": 100}
        return counts

    def m3_records():
        actor_summaries = []
        comparisons = 0
        for actor in (0, 1):
            name = f"m3_seed{actor}_corrected_500.json"
            source = revision / "historical_sources" / name
            historical = read_json(source)
            revised = read_json(evidence / f"m3_actor{actor}_unique200_historical.json")
            assert digest(source) == revised["source_sha256"]
            canonical = repo / "results/memory_experiment_3_m3" / name
            assert read_json(canonical) == historical, f"Historical JSON content differs: {name}"
            groups = defaultdict(list)
            for row in historical["records"]:
                groups[row["seed"] % 200].append(row)
            assert len(historical["records"]) == 500 and len(groups) == 200
            for rows in groups.values():
                normalized = [{key: value for key, value in row.items() if key != "seed"}
                              for row in rows]
                assert all(row == normalized[0] for row in normalized)
            conditions = ("baseline", "cue_early", "cue_mid", "cue_late", "wall_mid")
            for row in revised["records"]:
                original = groups[row["actual_seed"]][0]
                assert all(row[name] == original[name] for name in conditions)
            correct = [row for row in revised["records"] if row["baseline"]["outcome"] == "target"]
            endpoints = Counter(row["cue_mid"]["outcome"] for row in correct)
            assert len(correct) == 92 and endpoints == {"target": 33, "timeout": 59}
            diagnostic = read_json(evidence / f"m3_actor{actor}_configuration_grounding.json")
            assert len(diagnostic["records"]) == 4 and diagnostic["bootstrap_resamples"] == 0
            by_signature = {row["initial_state_signature"]: row for row in diagnostic["records"]}
            assert len(by_signature) == 4
            for row in revised["records"]:
                representative = by_signature[row["initial_state_signature"]]
                for name in conditions:
                    assert row[name] == {key: representative[name][key] for key in ("outcome", "steps")}
                    comparisons += 1
            for row in diagnostic["records"]:
                for name in ("cue_early", "cue_mid", "cue_late", "wall_mid"):
                    result = row[name]
                    trace = result["intervention_trace"]
                    assert len(trace) == result["intervened_steps"]
                    for step in trace:
                        raw, supplied, index = step["predicted_atoms"], step["policy_atoms"], step["edited_atom_index"]
                        assert supplied[index] == -raw[index]
                        assert all(a == b for i, (a, b) in enumerate(zip(raw, supplied)) if i != index)
                    if name.startswith("cue_"):
                        assert trace and all(step["cue_sign_correct"] for step in trace)
                        assert all((step["predicted_cue"] > 0) == bool(step["cue_truth"]) for step in trace)
            actor_summaries.append({"actor": actor, "unique_seeds": 200,
                                    "initial_configurations": 4, "baseline_correct": 92,
                                    "cue_mid_target_to_other": 0, "cue_mid_target_to_timeout": 59})
        return {"actors": actor_summaries, "historical_endpoint_timing_comparisons": comparisons,
                "new_model_execution": False}

    check("retained_file_hashes", retained_bytes)
    check("canonical_evaluation_code_matches_snapshot", canonical_code)
    check("checkpoint_digest_references", checkpoint_references)
    if check_models:
        check("actual_model_file_hashes", model_bytes)
    check("M1_M2_direct_seed_partitions", direct_seed_splits)
    check("D3_input_control_counts", d3_inputs)
    check("M3_deduplication_and_grounding_traces", m3_records)
    return {"passed": all(row["passed"] for row in checks), "checks": checks,
            "scope": "File integrity and recomputation of retained observations; not proof of execution dates or timely submission."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check-models", action="store_true", help="Also hash downloaded Git-LFS model bytes")
    parser.add_argument("--out", type=Path, help="Optional report outside the retained v9 directory")
    args = parser.parse_args()
    if not __debug__:
        parser.error("Run without Python -O/-OO so all verification checks execute")
    repo = args.repo.resolve()
    if args.out and args.out.resolve().is_relative_to(repo / "v9"):
        parser.error("Write new reports outside v9 so retained evidence is preserved")
    result = verify(repo, args.check_models)
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
