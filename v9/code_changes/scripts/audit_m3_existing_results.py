"""Audit and deduplicate historical M3 records without rerunning the model.

This is a correction of retained observations, not new evaluation evidence.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'scripts')]
from evaluate_m3_corrected_flip import cohort_comparisons, seed_inventory


def audit(source, output):
    original = json.loads(source.read_text(encoding='utf-8'))
    groups = defaultdict(list)
    for row in original['records']:
        groups[int(row['seed']) % 200].append(row)
    inconsistent = []
    for seed, rows in groups.items():
        normalized = [{k: v for k, v in row.items() if k != 'seed'} for row in rows]
        if any(row != normalized[0] for row in normalized[1:]):
            inconsistent.append(seed)
    if inconsistent:
        raise ValueError(f'Duplicate-seed records differ: {inconsistent}')
    inventory = seed_inventory(sorted(groups), 'direct', 250)
    provenance = {p['actual_seed']: p for p in inventory}
    records = []
    configurations = defaultdict(list)
    for seed, rows in sorted(groups.items()):
        row = dict(rows[0], **provenance[seed])
        row['historical_requested_seeds'] = [r['seed'] for r in rows]
        row['seed'] = seed
        records.append(row)
        configurations[row['initial_state_signature']].append(row)
    config_summary = []
    for signature, rows in configurations.items():
        exclude = {'seed', 'requested_seed', 'actual_seed', 'eval_seed_mode',
                   'initial_state_signature', 'historical_requested_seeds'}
        values = [{k: v for k, v in row.items() if k not in exclude} for row in rows]
        config_summary.append({
            'signature': signature, 'n_actual_seeds': len(rows),
            'actual_seeds': [r['actual_seed'] for r in rows],
            'all_stored_outcomes_and_timings_identical': all(v == values[0] for v in values),
            'representative_record': rows[0],
        })
    comparisons = cohort_comparisons(records, ['cue_early', 'cue_mid', 'cue_late', 'wall_mid'], 5000)
    result = {
        'audit_type': 'deduplication of historical retained records; no new model execution',
        'source': str(source.resolve()), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'historical_nominal_records': len(original['records']),
        'distinct_actual_seeds': len(records), 'distinct_initial_configurations': len(configurations),
        'duplicate_actual_seed_groups': sum(len(r) > 1 for r in groups.values()),
        'inconsistent_duplicate_seed_groups': inconsistent,
        'historical_seed_mapping': 'requested_seed % TEST_POOL, TEST_POOL=200',
        'statistics_scope': 'Seed-weighted description of a fixed finite pool. Binomial/bootstrap intervals '
            'are conditional seed-sampling references, not configuration-independent generalisation bounds. '
            'The configuration table makes the much smaller structural support explicit.',
        'zero_controls_in_original_source': False,
        'grounding_at_intervention_in_original_source': 'not recorded',
        'comparisons': comparisons, 'configuration_summary': config_summary, 'records': records,
    }
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['historical_nominal_records', 'distinct_actual_seeds',
        'distinct_initial_configurations', 'duplicate_actual_seed_groups', 'inconsistent_duplicate_seed_groups']},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    audit(args.source, args.out)
