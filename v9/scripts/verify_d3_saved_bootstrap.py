"""Recompute all saved D3 clustered intervals; never modifies source evidence."""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo))
    from scripts.evaluate_d3_directed_actions import bootstrap_episode_mean
    saved = json.loads(args.source.read_text(encoding='utf-8'))
    results = []
    for index, (label, records) in enumerate(saved['state_records'].items()):
        if not records or 'signed_effect' not in records[0]:
            continue
        expected = saved['atoms'][label]['semantic_signed_effect_ci95']
        actual = bootstrap_episode_mean([r['signed_effect'] for r in records],
                    [r['actual_seed'] for r in records], saved['actual_evaluation_seeds'],
                    5000, 20260815 + index)
        results.append({'label': label, 'bootstrap_draws': 5000,
                        'bootstrap_seed': 20260815 + index, 'saved_interval': expected,
                        'recomputed_interval': actual, 'exact_match': actual == expected})
    summary = {'source': args.source.name,
               'source_sha256': hashlib.sha256(args.source.read_bytes()).hexdigest(),
               'method': 'Percentile bootstrap of 100 complete episodes; pooled-state mean',
               'all_intervals_match_exactly': all(r['exact_match'] for r in results),
               'intervals': results}
    args.out.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    if not summary['all_intervals_match_exactly']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
