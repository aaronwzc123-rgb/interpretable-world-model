# Original result artifacts

This directory preserves the original experiment records. The final manuscript also uses the retained revision records in [`v9/evidence/`](../v9/evidence/); start with the [revision index](../v9/README.md) and [current result ledger](../docs/RESULTS.md). The revision snapshot is published on 29 October 2026 and preserves evaluations recorded in September 2026. Historical files here have not been overwritten.

- `doorkey_experiment_1_d1/`: frozen trajectory archive and unified episode-disjoint probe result.
- `doorkey_experiment_3_d3/`: 100-episode ablation and fixed-state directed edits.
- `memory_experiment_1_m1/`: matched 52.5k and final-budget Grounded/Plain seed evaluations.
- `memory_experiment_2_m2/`: two final 84k BCE-head evaluations.
- `memory_experiment_3_m3/`: Stage-A/B compact metadata plus corrected 500-pair endpoint records for two actor seeds. These are nominal runs with repeated actual seeds, not 500 independent seeds; the final deduplicated counts are 92/200 baseline-correct, 0/92 semantic redirections, and 59/92 cue-mid timeouts per actor.
- `relational_stress_test/`: curated frozen summary of the final executed stress-test notebook.
- `appendix_history/`: compact historical evidence referenced only by the dissertation appendix.

The old M2 seed-1 BA of 45.9% and old D3 state-bootstrap intervals are retained for traceability; the final direct-seed BA is 44.8%, and the final D3 intervals resample complete episodes. See the revision records for these changes and for M1 direct-seed evaluations and M3 grounding diagnostics.

Raw training logs, scheduler state, PID files, TensorBoard data, replay buffers, and intermediate evaluation probes are intentionally excluded from this original result directory. Do not treat a curated summary as a raw evaluator dump; each file states its provenance. Write new reproduction outputs under ignored `runs/reproduced_v9/` instead of replacing either original or revision evidence.
