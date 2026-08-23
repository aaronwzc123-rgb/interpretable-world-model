# Result artifacts

- `doorkey_experiment_1_d1/`: frozen trajectory archive and unified episode-disjoint probe result.
- `doorkey_experiment_3_d3/`: 100-episode ablation and fixed-state directed edits.
- `memory_experiment_1_m1/`: matched 52.5k and final-budget Grounded/Plain seed evaluations.
- `memory_experiment_2_m2/`: two final 84k BCE-head evaluations.
- `memory_experiment_3_m3/`: Stage-A/B compact metadata plus corrected 500-pair endpoint records for two actor seeds.
- `relational_stress_test/`: curated frozen summary of the final executed stress-test notebook.
- `appendix_history/`: compact historical evidence referenced only by the dissertation appendix.

Raw training logs, scheduler state, PID files, TensorBoard data, replay buffers, and intermediate evaluation probes are intentionally excluded. Do not treat a curated summary as a raw evaluator dump; each file states its provenance.