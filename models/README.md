# Model artifacts

Every `.pt` file is a complete Dreamer agent checkpoint except `memory_experiment_3_m3/frozen_symbolic_head.pt`, which is the standalone Stage-A head. See `SHA256SUMS.txt` for integrity checks.

- `doorkey/doorkey_experiment_1_d1/`: continuous-policy DoorKey baseline used for D1.
- `doorkey/doorkey_experiment_3_d3/`: nine-atom policy used for D3.
- `doorkey/relational_stress_test/`: 19-atom, 50k R1 checkpoint.
- `memory/memory_experiment_1_m1/`: Grounded/Plain seed checkpoints at paper-reported comparison budgets.
- `memory/memory_experiment_2_m2/`: two 84k online-BCE checkpoints.
- `memory/memory_experiment_3_m3/`: shared world model, frozen head, two 400-demo actors, and zero-demo control.

Training overlays and commands are in each experiment's `config.yaml` and `docs/EXPERIMENTS.md`. Model filenames state seed, budget, and role; no file called simply `latest.pt` remains.