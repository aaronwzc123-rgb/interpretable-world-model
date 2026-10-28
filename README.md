# XDreamer: symbolic interfaces for DreamerV3 beliefs

This repository is the research artifact for the dissertation *Readable and Causally Used Symbolic Interfaces for World-Model Beliefs*. It contains the MiniGrid implementation, paper-reported checkpoints, training overlays, retained results, evaluation code, three acceptance notebooks, and an interactive visualisation.

**Start with the [final revision evidence index](v9/README.md) and [current result ledger](docs/RESULTS.md).** The `v9/evidence/` snapshot contains the retained September 2026 evaluations and audits used by the final manuscript. Its corresponding code and tests are now included in the main source tree, with the original code snapshot and patch under `v9/code_changes/`.

This repository update is published on **29 October 2026**. It adds the retained revision evidence that was absent from the original compact submission package and its frozen baseline commit, `7805619c4cb7b4ab51945dc2396351085c431b46`. Publication now does not establish that these files were uploaded with the original submission. Historical result files remain available for comparison; they must not be read as the final revised numbers.

## What the experiment IDs mean

| ID | Paper experiment | Environment | Main artifact |
|---|---|---|---|
| D1 | DoorKey Experiment 1: continuous-belief decoding | `MiniGrid-DoorKey-6x6-v0` | `models/doorkey/doorkey_experiment_1_d1/checkpoint.pt` |
| D2 | DoorKey online N-DNF transition (historical negative result) | DoorKey-6x6 | archive/d2_model2_legacy/runs/m2rebuild_run/m2rebuild_s0/latest.pt |
| D3 | DoorKey Experiment 3: atoms-only causal control | `MiniGrid-DoorKey-6x6-v0` | `models/doorkey/doorkey_experiment_3_d3/checkpoint.pt` |
| M1 | Memory Experiment 1: supervised continuous memory | `memoryS7_cuestart` | `models/memory/memory_experiment_1_m1/` |
| M2 | Memory Experiment 2: online BCE symbolic readout | `memoryS7_cuestart` | `models/memory/memory_experiment_2_m2/` |
| M3 | Memory Experiment 3: staged frozen symbolic interface | `memoryS7_cuestart` | `models/memory/memory_experiment_3_m3/` |
| R1 | Relational map-size stress test | DoorKey 6x6/8x8/16x16 | `models/doorkey/relational_stress_test/` |

D1 and D3 use seed-held-out evaluations from the same DoorKey generator. They are not unseen-layout tests: the generator audit found complete layout-support overlap. MemoryS7 evaluations cover only four complete initial configurations. M3's two actors share the same world model and symbolic head and have unequal training budgets and demonstration sets.

## Quick start

The frozen environment used Python 3.11 and PyTorch 2.8.0+cu126. Install the PyTorch build appropriate for your CPU/CUDA platform first, then install the remaining dependencies:

For a complete setup walkthrough, see [INSTALL.md](INSTALL.md).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch
python -m pip install -r requirements.txt
```

Run the regression suite:

```bash
python -m unittest discover -s tests -v
```

Check the retained revision's file hashes and core result counts without running the models:

```bash
python scripts/verify_revision_evidence.py
```

Add `--check-models` to also verify the retained checkpoint files against their recorded hashes.

The checkpoints are larger than ordinary Git blobs and are configured for Git LFS. Install Git LFS before cloning or pushing a complete copy:

```bash
git lfs install
git lfs pull
```

## Inspect and reproduce the final evidence

The [revision index](v9/README.md) maps results to retained files and distinguishes historical seed deduplication from newly executed diagnostics. The [experiment guide](docs/EXPERIMENTS.md) gives current evaluation commands. No retraining is needed to evaluate the retained checkpoints.

Create a separate output directory before running the examples:

```bash
python -c "from pathlib import Path; Path('runs/reproduced_v9').mkdir(parents=True, exist_ok=True)"
```

The three notebooks in `notebooks/` retain visual inspections and historical outputs:

- `D1_DoorKey_Belief_Decoding.ipynb` (D1 visualisation)
- `D3_DoorKey_Atoms_Only_Causal_Control.ipynb` (D3 visualisation and ablation)
- `M3_MemoryS7_Staged_Frozen_Interface.ipynb` (M3 visualisation; the opening note identifies superseded historical metrics)

On Windows, double-click `run_acceptance_notebooks.bat` to open them with the project environment. Saved notebook output is not a new execution and can use a historical protocol. The final numerical claims are indexed in `docs/RESULTS.md` and `v9/evidence/`. M1 and M2 have scripted direct-seed evaluations; their project-defined gates are not external preregistrations.

Examples:

```bash
python scripts/evaluate_d1_unified.py \
  --traj results/doorkey_experiment_1_d1/trajectory.npz \
  --checkpoint models/doorkey/doorkey_experiment_1_d1/checkpoint.pt \
  --out runs/reproduced_v9/d1_unified.json

python scripts/evaluate_d3_directed_actions.py \
  --checkpoint models/doorkey/doorkey_experiment_3_d3/checkpoint.pt \
  --episodes 100 --seed-start 0 --bootstrap 5000 --torch-threads 1 \
  --device cpu --out runs/reproduced_v9/d3_episode_clustered.json

python scripts/evaluate_d3_interface_controls.py \
  --repo . --episodes 100 --device cpu \
  --out runs/reproduced_v9/d3_interface_controls.json

python scripts/evaluate_m1_cuestart.py \
  --checkpoint models/memory/memory_experiment_1_m1/grounded_seed0/checkpoint_084000.pt \
  --condition grounded --episodes 200 --seed-start 1000 --eval-seed-mode direct \
  --coverage-controller fixed_upper --split-seed 20260813 --torch-threads 1 --device cpu \
  --out runs/reproduced_v9/m1_grounded_seed0_fixed_upper_direct200.json

python scripts/evaluate_m2_head.py \
  --checkpoint models/memory/memory_experiment_2_m2/seed0/checkpoint_084000.pt \
  --episodes 200 --seed-start 1000 --eval-seed-mode direct --offline-probe \
  --split-seed 20260813 --torch-threads 1 --device cpu \
  --out runs/reproduced_v9/m2_seed0_direct200_with_probe.json

python scripts/evaluate_m3_corrected_flip.py \
  --checkpoint models/memory/memory_experiment_3_m3/actor_seed0_400demo.pt \
  --rundir runs/reproduced_v9 \
  --distill-meta models/memory/memory_experiment_3_m3/frozen_symbolic_head_meta.json \
  --episodes 200 --seed-start 0 --seed-mode direct --representative-configs --zero-controls \
  --bootstrap 0 --torch-threads 1 --device cpu \
  --out runs/reproduced_v9/m3_actor0_configuration_grounding.json
```

Repeat the M1, M2, and M3 examples for seed 1 using the corresponding checkpoint and a distinct output file. The M3 command inventories 200 seeds but executes only four representative configurations in seven conditions: 28 rollouts per actor. It does not produce 200 new rollouts per condition. The original revision used CPU for M1/M2/D3 and CUDA for M3; CPU M3 evaluation is supported but numerical details can depend on the runtime. Checkpoint hashes are in `models/SHA256SUMS.txt`.

## Interactive visualisation

Run:

```bash
python interactive_demo.py
```

or on Windows double-click `run_interactive_demo.bat`. The launcher uses `.venv` when present, otherwise it tries the active system Python and prints the setup commands if the environment is incomplete. The backend discovers the renamed checkpoints in `models/`; `interactive_demo.html` is the single retained frontend. Duplicate and historical HTML reports have been removed.

## Repository policy

- `models/` contains only checkpoints needed for paper-reported comparisons.
- `results/` preserves the original JSON/NPZ evidence and compact appendix records; `v9/evidence/` contains the final revision records. Historical files are not overwritten.
- `v9/code_changes/` preserves the original revision source snapshot and patch; executable evaluation code is also in the main source tree.
- `runs/` is intentionally absent and ignored. New training creates it locally.
- No numerical claim should be upgraded beyond `docs/RESULTS.md`: M3 has 0/92 semantic redirections in the deduplicated baseline-correct cohort, covering only two successful configurations; R1 is a negative transfer stress test.
- Original logs and local paths inside the retained revision snapshot are provenance records. Use current reproduction commands with a separate output directory instead of running an auxiliary script that would overwrite retained files.

