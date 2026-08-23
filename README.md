# XDreamer: symbolic interfaces for DreamerV3 beliefs

This repository is the cleaned research artifact for the dissertation *Readable and Causally Used Symbolic Interfaces for World-Model Beliefs*. It contains the final MiniGrid implementation, paper-reported checkpoints, exact training overlays, machine-readable results, three acceptance notebooks, and one interactive HTML visualisation. Prompt files, overnight orchestration, replay buffers, intermediate checkpoints, TensorBoard files, PID files, and exploratory notebooks have been removed.

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

D1 and D3 use seed-held-out evaluations from the same DoorKey generator. They are not unseen-layout tests: the generator audit found complete layout-support overlap. M3's two actors use different actor seeds but share the same world model and symbolic head.

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
python -m unittest tests.test_regressions
```

The checkpoints are larger than ordinary Git blobs and are configured for Git LFS. Install Git LFS before cloning or pushing a complete copy:

```bash
git lfs install
git lfs pull
```

## Reproduce the paper checks

The shortest route is the three notebooks in `notebooks/`:

- `D1_DoorKey_Belief_Decoding.ipynb` (D1 visualisation)
- `D3_DoorKey_Atoms_Only_Causal_Control.ipynb` (D3 visualisation and ablation)
- `M3_MemoryS7_Staged_Frozen_Interface.ipynb` (M3 visualisation; the opening note identifies superseded historical metrics)

On Windows, double-click `run_acceptance_notebooks.bat` to open them with the project environment. The notebooks load the final paper-named artifacts and retain saved animations; frozen headline metrics and corrected M3 outcomes are in `results/` and `docs/RESULTS.md`. M1 and M2 are evaluated by deterministic scripts because the former is a multi-checkpoint seed comparison and the latter is a registered-gate negative result.

Examples:

```bash
python scripts/evaluate_d1_unified.py \
  --traj results/doorkey_experiment_1_d1/trajectory.npz \
  --checkpoint models/doorkey/doorkey_experiment_1_d1/checkpoint.pt \
  --out results/doorkey_experiment_1_d1/rerun.json

python scripts/evaluate_d3_directed_actions.py \
  --checkpoint models/doorkey/doorkey_experiment_3_d3/checkpoint.pt \
  --episodes 100 --seed-start 1000 --bootstrap 5000 \
  --device cpu --out results/doorkey_experiment_3_d3/rerun_directed.json

python scripts/evaluate_m1_cuestart.py \
  --checkpoint models/memory/memory_experiment_1_m1/grounded_seed0/checkpoint_084000.pt \
  --condition grounded --episodes 150 --device cpu \
  --out results/memory_experiment_1_m1/rerun_grounded_seed0.json

python scripts/evaluate_m2_head.py \
  --checkpoint models/memory/memory_experiment_2_m2/seed0/checkpoint_084000.pt \
  --episodes 100 --device cpu \
  --out results/memory_experiment_2_m2/rerun_seed0.json

python scripts/evaluate_m3_corrected_flip.py \
  --checkpoint models/memory/memory_experiment_3_m3/actor_seed0_400demo.pt \
  --rundir models/memory/memory_experiment_3_m3 \
  --distill-meta models/memory/memory_experiment_3_m3/frozen_symbolic_head_meta.json \
  --episodes 500 --seed-start 1000 --bootstrap 5000 --device cpu \
  --out results/memory_experiment_3_m3/rerun_seed0.json
```

CPU evaluation is valid but slow; use `--device cuda:0` when CUDA is available. Training and full evaluation commands are indexed in `docs/EXPERIMENTS.md`. Frozen headline results and claim boundaries are in `docs/RESULTS.md`. Checkpoint hashes are in `models/SHA256SUMS.txt`.

## Interactive visualisation

Run:

```bash
python interactive_demo.py
```

or on Windows double-click `run_interactive_demo.bat`. The launcher uses `.venv` when present, otherwise it tries the active system Python and prints the setup commands if the environment is incomplete. The backend discovers the renamed checkpoints in `models/`; `interactive_demo.html` is the single retained frontend. Duplicate and historical HTML reports have been removed.

## Repository policy

- `models/` contains only checkpoints needed for paper-reported comparisons.
- `results/` contains frozen JSON/NPZ evidence and compact appendix records; it is not a training-log dump.
- `runs/` is intentionally absent and ignored. New training creates it locally.
- No numerical claim should be upgraded beyond `docs/RESULTS.md`; in particular, M3 shows disruption but 0/237 semantic redirections, and R1 is a negative transfer stress test.

