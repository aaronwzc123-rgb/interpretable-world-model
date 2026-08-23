# Experiment index and reproduction commands

All paths are repository-relative. `configs.yaml` uses ordered overlay merging: later overlays override earlier ones. Training outputs should go under ignored `runs/`; paper artifacts in `models/` and `results/` must not be overwritten.

## Six executed experiment artifact map

D2 is registered but unexecuted and is therefore excluded from these six executed artifact rows.

| ID | Training parameters | Retained checkpoint(s) | Frozen evidence | Evaluation entry |
|---|---|---|---|---|
| D1 | `models/doorkey/doorkey_experiment_1_d1/config.yaml` | `models/doorkey/doorkey_experiment_1_d1/checkpoint.pt` | `results/doorkey_experiment_1_d1/d1_unified.json`, `trajectory.npz` | `scripts/evaluate_d1_unified.py` |
| D3 | `models/doorkey/doorkey_experiment_3_d3/config.yaml` | `models/doorkey/doorkey_experiment_3_d3/checkpoint.pt` | `d3_ablation.json`, `d3_directed_actions.json` | `scripts/ablate_sym.py`, `scripts/evaluate_d3_directed_actions.py` |
| M1 | `models/memory/memory_experiment_1_m1/config.yaml` | Grounded/Plain seed 0/1 checkpoints | `results/memory_experiment_1_m1/` | `scripts/evaluate_m1_cuestart.py` |
| M2 | `models/memory/memory_experiment_2_m2/config.yaml` | seed 0/1 84k checkpoints | `results/memory_experiment_2_m2/` | `scripts/evaluate_m2_head.py` |
| M3 | `models/memory/memory_experiment_3_m3/config.yaml` | shared WM, frozen head, two 400-demo actors, zero-demo actor | corrected seed 0/1 JSON plus Stage A/B reports | Stage A/B scripts and `scripts/evaluate_m3_corrected_flip.py` |
| R1 | `models/doorkey/relational_stress_test/config.yaml` | `checkpoint_050000.pt` | `results/relational_stress_test/summary.json` | `scripts/doorkey_relational_scripted.py` |

## D1 — DoorKey Experiment 1

Purpose: test whether nine DoorKey predicates are recoverable from a continuous 1280-dimensional DreamerV3 belief. The actor and critic use the full continuous feature.

Train:

```bash
python dreamer.py --configs minigrid --steps 500000 --seed 0 --logdir runs/doorkey_experiment_1_d1
```

Evaluate the frozen episode-disjoint probe dataset:

```bash
python scripts/evaluate_d1_unified.py --traj results/doorkey_experiment_1_d1/trajectory.npz --checkpoint models/doorkey/doorkey_experiment_1_d1/checkpoint.pt --out results/doorkey_experiment_1_d1/rerun.json --seed 20260812 --top-per-label 6 --conjunctions 18 --steps 4000
```

The trajectory split is 70 train / 15 validation / 15 test episodes. This is a same-generator, seed-held-out test, not unseen-layout generalisation.

## D2 — registered but unexecuted

The planned detached/MSE/BCE DoorKey matrix was not run. There is intentionally no D2 checkpoint, config, or synthetic placeholder result.

## D3 — DoorKey Experiment 3

Purpose: force actor and critic to consume only nine named atoms, then test whole-interface removal, per-atom removal, and valid-state fixed-belief sign edits.

Train:

```bash
python dreamer.py --configs minigrid minigrid_symbolic_nav --steps 500000 --seed 0 --logdir runs/doorkey_experiment_3_d3
```

Evaluate:

```bash
python scripts/ablate_sym.py --checkpoint models/doorkey/doorkey_experiment_3_d3/checkpoint.pt --episodes 100 --device cuda:0
python scripts/evaluate_d3_directed_actions.py --checkpoint models/doorkey/doorkey_experiment_3_d3/checkpoint.pt --episodes 100 --seed-start 1000 --bootstrap 5000 --device cuda:0 --out results/doorkey_experiment_3_d3/rerun_directed.json
```

The fixed-state edit tests a local action-semantic effect. It does not establish episode-level goal redirection or retraining robustness.

## M1 — Memory Experiment 1

Environment: `minigrid_memoryS7_cuestart`. Grounded conditions add `minigrid_memory_shape`; Plain conditions explicitly disable all symbolic/shaping heads.

Grounded:

```bash
python dreamer.py --configs minigrid minigrid_memory minigrid_memory_cuestart minigrid_memory_shape --steps 84000 --seed 0 --logdir runs/memory_experiment_1_m1/grounded_seed0
python dreamer.py --configs minigrid minigrid_memory minigrid_memory_cuestart minigrid_memory_shape --steps 84000 --seed 1 --logdir runs/memory_experiment_1_m1/grounded_seed1
```

Plain matched controls:

```bash
python dreamer.py --configs minigrid minigrid_memory minigrid_memory_cuestart --ndnf_enabled False --shape_enabled False --sym_enabled False --steps 52500 --seed 0 --logdir runs/memory_experiment_1_m1/plain_seed0
python dreamer.py --configs minigrid minigrid_memory minigrid_memory_cuestart --ndnf_enabled False --shape_enabled False --sym_enabled False --steps 52500 --seed 1 --logdir runs/memory_experiment_1_m1/plain_seed1
```

Evaluate with `scripts/evaluate_m1_cuestart.py`; choose `--condition grounded` or `plain`. The main matched comparison is 52.5k across two seeds. The 84k claim applies to the two Grounded seeds. Plain seed 0's later 82.5k run is outside the matched comparison, so neither its checkpoint nor its redundant evaluation JSON is retained.

## M2 — Memory Experiment 2

Purpose: test an online BCE N-DNF readout while actor and critic continue to use the continuous feature. Both training seeds use the same 84k budget.

```bash
python dreamer.py --configs minigrid minigrid_memory minigrid_memory_cuestart minigrid_memory_symbolic_nav_v2 minigrid_memory_symbolic_nav_v2_entropy minigrid_memory_symbolic_nav_v2_bce --steps 84000 --seed 0 --logdir runs/memory_experiment_2_m2/seed0
python dreamer.py --configs minigrid minigrid_memory minigrid_memory_cuestart minigrid_memory_symbolic_nav_v2 minigrid_memory_symbolic_nav_v2_entropy minigrid_memory_symbolic_nav_v2_bce --steps 84000 --seed 1 --logdir runs/memory_experiment_2_m2/seed1
```

```bash
python scripts/evaluate_m2_head.py --checkpoint models/memory/memory_experiment_2_m2/seed0/checkpoint_084000.pt --episodes 100 --device cuda:0 --out results/memory_experiment_2_m2/rerun_seed0.json
```

The registered gate is pure-memory balanced accuracy at least 0.65 with non-constant predictions. Both seeds fail; no atoms-only stage follows.

## M3 — Memory Experiment 3

M3 has three distinct artifacts: one shared memory-bearing world model, one frozen ten-atom head, and three actor checkpoints (two 400-demo actor seeds plus a zero-demo control). The two positive actors are not independent end-to-end replicas because the first two artifacts are shared.

Recreate Stage A:

```bash
python scripts/staged_stage_a_distill.py --checkpoint models/memory/memory_experiment_3_m3/shared_world_model.pt --episodes 300 --distill-steps 8000 --seed0 0 --device cuda:0
```

Recreate Stage B actors (use a fresh log directory for each):

```bash
python scripts/staged_stage_b_train.py --wm-checkpoint models/memory/memory_experiment_3_m3/shared_world_model.pt --distill-head models/memory/memory_experiment_3_m3/frozen_symbolic_head.pt --distill-meta models/memory/memory_experiment_3_m3/frozen_symbolic_head_meta.json --steps 84000 --n-demo 400 --seed 0 --device cuda:0 --logdir runs/memory_experiment_3_m3/actor_seed0
python scripts/staged_stage_b_train.py --wm-checkpoint models/memory/memory_experiment_3_m3/shared_world_model.pt --distill-head models/memory/memory_experiment_3_m3/frozen_symbolic_head.pt --distill-meta models/memory/memory_experiment_3_m3/frozen_symbolic_head_meta.json --steps 300000 --n-demo-per-cue 200 --seed 1 --device cuda:0 --logdir runs/memory_experiment_3_m3/actor_seed1
python scripts/staged_stage_b_train.py --wm-checkpoint models/memory/memory_experiment_3_m3/shared_world_model.pt --distill-head models/memory/memory_experiment_3_m3/frozen_symbolic_head.pt --distill-meta models/memory/memory_experiment_3_m3/frozen_symbolic_head_meta.json --steps 84000 --n-demo 0 --seed 0 --device cuda:0 --logdir runs/memory_experiment_3_m3/actor_seed0_0demo
```

Because training advances in evaluation chunks, the archived seed-0 and seed-1 actors reached 100k and 310k steps respectively. Corrected evaluation uses exhaustive `target / other / timeout / neither` outcomes:

```bash
python scripts/evaluate_m3_corrected_flip.py --checkpoint models/memory/memory_experiment_3_m3/actor_seed0_400demo.pt --rundir models/memory/memory_experiment_3_m3 --distill-meta models/memory/memory_experiment_3_m3/frozen_symbolic_head_meta.json --episodes 500 --seed-start 1000 --bootstrap 5000 --device cuda:0 --out results/memory_experiment_3_m3/rerun_seed0.json
```

Only baseline-target to alternative-target counts as semantic redirection. Timeout is disruption, not redirection.

## R1 — relational map-size stress test

```bash
python dreamer.py --configs minigrid minigrid_symbolic_nav_rel --steps 50000 --seed 0 --logdir runs/relational_stress_test
python scripts/doorkey_relational_scripted.py --episodes 100
```

R1 is a bounded negative transfer result, not a generalisation success claim.