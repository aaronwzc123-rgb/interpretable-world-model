# Experiment index and reproduction commands

All paths are repository-relative. `configs.yaml` uses ordered overlay merging: later overlays override earlier ones. New training and evaluation outputs should go under ignored `runs/`; retained artifacts in `models/`, `results/`, and `v9/evidence/` must not be overwritten.

## Final revision and historical baseline

Start with the [revision index](../v9/README.md) and [current claim ledger](RESULTS.md). The final manuscript uses retained September 2026 records in `v9/evidence/` in addition to the original `results/` files. The repository publishes that revision on 29 October 2026; the compact original evidence package and baseline commit `7805619c4cb7b4ab51945dc2396351085c431b46` did not include these new records or code fixes. Publication now is not evidence of inclusion in the original submission.

Current evaluation code in `scripts/` and `envs/` includes the revision fixes. The original full source snapshots, 11-file manifest, and patch are retained under `v9/code_changes/`. Do not apply that old patch again to this checkout. Auxiliary scripts inside the frozen snapshot can contain original absolute paths and output locations; use the commands below or adapt paths before running them.

The evaluation commands below use existing checkpoints without training updates; separate training commands document the original recipes. Create the evaluation output directory first:

```bash
python -c "from pathlib import Path; Path('runs/reproduced_v9').mkdir(parents=True, exist_ok=True)"
python -m unittest discover -s tests -v
```

The original revised D3/M1/M2 evaluations used CPU with one PyTorch thread; M3 diagnostics used CUDA. Use `--device cuda:0` for M3 when available. Different devices or library versions can change floating-point details. New executions are replication checks and do not increase the manuscript's independent configuration count.

## Seven executed experiment artifact map

D2 is the executed historical N-DNF transition experiment. The later detached/MSE/BCE head-comparison extension remains unexecuted.

| ID | Training parameters | Retained checkpoint(s) | Frozen evidence | Evaluation entry |
|---|---|---|---|---|
| D1 | `models/doorkey/doorkey_experiment_1_d1/config.yaml` | `models/doorkey/doorkey_experiment_1_d1/checkpoint.pt` | `results/doorkey_experiment_1_d1/d1_unified.json`, `trajectory.npz` | `scripts/evaluate_d1_unified.py` |
| D2 | `archive/d2_model2_legacy/source/model_base.yaml`, historical resolved configs under `archive/d2_model2_legacy/runs/`; controlled-rebuild parameters in `docs/d2_model2/m2rebuild_log.md` | `archive/d2_model2_legacy/runs/m2rebuild_run/m2rebuild_s0/latest.pt` | `docs/d2_model2/`, archived console logs and `metrics.jsonl` | historical runs have resolved configs; controlled rebuild uses its parameter log |
| D3 | `models/doorkey/doorkey_experiment_3_d3/config.yaml` | `models/doorkey/doorkey_experiment_3_d3/checkpoint.pt` | original ablations plus `v9/evidence/d3_episode_clustered.json`, `d3_interface_controls.json` | `scripts/ablate_sym.py`, `scripts/evaluate_d3_directed_actions.py`, `scripts/evaluate_d3_interface_controls.py` |
| M1 | `models/memory/memory_experiment_1_m1/config.yaml` | Grounded/Plain seed 0/1 checkpoints | original controls plus `v9/evidence/m1_grounded_seed{0,1}_fixed_upper_direct200.json` and autonomous coverage records | `scripts/evaluate_m1_cuestart.py` |
| M2 | `models/memory/memory_experiment_2_m2/config.yaml` | seed 0/1 84k checkpoints | `v9/evidence/m2_seed{0,1}_direct200_with_probe.json`; original JSON retained historically | `scripts/evaluate_m2_head.py` |
| M3 | `models/memory/memory_experiment_3_m3/config.yaml` | shared WM, frozen head, two 400-demo actors, zero-demo actor | original 500-record JSON; `v9/evidence/` deduplication, grounding, and replay records | Stage A/B scripts, `scripts/audit_m3_existing_results.py`, `scripts/evaluate_m3_corrected_flip.py` |
| R1 | `models/doorkey/relational_stress_test/config.yaml` | `checkpoint_050000.pt` | `results/relational_stress_test/summary.json` | `scripts/doorkey_relational_scripted.py` |

## D1 — DoorKey Experiment 1

Purpose: test whether nine DoorKey predicates are recoverable from a continuous 1280-dimensional DreamerV3 belief. The actor and critic use the full continuous feature.

Train:

```bash
python dreamer.py --configs minigrid --steps 500000 --seed 0 --logdir runs/doorkey_experiment_1_d1
```

Evaluate the frozen episode-disjoint probe dataset:

```bash
python scripts/evaluate_d1_unified.py --traj results/doorkey_experiment_1_d1/trajectory.npz --checkpoint models/doorkey/doorkey_experiment_1_d1/checkpoint.pt --out runs/reproduced_v9/d1_unified.json --seed 20260812 --top-per-label 6 --conjunctions 18 --steps 4000
```

The trajectory split is 70 train / 15 validation / 15 test episodes. This is a same-generator, seed-held-out test, not unseen-layout generalisation.

## D2 — DoorKey Experiment 2: online N-DNF transition (negative)

Purpose: test whether a Neural-DNF transition can replace the RSSM recurrent transition in end-to-end Dreamer training on DoorKey-6x6.

This is the executed historical D2 experiment. The N-DNF mechanism is trained online as the recurrent transition, not fitted as a detached post-hoc readout. The restored source is under `archive/d2_model2_legacy/source/`; historical runs retain resolved parameters under `archive/d2_model2_legacy/runs/`, while the controlled rebuild parameter record is `docs/d2_model2/m2rebuild_log.md`.

The historical line contains eight long runs plus a controlled 300k-step rebuild. The controlled rebuild reached 312,456 steps; from steps 182,500--302,500, 13 consecutive evaluation checkpoints covering 260 episodes had `eval_return = 0.000`. The final failed checkpoint is `archive/d2_model2_legacy/runs/m2rebuild_run/m2rebuild_s0/latest.pt`.

Supported claim: in the tested DoorKey-6x6 implementation, replacing the complete stochastic RSSM transition with an online N-DNF transition was not trainable. This is a bounded architecture result, not evidence that every N-DNF head or every loss function is impossible.

### D2 extension — fair online-head comparison (not executed)

A later detached/MSE/BCE comparison would keep the RSSM transition and vary only an auxiliary online readout/head. That matched matrix was not run and remains an explicit future extension; it is not the definition of the executed D2 result above.
## D3 — DoorKey Experiment 3

Purpose: force actor and critic to consume only nine named atoms, then test whole-interface removal, per-atom removal, and valid-state fixed-belief sign edits.

Train:

```bash
python dreamer.py --configs minigrid minigrid_symbolic_nav --steps 500000 --seed 0 --logdir runs/doorkey_experiment_3_d3
```

Evaluate:

```bash
python scripts/ablate_sym.py --checkpoint models/doorkey/doorkey_experiment_3_d3/checkpoint.pt --episodes 100 --device cuda:0
python scripts/evaluate_d3_directed_actions.py --checkpoint models/doorkey/doorkey_experiment_3_d3/checkpoint.pt --episodes 100 --seed-start 0 --bootstrap 5000 --torch-threads 1 --device cpu --out runs/reproduced_v9/d3_episode_clustered.json
python scripts/evaluate_d3_interface_controls.py --repo . --episodes 100 --device cpu --out runs/reproduced_v9/d3_interface_controls.json
python v9/scripts/verify_d3_saved_bootstrap.py --repo . --source v9/evidence/d3_episode_clustered.json --out runs/reproduced_v9/d3_bootstrap_verification.json
```

The final intervals resample complete episodes from saved state-level effects, using 5,000 bootstrap draws. The old state-bootstrap intervals are retained in the original JSON for comparison. The same actor scores 100/100 with continuous atoms, 13/100 with sign-only atoms, and 100/100 with oracle signs; the oracle is a privileged diagnostic. The fixed-state edit tests a local action-semantic effect. It does not establish episode-level goal redirection, a Boolean policy, or retraining robustness.

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

For each Grounded checkpoint, the final confirmation uses 200 direct actual seeds and a grouped 140/60 probe split. Run the following with seed 0 and then replace both checkpoint seed and output filename with seed 1:

```bash
python scripts/evaluate_m1_cuestart.py --checkpoint models/memory/memory_experiment_1_m1/grounded_seed0/checkpoint_084000.pt --condition grounded --episodes 200 --seed-start 1000 --eval-seed-mode direct --split-seed 20260813 --coverage-controller fixed_upper --torch-threads 1 --device cpu --out runs/reproduced_v9/m1_grounded_seed0_fixed_upper_direct200.json
python scripts/evaluate_m1_cuestart.py --checkpoint models/memory/memory_experiment_1_m1/grounded_seed0/checkpoint_084000.pt --condition grounded --episodes 200 --seed-start 1000 --eval-seed-mode direct --split-seed 20260813 --coverage-controller actor --torch-threads 1 --device cpu --out runs/reproduced_v9/m1_grounded_seed0_direct200.json
```

Fixed-upper coverage is independent of cue and object identity and supports the matched storage test. Actor coverage is reported separately because its state distribution differs. Both models attain 100% pure-memory belief/deterministic-state BA on fixed-upper coverage, while frame BA is 44.8%; this does not imply successful autonomous control. The historical 500-episode probe confirmation is excluded because its nominal train/test split repeated actual seeds. Direct mode removes seed repetition, but all four complete initial configurations still occur on both sides of the new split.

## M2 — Memory Experiment 2

Purpose: test an online BCE N-DNF readout while actor and critic continue to use the continuous feature. Both training seeds use the same 84k budget.

```bash
python dreamer.py --configs minigrid minigrid_memory minigrid_memory_cuestart minigrid_memory_symbolic_nav_v2 minigrid_memory_symbolic_nav_v2_entropy minigrid_memory_symbolic_nav_v2_bce --steps 84000 --seed 0 --logdir runs/memory_experiment_2_m2/seed0
python dreamer.py --configs minigrid minigrid_memory minigrid_memory_cuestart minigrid_memory_symbolic_nav_v2 minigrid_memory_symbolic_nav_v2_entropy minigrid_memory_symbolic_nav_v2_bce --steps 84000 --seed 1 --logdir runs/memory_experiment_2_m2/seed1
```

```bash
python scripts/evaluate_m2_head.py --checkpoint models/memory/memory_experiment_2_m2/seed0/checkpoint_084000.pt --episodes 200 --seed-start 1000 --eval-seed-mode direct --offline-probe --split-seed 20260813 --torch-threads 1 --device cpu --out runs/reproduced_v9/m2_seed0_direct200_with_probe.json
```

Repeat for the seed-1 checkpoint and output filename. The project-defined gate is pure-memory balanced accuracy at least 0.65 with non-constant predictions; it is not an externally preregistered protocol. Both seeds fail at 60.0% and 44.8%. The seed-1 historical repetition-weighted 45.9% remains in `results/` but is superseded. `--offline-probe` fits readouts to the same checkpoint and trajectories, with a grouped 140/60 split; belief/deterministic/frame BA are 56.9%/57.9%/55.8%. These scores do not isolate storage from readout failure. The privileged scripted controller supplies coverage actions, so its success is not learned-actor task success. No atoms-only stage follows.

## M3 — Memory Experiment 3

M3 has three distinct artifacts: one shared memory-bearing world model, one frozen ten-atom head, and three actor checkpoints (two 400-demo actor seeds plus a zero-demo control). The two 400-demo actors are not independent end-to-end replicas because the first two artifacts are shared.

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

Because training advances in evaluation chunks, the archived seed-0 and seed-1 actors reached 100k and 310k steps respectively. Demonstration distributions also differ (209/191 versus 200/200 by cue). The actor seed alone therefore does not explain any behavioural differences. Corrected evaluation uses exhaustive `target / other / timeout / neither` outcomes.

First audit historical observations without executing the model; repeat with seed 1:

```bash
python scripts/audit_m3_existing_results.py --source results/memory_experiment_3_m3/m3_seed0_corrected_500.json --out runs/reproduced_v9/m3_actor0_unique200_historical.json
```

The 500 nominal records repeat 200 actual seeds. Deduplication gives 92/200 baseline-correct per actor, with 0/92 early/mid semantic redirections, 59/92 timeouts, and 33/92 retaining the key endpoint. Only four complete initial configurations are represented, two baseline-correct. Historical seed weights are descriptive counts rather than independent configuration replications.

Separately execute the final representative-configuration diagnostic; repeat with the seed-1 actor and output filename:

```bash
python scripts/evaluate_m3_corrected_flip.py --checkpoint models/memory/memory_experiment_3_m3/actor_seed0_400demo.pt --rundir runs/reproduced_v9 --distill-meta models/memory/memory_experiment_3_m3/frozen_symbolic_head_meta.json --episodes 200 --seed-start 0 --seed-mode direct --representative-configs --zero-controls --bootstrap 0 --torch-threads 1 --device cuda:0 --out runs/reproduced_v9/m3_actor0_configuration_grounding.json
```

This inventories seeds 0–199 and newly executes only one representative of each configuration in baseline, three cue-edit timings, wall-mid, zero-all, and zero-cue: 28 rollouts per actor, 56 total. The original revision additionally executed a second actual seed per configuration in five conditions, giving 40 replay checks and 96 new diagnostic rollouts overall. Those replay records and the original generating script are retained at `v9/evidence/m3_replay_validation.json` and `validate_m3_replay.py`; adapt that auxiliary script's original absolute paths and output directory before rerunning it.

The new cue-negation traces retain the correct raw cue sign throughout. Both baseline-correct configurations still fail to redirect; the key case retains its endpoint after 147/149 steps rather than six at baseline. Sign stability is not asserted for all wall-edited trajectories. Zero-all yields four timeouts per actor, while zero-cue yields one target, one other, and two timeouts. These are four configuration-level executions per condition, not 200 fresh weighted episodes.

Only baseline-target to the specific alternative-target counts as semantic redirection. Timeout is disruption, not redirection. The direct-seed evaluator rejects repeated actual seeds; do not request 500 episodes in the historical 200-seed pooled mode. A new 200-direct-seed cohort (for example, seeds 1000–1199 without `--representative-configs`) is a separate descriptive evaluation and does not reproduce the historical pool weights.

## R1 — relational map-size stress test

```bash
python dreamer.py --configs minigrid minigrid_symbolic_nav_rel --steps 50000 --seed 0 --logdir runs/relational_stress_test
python scripts/doorkey_relational_scripted.py --episodes 100
```

R1 is a bounded negative transfer result, not a generalisation success claim.
