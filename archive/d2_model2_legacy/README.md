# Historical D2 / Model2 evidence

This directory restores the historical Model2 N-DNF-transition experiment that was moved to the local quarantine during repository pruning.

This is the executed historical D2 in the original DoorKey experiment programme: an online/end-to-end N-DNF recurrent transition. The later detached/MSE/BCE head comparison is a separate, unexecuted D2 extension.

## Restored contents

- `source/`: the archived `model2/r2dreamer` implementation and launch scripts.
- `runs/`: console logs, metrics, and Hydra configurations for the historical runs.
- `runs/m2rebuild_run/`: the controlled 300k-step rebuild and its postmortems.
- `../../docs/d2_model2/`: the implementation comparison, debug log, and final evidence reports.

TensorBoard event files and the other missing checkpoints are intentionally omitted from this evidence commit. The final `m2rebuild_s0/latest.pt` checkpoint is included below; any checkpoint not listed there was not present in the quarantine copy.

## Main negative result

The controlled `m2rebuild` run reached 312,456 environment steps. Evaluation declined as the N-DNF transition hardened; from steps 182,500–302,500, 12 consecutive evaluation checkpoints covering 240 episodes had `eval_return = 0.000`.

The evidence supports the bounded D2 claim that this historical N-DNF-only transition was not trainable in the tested DoorKey-6x6 setup. It does not answer the later detached/MSE/BCE head-comparison extension.
## Run-to-artifact map

Every restored run keeps its resolved Hydra parameters in `.hydra/config.yaml` and `.hydra/overrides.yaml`, with raw console output and `metrics.jsonl` beside them.

| Run | Parameters / results | Failed model checkpoint |
|---|---|---|
| `2026-07-06_22-00-44` | `runs/2026-07-06_22-00-44/` | not saved in quarantine |
| `ndnf_dreamer_demo_s0` | `runs/ndnf_dreamer_demo_s0/` | not saved in quarantine |
| `ndnf_dreamer_demo_anneal_s0` | `runs/ndnf_dreamer_demo_anneal_s0/` | not saved in quarantine |
| `ndnf_demo_deltafix_s0` | `runs/ndnf_demo_deltafix_s0/` | not saved in quarantine |
| `m2fix_exp1_delta_full_s0` | `runs/m2fix_exp1_delta_full_s0/` | not saved in quarantine |
| `m2fix_exp2_nodemo_s0` | `runs/m2fix_exp2_nodemo_s0/` | not saved in quarantine |
| `m2fix_exp3_optsplit_s0` | `runs/m2fix_exp3_optsplit_s0/` | not saved in quarantine |
| `m2rebuild_s0` | `runs/m2rebuild_run/` | `runs/m2rebuild_run/m2rebuild_s0/latest.pt` |

The restored `m2rebuild_s0/latest.pt` is the final failed checkpoint from the controlled 300k-step rebuild. Size: 107.60 MiB. SHA256: `8e42e630c3b88de2daac101fb555a10d2583b29825dd537790d427fa5ab507a3`.