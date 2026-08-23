# Historical D2 / Model2 evidence

This directory restores the historical Model2 N-DNF-transition experiment that was moved to the local quarantine during repository pruning.

It is not the later D2 detached/MSE/BCE comparison. That comparison remains unrun. Model2 is retained here because it is the negative architectural precursor that motivated the later D2 protocol.

## Restored contents

- `source/`: the archived `model2/r2dreamer` implementation and launch scripts.
- `runs/`: console logs, metrics, and Hydra configurations for the historical runs.
- `runs/m2rebuild_run/`: the controlled 300k-step rebuild and its postmortems.
- `../../docs/d2_model2/`: the implementation comparison, debug log, and final evidence reports.

Large checkpoints (`latest.pt`) and TensorBoard event files are intentionally omitted from this evidence commit. They remain in the local quarantine copy and are not needed to verify the reported failure trajectory.

## Main negative result

The controlled `m2rebuild` run reached 312,456 environment steps. Evaluation declined as the N-DNF transition hardened; from steps 182,500–302,500, 12 consecutive evaluation checkpoints covering 240 episodes had `eval_return = 0.000`.

The evidence supports the bounded claim that this historical N-DNF-only transition was not trainable in the tested DoorKey-6x6 setup. It does not support a claim that the later D2 detached/MSE/BCE matrix was executed.