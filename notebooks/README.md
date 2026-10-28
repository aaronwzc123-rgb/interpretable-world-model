# Paper-aligned acceptance notebooks

Run `run_acceptance_notebooks.bat` from the repository root. All notebooks use the project `.venv` and keep saved visual output. Historical outputs are retained for inspection and do not establish that a cell was newly executed. Start with the [final revision evidence index](../v9/README.md) for the current numerical record.

| Notebook | Paper role | Important boundary |
|---|---|---|
| `D1_DoorKey_Belief_Decoding.ipynb` | D1 continuous-belief decoding plus original-policy animation | Includes an extra post-hoc linear action-distillation animation; that clone is not paper D1 evidence |
| `D3_DoorKey_Atoms_Only_Causal_Control.ipynb` | D3 atoms-only policy, animation and causal checks | One checkpoint; same-generator layouts; not D1 distillation |
| `M3_MemoryS7_Staged_Frozen_Interface.ipynb` | M3 frozen memory interface and paired cue-flip case | Final deduplicated result is 0/92 semantic redirections and 59/92 early/mid cue-edit timeouts per actor; old 500-record output is historical |

M3's 200 actual seeds cover only four initial configurations; its 92 baseline-correct instances cover two. New grounding and replay diagnostics total 96 rollouts across both actors and do not expand configuration coverage. D3 sign-only deployment scores 13/100; final D3 intervals use episode bootstrap. These revision records live in `v9/evidence/`, not in the saved historical notebook outputs.

M1, M2 and R1 use scripts and retained machine-readable results rather than duplicate notebooks. See [reproduction commands](../docs/EXPERIMENTS.md) and the [current result ledger](../docs/RESULTS.md). Running the notebooks requires the repository code and Git-LFS checkpoints; a compact evidence ZIP with saved outputs alone is not a standalone execution environment.
