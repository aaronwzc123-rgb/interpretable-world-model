# Paper-aligned acceptance notebooks

Run `run_acceptance_notebooks.bat` from the repository root. All notebooks use the project `.venv` and keep executed visual output.

| Notebook | Paper role | Important boundary |
|---|---|---|
| `D1_DoorKey_Belief_Decoding.ipynb` | D1 continuous-belief decoding plus original-policy animation | Includes an extra post-hoc linear action-distillation animation; that clone is not paper D1 evidence |
| `D3_DoorKey_Atoms_Only_Causal_Control.ipynb` | D3 atoms-only policy, animation and causal checks | One checkpoint; same-generator layouts; not D1 distillation |
| `M3_MemoryS7_Staged_Frozen_Interface.ipynb` | M3 frozen memory interface and paired cue-flip case | Final result is 0/237 semantic redirections and 151/237 timeouts |

M1, M2 and R1 use deterministic scripts and frozen machine-readable results rather than duplicate notebooks. See `docs/EXPERIMENTS.md` and `docs/RESULTS.md`.
