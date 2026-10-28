# M3 code changes and reproduction

These files belong in the corresponding Xdreamer paths. The working test repository already has the initial rollout/evaluator edits from before the workspace permission update. The copies here contain the final versions, including descriptive-interval suppression, legacy Stage-C timeout-safe redirection counts, explicit ablation sample sizes, and six focused tests.

Files to apply:

- scripts/evaluate_m3_corrected_flip.py
- scripts/staged_stage_c_eval.py
- scripts/audit_m3_existing_results.py
- tests/test_m3_evaluation_audit.py

The environment provenance API is provided by the separate M1/M2 change: MiniGrid(eval_seed_mode="pool"|"direct") and episode_provenance after reset. Historical behavior remains available with pool; the corrected M3 evaluator defaults to direct and rejects repeated actual seeds before executing rollouts.

In Xdreamer, for actor 0 or actor 1 (replace actor_seed0 below), with output pointing to dissertation/v9/evidence:

```powershell
.\.venv\Scripts\python.exe -B scripts/evaluate_m3_corrected_flip.py --checkpoint models/memory/memory_experiment_3_m3/actor_seed0_400demo.pt --rundir runs/memory_experiment_3_m3_reproduction --distill-meta models/memory/memory_experiment_3_m3/frozen_symbolic_head_meta.json --device cuda:0 --episodes 200 --seed-start 0 --seed-mode direct --representative-configs --zero-controls --bootstrap 0 --out C:/Users/super/Desktop/UCL/dissertation/v9/evidence/m3_actor0_configuration_grounding.json
.\.venv\Scripts\python.exe -B scripts/audit_m3_existing_results.py --source results/memory_experiment_3_m3/m3_seed0_corrected_500.json --out C:/Users/super/Desktop/UCL/dissertation/v9/evidence/m3_actor0_unique200_historical.json
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_m3_evaluation_audit.py
```

The second-seed replay script is stored as dissertation/v9/evidence/validate_m3_replay.py. It uses absolute local paths and can run directly with the same -B Python executable.

No world model, symbolic head, actor, optimizer state, or original result JSON was updated. Current-output negation remains the actual experimental operation. The optional fixed_value rollout argument is explicit, tested, and was not evaluated as a new experimental condition.
