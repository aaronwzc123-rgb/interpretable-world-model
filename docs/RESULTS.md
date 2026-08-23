# Frozen results and claim boundaries

These are the results reported in the dissertation. JSON files are authoritative when notebook prose or historical reports differ.

| ID | Frozen result | Supported claim | Not supported |
|---|---|---|---|
| D1 | 100/100 task success; linear accuracy 95.4–100%; hardened N-DNF accuracy 94.9–99.5% | Nine selected facts are decodable on episode-disjoint same-generator data | Policy use; unseen-layout generalisation |
| D2 | Historical online N-DNF transition: eight long runs plus a controlled rebuild; late controlled evaluation 0/240; 8/9 predicates near base rate | In the tested DoorKey-6x6 implementation, replacing the complete RSSM transition with online N-DNF was not trainable | Fair detached/MSE/BCE head comparison; impossibility of all N-DNF heads |
| D3 | 100/100 normal, 0/100 zero-all; `t_ahead` signed effect 0.507 and `wall_ahead` 0.384 | One checkpoint uses the atom boundary and has local semantically aligned action effects | Retraining robustness; episode-level goal redirection; unseen layouts |
| M1 | Grounded seeds 0/1 reach 100% pure-memory BA at 84k; current-frame BA 50%; Plain matched controls 50% | Supervision can make the recurrent belief retain the hidden cue in two training seeds | Stable policy success; unsupervised memory |
| M2 | BCE seeds 0/1 reach 60.0% and 45.9% pure-memory BA, below the 65% gate | This online-head recipe fails its registered gate at 84k in two seeds | Impossibility of online symbolic memory |
| M3 | Both actors: 237/500 baseline correct; cue-mid causes 0/237 target→other and 151/237 target→timeout | Readable frozen cue and policy/interface dependence; corrected negative causal result | Semantically editable memory; independent representation replication |
| R1 | Autonomous 6x6/8x8/16x16: 81/100, 23/100, 0/100 | A control-sufficient vocabulary does not guarantee learned size transfer | Robust map-size generalisation |

## Important corrections

- The old M3 “78% redirection” merged alternative-target outcomes with timeouts. It is invalid. The corrected four-way result is 0/237 semantic redirections for each actor.
- D1 has no uniform probe winner. Linear and hardened N-DNF each lead on some raw or balanced metrics.
- D1/D3 seed separation is not layout separation. The 200 evaluation seeds contain 36 static layouts, all present in candidate training support.
- R1 is a stress-test failure boundary, not a replacement for D3.
- `results/appendix_history/door_key_macro_rule_extraction.json` reports `rule_match=1.0`, but the cleaned archive does not preserve a complete generating-script/split provenance chain. Treat it as historical appendix evidence only.