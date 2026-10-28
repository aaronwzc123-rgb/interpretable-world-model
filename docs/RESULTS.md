# Final results and evidence boundaries

This ledger follows the final manuscript revision. The original experiment record was frozen in August 2026; retained revision evaluations were executed on 5 September and integrated on 6 September 2026. They are published in this repository on **29 October 2026**. This publication date is distinct from the experiment dates and from the original assessment submission.

Use the [revision evidence index](../v9/README.md), the original [evidence inventory](../v9/evidence/README.md), and the [current reproduction guide](EXPERIMENTS.md). Files under `results/` remain unchanged historical evidence. Their old sampling weights, intervals, or notebook summaries do not override the revision records.

| ID | Final retained result | Supported claim | Not supported |
|---|---|---|---|
| D1 | 100/100 task success; linear accuracy 95.4–100%; hardened N-DNF accuracy 94.9–99.5% | Nine selected facts are decodable on episode-disjoint same-generator data | Policy use; unseen-layout generalisation |
| D2 | Historical online N-DNF transition: eight long runs plus a controlled rebuild; late controlled evaluation 0/260; 8/9 predicates near base rate | In the tested DoorKey-6x6 implementation, replacing the complete RSSM transition with online N-DNF was not trainable | Fair detached/MSE/BCE head comparison; impossibility of all N-DNF heads |
| D3 | 100/100 normal, 0/100 zero-all; `t_ahead` signed effect 0.507 (episode-bootstrap 95% CI 0.496–0.519), `wall_ahead` 0.384 (0.372–0.397); soft/sign-only/oracle inputs score 100/100, 13/100, 100/100 | One checkpoint depends on the atom boundary and has selected local action effects; the continuous interface does not retain performance after sign-only deployment | Retraining robustness; trajectory-level goal redirection; unseen layouts; equivalence to a Boolean policy |
| M1 | Grounded seeds 0/1 at 84k: belief and deterministic-state pure-memory BA 100% on fixed-upper coverage, frame BA 44.8%; 200 direct seeds per model, 140/60 probe split. Historical 52.5k Plain controls have 50% memory BA | Supervised recurrent belief can retain the vanished cue in two independently trained world models on the observed four configurations | Stable autonomous policy success; unsupervised memory; unseen-configuration generalisation |
| M2 | Online BCE seeds 0/1: pure-memory BA 60.0% and 44.8% on 200 direct seeds, below the project-defined 65% gate; same-checkpoint offline belief BA 56.9%, deterministic-state 57.9%, frame 55.8% | The tested online head remains below its gate; storage versus readout failure is not isolated by these probes | Impossibility of online symbolic memory; treating the privileged coverage controller's success as learned-actor success |
| M3 | Each actor: 92/200 actual seeds baseline-correct across only four complete initial configurations. Early/mid cue edits cause 0/92 correct-target→other-target and 59/92 correct-target→timeout; the remaining 33 retain the key endpoint after 147/149 steps. New configuration diagnostics and second-seed replay total 96 rollouts | Readable frozen cue, policy/interface dependence, and no semantic redirection in the tested configurations, even with correct raw cue signs throughout cue-negation trajectories | Above-chance competence; semantically editable memory; independent representation replication; 200 independent configurations |
| R1 | Autonomous 6x6/8x8/16x16: 81/100, 23/100, 0/100 | A control-sufficient vocabulary does not guarantee learned size transfer | Robust map-size generalisation; a complete retained raw episode record |

## Record-to-claim map

| Claim | Retained source |
|---|---|
| D1/D2 and historical controls | Original `results/`, `docs/d2_model2/`, and `archive/d2_model2_legacy/` records |
| D3 episode intervals | [`d3_episode_clustered.json`](../v9/evidence/d3_episode_clustered.json), [`d3_bootstrap_verification.json`](../v9/evidence/d3_bootstrap_verification.json) |
| D3 soft/sign-only/oracle control | [`d3_interface_controls.json`](../v9/evidence/d3_interface_controls.json) |
| M1 fixed-path confirmation and autonomous coverage | `v9/evidence/m1_grounded_seed{0,1}_fixed_upper_direct200.json` and `m1_grounded_seed{0,1}_direct200.json` |
| M2 online scores and offline probes | `v9/evidence/m2_seed{0,1}_direct200_with_probe.json` |
| M3 historical seed deduplication | `v9/evidence/m3_actor{0,1}_unique200_historical.json`, derived from the unchanged original 500-record files |
| M3 fresh grounding, zero controls, and replay | `v9/evidence/m3_actor{0,1}_configuration_grounding.json`, [`m3_replay_validation.json`](../v9/evidence/m3_replay_validation.json), [`m3_revision_summary.json`](../v9/evidence/m3_revision_summary.json) |

## Historical corrections

- M3's former “78% redirection” merged alternative-target outcomes with timeouts and is invalid. The later 237/500 baseline, 0/237 redirection, and 151/237 timeout figures describe 500 nominal records that repeat actual seeds. Deduplication gives the final 92/200, 0/92, and 59/92 counts. Duplicate records agree; this is a correction of retained observations, not new execution for every seed.
- M3's 200 actual seeds cover only four complete initial configurations; its 92 correct seed instances cover two configurations. No binomial generalisation bound or independent-configuration sample size is inferred from these repeated deterministic responses. Both actors share one world model and head, have unequal budgets (100k/310k), and use different demonstration distributions.
- The 96 new M3 diagnostics are 56 representative-condition rollouts plus 40 second-seed checks. Historical seed weights are not additional fresh runs. Raw cue sign stability is verified specifically on cue-negation trajectories, not all wall-edit trajectories.
- M1's historical 500-episode confirmation has repeated actual seeds across its nominal probe split and is excluded from the final confirmation. New direct-seed evaluations have no actual-seed overlap, but all four initial configurations occur on both sides. Fixed-upper coverage and autonomous-actor coverage are separate collection protocols.
- M2's historical repetition-weighted seed-1 BA of 45.9% is superseded by 44.8% on 200 direct seeds. The 65% gates are project-defined acceptance rules, not external preregistrations.
- D3's old state-bootstrap intervals remain in historical JSON. The final intervals resample complete episodes from saved state records. The sign-only condition changes the deployed input without retraining; oracle performance is a privileged diagnostic.
- D1 has no uniform probe winner. Linear and hardened N-DNF each lead on some raw or balanced metrics.
- D1/D3 seed separation is not layout separation. The 200-seed DoorKey support audit found 36 static layouts, all present in candidate training support; it does not prove every layout was actually sampled during training.
- R1 is a negative transfer boundary. Its retained summary is transcribed from executed notebook results; a complete original episode-output chain was not preserved.
- `results/appendix_history/door_key_macro_rule_extraction.json` reports `rule_match=1.0`, but lacks a complete generating-script/split provenance chain in the cleaned archive. It is historical appendix evidence only.
