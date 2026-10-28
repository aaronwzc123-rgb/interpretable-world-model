# M3 revision evidence

Historical 200 actual seeds per actor, four complete initial states. New diagnostics are 4 actual model runs per condition per actor, unweighted finite-configuration descriptions, not 200 new runs.

## Actor 0

Checkpoint SHA256: `7c9c170437b0e202ce3e2ea635537e72bd86d059e762a0b55a768086c3ac8f35`

| Representative seed | Pool weight | True cue | Baseline | Early / mid / late cue atom at onset | Early / mid / late endpoint |
|---|---:|---|---|---|---|
| 0 | 59 | ball | target | -0.997985 / -0.970769 / -0.990619 | timeout / timeout / target |
| 2 | 33 | key | target | 0.999996 / 1.000000 / 1.000000 | target / target / target |
| 5 | 58 | ball | other | -0.997985 / -0.974081 / -0.960094 | timeout / timeout / other |
| 11 | 50 | key | timeout | 0.999996 / 0.999999 / 0.999999 | other / timeout / timeout |

All 12 cue-edit onsets (4 configurations x 3 timings) have a correct nonzero predicted cue sign. The raw cue sign remains correct at every edited step in each of those 12 trajectories; no predicted sign switches occur. Both baseline-correct configurations therefore remain in the grounding-correct cohort.

## Actor 1

Checkpoint SHA256: `8b2888792f7601a8283b1583ef832b4d663275da93cdd9dc13a46241a9cf5d35`

| Representative seed | Pool weight | True cue | Baseline | Early / mid / late cue atom at onset | Early / mid / late endpoint |
|---|---:|---|---|---|---|
| 0 | 59 | ball | target | -0.997985 / -0.970769 / -0.990619 | timeout / timeout / timeout |
| 2 | 33 | key | target | 0.999996 / 1.000000 / 1.000000 | target / target / target |
| 5 | 58 | ball | other | -0.997985 / -0.974081 / -0.960094 | timeout / timeout / other |
| 11 | 50 | key | other | 0.999996 / 0.999999 / 0.931378 | other / other / other |

All 12 cue-edit onsets (4 configurations x 3 timings) have a correct nonzero predicted cue sign. The raw cue sign remains correct at every edited step in each of those 12 trajectories; no predicted sign switches occur. Both baseline-correct configurations therefore remain in the grounding-correct cohort.

## Interpretation

The baseline-correct ball configuration has weight 59/200, and the correct key configuration has weight 33/200. Early/mid negation produces a timeout for the former and eventual correct completion for the latter: zero redirection in either of the two successful configurations. The 0/92 historical count describes the seed-weighted finite pool, not 92 different layouts. Actor 0 retains both successful configurations under late negation; actor 1 loses the ball configuration to timeout.

Zero controls here have exactly four configurations per actor: zero_all gives four timeouts; zero_cue gives one correct key endpoint, one incorrect endpoint and two timeouts. They were not in the historical 500-record correction files.

The operation remains negation of each current predicted output. A sign-correct raw cue throughout these diagnostic trajectories means the supplied sign is persistently counterfactual here, but magnitudes continue varying; this does not establish results for a fixed-value intervention or new environments.


## Validation and exact new-run counts

Each actor used representative actual seeds 0, 2, 5 and 11. Independent execution of second actual seeds 1, 3, 10 and 12 reproduced each representative's complete recorded grounding trace, outcome and timing in baseline plus all four negation conditions: **40/40 new pair checks identical** across both actors. Every historical deduplicated record agrees with its new configuration representative on outcome and timing in those five conditions: **2,000/2,000 comparisons identical**. This validates deterministic replay on the observed configurations; it does not enlarge the number of configurations.

The configuration diagnostics executed 4 x 7 = 28 new rollouts per actor (baseline, three cue timings, wall-mid, zero-all and zero-cue). The second-seed validation executed another 4 x 5 = 20 per actor. Thus 96 new rollouts were used in the paired diagnostics/validation across both actors, plus one six-step actor-0 baseline used only as a speed benchmark. None was a training update.

Six focused regression tests passed: pooled-seed duplication rejection, direct-seed identity, timeout precedence near the alternate endpoint, exact time-limit termination, current-output negation versus fixed-value semantics, and suppression of sampling intervals in descriptive configuration summaries.
