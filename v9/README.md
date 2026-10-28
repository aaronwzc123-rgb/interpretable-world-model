# Final dissertation revision evidence (manuscript v13)

This directory publishes the retained revision evidence cited in Appendix B.2 of *Readable and Causally Used Symbolic Interfaces for World-Model Beliefs*. The local revision folder was named `v9`; its manuscript version is **v13**. The reference 69-page PDF has SHA-256 `f92c3849781a1a05ae7764ebea73b3bd373fa5a96b9daaaad0da10759c28ce2a`.

The revision evaluations were recorded on **5–6 September 2026**. These retained files are being added to GitHub on **29 October 2026**. This publication does not establish that they were included in the original assessment submission. No new training or new experimental claims are introduced by this publication.

## Start here

- [Current results and claim boundaries](../docs/RESULTS.md): the final manuscript's numerical ledger.
- [Original evidence index](evidence/README.md): the retained file-to-claim map.
- [Reproduction commands](../docs/EXPERIMENTS.md): commands using the integrated evaluation code and separate output directories.
- [Original evaluation patch and reproduction notes](code_changes/CODE_CHANGES.md): the preserved September code package, based on commit `7805619c4cb7b4ab51945dc2396351085c431b46`.
- [Snapshot manifest](manifest.json): SHA-256 and size for all 54 retained files copied into this directory.

The `evidence/` and `code_changes/` files, auxiliary bootstrap script and `SOURCE_MANIFEST.json` are copied **without changing their bytes**. Local absolute paths and historical wording inside them remain part of the record. Git attributes disable newline conversion for this directory so the original digests also match Git blobs and fresh checkouts.

`historical_sources/` also preserves the two original M3 500-record input files with the exact Windows line endings used by the retained `source_sha256` values. Their parsed JSON content is identical to the original `results/` files. The older Git blobs used LF, so their raw-file hashes differ solely because of line endings; the verifier checks exact archived bytes and compares the canonical JSON by content.

`SOURCE_MANIFEST.json` is the unchanged manifest of the larger original local revision directory, including manuscript/build files that are not all copied into this repository. Use `manifest.json` to check the files actually published here. The PDF itself is identified by its digest rather than duplicated here.

## Which evidence supports which result?

| Final result | Retained evidence | Unit and interpretation |
|---|---|---|
| D3: episode-clustered intervals for local action effects | `evidence/d3_episode_clustered.json`, `d3_bootstrap_verification.json` | 100 actual seeds; 5,000 complete-episode bootstrap draws; one trained checkpoint |
| D3: soft / sign-only / oracle success 100/13/100 | `evidence/d3_interface_controls.json` | 100 paired seeds in each condition; oracle is a privileged control |
| M1: supervised cue storage under controlled coverage | `evidence/m1_grounded_seed{0,1}_fixed_upper_direct200.json` | 200 direct seeds per model; 140/60 probe split; four configurations overlap across the split |
| M1: autonomous coverage supplement | `evidence/m1_grounded_seed{0,1}_direct200.json` | Different collection protocol; do not conflate probe accuracy with actor success |
| M2: online pure-memory BA 60.0% / 44.8% and same-checkpoint probes | `evidence/m2_seed{0,1}_direct200_with_probe.json` | 200 direct seeds; below the project-defined 65% gate; privileged collection controller |
| M3: 92/200 baseline-correct, 0/92 semantic redirections, 59/92 cue-mid timeouts | `evidence/m3_actor{0,1}_unique200_historical.json` | Deduplication of old records, not newly executed 200-seed rollouts; only four initial configurations |
| M3: raw cue signs, zero controls and configuration traces | `evidence/m3_actor{0,1}_configuration_grounding.json` | Four configurations × seven conditions × two actors = 56 executions |
| M3: second-seed validation | `evidence/m3_replay_validation.json`, `m3_revision_summary.json` | Forty additional replay checks; 96 total revision diagnostics; no expanded configuration support |

The existing `results/` directory retains the original observations. In particular, the two old M3 `*_corrected_500.json` files contain 500 nominal records but repeat 200 actual seeds. Their old 237/500, 0/237 and 151/237 counts use repetition weights. The final 92/200, 0/92 and 59/92 counts use each actual seed once. M2's 45.9% historical seed-1 score is superseded by the direct-200 44.8% score. Historical files and saved notebook outputs are preserved, with current interpretation in the main documentation.

## Verify without running models

From the repository root, using Python 3.11:

```bash
python scripts/verify_revision_evidence.py --out runs/reproduced_v9/integrity.json
```

This uses only the standard library. It verifies retained hashes, the eleven integrated evaluation-code files, checkpoint digest references, seed splits, D3 input-control counts and M3 deduplication/grounding records. It does not perform model inference or verify historical execution dates. After downloading Git-LFS models, optionally add `--check-models` to hash their actual bytes.

To recompute all four D3 episode-bootstrap intervals with the project dependencies installed:

```bash
python v9/scripts/verify_d3_saved_bootstrap.py --repo . --source v9/evidence/d3_episode_clustered.json --out runs/reproduced_v9/d3_bootstrap_verification.json
```

Create `runs/reproduced_v9/` before using the standalone bootstrap script. Full model re-evaluation commands are in [docs/EXPERIMENTS.md](../docs/EXPERIMENTS.md). Write new outputs under `runs/` so the retained evidence stays intact.

## Preserved scripts and evidence limits

The original helper scripts in `evidence/` retain their local absolute paths. Some write beside themselves and must **not** be run directly against this archived directory: doing so can overwrite the retained JSON. Use the integrated CLI commands in `docs/EXPERIMENTS.md` with separate output paths; the original scripts remain available for provenance inspection.

The original code patch is already integrated into the repository's canonical `envs/`, `scripts/` and `tests/` files. Do not apply it again to this updated checkout. It is retained to reconstruct the change from the old frozen commit.

Checksums demonstrate content consistency, not independent evidence of when a file was created or submitted. MemoryS7 covers only four complete initial configurations, M3 actors share a world model and symbolic head, D3 has one checkpoint, and R1 still lacks complete original episode records. These limitations remain part of the final claims.
