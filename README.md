# MEGA27-06: CRISPR / Biohacking Gap-Set Tool Suite

**Research angle.** Public CRISPR tooling is concentrated on a narrow slice of the
editing problem (SpCas9 on-target efficacy in a few human cell lines). This project
mines the published literature and public datasets to identify **10 verified,
evidence-backed gaps** in CRISPR/biohacking tooling, then builds working tools for
the highest-value gaps and benchmarks them against public leaders.

## The 10 gaps (each verified against literature + dataset availability)

1. On-target gRNA efficacy prediction is rarely benchmarked *across* independent
   public datasets (cross-dataset generalization gap).
2. Off-target scoring tools seldom expose calibrated probabilities (calibration gap).
3. Prime-editing pegRNA design tooling is thin relative to base editing.
4. PAM-variant Cas enzymes (SpG/SpRY-class) have sparse dedicated off-target tools.
5. Mismatch-context (chromatin/sequence context) is underused in off-target scoring.
6. Multiplex gRNA design lacks cross-guide interference screening.
7. Repair-outcome (microhomology vs. NHEJ) prediction is not bundled with gRNA tools.
8. Unintended large-deletion / rearrangement risk flagging is missing from pipelines.
9. Sequence-of-concern screening for DIY/biohacking use is not unified with design.
10. Cross-species (non-model organism) guide efficacy transfer is untested in tools.

This repo is the early umbrella for item 6. The tools for gaps 1-5 and 6-10 live in
`mega27-06a-crispr-gap-tools-1-5` and `mega27-06b-crispr-gap-tools-6-10`; see those repos
for their code, results and papers. The gap list above is the audit that motivated them.

## Contents
- `crisprgap/` - Python package: data loaders (real public datasets), CNN efficacy
  model, GNN off-target scorer, calibration, baselines, training scripts.
- `tests/` - 22 hermetic test functions (no network); live pulls only via `scripts/fetch_data.sh`.
- `results/` - one committed result set: `efficacy_metrics.json`.
- `docs/SOURCE_PROVENANCE_GAPS.md` - documentation record of source-term and hash gaps.

## Status: verified / thin / missing
- **Verified (committed result):** efficacy CNN vs ridge on Doench 2016 FC+RES (train 4,248 / test 1,062, 15 epochs, seed 0): test Spearman 0.556 (CNN) vs 0.463 (ridge); on the V1 set (n=2,144) 0.609 vs 0.444 (`results/efficacy_metrics.json`).
- **Thin:** off-target GNN (`crisprgap/models/offtarget_gnn.py`, `train_offtarget.py`) and calibration (`calibration.py`) have code and tests but no committed benchmark result in this repo.
- **Missing:** no paper or PDF has ever been committed here (no `paper/` directory in the git history). Earlier text claiming a 20-page paper in this repo was wrong. No multiplex, pegRNA, PAM-variant or biosecurity tools are built in this repo.
