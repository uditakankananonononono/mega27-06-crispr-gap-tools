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

Tools built here attack gaps 1, 2, 5, 6, 9 directly; the rest are documented with
dataset-level evidence in the paper.

## Contents
- `crisprgap/` - Python package: data loaders (real public datasets), CNN efficacy
  model, GNN off-target scorer, calibration, benchmark harness.
- `tests/` - hermetic pytest suite (no network); live pulls only via CLI scripts.
- `paper/` - 20-page Times New Roman research paper + figures.
