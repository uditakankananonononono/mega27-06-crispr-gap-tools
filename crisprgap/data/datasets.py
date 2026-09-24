"""Loaders for the public datasets used in this project.

Datasets (all public, fetched by scripts/fetch_data.sh):
- Doench 2016 FC+RES: 5,310 30-mers with drug-gene rank efficacy scores
  (MicrosoftResearch/azimuth mirror of Doench et al. 2016 Nat Biotechnol).
- Doench 2016 V1: 2,144 guides with log-fold-change activity (same source).
- crisprSQL 100720: 25,632 experimentally measured off-target cleavage records
  aggregated from GUIDE-seq / CIRCLE-seq / CHANGE-seq-class studies
  (Stortz & Minary, crisprsql.com).
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


@dataclass
class EfficacyDataset:
    sequences: list[str]  # 30-mer context sequences
    scores: np.ndarray    # continuous efficacy score
    source: str


@dataclass
class OffTargetDataset:
    guides: list[str]        # guide (on-target) protospacer sequences
    offtargets: list[str]    # candidate off-target protospacer sequences
    labels: np.ndarray       # 1 = experimentally cleaved, 0 = not cleaved
    freqs: np.ndarray        # raw cleavage frequencies
    studies: list[str]


def _clean_seq(s: str) -> str:
    return s.strip().upper()


def load_doench_fcres(path: str | None = None) -> EfficacyDataset:
    path = path or os.path.join(DATA_DIR, "FC_plus_RES_withPredictions.csv")
    seqs, scores = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            s = _clean_seq(row["30mer"])
            if len(s) != 30 or any(b not in "ACGT" for b in s):
                continue
            seqs.append(s)
            scores.append(float(row["score_drug_gene_rank"]))
    return EfficacyDataset(seqs, np.array(scores, dtype=np.float32), "doench2016_fcres")


def load_doench_v1(path: str | None = None) -> EfficacyDataset:
    path = path or os.path.join(DATA_DIR, "V1_suppl_data.txt")
    seqs, scores = [], []
    with open(path) as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 11:
                continue
            # V1 "Extended Spacer" is a 34-mer: NNNN + 20nt guide + NGG + NNNNNNN.
            # The Azimuth 30-mer model input is its first 30 bases.
            s = _clean_seq(parts[1])[:30]
            if len(s) != 30 or any(b not in "ACGT" for b in s):
                continue
            seqs.append(s)
            scores.append(float(parts[10]))  # Percent Rank
    return EfficacyDataset(seqs, np.array(scores, dtype=np.float32), "doench2016_v1")


def load_crisprsql(path: str | None = None) -> OffTargetDataset:
    path = path or os.path.join(DATA_DIR, "crisprsql", "100720.csv")
    guides, offts, labels, freqs, studies = [], [], [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            g = _clean_seq(row["grna_target_sequence"])
            o = _clean_seq(row["target_sequence"])
            # strip PAM for the 20-nt protospacer when sequences are 23-nt NGG-flanked
            g20, o20 = g[:20], o[:20]
            if len(g20) != 20 or len(o20) != 20 or len(g20) != len(o20):
                continue
            if any(b not in "ACGT" for b in g20 + o20):
                continue
            freq = float(row["cleavage_freq"] or 0.0)
            guides.append(g20)
            offts.append(o20)
            freqs.append(freq)
            labels.append(1 if freq > 0 else 0)
            studies.append(row["study_name"])
    return OffTargetDataset(guides, offts, np.array(labels, dtype=np.float32),
                            np.array(freqs, dtype=np.float32), studies)
