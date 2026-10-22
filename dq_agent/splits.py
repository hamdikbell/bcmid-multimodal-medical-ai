"""Module 8 — Splits: stratified train/val/test split by patient (no data leakage).

label_group is added as metadata but splits are stratified on it (not on raw birads).
The grouping must be validated by the user before calling make_splits.
"""

from __future__ import annotations

import json
import logging
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .config import Config

logger = logging.getLogger(__name__)

# ── Proposed clinical grouping (to be validated by user) ─────────────────────
DEFAULT_LABEL_GROUPS: dict[str, str] = {
    "0": "incomplete",
    "1": "benign",
    "2": "benign",
    "3": "probably_benign",
    "4": "suspicious",
    "4A": "suspicious",
    "4B": "suspicious",
    "4C": "suspicious",
    "5": "highly_suspicious",
    "6": "known_malignant",
    "UNKNOWN": "unknown",
    "RARE": "rare",
}


def get_label_group(birads: str, groups: dict[str, str] | None = None) -> str:
    """Map a normalized BI-RADS label to its clinical group."""
    g = groups or DEFAULT_LABEL_GROUPS
    return g.get(birads, "unknown")


def show_distribution(patient_labels: dict[str, str]) -> None:
    """Log BI-RADS distribution and proposed groups."""
    counts = Counter(patient_labels.values())
    logger.info("BI-RADS distribution (raw):")
    for cls in sorted(counts):
        grp = get_label_group(cls)
        logger.info("  BI-RADS %s → group '%s' : %d patients", cls, grp, counts[cls])


def _group_rare_classes(
    patient_labels: dict[str, str],
    threshold: int,
    label_groups: dict[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    """
    Group BI-RADS classes with fewer than threshold patients into 'RARE'.
    Classes 0 and 1 are NEVER merged with 2 automatically.

    Returns
    -------
    (strat_labels, rare_mapping)
    strat_labels: patient_id -> label_group (used for stratification)
    rare_mapping: original_class -> 'rare' (documented)
    """
    group_counts = Counter(
        get_label_group(lbl, label_groups) for lbl in patient_labels.values()
    )
    rare_groups = {grp for grp, cnt in group_counts.items() if cnt < threshold}

    if rare_groups:
        logger.info(
            "Groups with < %d patients (rare): %s — will be pooled as 'rare' for splits only",
            threshold, sorted(rare_groups),
        )

    strat_labels: dict[str, str] = {}
    rare_mapping: dict[str, str] = {}

    for pid, birads in patient_labels.items():
        grp = get_label_group(birads, label_groups)
        if grp in rare_groups:
            strat_labels[pid] = "rare"
            rare_mapping[birads] = "rare"
        else:
            strat_labels[pid] = grp

    return strat_labels, rare_mapping


def make_splits(
    cfg: Config,
    multimodal_ids: list[str],
    labels: dict[str, str],
    label_groups: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Create stratified train/val/test splits by patient.

    Stratification is by label_group. BI-RADS 0 and 1 are NEVER auto-merged with 2.
    No patient appears in more than one split.

    Returns
    -------
    dict with train/val/test lists of patient IDs and metadata.
    """
    rng = random.Random(cfg.splits.random_seed)
    threshold = cfg.splits.rare_class_threshold
    lg = label_groups or DEFAULT_LABEL_GROUPS

    # Separate known vs unknown labels
    patient_labels = {
        pid: labels.get(pid, "UNKNOWN")
        for pid in multimodal_ids
        if labels.get(pid, "UNKNOWN") != "UNKNOWN"
    }
    unknown_ids = [pid for pid in multimodal_ids if labels.get(pid, "UNKNOWN") == "UNKNOWN"]
    if unknown_ids:
        logger.warning(
            "%d patients with UNKNOWN label — added to train set.", len(unknown_ids)
        )

    show_distribution(patient_labels)

    # Group by clinical category for stratification
    strat_labels, rare_mapping = _group_rare_classes(patient_labels, threshold, lg)

    # Stratify buckets
    class_buckets: dict[str, list[str]] = defaultdict(list)
    for pid, grp in strat_labels.items():
        class_buckets[grp].append(pid)

    for grp in class_buckets:
        rng.shuffle(class_buckets[grp])

    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []

    train_r = cfg.splits.train_ratio
    val_r = cfg.splits.val_ratio

    for grp, pids in class_buckets.items():
        n = len(pids)
        n_train = max(1, math.floor(n * train_r))
        n_val = max(0, math.floor(n * val_r))
        n_val = min(n_val, n - n_train)
        n_test = n - n_train - n_val

        train_ids.extend(pids[:n_train])
        val_ids.extend(pids[n_train: n_train + n_val])
        test_ids.extend(pids[n_train + n_val:])

    train_ids.extend(unknown_ids)

    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    rng.shuffle(test_ids)

    # ── No-leakage invariant ──
    assert set(train_ids).isdisjoint(set(val_ids)),  "LEAK: patient(s) in train AND val"
    assert set(train_ids).isdisjoint(set(test_ids)), "LEAK: patient(s) in train AND test"
    assert set(val_ids).isdisjoint(set(test_ids)),   "LEAK: patient(s) in val AND test"

    logger.info(
        "Splits — train=%d, val=%d, test=%d (seed=%d)",
        len(train_ids), len(val_ids), len(test_ids), cfg.splits.random_seed,
    )

    return {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "n_test": len(test_ids),
        "random_seed": cfg.splits.random_seed,
        "rare_groups_pooled": rare_mapping,
        "rare_threshold": threshold,
        "unknown_label_patients_in_train": unknown_ids,
        "label_groups_used": lg,
        "class_distribution": {
            "train": dict(Counter(strat_labels.get(p, "unknown") for p in train_ids)),
            "val":   dict(Counter(strat_labels.get(p, "unknown") for p in val_ids)),
            "test":  dict(Counter(strat_labels.get(p, "unknown") for p in test_ids)),
        },
    }


def save_splits(splits: dict[str, Any], out_dir: Path) -> None:
    """Save splits.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "splits.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2, ensure_ascii=False)
    logger.info("Splits saved: %s", path)
