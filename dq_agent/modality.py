"""Module 3 — Modality classification: complete / mammo_only / us_only / empty."""

from __future__ import annotations

import logging
from typing import Any

from .config import Config

logger = logging.getLogger(__name__)

# Canonical status values
STATUS_COMPLETE = "complete"
STATUS_MAMMO_ONLY = "mammo_only"
STATUS_US_ONLY = "us_only"
STATUS_EMPTY = "empty"
STATUS_PARTIAL = "partial"  # has both modalities but missing report


def classify_patients(
    cfg: Config,
    patients: dict[str, dict],
    valid_members: set[str],
) -> dict[str, dict]:
    """
    Classify each patient by modality completeness.

    Parameters
    ----------
    cfg : Config
    patients : zip inventory from scan_zip()
    valid_members : set of zip members that passed integrity checks

    Returns
    -------
    dict mapping patient_id -> {
        "status": str,
        "n_valid_us": int,
        "n_valid_mammo": int,
        "has_report": bool,
        "valid_us_members": list,
        "valid_mammo_members": list,
    }
    """
    result: dict[str, dict] = {}
    require_report = cfg.modality_policy.require_report

    for pid, inv in patients.items():
        valid_us = [m for m in inv["us"] if m in valid_members]
        valid_mammo = [m for m in inv["mammo"] if m in valid_members]
        has_report = len(inv["report"]) > 0

        has_us = len(valid_us) > 0
        has_mammo = len(valid_mammo) > 0

        # Status based ONLY on images — report presence is metadata, not a gate
        if has_us and has_mammo:
            status = STATUS_COMPLETE
        elif has_mammo:
            status = STATUS_MAMMO_ONLY
        elif has_us:
            status = STATUS_US_ONLY
        else:
            status = STATUS_EMPTY

        result[pid] = {
            "status": status,
            "n_valid_us": len(valid_us),
            "n_valid_mammo": len(valid_mammo),
            "has_report": has_report,
            "valid_us_members": valid_us,
            "valid_mammo_members": valid_mammo,
        }

    counts = {}
    for v in result.values():
        s = v["status"]
        counts[s] = counts.get(s, 0) + 1

    logger.info("Modality classification — %s", counts)
    return result


def modality_summary(modalities: dict[str, dict]) -> dict[str, int]:
    """Return status counts."""
    counts: dict[str, int] = {}
    for v in modalities.values():
        s = v["status"]
        counts[s] = counts.get(s, 0) + 1
    return counts


def filter_for_training(
    modalities: dict[str, dict],
) -> tuple[list[str], list[str]]:
    """
    Split patients into multimodal (for training) and partial (for unimodal use).

    Returns
    -------
    (multimodal_ids, partial_ids)
    """
    multimodal = [pid for pid, v in modalities.items() if v["status"] == STATUS_COMPLETE]
    partial = [pid for pid, v in modalities.items() if v["status"] != STATUS_COMPLETE]
    logger.info(
        "Training split — multimodal=%d, partial/excluded=%d",
        len(multimodal),
        len(partial),
    )
    return multimodal, partial
