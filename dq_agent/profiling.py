"""Module 1 — Profiling: dataset statistics and BI-RADS distribution."""

from __future__ import annotations

import json
import logging
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import Config
from .zip_io import read_image_from_zip, scan_zip

logger = logging.getLogger(__name__)

# Pattern: {digits}y{suffix} — suffix encodes study year (e.g. "19" = 2019)
_ID_PATTERN = re.compile(r"^(\d+)y(\d{2,3})$")


def _parse_age_from_id(pid: str) -> int | None:
    """Extract year suffix from patient ID (not the real age — just the year code)."""
    m = _ID_PATTERN.match(pid)
    if m:
        return int(m.group(2))
    return None


def profile_dataset(
    cfg: Config,
    patients: dict[str, dict],
    labels: dict[str, str],
    n_total_zip_patients: int | None = None,
) -> dict[str, Any]:
    """
    Compute dataset-level profiling statistics.

    Parameters
    ----------
    cfg : Config
    patients : output of zip_io.scan_zip() (may be limited)
    labels : dict mapping patient_id -> BI-RADS label (from validation step)
    n_total_zip_patients : total patient dirs in ZIP (before --limit). If None, uses len(patients).

    Returns
    -------
    dict with profiling stats
    """
    zip_path = cfg.zip_path
    n_zip = n_total_zip_patients if n_total_zip_patients is not None else len(patients)
    n_tsv = len(set(labels.values()) - {"UNKNOWN"})  # patients actually in TSV
    # n_patients_in_annotations is set by validation and passed through labels
    # But labels dict includes patients with UNKNOWN label (not in TSV)
    # So: patients with real labels = patients where label != UNKNOWN
    n_annotated = sum(1 for v in labels.values() if v != "UNKNOWN")
    stats: dict[str, Any] = {
        "n_patient_folders": n_zip,
        "n_patients_in_annotations": n_annotated,
        "ecart_patients": abs(n_zip - n_annotated),
        "patients_only_in_folders": sorted(set(patients) - set(labels)),
        "patients_only_in_annotations": sorted(set(labels) - set(patients)),
        "image_extensions": {},
        "modality_counts": {"us": 0, "mammo": 0},
        "per_patient": [],
        "birads_distribution": {},
        "year_suffix_distribution": {},
        "images_without_report": 0,
    }

    ext_counter: dict[str, int] = {}
    birads_counter: dict[str, int] = {}
    year_counter: dict[str, int] = {}

    us_counts: list[int] = []
    mammo_counts: list[int] = []

    logger.info("Profiling %d patients …", len(patients))

    with zipfile.ZipFile(zip_path, "r") as zf:
        for pid, inv in tqdm(patients.items(), desc="Profiling", unit="patient"):
            n_us = len(inv["us"])
            n_mammo = len(inv["mammo"])
            has_report = len(inv["report"]) > 0
            label = labels.get(pid, "UNKNOWN")

            us_counts.append(n_us)
            mammo_counts.append(n_mammo)

            if not has_report:
                stats["images_without_report"] += 1

            # Count extensions
            for member in inv["us"] + inv["mammo"]:
                ext = Path(member).suffix.lower().lstrip(".")
                ext_counter[ext] = ext_counter.get(ext, 0) + 1

            # Sample one US + one mammo image for intensity stats
            us_intensity: float | None = None
            mammo_intensity: float | None = None

            if inv["us"]:
                img = read_image_from_zip(zf, inv["us"][0])
                if img is not None:
                    arr = np.array(img.convert("L"), dtype=float)
                    us_intensity = float(arr.mean())

            if inv["mammo"]:
                img = read_image_from_zip(zf, inv["mammo"][0])
                if img is not None:
                    arr = np.array(img.convert("L"), dtype=float)
                    mammo_intensity = float(arr.mean())

            # Year suffix
            yr = _parse_age_from_id(pid)
            if yr is not None:
                year_counter[str(yr)] = year_counter.get(str(yr), 0) + 1

            # BI-RADS
            birads_counter[label] = birads_counter.get(label, 0) + 1

            stats["per_patient"].append(
                {
                    "patient_id": pid,  # will be pseudonymised later
                    "n_us": n_us,
                    "n_mammo": n_mammo,
                    "has_report": has_report,
                    "birads": label,
                    "year_suffix": yr,
                    "us_mean_intensity": us_intensity,
                    "mammo_mean_intensity": mammo_intensity,
                }
            )

    stats["image_extensions"] = ext_counter
    stats["modality_counts"]["us"] = sum(us_counts)
    stats["modality_counts"]["mammo"] = sum(mammo_counts)
    stats["birads_distribution"] = dict(sorted(birads_counter.items()))
    stats["year_suffix_distribution"] = dict(sorted(year_counter.items(), key=lambda x: int(x[0])))

    if us_counts:
        stats["us_per_patient"] = {
            "min": int(min(us_counts)),
            "max": int(max(us_counts)),
            "mean": round(float(np.mean(us_counts)), 2),
            "std": round(float(np.std(us_counts)), 2),
        }
    if mammo_counts:
        stats["mammo_per_patient"] = {
            "min": int(min(mammo_counts)),
            "max": int(max(mammo_counts)),
            "mean": round(float(np.mean(mammo_counts)), 2),
            "std": round(float(np.std(mammo_counts)), 2),
        }

    logger.info(
        "Profiling done — %d patients, BI-RADS classes: %s",
        len(patients),
        list(stats["birads_distribution"].keys()),
    )
    return stats


def save_profile(profile: dict[str, Any], out_dir: Path) -> None:
    """Persist profiling data (without real IDs in the public version)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save public version (no patient IDs)
    public = {k: v for k, v in profile.items() if k != "per_patient"}
    with open(out_dir / "profile_summary.json", "w", encoding="utf-8") as f:
        json.dump(public, f, indent=2, ensure_ascii=False)
    logger.info("Profile summary saved to %s", out_dir / "profile_summary.json")
