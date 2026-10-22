"""Module 6 — Validation: join images with BI-RADS labels, detect anomalies."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Config

logger = logging.getLogger(__name__)

# BI-RADS normalization patterns
_BIRADS_PATTERNS = [
    # Match "BI-RADS 4A", "BIRADS4a", "bi rads 4A" etc.
    re.compile(r"bi[\s\-_]*rads[\s\-_]*(\d[A-Ca-c]?)", re.IGNORECASE),
    # Match bare numbers/letters like "4A", "4a", "4", "0"
    re.compile(r"^(\d[A-Ca-c]?)$", re.IGNORECASE),
]

_VALID_BIRADS = {
    "0", "1", "2", "3",
    "4", "4A", "4B", "4C",
    "5", "6",
}


def normalize_birads(raw: str) -> str:
    """
    Normalize a raw BI-RADS label string to canonical form.

    Examples
    --------
    "4A" -> "4A"
    "4a" -> "4A"
    "BI-RADS 4A" -> "4A"
    "bi rads 4b" -> "4B"
    "4" -> "4"
    """
    raw = str(raw).strip()
    for pat in _BIRADS_PATTERNS:
        m = pat.search(raw)
        if m:
            val = m.group(1).upper()
            return val
    return raw.upper().strip()


def load_annotations(tsv_path: Path) -> tuple[pd.DataFrame, list[str]]:
    """
    Load and inspect the annotations TSV.

    Returns
    -------
    (df, column_names)
    df has columns auto-detected; patient_id_col and birads_col identified heuristically.
    """
    if not tsv_path.exists():
        raise FileNotFoundError(f"Annotations TSV not found: {tsv_path}")

    df = pd.read_csv(tsv_path, sep="\t", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    logger.info("TSV loaded — columns: %s, rows: %d", list(df.columns), len(df))
    return df, list(df.columns)


def detect_id_and_label_columns(df: pd.DataFrame) -> tuple[str, str]:
    """
    Heuristically identify the patient ID column and the BI-RADS label column.
    Works with the real BCMID TSV columns: 'Patient ID', 'BI-RADS Label', etc.
    """
    cols = [c.lower() for c in df.columns]
    orig_cols = list(df.columns)

    id_col = None
    label_col = None

    # ID column heuristics — covers "Patient ID", "patient_id", "id", etc.
    for kw in ["patient id", "patient_id", "patientid", "id", "patient", "case", "subject"]:
        for i, c in enumerate(cols):
            if kw in c and "label" not in c:
                id_col = orig_cols[i]
                break
        if id_col:
            break

    # Label column heuristics — covers "BI-RADS Label", "birads", "label", etc.
    for kw in ["bi-rads", "birads", "bi_rads", "bi rads"]:
        for i, c in enumerate(cols):
            if kw in c:
                label_col = orig_cols[i]
                break
        if label_col:
            break
    # Fallback to any column named "label" or "class"
    if label_col is None:
        for kw in ["label", "class", "category", "grade"]:
            for i, c in enumerate(cols):
                if kw in c and i != cols.index(id_col.lower() if id_col else ""):
                    label_col = orig_cols[i]
                    break
            if label_col:
                break

    # Last-resort fallback
    if id_col is None:
        id_col = orig_cols[0]
        logger.warning("Could not detect ID column — using first column: %s", id_col)
    if label_col is None:
        label_col = orig_cols[-1]
        logger.warning("Could not detect label column — using last column: %s", label_col)

    logger.info("Detected columns — ID: '%s', Label: '%s'", id_col, label_col)
    return id_col, label_col


def validate_labels(
    cfg: Config,
    patients: dict[str, dict],
) -> dict[str, Any]:
    """
    Full validation of BI-RADS labels against patient folders.

    Returns
    -------
    dict with:
      - labels: dict patient_id -> normalized_birads
      - issues: list of issue dicts
      - duplicate_ids_in_annotations: list
      - patients_without_label: list
      - labels_without_folder: list
      - malformed_labels: list
    """
    tsv_path = cfg.tsv_path
    issues: list[dict] = []

    # Handle missing TSV gracefully
    if not tsv_path.exists():
        logger.warning("TSV not found at %s — labels will be UNKNOWN", tsv_path)
        labels = {pid: "UNKNOWN" for pid in patients}
        return {
            "labels": labels,
            "issues": [{"type": "missing_tsv", "detail": str(tsv_path)}],
            "duplicate_ids_in_annotations": [],
            "patients_without_label": list(patients.keys()),
            "labels_without_folder": [],
            "malformed_labels": [],
            "tsv_columns": [],
            "id_column": None,
            "label_column": None,
        }

    df, cols = load_annotations(tsv_path)
    id_col, label_col = detect_id_and_label_columns(df)

    # Capture any extra informative columns
    extra_cols = [c for c in cols if c not in (id_col, label_col)]
    extra_data: dict[str, dict] = {}

    # Detect duplicate IDs in TSV
    dup_ids = df[df.duplicated(subset=[id_col], keep=False)][id_col].tolist()
    if dup_ids:
        issues.append({"type": "duplicate_ids_in_annotations", "ids": list(set(dup_ids))})
        logger.warning("Duplicate IDs in annotations: %s", dup_ids[:10])

    # Deduplicate (keep first)
    df = df.drop_duplicates(subset=[id_col], keep="first")

    # Store extra column data per patient
    for _, row in df.iterrows():
        pid = str(row[id_col]).strip()
        extra_data[pid] = {c: str(row[c]).strip() for c in extra_cols}

    # Normalize labels
    raw_labels = df.set_index(id_col)[label_col].to_dict()
    normalized_labels: dict[str, str] = {}
    malformed: list[dict] = []

    for pid_raw, lbl in raw_labels.items():
        pid = str(pid_raw).strip()
        norm = normalize_birads(str(lbl))
        if norm not in _VALID_BIRADS:
            malformed.append({"patient_id": pid, "raw_label": lbl, "normalized": norm})
            issues.append({"type": "malformed_label", "patient_id": pid, "raw": lbl, "normalized": norm})
        normalized_labels[pid] = norm

    # Cross-join
    folder_ids = set(patients.keys())
    annotation_ids = set(normalized_labels.keys())

    patients_without_label = sorted(folder_ids - annotation_ids)
    labels_without_folder = sorted(annotation_ids - folder_ids)

    for pid in patients_without_label:
        issues.append({"type": "patient_without_label", "patient_id": pid})
        normalized_labels[pid] = "UNKNOWN"

    if labels_without_folder:
        issues.append({"type": "labels_without_folder", "ids": labels_without_folder})

    logger.info(
        "Validation — %d labels loaded, %d without label, %d labels without folder, %d malformed",
        len(raw_labels),
        len(patients_without_label),
        len(labels_without_folder),
        len(malformed),
    )

    return {
        "labels": normalized_labels,
        "issues": issues,
        "duplicate_ids_in_annotations": list(set(dup_ids)),
        "patients_without_label": patients_without_label,
        "labels_without_folder": labels_without_folder,
        "malformed_labels": malformed,
        "tsv_columns": cols,
        "id_column": id_col,
        "label_column": label_col,
        "extra_columns": extra_cols if 'extra_cols' in dir() else [],
        "extra_data": extra_data if 'extra_data' in dir() else {},
        "n_patients_in_annotations": len(df) if 'df' in dir() else 0,
    }
