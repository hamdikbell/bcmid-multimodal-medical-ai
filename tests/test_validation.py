"""Tests for validation module."""

from __future__ import annotations

import pytest

from dq_agent.validation import normalize_birads, validate_labels
from dq_agent.zip_io import scan_zip


@pytest.mark.parametrize("raw, expected", [
    ("4A", "4A"),
    ("4a", "4A"),
    ("BI-RADS 4A", "4A"),
    ("bi-rads 4b", "4B"),
    ("BIRADS 3", "3"),
    ("BI RADS 4C", "4C"),
    ("2", "2"),
    ("0", "0"),
    ("5", "5"),
])
def test_birads_normalization(raw, expected):
    assert normalize_birads(raw) == expected


def test_duplicate_id_in_tsv_detected(config_for_tests):
    """P001y19 appears twice in test TSV — must be reported."""
    patients = scan_zip(config_for_tests.zip_path)
    result = validate_labels(config_for_tests, patients)
    assert "P001y19" in result["duplicate_ids_in_annotations"]


def test_label_without_folder_detected(config_for_tests):
    """P005y23 exists in TSV but has no folder — must be reported."""
    patients = scan_zip(config_for_tests.zip_path)
    result = validate_labels(config_for_tests, patients)
    assert "P005y23" in result["labels_without_folder"]


def test_malformed_label_4b(config_for_tests):
    """'4b' (lowercase) must be normalized to '4B' (not flagged as malformed)."""
    patients = scan_zip(config_for_tests.zip_path)
    result = validate_labels(config_for_tests, patients)
    # 4b → 4B is valid
    malformed_pids = [m["patient_id"] for m in result["malformed_labels"]]
    assert "P003y21" not in malformed_pids
    # Label must be normalized
    assert result["labels"]["P003y21"] == "4B"


def test_unknown_label_for_missing_patient(config_for_tests, tmp_path):
    """Patient in folder but not in TSV gets UNKNOWN label."""
    # Modify config to use a TSV without P004y22
    tsv_path = tmp_path / "partial.tsv"
    tsv_path.write_text("patient_id\tbirads_label\nP001y19\t4A\n", encoding="utf-8")

    from dq_agent.config import Config
    cfg = Config(
        zip_path=config_for_tests.zip_path,
        tsv_path=tsv_path,
        work_dir=tmp_path,
        private_dir=tmp_path,
    )
    patients = scan_zip(cfg.zip_path)
    result = validate_labels(cfg, patients)

    assert "P004y22" in result["patients_without_label"]
    assert result["labels"]["P004y22"] == "UNKNOWN"
