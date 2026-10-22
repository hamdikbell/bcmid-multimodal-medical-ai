"""Tests for report module."""

from __future__ import annotations

import json
from pathlib import Path


def _base_profile():
    return {
        "n_patient_folders": 332,
        "n_patients_in_annotations": 323,
        "ecart_patients": 9,
        "patients_only_in_folders": ["X1", "X2"],
        "patients_only_in_annotations": [],
        "image_extensions": {"png": 2001},
        "birads_distribution": {"3": 50, "4A": 120, "4B": 80, "5": 73},
        "images_without_report": 11,
        "us_per_patient": {"min": 1, "max": 16, "mean": 3.2, "std": 1.1},
        "mammo_per_patient": {"min": 1, "max": 4, "mean": 2.9, "std": 0.6},
        "per_patient": [],
    }


def _base_integrity():
    return {
        "corrupted_count": 3,
        "corrupted_sample": [{"member": "a/b.png", "reason": "corrupt"}],
        "duplicate_count": 0,
        "duplicate_files": [],
    }


def _base_validation():
    return {
        "n_patients_in_annotations": 323,
        "patients_without_label": ["P_X1"],
        "labels_without_folder": ["P_Y2"],
        "duplicate_ids_in_annotations": [],
        "malformed_labels": [],
        "tsv_columns": ["patient_id", "birads"],
        "id_column": "patient_id",
        "label_column": "birads",
        "issues": [],
    }


def test_report_contains_required_keys(tmp_path):
    from dq_agent.report import build_report, save_report

    report = build_report(
        profile=_base_profile(),
        integrity_result=_base_integrity(),
        modality_counts={"complete": 321, "us_only": 0},
        validation_result=_base_validation(),
        preprocessing_params={"target_size": 512},
        n_multimodal_ready=321,
    )

    required_keys = [
        "n_patient_folders", "n_patients_in_annotations", "ecart_patients",
        "modality_status", "birads_distribution", "corrupted_or_invalid_count",
        "duplicate_files_count", "patients_without_label", "labels_without_folder",
        "duplicate_ids_in_annotations", "patients_ready_multimodal",
        "parametres_preprocessing", "library_versions", "date_generated",
    ]
    for key in required_keys:
        assert key in report, f"Missing required key in report: {key}"


def test_markdown_report_created(tmp_path):
    from dq_agent.report import build_report, save_report

    report = build_report(
        profile=_base_profile(),
        integrity_result=_base_integrity(),
        modality_counts={"complete": 321},
        validation_result=_base_validation(),
        preprocessing_params={"target_size": 512},
        n_multimodal_ready=321,
    )
    save_report(report, tmp_path)

    md_path = tmp_path / "quality_report.md"
    json_path = tmp_path / "quality_report.json"
    assert md_path.exists()
    assert json_path.exists()

    # Check MD has tables
    content = md_path.read_text(encoding="utf-8")
    assert "BI-RADS" in content
    assert "|" in content  # has table

    # Check JSON is valid
    with open(json_path, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["ecart_patients"] == 9
