"""Tests for modality classification."""

from __future__ import annotations

import pytest

from dq_agent.integrity import check_integrity
from dq_agent.modality import classify_patients, filter_for_training
from dq_agent.zip_io import scan_zip


def test_us_only_patient(config_for_tests):
    """P003y21 has only US images — must be classified as us_only."""
    patients = scan_zip(config_for_tests.zip_path)
    integrity = check_integrity(config_for_tests, patients)
    modalities = classify_patients(config_for_tests, patients, integrity["valid_members"])

    assert "P003y21" in modalities
    assert modalities["P003y21"]["status"] == "us_only"


def test_complete_patient(config_for_tests):
    """P001y19 has US + Mammo — must be complete."""
    patients = scan_zip(config_for_tests.zip_path)
    integrity = check_integrity(config_for_tests, patients)
    modalities = classify_patients(config_for_tests, patients, integrity["valid_members"])

    assert modalities["P001y19"]["status"] == "complete"


def test_filter_for_training(config_for_tests):
    """Multimodal list must not include us_only patients."""
    patients = scan_zip(config_for_tests.zip_path)
    integrity = check_integrity(config_for_tests, patients)
    modalities = classify_patients(config_for_tests, patients, integrity["valid_members"])
    multimodal, partial = filter_for_training(modalities)

    assert "P003y21" not in multimodal
    assert "P003y21" in partial
