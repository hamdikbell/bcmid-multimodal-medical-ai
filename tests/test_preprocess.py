"""Tests for preprocessing module."""

from __future__ import annotations

import io
import numpy as np
import pytest
from PIL import Image


def _make_arr(w=64, h=64, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(20, 235, (h, w), dtype=np.uint8).astype(np.float32) / 255.0


def test_preprocess_us_output_shape(config_for_tests):
    """US preprocessing must output target_size × target_size array."""
    from dq_agent.preprocess import preprocess_us

    arr = _make_arr()
    result = preprocess_us(arr, config_for_tests)
    assert result.shape == (64, 64), f"Expected (64,64), got {result.shape}"


def test_preprocess_mammo_output_shape(config_for_tests):
    """Mammography preprocessing must output target_size × target_size array."""
    from dq_agent.preprocess import preprocess_mammo

    arr = _make_arr()
    result = preprocess_mammo(arr, config_for_tests)
    assert result.shape == (64, 64)


def test_preprocess_output_uint8_range(config_for_tests):
    """Preprocessed arrays must have values in [0, 255] (uint8)."""
    from dq_agent.preprocess import preprocess_us

    arr = _make_arr()
    result = preprocess_us(arr, config_for_tests)
    assert result.min() >= 0
    assert result.max() <= 255


def test_normalize_01():
    """_normalize_01 must map min→0 and max→1."""
    from dq_agent.preprocess import _normalize_01

    arr = np.array([10.0, 50.0, 100.0])
    normed = _normalize_01(arr)
    assert abs(normed.min()) < 1e-6
    assert abs(normed.max() - 1.0) < 1e-6


def test_process_patient_saves_files(config_for_tests):
    """process_patient must create PNG files on disk."""
    import zipfile
    from dq_agent.zip_io import scan_zip
    from dq_agent.integrity import check_integrity
    from dq_agent.modality import classify_patients
    from dq_agent.preprocess import process_patient

    cfg = config_for_tests
    patients = scan_zip(cfg.zip_path)
    integrity = check_integrity(cfg, patients)
    modalities = classify_patients(cfg, patients, integrity["valid_members"])

    pid = "P001y19"
    pseudo = "Ptest12345a"

    cfg.mammo_dir.mkdir(parents=True, exist_ok=True)
    cfg.us_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(cfg.zip_path, "r") as zf:
        result = process_patient(pid, pseudo, modalities[pid], cfg, zf)

    assert result["n_us"] > 0
    assert result["n_mammo"] > 0
    for p in result["us_paths"]:
        assert (cfg.validated_dir / p).exists()
    for p in result["mammo_paths"]:
        assert (cfg.validated_dir / p).exists()
