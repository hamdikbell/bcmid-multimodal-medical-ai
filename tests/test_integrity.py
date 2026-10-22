"""Tests for integrity module."""

from __future__ import annotations

import pytest

from dq_agent.integrity import check_integrity
from dq_agent.zip_io import scan_zip


def test_corrupt_image_detected(config_for_tests):
    """Corrupted PNG must appear in corrupted_or_invalid list."""
    patients = scan_zip(config_for_tests.zip_path)
    result = check_integrity(config_for_tests, patients)
    corrupted_members = [c["member"] for c in result["corrupted_or_invalid"]]
    assert any("corrupt" in m for m in corrupted_members), (
        f"Expected corrupt image in results, got: {corrupted_members}"
    )


def test_uniform_image_detected(config_for_tests):
    """Uniform image (constant pixel value) must be flagged."""
    patients = scan_zip(config_for_tests.zip_path)
    result = check_integrity(config_for_tests, patients)
    corrupted_members = [c["member"] for c in result["corrupted_or_invalid"]]
    assert any("uniform" in m for m in corrupted_members), (
        f"Expected uniform image in results, got: {corrupted_members}"
    )


def test_valid_images_pass(config_for_tests):
    """Valid PNG images must appear in valid_members."""
    patients = scan_zip(config_for_tests.zip_path)
    result = check_integrity(config_for_tests, patients)
    assert len(result["valid_members"]) >= 7  # P001 (5) + P002 (2) at minimum


def test_duplicate_detection(tmp_path):
    """Identical file content must be reported as duplicate."""
    import io, zipfile
    from PIL import Image
    import numpy as np
    from dq_agent.config import Config, IntegrityConfig

    same_content = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    # Create a proper PNG
    buf = io.BytesIO()
    arr = np.random.randint(20, 200, (64, 64), dtype=np.uint8)
    Image.fromarray(arr, "L").save(buf, "PNG")
    png_bytes = buf.getvalue()

    zip_path = tmp_path / "dup.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("PAT1y18/Ultrasound/img1.png", png_bytes)
        zf.writestr("PAT1y18/Mammogram/img2.png", png_bytes)  # same content = duplicate

    cfg = Config(zip_path=zip_path, tsv_path=tmp_path / "none.tsv",
                 work_dir=tmp_path, private_dir=tmp_path)
    cfg.integrity = IntegrityConfig(min_dimension=8, std_threshold=0.5)

    patients = scan_zip(zip_path)
    result = check_integrity(cfg, patients)
    assert result["duplicate_count"] >= 1
