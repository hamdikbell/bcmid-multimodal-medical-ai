"""Shared pytest fixtures: synthetic mini-dataset (in-memory ZIP + TSV)."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _make_png_bytes(w: int = 64, h: int = 64, value: int | None = None) -> bytes:
    """Create a valid PNG image as bytes."""
    if value is not None:
        arr = np.full((h, w), value, dtype=np.uint8)
    else:
        rng = np.random.default_rng(42)
        arr = rng.integers(10, 240, (h, w), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def _make_corrupt_bytes() -> bytes:
    return b"NOT_A_VALID_IMAGE_FILE_XYZ"


@pytest.fixture()
def synthetic_zip(tmp_path: Path) -> Path:
    """
    Build a synthetic ZIP with:
      - P001y19 : US (3 valid) + Mammo (2 valid) + report PDF  → complete
      - P002y20 : US (1 valid) + Mammo (1 valid) + no report   → complete (report optional)
      - P003y21 : US only (2 valid)                            → us_only
      - P004y22 : US (1 corrupt + 1 uniform) + Mammo (1 valid) → partial
    """
    zip_path = tmp_path / "test_dataset.zip"
    pdf_bytes = b"%PDF-1.4 fake pdf content"

    with zipfile.ZipFile(zip_path, "w") as zf:
        # P001: complete
        for i in range(3):
            zf.writestr(f"P001y19/Ultrasound/img_us_{i}.png", _make_png_bytes())
        for i in range(2):
            zf.writestr(f"P001y19/Mammogram/img_mam_{i}.png", _make_png_bytes())
        zf.writestr("P001y19/P001y19_Report.pdf", pdf_bytes)

        # P002: complete, no report
        zf.writestr("P002y20/Ultrasound/img_us_0.png", _make_png_bytes())
        zf.writestr("P002y20/Mammogram/img_mam_0.png", _make_png_bytes())

        # P003: us_only
        for i in range(2):
            zf.writestr(f"P003y21/Ultrasound/img_us_{i}.png", _make_png_bytes())

        # P004: 1 corrupt US + 1 uniform US + 1 valid mammo
        zf.writestr("P004y22/Ultrasound/img_us_corrupt.png", _make_corrupt_bytes())
        zf.writestr("P004y22/Ultrasound/img_us_uniform.png", _make_png_bytes(value=128))
        zf.writestr("P004y22/Mammogram/img_mam_0.png", _make_png_bytes())

    return zip_path


@pytest.fixture()
def synthetic_tsv(tmp_path: Path) -> Path:
    """TSV with labels including malformed and duplicate entries."""
    tsv_path = tmp_path / "annotations.tsv"
    lines = [
        "patient_id\tbirads_label",
        "P001y19\t4A",
        "P002y20\tBI-RADS 3",
        "P003y21\t4b",         # malformed casing → normalized to 4B
        "P004y22\t4A",
        "P005y23\t2",           # label without folder
        "P001y19\t5",           # DUPLICATE ID
    ]
    tsv_path.write_text("\n".join(lines), encoding="utf-8")
    return tsv_path


@pytest.fixture()
def config_for_tests(tmp_path: Path, synthetic_zip: Path, synthetic_tsv: Path):
    """Config object pointing to test fixtures."""
    from dq_agent.config import Config, IntegrityConfig, ModalityConfig, PreprocessingConfig, SplitsConfig

    cfg = Config(
        zip_path=synthetic_zip,
        tsv_path=synthetic_tsv,
        work_dir=tmp_path / "work",
        private_dir=tmp_path / "work",
    )
    cfg.integrity = IntegrityConfig(min_dimension=8, std_threshold=1.0)
    cfg.preprocessing = PreprocessingConfig(target_size=64)
    cfg.modality_policy = ModalityConfig(require_report=False)
    cfg.splits = SplitsConfig(random_seed=0, rare_class_threshold=2)
    return cfg
