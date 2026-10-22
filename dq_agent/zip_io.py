"""ZIP I/O utilities — dual-mode: ZIP file OR extracted directory (Kaggle-compatible)."""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path, PurePosixPath
from typing import Generator

from PIL import Image

logger = logging.getLogger(__name__)

# Folder name aliases for modalities
_US_ALIASES = {"ultrasound", "us", "echo", "echography", "ultrason"}
_MAMMO_ALIASES = {"mammogram", "mammography", "mammo", "mg", "mammographie"}


def _is_us_folder(name: str) -> bool:
    return name.lower() in _US_ALIASES


def _is_mammo_folder(name: str) -> bool:
    return name.lower() in _MAMMO_ALIASES


def _is_image(fname: str) -> bool:
    return fname.lower().endswith((".png", ".jpg", ".jpeg", ".dcm", ".bmp", ".tiff", ".tif"))


def _is_pdf(fname: str) -> bool:
    return fname.lower().endswith(".pdf")


# ══════════════════════════════════════════════════════════════════
# AUTO-DETECTION (like notebook1)
# ══════════════════════════════════════════════════════════════════

def auto_find_dataset(search_root: Path = Path("/kaggle/input")) -> tuple[Path | None, Path | None]:
    """
    Auto-detect BCMID dataset like notebook1 does.
    Recursively searches for BCMID_annotations.tsv and derives data root.

    Returns (data_root, tsv_path) or (None, None) if not found.
    """
    # Search in kaggle input first, then cwd
    for root in [search_root, Path(".")]:
        if not root.exists():
            continue
        candidates = list(root.rglob("BCMID_annotations.tsv"))
        if candidates:
            tsv_path = candidates[0]
            data_root = tsv_path.parent
            logger.info("Auto-detected TSV: %s", tsv_path)
            logger.info("Auto-detected data root: %s", data_root)
            return data_root, tsv_path

    return None, None


def auto_find_zip(search_root: Path = Path("/kaggle/input")) -> Path | None:
    """Find a BCMID ZIP file under search_root."""
    for root in [search_root, Path(".")]:
        if not root.exists():
            continue
        for candidate in root.rglob("*.zip"):
            name = candidate.name.lower()
            if "bcmid" in name or "breast" in name:
                return candidate
    return None


# ══════════════════════════════════════════════════════════════════
# DIRECTORY MODE (Kaggle extracted dataset)
# ══════════════════════════════════════════════════════════════════

def scan_dir(data_root: Path) -> dict[str, dict]:
    """
    Scan an extracted directory structure (Kaggle layout).

    Expected structure:
        data_root/
          [BCMID/]
            <patient_id>/
              Mammogram/
                *.png
              Ultrasound/
                *.png
              Report.pdf  (optional)

    Returns dict mapping patient_id -> {
        "us": [Path, ...],
        "mammo": [Path, ...],
        "report": [Path, ...],
        "source": "dir",
        "patient_dir": Path,
    }
    """
    patients: dict[str, dict] = {}

    for directory in sorted(data_root.rglob("*")):
        if not directory.is_dir():
            continue
        child_names = {c.name.lower() for c in directory.iterdir() if c.is_dir()}
        if not any(_is_mammo_folder(n) or _is_us_folder(n) for n in child_names):
            continue

        pid = directory.name
        entry: dict = {
            "us": [],
            "mammo": [],
            "report": [],
            "other": [],
            "source": "dir",
            "patient_dir": directory,
        }

        for child in directory.iterdir():
            if child.is_dir():
                if _is_us_folder(child.name):
                    entry["us"] = sorted([f for f in child.iterdir() if f.is_file() and _is_image(f.name)])
                elif _is_mammo_folder(child.name):
                    entry["mammo"] = sorted([f for f in child.iterdir() if f.is_file() and _is_image(f.name)])
            elif child.is_file():
                if _is_pdf(child.name):
                    entry["report"].append(child)

        patients[pid] = entry

    logger.info("scan_dir: found %d patient directories under %s", len(patients), data_root)
    return patients


# ══════════════════════════════════════════════════════════════════
# ZIP MODE (local or Kaggle with uploaded ZIP)
# ══════════════════════════════════════════════════════════════════

def scan_zip(zip_path: Path) -> dict[str, dict]:
    """
    Scan zip structure and return per-patient inventory.

    Returns dict mapping patient_id -> {
        "us": [zip_member_str, ...],
        "mammo": [zip_member_str, ...],
        "report": [zip_member_str, ...],
        "source": "zip",
    }
    """
    patients: dict[str, dict] = {}

    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            parts = PurePosixPath(info.filename).parts
            if len(parts) < 1 or not parts[0]:
                continue

            pid = parts[0]
            if pid not in patients:
                patients[pid] = {"us": [], "mammo": [], "report": [], "other": [], "source": "zip"}

            if len(parts) == 1:
                continue

            fname = parts[-1]
            if not fname:
                continue

            if len(parts) == 3:
                sub = parts[1]
                if _is_us_folder(sub) and _is_image(fname):
                    patients[pid]["us"].append(info.filename)
                elif _is_mammo_folder(sub) and _is_image(fname):
                    patients[pid]["mammo"].append(info.filename)
                else:
                    patients[pid]["other"].append(info.filename)
            elif len(parts) == 2:
                if _is_pdf(fname):
                    patients[pid]["report"].append(info.filename)
                elif _is_image(fname):
                    patients[pid]["other"].append(info.filename)

    logger.info("scan_zip: found %d patient directories in %s", len(patients), zip_path)
    return patients


# ══════════════════════════════════════════════════════════════════
# UNIFIED SCAN — auto-detects ZIP vs directory
# ══════════════════════════════════════════════════════════════════

def scan_source(zip_path: Path | None = None, data_root: Path | None = None) -> dict[str, dict]:
    """
    Unified scan: uses ZIP if available, otherwise extracted directory.
    Priority: zip_path > data_root
    """
    if zip_path and Path(zip_path).is_file():
        logger.info("Source mode: ZIP")
        return scan_zip(Path(zip_path))
    elif data_root and Path(data_root).is_dir():
        logger.info("Source mode: directory")
        return scan_dir(Path(data_root))
    else:
        raise FileNotFoundError(
            f"No valid data source. zip_path={zip_path}, data_root={data_root}"
        )


# ══════════════════════════════════════════════════════════════════
# IMAGE READING — unified for ZIP and directory
# ══════════════════════════════════════════════════════════════════

def read_image_from_zip(zf: zipfile.ZipFile, member: str) -> Image.Image | None:
    """Read a single image from an open ZipFile."""
    try:
        data = zf.read(member)
    except Exception as exc:
        logger.warning("Cannot read zip member %s: %s", member, exc)
        return None

    if member.lower().endswith(".dcm"):
        try:
            import pydicom
            import numpy as np
            ds = pydicom.dcmread(io.BytesIO(data))
            arr = ds.pixel_array.astype(float)
            if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
                arr = arr.max() - arr
            arr = (arr - arr.min()) / (arr.ptp() + 1e-8) * 255
            return Image.fromarray(arr.astype("uint8"), mode="L")
        except Exception as exc:
            logger.warning("DICOM read error for %s: %s", member, exc)
            return None
    else:
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
            return img
        except Exception as exc:
            logger.warning("Image read error for %s: %s", member, exc)
            return None


def read_image_from_path(path: Path) -> Image.Image | None:
    """Read a single image from a filesystem path."""
    try:
        img = Image.open(path)
        img.load()
        return img
    except Exception as exc:
        logger.warning("Image read error for %s: %s", path, exc)
        return None


def read_image(source: str | Path, zf: zipfile.ZipFile | None = None) -> Image.Image | None:
    """Unified image reader: ZIP member (string + zf) or filesystem path."""
    if zf is not None and isinstance(source, str):
        return read_image_from_zip(zf, source)
    return read_image_from_path(Path(source))


# ══════════════════════════════════════════════════════════════════
# PATIENT ITERATOR
# ══════════════════════════════════════════════════════════════════

def iter_patients(
    zip_path: Path | None,
    patients: dict[str, dict],
    limit: int | None = None,
) -> Generator[tuple[str, dict, zipfile.ZipFile | None], None, None]:
    """
    Yield (patient_id, inventory, open_zipfile_or_None) for each patient.
    Automatically handles ZIP mode (zf open) or directory mode (zf=None).
    """
    pids = list(patients.keys())
    if limit is not None:
        pids = pids[:limit]

    # Detect mode from inventory
    source_mode = patients[pids[0]].get("source", "zip") if pids else "zip"

    if source_mode == "zip" and zip_path and Path(zip_path).is_file():
        with zipfile.ZipFile(zip_path, "r") as zf:
            for pid in pids:
                yield pid, patients[pid], zf
    else:
        for pid in pids:
            yield pid, patients[pid], None
