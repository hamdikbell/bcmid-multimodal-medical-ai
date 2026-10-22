"""Module 2 — Integrity: corruption detection, duplicates, uniform images.
Supports both ZIP mode and directory mode (Kaggle extracted dataset).
"""

from __future__ import annotations

import hashlib
import io
import logging
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

from .config import Config
from .zip_io import read_image, read_image_from_zip

logger = logging.getLogger(__name__)


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_uniform(arr: np.ndarray, threshold: float) -> bool:
    return float(arr.std()) < threshold


def _read_raw_bytes(source: str | Path, zf: zipfile.ZipFile | None) -> bytes | None:
    """Read raw bytes from ZIP member or filesystem path."""
    if zf is not None and isinstance(source, str):
        try:
            return zf.read(source)
        except Exception as exc:
            logger.warning("Cannot read zip member %s: %s", source, exc)
            return None
    else:
        try:
            return Path(source).read_bytes()
        except Exception as exc:
            logger.warning("Cannot read file %s: %s", source, exc)
            return None


def check_integrity(
    cfg: Config,
    patients: dict[str, dict],
) -> dict[str, Any]:
    """
    Check integrity of all images — works for ZIP and directory sources.

    Returns
    -------
    dict with:
      - corrupted_or_invalid: list of {member, reason}
      - uniform_images: list of members
      - duplicate_files: list of lists (groups of identical files by hash)
      - valid_members: set of members that passed all checks
    """
    min_dim = cfg.integrity.min_dimension
    std_thresh = cfg.integrity.std_threshold

    corrupted: list[dict] = []
    uniform: list[str] = []
    hash_map: dict[str, list[str]] = {}
    valid_members: set[str] = set()

    # Collect all image references
    all_images: list[tuple[str | Path, str | Path]] = []  # (source_ref, display_name)
    for inv in patients.values():
        for m in inv.get("us", []) + inv.get("mammo", []):
            display = str(m)
            all_images.append((m, display))

    logger.info("Checking integrity of %d images ...", len(all_images))

    # Determine if we're in ZIP or directory mode
    first_inv = next(iter(patients.values()), {})
    source_mode = first_inv.get("source", "zip")

    if source_mode == "zip":
        zip_path = cfg.zip_path
        if not Path(zip_path).is_file():
            raise FileNotFoundError(f"ZIP not found: {zip_path}")
        _check_zip_mode(all_images, zip_path, corrupted, uniform, hash_map, valid_members, min_dim, std_thresh)
    else:
        _check_dir_mode(all_images, corrupted, uniform, hash_map, valid_members, min_dim, std_thresh)

    duplicates = [members for members in hash_map.values() if len(members) > 1]

    logger.info(
        "Integrity done — valid=%d, corrupted=%d, uniform=%d, duplicate_groups=%d",
        len(valid_members), len(corrupted), len(uniform), len(duplicates),
    )

    return {
        "corrupted_or_invalid": corrupted,
        "corrupted_count": len(corrupted),
        "corrupted_sample": corrupted[:20],
        "uniform_images": uniform,
        "duplicate_files": duplicates,
        "duplicate_count": len(duplicates),
        "valid_members": valid_members,
    }


def _check_zip_mode(all_images, zip_path, corrupted, uniform, hash_map, valid_members, min_dim, std_thresh):
    with zipfile.ZipFile(zip_path, "r") as zf:
        for source, display in tqdm(all_images, desc="Integrity", unit="img"):
            member = str(source)
            data = _read_raw_bytes(member, zf)
            if data is None:
                corrupted.append({"member": display, "reason": "unreadable"})
                continue

            h = _hash_bytes(data)
            hash_map.setdefault(h, []).append(display)

            img = read_image_from_zip(zf, member)
            if img is None:
                corrupted.append({"member": display, "reason": "cannot decode image"})
                continue

            w, h_px = img.size
            if w < min_dim or h_px < min_dim:
                corrupted.append({"member": display, "reason": f"too small: {w}x{h_px}"})
                continue

            arr = np.array(img.convert("L"), dtype=float)
            if float(arr.std()) < std_thresh:
                uniform.append(display)
                corrupted.append({"member": display, "reason": "uniform (std < threshold)"})
                continue

            valid_members.add(display)


def _check_dir_mode(all_images, corrupted, uniform, hash_map, valid_members, min_dim, std_thresh):
    for source, display in tqdm(all_images, desc="Integrity", unit="img"):
        path = Path(source)
        data = _read_raw_bytes(path, None)
        if data is None:
            corrupted.append({"member": display, "reason": "unreadable"})
            continue

        h = _hash_bytes(data)
        hash_map.setdefault(h, []).append(display)

        try:
            img = Image.open(io.BytesIO(data))
            img.load()
        except Exception as exc:
            corrupted.append({"member": display, "reason": f"cannot decode image: {exc}"})
            continue

        w, h_px = img.size
        if w < min_dim or h_px < min_dim:
            corrupted.append({"member": display, "reason": f"too small: {w}x{h_px}"})
            continue

        arr = np.array(img.convert("L"), dtype=float)
        if float(arr.std()) < std_thresh:
            uniform.append(display)
            corrupted.append({"member": display, "reason": "uniform (std < threshold)"})
            continue

        valid_members.add(display)
