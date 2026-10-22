"""Module 4 — Preprocessing: normalization, denoising, CLAHE, resize.
Supports both ZIP mode and directory mode (Kaggle extracted dataset).
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

from .config import Config
from .zip_io import read_image

logger = logging.getLogger(__name__)


def _to_gray_array(img: Image.Image) -> np.ndarray:
    """Convert PIL image to grayscale float32 array [0, 1]."""
    gray = img.convert("L")
    return np.array(gray, dtype=np.float32) / 255.0


def _normalize_01(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def _median_denoise(arr: np.ndarray, ksize: int) -> np.ndarray:
    """Apply median filter (for ultrasound speckle noise)."""
    try:
        import cv2  # type: ignore
        u8 = (arr * 255).clip(0, 255).astype(np.uint8)
        filtered = cv2.medianBlur(u8, ksize)
        return filtered.astype(np.float32) / 255.0
    except ImportError:
        from scipy.ndimage import median_filter  # type: ignore
        return median_filter(arr, size=ksize)


def _nlm_denoise(arr: np.ndarray, h: int) -> np.ndarray:
    """Apply Non-Local Means denoising (for mammography)."""
    try:
        import cv2  # type: ignore
        u8 = (arr * 255).clip(0, 255).astype(np.uint8)
        denoised = cv2.fastNlMeansDenoising(u8, h=h, templateWindowSize=7, searchWindowSize=21)
        return denoised.astype(np.float32) / 255.0
    except ImportError:
        logger.warning("cv2 not available, skipping NLM denoising")
        return arr


def _apply_clahe(arr: np.ndarray, clip_limit: float, tile_grid: int) -> np.ndarray:
    """Apply CLAHE to enhance local contrast (for mammography)."""
    try:
        import cv2  # type: ignore
        u8 = (arr * 255).clip(0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
        enhanced = clahe.apply(u8)
        return enhanced.astype(np.float32) / 255.0
    except ImportError:
        logger.warning("cv2 not available, skipping CLAHE")
        return arr


def _resize(arr: np.ndarray, target: int) -> np.ndarray:
    """Resize to target×target using INTER_AREA (best for downscaling)."""
    try:
        import cv2  # type: ignore
        u8 = (arr * 255).clip(0, 255).astype(np.uint8)
        return cv2.resize(u8, (target, target), interpolation=cv2.INTER_AREA)
    except ImportError:
        pil = Image.fromarray((arr * 255).astype(np.uint8))
        pil = pil.resize((target, target), Image.LANCZOS)
        return np.array(pil)


def preprocess_us(arr: np.ndarray, cfg: Config) -> np.ndarray:
    """Full preprocessing pipeline for ultrasound images."""
    arr = _normalize_01(arr)
    arr = _median_denoise(arr, cfg.preprocessing.us_median_ksize)
    arr = _normalize_01(arr)
    return _resize(arr, cfg.preprocessing.target_size)


def preprocess_mammo(arr: np.ndarray, cfg: Config) -> np.ndarray:
    """Full preprocessing pipeline for mammography images."""
    arr = _normalize_01(arr)
    arr = _nlm_denoise(arr, cfg.preprocessing.nlm_h)
    arr = _apply_clahe(arr, cfg.preprocessing.mammo_clahe_clip_limit, cfg.preprocessing.mammo_clahe_tile_grid)
    arr = _normalize_01(arr)
    return _resize(arr, cfg.preprocessing.target_size)


def process_patient(
    pid: str,
    pseudo_id: str,
    modality_info: dict,
    cfg: Config,
    zf: zipfile.ZipFile | None,
) -> dict[str, Any]:
    """
    Preprocess all images for one patient and save to validated/.
    Works for both ZIP (zf provided) and directory (zf=None) modes.
    Skips already-processed images (idempotent).
    """
    mammo_dir = cfg.mammo_dir
    us_dir = cfg.us_dir
    mammo_dir.mkdir(parents=True, exist_ok=True)
    us_dir.mkdir(parents=True, exist_ok=True)

    saved_us: list[str] = []
    saved_mammo: list[str] = []

    # --- Ultrasound ---
    for idx, member in enumerate(modality_info.get("valid_us_members", [])):
        out_name = f"{pseudo_id}_us_{idx}.png"
        out_path = us_dir / out_name

        if out_path.exists():
            saved_us.append(str(out_path.relative_to(cfg.validated_dir)))
            continue

        img = read_image(member, zf)
        if img is None:
            logger.warning("Skip US image (unreadable): %s", member)
            continue

        arr = _to_gray_array(img)
        processed = preprocess_us(arr, cfg)
        Image.fromarray(processed, mode="L").save(out_path, format="PNG")
        saved_us.append(str(out_path.relative_to(cfg.validated_dir)))

    # --- Mammography ---
    for idx, member in enumerate(modality_info.get("valid_mammo_members", [])):
        out_name = f"{pseudo_id}_mammo_{idx}.png"
        out_path = mammo_dir / out_name

        if out_path.exists():
            saved_mammo.append(str(out_path.relative_to(cfg.validated_dir)))
            continue

        img = read_image(member, zf)
        if img is None:
            logger.warning("Skip mammo image (unreadable): %s", member)
            continue

        arr = _to_gray_array(img)
        processed = preprocess_mammo(arr, cfg)
        Image.fromarray(processed, mode="L").save(out_path, format="PNG")
        saved_mammo.append(str(out_path.relative_to(cfg.validated_dir)))

    return {
        "pseudo_id": pseudo_id,
        "us_paths": saved_us,
        "mammo_paths": saved_mammo,
        "n_us": len(saved_us),
        "n_mammo": len(saved_mammo),
    }
