"""Profiling plots: BI-RADS distribution, images per patient, modality breakdown, before/after."""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

from .config import Config
from .zip_io import read_image_from_zip

logger = logging.getLogger(__name__)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot saved: %s", path)


def plot_birads_distribution(birads: dict[str, int], out_dir: Path) -> None:
    """Bar chart of BI-RADS class distribution."""
    fig, ax = plt.subplots(figsize=(8, 5))
    classes = list(birads.keys())
    counts = list(birads.values())
    colors = plt.cm.tab10(range(len(classes)))
    ax.bar(classes, counts, color=colors)
    ax.set_title("Distribution des classes BI-RADS")
    ax.set_xlabel("Classe BI-RADS")
    ax.set_ylabel("Nombre de patients")
    for i, v in enumerate(counts):
        ax.text(i, v + 0.3, str(v), ha="center", fontsize=9)
    ax.set_ylim(0, max(counts) * 1.15 if counts else 1)
    _save(fig, out_dir / "birads_distribution.png")


def plot_images_per_patient(profile: dict[str, Any], out_dir: Path) -> None:
    """Histograms of US and mammo image counts per patient."""
    per_patient = profile.get("per_patient", [])
    if not per_patient:
        return

    us_counts = [p["n_us"] for p in per_patient]
    mammo_counts = [p["n_mammo"] for p in per_patient]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, title in [
        (axes[0], us_counts, "Images échographie (US) par patient"),
        (axes[1], mammo_counts, "Images mammographie par patient"),
    ]:
        bins = range(0, max(data) + 2) if data else [0, 1]
        ax.hist(data, bins=bins, color="#4C72B0", edgecolor="white", rwidth=0.85)
        ax.set_title(title)
        ax.set_xlabel("Nombre d'images")
        ax.set_ylabel("Nombre de patients")
        ax.set_xticks(list(bins))

    fig.suptitle("Nombre d'images par patient", fontsize=13)
    _save(fig, out_dir / "images_per_patient.png")


def plot_modality_distribution(modality_counts: dict[str, int], out_dir: Path) -> None:
    """Pie chart of modality status."""
    labels = list(modality_counts.keys())
    sizes = list(modality_counts.values())
    if not any(sizes):
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=plt.cm.Set2(range(len(labels))),
    )
    ax.set_title("Répartition des statuts de modalité")
    _save(fig, out_dir / "modality_distribution.png")


def plot_image_size_histogram(profile: dict[str, Any], zip_path: Path, out_dir: Path) -> None:
    """Sample image dimensions histogram (from first 50 patients)."""
    per_patient = profile.get("per_patient", [])
    widths, heights = [], []

    sample_members: list[str] = []
    for pp in per_patient[:50]:
        pid = pp["patient_id"]
        # We can't easily get members back from profile alone; skip silently
        pass

    # Quick scan from zip
    try:
        from .zip_io import scan_zip
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            sample = [n for n in names if n.lower().endswith(".png")][:200]
            for member in sample:
                try:
                    from PIL import Image
                    import io
                    data = zf.read(member)
                    img = Image.open(io.BytesIO(data))
                    widths.append(img.width)
                    heights.append(img.height)
                except Exception:
                    pass
    except Exception as exc:
        logger.warning("Could not sample image sizes: %s", exc)
        return

    if not widths:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(widths, bins=30, color="#DD8452", edgecolor="white")
    axes[0].set_title("Distribution largeur (pixels)")
    axes[0].set_xlabel("Largeur")
    axes[0].set_ylabel("Nombre d'images")

    axes[1].hist(heights, bins=30, color="#55A868", edgecolor="white")
    axes[1].set_title("Distribution hauteur (pixels)")
    axes[1].set_xlabel("Hauteur")

    fig.suptitle("Tailles des images originales (échantillon 200)", fontsize=13)
    _save(fig, out_dir / "image_size_histogram.png")


def plot_before_after(
    cfg: Config,
    patients: dict[str, dict],
    modalities: dict[str, dict],
    out_dir: Path,
) -> None:
    """
    Show before/after preprocessing for 1 mammo + 1 US patient.
    Reads directly from ZIP without saving full preprocessed image.
    """
    from .preprocess import _to_gray_array, preprocess_mammo, preprocess_us

    zip_path = cfg.zip_path

    # Pick first complete patients
    mammo_pid = us_pid = None
    for pid, info in modalities.items():
        if info["valid_mammo_members"] and mammo_pid is None:
            mammo_pid = pid
        if info["valid_us_members"] and us_pid is None:
            us_pid = pid
        if mammo_pid and us_pid:
            break

    with zipfile.ZipFile(zip_path, "r") as zf:
        for pid, member_key, title, preprocess_fn in [
            (mammo_pid, "valid_mammo_members", "Mammographie", preprocess_mammo),
            (us_pid, "valid_us_members", "Échographie", preprocess_us),
        ]:
            if pid is None:
                continue
            members = modalities[pid][member_key]
            if not members:
                continue

            img = read_image_from_zip(zf, members[0])
            if img is None:
                continue

            arr_before = _to_gray_array(img)
            arr_after = preprocess_fn(arr_before.copy(), cfg)

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(arr_before, cmap="gray")
            axes[0].set_title(f"{title} — Original")
            axes[0].axis("off")

            axes[1].imshow(arr_after, cmap="gray")
            axes[1].set_title(f"{title} — Après prétraitement (512×512)")
            axes[1].axis("off")

            fig.suptitle(f"Avant / Après — {title}", fontsize=13)
            fname = "before_after_mammo.png" if "mammo" in member_key else "before_after_us.png"
            _save(fig, out_dir / fname)


def generate_all_plots(
    cfg: Config,
    profile: dict[str, Any],
    modality_counts: dict[str, int],
    patients: dict[str, dict],
    modalities: dict[str, dict],
) -> None:
    """Generate and save all profiling plots."""
    out_dir = cfg.plots_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_birads_distribution(profile["birads_distribution"], out_dir)
    plot_images_per_patient(profile, out_dir)
    plot_modality_distribution(modality_counts, out_dir)
    plot_image_size_histogram(profile, cfg.zip_path, out_dir)
    plot_before_after(cfg, patients, modalities, out_dir)
    logger.info("All plots saved to %s", out_dir)
