"""
CLI entry point for the Data & Quality Agent.

Usage
-----
    python -m dq_agent run    --config config.yaml [--limit N]
    python -m dq_agent profile --config config.yaml [--limit N]
    python -m dq_agent splits  --config config.yaml
    python -m dq_agent report  --config config.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import zipfile
from pathlib import Path
from typing import Any

from .config import Config, load_config, setup_logging
from .splits import DEFAULT_LABEL_GROUPS, get_label_group

logger = logging.getLogger(__name__)


# ──────────────────────────── helpers ────────────────────────────────────────

def _load_progress(cfg: Config) -> dict[str, Any]:
    if cfg.progress_file.exists():
        with open(cfg.progress_file, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_progress(cfg: Config, progress: dict[str, Any]) -> None:
    cfg.progress_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.progress_file, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)


def _write_manifests(
    cfg: Config,
    processed: list[dict],
    modalities: dict[str, dict],
    labels: dict[str, str],
    extra_data: dict[str, dict],
    mapping: dict[str, str],
    label_groups: dict[str, str],
) -> None:
    """Write manifest.json and manifest.csv — no real IDs anywhere."""
    out_dir = cfg.validated_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    reverse_map = {v: k for k, v in mapping.items()}
    records = []

    for item in processed:
        pseudo = item["pseudo_id"]
        real_id = reverse_map.get(pseudo, "")
        mod_info = modalities.get(real_id, {})
        birads = labels.get(real_id, "UNKNOWN")
        lbl_grp = get_label_group(birads, label_groups)
        ex = extra_data.get(real_id, {})

        records.append({
            "pseudo_id": pseudo,
            "n_us": item["n_us"],
            "n_mammo": item["n_mammo"],
            "us_paths": item["us_paths"],
            "mammo_paths": item["mammo_paths"],
            "modality_status": mod_info.get("status", "unknown"),
            "has_report": mod_info.get("has_report", False),
            "birads": birads,
            "label_group": lbl_grp,
            # Extra TSV columns (no real ID)
            "binary_label": ex.get("Binary Label", ""),
            "pathology_label": ex.get("Pathology Label", ""),
        })

    # JSON
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    # CSV
    fieldnames = [
        "pseudo_id", "n_us", "n_mammo", "us_paths", "mammo_paths",
        "modality_status", "has_report", "birads", "label_group",
        "binary_label", "pathology_label",
    ]
    with open(out_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            row = dict(r)
            row["us_paths"] = ";".join(r["us_paths"])
            row["mammo_paths"] = ";".join(r["mammo_paths"])
            writer.writerow(row)

    logger.info("Manifests written → %s", out_dir)


# ──────────────────────────── commands ───────────────────────────────────────

def cmd_run(cfg: Config, limit: int | None) -> None:
    """Full pipeline."""
    from tqdm import tqdm

    from .anonymize import build_mapping, save_private_mapping
    from .integrity import check_integrity
    from .modality import classify_patients, filter_for_training, modality_summary
    from .plots import generate_all_plots
    from .preprocess import process_patient
    from .profiling import profile_dataset, save_profile
    from .report import build_report, save_report
    from .reports import process_reports
    from .splits import make_splits, save_splits
    from .validation import validate_labels
    from .zip_io import scan_source, auto_find_dataset, auto_find_zip, iter_patients

    # ── Directories
    for d in [cfg.work_dir, cfg.validated_dir, cfg.mammo_dir,
              cfg.us_dir, cfg.reports_dir, cfg.plots_dir]:
        d.mkdir(parents=True, exist_ok=True)

    progress = _load_progress(cfg)

    # ── Step 1: Auto-detect dataset (ZIP or extracted directory)
    if "patients" not in progress:
        kaggle_input = Path("/kaggle/input")
        search_root = kaggle_input if kaggle_input.exists() else Path(".")

        # Priority 1: zip_path from config
        zip_path = cfg.zip_path if cfg.zip_path and Path(cfg.zip_path).is_file() else None

        # Priority 2: auto-find ZIP in /kaggle/input (like notebook1 finds TSV)
        if zip_path is None:
            zip_path = auto_find_zip(search_root)

        # Priority 3: auto-find extracted directory
        data_root = None
        tsv_auto = None
        if zip_path is None:
            data_root, tsv_auto = auto_find_dataset(search_root)
            if data_root is None:
                raise FileNotFoundError(
                    "Could not find BCMID dataset. "
                    "Import BCMID.zip or the BCMID folder into Kaggle, "
                    "or set zip_path in config.yaml."
                )

        # Use auto-detected TSV if config path not valid
        if tsv_auto and (not cfg.tsv_path or not Path(cfg.tsv_path).exists()):
            cfg.tsv_path = tsv_auto
            logger.info("Auto-detected TSV: %s", tsv_auto)

        if zip_path:
            cfg.zip_path = zip_path
            logger.info("Source: ZIP -> %s", zip_path)
        else:
            logger.info("Source: directory -> %s", data_root)

        all_patients_full = scan_source(zip_path=zip_path, data_root=data_root)
        n_total_zip = len(all_patients_full)
        patients = all_patients_full
        if limit:
            patients = dict(list(patients.items())[:limit])
            logger.info("--limit %d: keeping %d patients (total: %d)", limit, len(patients), n_total_zip)
        progress["patients"] = patients
        progress["n_total_zip"] = n_total_zip
        _save_progress(cfg, progress)
    else:
        patients = progress["patients"]
        n_total_zip = progress.get("n_total_zip", len(patients))
        if limit:
            patients = dict(list(patients.items())[:limit])
        logger.info("Loaded %d patients from progress cache (total: %d)", len(patients), n_total_zip)


    # ── Step 2: Validate labels
    logger.info("Validating labels …")
    val_result = validate_labels(cfg, patients)
    labels = val_result["labels"]
    extra_data = val_result.get("extra_data", {})
    logger.info(
        "TSV columns: %s | ID col: '%s' | Label col: '%s' | Extra: %s",
        val_result["tsv_columns"], val_result["id_column"],
        val_result["label_column"], val_result.get("extra_columns", []),
    )

    # ── Step 3: Integrity
    logger.info("Checking image integrity …")
    integrity_result = check_integrity(cfg, patients)

    # ── Step 4: Modality classification (images only, not report)
    logger.info("Classifying modalities …")
    modalities = classify_patients(cfg, patients, integrity_result["valid_members"])
    multimodal_ids, partial_ids = filter_for_training(modalities)
    mod_counts = modality_summary(modalities)

    # ── Step 5: Profiling
    logger.info("Profiling dataset ...")
    profile = profile_dataset(cfg, patients, labels, n_total_zip_patients=n_total_zip)
    save_profile(profile, cfg.validated_dir)

    # ── Step 6: Anonymization
    logger.info("Building anonymization mapping …")
    all_pids = list(patients.keys())
    mapping = build_mapping(all_pids)
    save_private_mapping(mapping, cfg.mapping_file)

    # ── Step 7: Preprocess images (idempotent, ZIP + dir mode)
    logger.info("Preprocessing images …")
    processed_results: list[dict] = []

    _zip_for_iter = cfg.zip_path if (cfg.zip_path and Path(cfg.zip_path).is_file()) else None

    for pid, inv, zf in tqdm(
        iter_patients(_zip_for_iter, patients),
        desc="Preprocess", unit="patient", total=len(all_pids)
    ):
        pseudo = mapping[pid]
        mod_info = modalities[pid]

        first_us = cfg.us_dir / f"{pseudo}_us_0.png"
        first_mammo = cfg.mammo_dir / f"{pseudo}_mammo_0.png"
        has_us_done = (mod_info["n_valid_us"] == 0) or first_us.exists()
        has_mammo_done = (mod_info["n_valid_mammo"] == 0) or first_mammo.exists()

        if has_us_done and has_mammo_done:
            us_paths = sorted(str(p.relative_to(cfg.validated_dir))
                              for p in cfg.us_dir.glob(f"{pseudo}_us_*.png"))
            mammo_paths = sorted(str(p.relative_to(cfg.validated_dir))
                                 for p in cfg.mammo_dir.glob(f"{pseudo}_mammo_*.png"))
            processed_results.append({
                "pseudo_id": pseudo,
                "us_paths": us_paths,
                "mammo_paths": mammo_paths,
                "n_us": len(us_paths),
                "n_mammo": len(mammo_paths),
            })
            continue

        result = process_patient(pid, pseudo, mod_info, cfg, zf)
        processed_results.append(result)


    logger.info("Extracting PDF reports …")
    reports_stats = process_reports(
        patients=patients,
        mapping=mapping,
        zip_path=cfg.zip_path,
        reports_dir=cfg.reports_dir,
    )
    logger.info(
        "Label leakage risk: %.1f%% of reports mention BI-RADS",
        reports_stats["label_leakage_risk_pct"],
    )

    # ── Step 9: Write manifests (with label_group, no real IDs)
    logger.info("Writing manifests …")
    _write_manifests(cfg, processed_results, modalities, labels,
                     extra_data, mapping, DEFAULT_LABEL_GROUPS)

    # ── Step 10: Splits (stratified by label_group)
    pseudo_multimodal = [mapping[pid] for pid in multimodal_ids if pid in mapping]
    pseudo_labels = {mapping[pid]: labels.get(pid, "UNKNOWN")
                     for pid in all_pids if pid in mapping}
    splits = make_splits(cfg, pseudo_multimodal, pseudo_labels, DEFAULT_LABEL_GROUPS)
    save_splits(splits, cfg.validated_dir)

    # ── Step 11: Plots
    logger.info("Generating plots …")
    try:
        generate_all_plots(cfg, profile, mod_counts, patients, modalities)
    except Exception as exc:
        logger.warning("Plot generation failed (non-fatal): %s", exc)

    # ── Step 12: Quality report
    logger.info("Building quality report …")
    preprocessing_params = {
        "target_size": cfg.preprocessing.target_size,
        "mammo_clahe_clip_limit": cfg.preprocessing.mammo_clahe_clip_limit,
        "mammo_clahe_tile_grid": cfg.preprocessing.mammo_clahe_tile_grid,
        "us_median_ksize": cfg.preprocessing.us_median_ksize,
        "nlm_h": cfg.preprocessing.nlm_h,
        "bit_depth": cfg.preprocessing.bit_depth,
    }
    report = build_report(
        profile=profile,
        integrity_result=integrity_result,
        modality_counts=mod_counts,
        validation_result=val_result,
        preprocessing_params=preprocessing_params,
        n_multimodal_ready=len(multimodal_ids),
        reports_stats=reports_stats,
    )
    save_report(report, cfg.validated_dir)

    logger.info("✅ Pipeline complete! Outputs in: %s", cfg.validated_dir)
    _print_summary(report, splits, reports_stats)


def _print_summary(report: dict, splits: dict, reports_stats: dict) -> None:
    sep = "=" * 62
    print("\n" + sep)
    print("  DATA & QUALITY AGENT -- RESUME FINAL")
    print(sep)
    print(f"  Dossiers patients (ZIP)      : {report['n_patient_folders']}")
    print(f"  Patients annotes (TSV)       : {report['n_patients_in_annotations']}")
    print(f"  Ecart detecte                : {report['ecart_patients']}")
    print(f"  Prets (multimodal complet)   : {report['patients_ready_multimodal']}")
    print(f"  Sans rapport PDF             : {report['patients_without_report_pdf']}")
    print(f"  Images corrompues            : {report['corrupted_or_invalid_count']}")
    print(f"  Groupes de doublons          : {report['duplicate_files_count']}")
    print(f"  Sans label BI-RADS           : {len(report['patients_without_label'])}")
    print("-" * 62)
    print(f"  BI-RADS distribution         : {report['birads_distribution']}")
    print("-" * 62)
    print(f"  Rapports PDF extraits (OK)   : {reports_stats.get('reports_extracted_ok', 0)}")
    print(f"  Rapports scannes (illisibles): {reports_stats.get('reports_scanned_unreadable', 0)}")
    print(f"  Risque label leakage         : {reports_stats.get('label_leakage_risk_pct', 0):.1f}%")
    print("-" * 62)
    print(f"  Split train/val/test         : {splits['n_train']}/{splits['n_val']}/{splits['n_test']}")
    if report.get("labels_without_folder"):
        print(f"  [!] Labels sans dossier      : {len(report['labels_without_folder'])}")
    print(sep + "\n")



def cmd_profile(cfg: Config, limit: int | None) -> None:
    from .profiling import profile_dataset, save_profile
    from .validation import validate_labels
    from .zip_io import scan_source, auto_find_dataset, auto_find_zip

    kaggle_input = Path("/kaggle/input")
    search_root = kaggle_input if kaggle_input.exists() else Path(".")
    zip_path = cfg.zip_path if cfg.zip_path and Path(cfg.zip_path).is_file() else auto_find_zip(search_root)
    data_root, tsv_auto = (None, None) if zip_path else auto_find_dataset(search_root)
    if tsv_auto and (not cfg.tsv_path or not Path(cfg.tsv_path).exists()):
        cfg.tsv_path = tsv_auto
    if zip_path:
        cfg.zip_path = zip_path

    patients = scan_source(zip_path=zip_path, data_root=data_root)
    if limit:
        patients = dict(list(patients.items())[:limit])

    val_result = validate_labels(cfg, patients)
    profile = profile_dataset(cfg, patients, val_result["labels"])
    save_profile(profile, cfg.validated_dir)

    print(f"\n{'='*50}")
    print(f"TSV columns     : {val_result['tsv_columns']}")
    print(f"ID column       : {val_result['id_column']}")
    print(f"Label column    : {val_result['label_column']}")
    print(f"Extra columns   : {val_result.get('extra_columns', [])}")
    print(f"Image extensions: {profile['image_extensions']}")
    print(f"Patients found  : {len(patients)}")
    print(f"Patients in TSV : {val_result.get('n_patients_in_annotations', 'N/A')}")
    print(f"BI-RADS dist.   : {profile['birads_distribution']}")
    print(f"{'='*50}\n")



def cmd_splits(cfg: Config) -> None:
    from .splits import make_splits, save_splits, DEFAULT_LABEL_GROUPS

    manifest_path = cfg.validated_dir / "manifest.json"
    if not manifest_path.exists():
        print("ERROR: manifest.json not found — run `run` first.")
        sys.exit(1)

    with open(manifest_path, encoding="utf-8") as f:
        records = json.load(f)

    multimodal = [r["pseudo_id"] for r in records if r["modality_status"] == "complete"]
    labels = {r["pseudo_id"]: r["birads"] for r in records}

    splits = make_splits(cfg, multimodal, labels, DEFAULT_LABEL_GROUPS)
    save_splits(splits, cfg.validated_dir)
    print(f"Splits: train={splits['n_train']}, val={splits['n_val']}, test={splits['n_test']}")


def cmd_report(cfg: Config) -> None:
    print("Use `run` to regenerate the full report with all pipeline state.")


# ──────────────────────────── main ───────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m dq_agent",
        description="Data & Quality Agent — BCMID Multimodal Medical AI",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    p_run = subparsers.add_parser("run", help="Run the full pipeline")
    p_run.add_argument("--config", default="config.yaml")
    p_run.add_argument("--limit", type=int, default=None)

    p_profile = subparsers.add_parser("profile", help="Profiling only")
    p_profile.add_argument("--config", default="config.yaml")
    p_profile.add_argument("--limit", type=int, default=None)

    p_splits = subparsers.add_parser("splits", help="Regenerate splits")
    p_splits.add_argument("--config", default="config.yaml")

    p_report = subparsers.add_parser("report", help="Regenerate report")
    p_report.add_argument("--config", default="config.yaml")

    args = parser.parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg)

    if args.command == "run":
        cmd_run(cfg, args.limit)
    elif args.command == "profile":
        cmd_profile(cfg, args.limit)
    elif args.command == "splits":
        cmd_splits(cfg)
    elif args.command == "report":
        cmd_report(cfg)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
