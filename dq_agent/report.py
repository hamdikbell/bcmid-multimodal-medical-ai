"""Module 7 — Report: generate quality_report.json and quality_report.md."""

from __future__ import annotations

import importlib.metadata
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_LIBRARY_VERSIONS = ["Pillow", "numpy", "pandas", "scikit-learn", "tqdm", "pyyaml", "pydicom"]


def _get_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for lib in _LIBRARY_VERSIONS:
        try:
            versions[lib] = importlib.metadata.version(lib)
        except Exception:
            versions[lib] = "N/A"
    return versions


def build_report(
    profile: dict[str, Any],
    integrity_result: dict[str, Any],
    modality_counts: dict[str, int],
    validation_result: dict[str, Any],
    preprocessing_params: dict[str, Any],
    n_multimodal_ready: int,
    reports_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the quality_report.json payload."""
    report: dict[str, Any] = {
        "date_generated": datetime.now(timezone.utc).isoformat(),
        "library_versions": _get_versions(),
        # Dataset counts
        "n_patient_folders": profile["n_patient_folders"],
        "n_patients_in_annotations": validation_result.get("n_patients_in_annotations", 0),
        "ecart_patients": profile["ecart_patients"],
        "patients_only_in_folders": profile["patients_only_in_folders"],
        "patients_only_in_annotations": profile["patients_only_in_annotations"],
        # Modality
        "modality_status": modality_counts,
        "patients_ready_multimodal": n_multimodal_ready,
        # BI-RADS
        "birads_distribution": profile["birads_distribution"],
        # Integrity
        "corrupted_or_invalid_count": integrity_result["corrupted_count"],
        "corrupted_or_invalid_sample": integrity_result["corrupted_sample"],
        "duplicate_files_count": integrity_result["duplicate_count"],
        "duplicate_groups": integrity_result["duplicate_files"],
        # Annotations quality
        "patients_without_label": validation_result["patients_without_label"],
        "labels_without_folder": validation_result["labels_without_folder"],
        "duplicate_ids_in_annotations": validation_result["duplicate_ids_in_annotations"],
        "malformed_labels": validation_result["malformed_labels"],
        "tsv_columns_detected": validation_result.get("tsv_columns", []),
        "id_column_detected": validation_result.get("id_column"),
        "label_column_detected": validation_result.get("label_column"),
        "extra_tsv_columns": validation_result.get("extra_columns", []),
        # Dataset stats
        "image_extensions": profile["image_extensions"],
        "us_per_patient": profile.get("us_per_patient", {}),
        "mammo_per_patient": profile.get("mammo_per_patient", {}),
        "patients_without_report_pdf": profile.get("images_without_report", 0),
        # Reports extraction
        "reports_stats": reports_stats or {},
        "label_leakage_risk_pct": (reports_stats or {}).get("label_leakage_risk_pct", None),
        # Preprocessing
        "parametres_preprocessing": preprocessing_params,
        # Validation issues
        "all_issues": validation_result.get("issues", []),
    }
    return report


def save_report(report: dict[str, Any], out_dir: Path) -> None:
    """Save JSON and Markdown reports."""
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "quality_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("JSON report saved: %s", json_path)

    md_path = out_dir / "quality_report.md"
    _write_markdown(report, md_path)
    logger.info("Markdown report saved: %s", md_path)


def _fmt_list(lst: list, max_items: int = 10) -> str:
    if not lst:
        return "_aucun_"
    shown = [str(x) for x in lst[:max_items]]
    rest = len(lst) - max_items
    result = ", ".join(shown)
    if rest > 0:
        result += f" _… et {rest} autres_"
    return result


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines: list[str] = []

    def h(level: int, text: str) -> None:
        lines.append("\n" + "#" * level + " " + text + "\n")

    def row(*cells: Any) -> str:
        return "| " + " | ".join(str(c) for c in cells) + " |"

    h(1, "Rapport de Qualité — Data & Quality Agent (BCMID)")
    lines.append(f"**Généré le :** {report['date_generated']}\n")

    h(2, "1. Résumé Dataset")
    lines.append(row("Métrique", "Valeur"))
    lines.append(row("---", "---"))
    lines.append(row("Dossiers patients (ZIP)", report["n_patient_folders"]))
    lines.append(row("Patients dans les annotations (TSV)", report["n_patients_in_annotations"]))
    lines.append(row("**Écart détecté**", f"**{report['ecart_patients']}**"))
    lines.append(row("Patients prêts (multimodal complet)", report["patients_ready_multimodal"]))
    lines.append(row("Patients sans rapport PDF", report["patients_without_report_pdf"]))

    h(2, "2. Distribution BI-RADS")
    lines.append(row("Classe BI-RADS", "Nombre de patients"))
    lines.append(row("---", "---"))
    for k, v in report["birads_distribution"].items():
        lines.append(row(k, v))

    h(2, "3. Statuts de Modalité")
    lines.append(row("Statut", "Nombre"))
    lines.append(row("---", "---"))
    for k, v in report["modality_status"].items():
        lines.append(row(k, v))

    h(2, "4. Intégrité des Images")
    lines.append(row("Indicateur", "Valeur"))
    lines.append(row("---", "---"))
    lines.append(row("Images corrompues / invalides", report["corrupted_or_invalid_count"]))
    lines.append(row("Groupes de doublons exacts", report["duplicate_files_count"]))

    if report["corrupted_or_invalid_sample"]:
        h(3, "Exemples d'images invalides (max 20)")
        lines.append(row("Fichier", "Raison"))
        lines.append(row("---", "---"))
        for item in report["corrupted_or_invalid_sample"]:
            lines.append(row(item.get("member", "?"), item.get("reason", "?")))

    h(2, "5. Annotations BI-RADS")
    lines.append(row("Indicateur", "Valeur"))
    lines.append(row("---", "---"))
    lines.append(row("Colonnes TSV détectées", ", ".join(report["tsv_columns_detected"])))
    lines.append(row("Colonne ID détectée", report["id_column_detected"] or "N/A"))
    lines.append(row("Colonne label détectée", report["label_column_detected"] or "N/A"))
    lines.append(row("Patients sans label", len(report["patients_without_label"])))
    lines.append(row("Labels sans dossier", len(report["labels_without_folder"])))
    lines.append(row("IDs dupliqués dans TSV", len(report["duplicate_ids_in_annotations"])))
    lines.append(row("Labels mal formés", len(report["malformed_labels"])))

    if report["patients_without_label"]:
        lines.append(f"\n**Patients sans label :** {_fmt_list(report['patients_without_label'])}\n")
    if report["labels_without_folder"]:
        lines.append(f"\n**Labels sans dossier :** {_fmt_list(report['labels_without_folder'])}\n")

    h(2, "6. Statistiques Images")
    for modality, key in [("Échographie (US)", "us_per_patient"), ("Mammographie", "mammo_per_patient")]:
        stats = report.get(key, {})
        if stats:
            lines.append(f"\n**{modality}** — min={stats.get('min')}, max={stats.get('max')}, "
                         f"moy={stats.get('mean')}, écart-type={stats.get('std')}\n")

    h(2, "7. Paramètres de Prétraitement")
    lines.append(row("Paramètre", "Valeur"))
    lines.append(row("---", "---"))
    for k, v in report["parametres_preprocessing"].items():
        lines.append(row(k, v))

    h(2, "8. Versions des Librairies")
    lines.append(row("Librairie", "Version"))
    lines.append(row("---", "---"))
    for lib, ver in report["library_versions"].items():
        lines.append(row(lib, ver))

    path.write_text("\n".join(lines), encoding="utf-8")
