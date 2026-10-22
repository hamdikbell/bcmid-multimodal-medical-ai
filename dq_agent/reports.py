"""
Module reports — PDF text extraction, PII removal, label leakage detection.

For each patient:
  1. Extract text from PDF report via pdfplumber.
  2. Strip identifying information (names, dates, numeric IDs, real patient ID).
  3. Save as reports/<pseudo_id>.txt in validated/.
  4. Detect whether the text mentions BI-RADS (label leakage risk for Vision agent).
"""

from __future__ import annotations

import io
import logging
import re
import zipfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── PII removal patterns ──────────────────────────────────────────────────────
_PII_PATTERNS: list[tuple[str, str]] = [
    # Dates: DD/MM/YYYY, DD-MM-YYYY, MM/DD/YYYY, YYYY-MM-DD, "le 12 janvier 2021"
    (r"\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b", "<DATE>"),
    (r"\b\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}\b", "<DATE>"),
    (r"\b\d{1,2}\s+(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{4}\b",
     "<DATE>", ),
    # Phone numbers
    (r"\b(?:\+?\d[\s\-]?){8,15}\b", "<PHONE>"),
    # Emails
    (r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", "<EMAIL>"),
    # Social security / ID numbers (long digit sequences 6-20 chars)
    (r"\b\d{6,20}\b", "<ID_NUM>"),
    # Names heuristic: two or more consecutive Capitalized words (Title Case)
    (r"\b([A-ZÁÉÈÊÀÙÛÔÎÂ][a-záéèêàùûôîâ]+)(\s+[A-ZÁÉÈÊÀÙÛÔÎÂ][a-záéèêàùûôîâ]+){1,3}\b",
     "<NOM>"),
]

# ── BI-RADS mention detection ─────────────────────────────────────────────────
_BIRADS_MENTION = re.compile(
    r"bi[\s\-]*rads|birads|acr\s*\d|cat[eé]gorie\s*\d|classification\s*\d",
    re.IGNORECASE,
)


def _strip_pii(text: str, real_id: str) -> str:
    """Remove PII from extracted text. Also removes the real patient ID."""
    # Remove real patient ID first (most specific)
    text = re.sub(re.escape(real_id), "<PATIENT_ID>", text, flags=re.IGNORECASE)
    # Remove year suffix pattern (NNNNyNN)
    text = re.sub(r"\b\d+y\d+\b", "<PATIENT_ID>", text)

    for pattern_args in _PII_PATTERNS:
        if len(pattern_args) == 3:
            pattern, replacement, *_ = pattern_args
        else:
            pattern, replacement = pattern_args
        try:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE | re.UNICODE)
        except re.error:
            pass
    return text


def extract_pdf_text(pdf_bytes: bytes) -> tuple[str, str]:
    """
    Extract text from PDF bytes using pdfplumber.

    Returns
    -------
    (text, status)  where status is 'ok' | 'scanned' | 'empty' | 'error'
    """
    try:
        import pdfplumber  # type: ignore
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages_text = []
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    pages_text.append(t)
            full_text = "\n".join(pages_text).strip()
    except Exception as exc:
        logger.warning("pdfplumber error: %s", exc)
        return "", "error"

    if not full_text:
        return "", "scanned"   # PDF with no extractable text (image-based)
    return full_text, "ok"


def has_birads_mention(text: str) -> bool:
    """Return True if text contains a BI-RADS classification mention."""
    return bool(_BIRADS_MENTION.search(text))


def process_reports(
    patients: dict[str, dict],
    mapping: dict[str, str],
    zip_path: Path,
    reports_dir: Path,
) -> dict[str, Any]:
    """
    Extract, anonymize, and save all patient PDF reports.

    Parameters
    ----------
    patients : zip inventory from scan_zip()
    mapping  : real_id -> pseudo_id
    zip_path : path to the BCMID ZIP
    reports_dir : output directory for .txt files

    Returns
    -------
    dict with summary stats for quality_report
    """
    from tqdm import tqdm

    reports_dir.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_scanned = 0
    n_error = 0
    n_no_pdf = 0
    n_birads_mention = 0
    n_total = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        for pid, inv in tqdm(patients.items(), desc="Reports", unit="patient"):
            pseudo = mapping.get(pid, pid)
            out_path = reports_dir / f"{pseudo}.txt"

            if not inv["report"]:
                n_no_pdf += 1
                # Write a placeholder
                out_path.write_text("[NO_REPORT_AVAILABLE]\n", encoding="utf-8")
                continue

            # Idempotent
            if out_path.exists():
                text = out_path.read_text(encoding="utf-8")
                if has_birads_mention(text):
                    n_birads_mention += 1
                n_total += 1
                n_ok += 1
                continue

            member = inv["report"][0]
            try:
                pdf_bytes = zf.read(member)
            except Exception as exc:
                logger.warning("Cannot read PDF %s: %s", member, exc)
                n_error += 1
                out_path.write_text("[PDF_READ_ERROR]\n", encoding="utf-8")
                continue

            text, status = extract_pdf_text(pdf_bytes)
            n_total += 1

            if status == "ok":
                n_ok += 1
                # Strip PII
                clean_text = _strip_pii(text, real_id=pid)
                if has_birads_mention(clean_text):
                    n_birads_mention += 1
                out_path.write_text(clean_text, encoding="utf-8")
            elif status == "scanned":
                n_scanned += 1
                out_path.write_text("[SCANNED_PDF_NO_TEXT]\n", encoding="utf-8")
            else:
                n_error += 1
                out_path.write_text("[PDF_ERROR]\n", encoding="utf-8")

    leakage_pct = round(n_birads_mention / n_total * 100, 1) if n_total > 0 else 0.0

    logger.info(
        "Reports — ok=%d, scanned=%d, error=%d, no_pdf=%d | BI-RADS mention: %d/%d (%.1f%%)",
        n_ok, n_scanned, n_error, n_no_pdf, n_birads_mention, n_total, leakage_pct,
    )

    return {
        "reports_extracted_ok": n_ok,
        "reports_scanned_unreadable": n_scanned,
        "reports_error": n_error,
        "reports_no_pdf": n_no_pdf,
        "reports_total_with_pdf": n_total,
        "birads_mention_in_reports": n_birads_mention,
        "label_leakage_risk_pct": leakage_pct,
    }
