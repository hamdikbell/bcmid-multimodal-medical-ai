"""
src/ingestion/ingestion_agent.py — Bloc 1: Data Ingestion Agent
Receives a patient folder path, verifies file presence, extensions,
image readability, and metadata consistency. Produces a structured
patient dossier inventory and flags missing/invalid items.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional


VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
REQUIRED_MODALITIES = {"mammo", "us"}


class IngestionAgent:
    """Bloc 1 — Ingestion Agent.

    Validates a patient folder structure and returns a structured
    inventory dict. Raises no exceptions — returns a status field
    with a list of issues instead.
    """

    def __init__(self, uploads_root: Optional[Path] = None):
        self.uploads_root = uploads_root or (Path.cwd() / "bcmid_work" / "uploads")

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def ingest(self, pseudo_id: str) -> Dict[str, Any]:
        """Validate and inventory a patient folder.

        Returns
        -------
        dict with keys:
          pseudo_id, status, issues, modalities_present, files, metadata_checksum
        """
        patient_dir = self.uploads_root / pseudo_id
        issues: List[str] = []
        files: Dict[str, Any] = {}

        # 1. Check patient directory exists
        if not patient_dir.exists() or not patient_dir.is_dir():
            return self._build_result(pseudo_id, "ERROR", [f"Patient directory not found: {patient_dir}"], {}, {})

        # 2. Enumerate and validate files
        modalities_found: Dict[str, List[str]] = {"mammo": [], "us": []}
        for f in patient_dir.iterdir():
            if not f.is_file():
                continue
            if f.suffix.lower() not in VALID_IMAGE_EXTENSIONS and f.suffix.lower() not in {".json", ".txt", ".pdf"}:
                issues.append(f"Unexpected file extension: {f.name}")
                continue

            # Detect modality
            fname = f.stem.lower()
            if "mammo" in fname or "mammograph" in fname:
                modalities_found["mammo"].append(f.name)
                if f.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                    ok, reason = self._validate_image(f)
                    if not ok:
                        issues.append(f"Image unreadable [{f.name}]: {reason}")
            elif "_us" in fname or "ultrasound" in fname or "_us_" in fname:
                modalities_found["us"].append(f.name)
                if f.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                    ok, reason = self._validate_image(f)
                    if not ok:
                        issues.append(f"Image unreadable [{f.name}]: {reason}")
            
            files[f.name] = {
                "size_bytes": f.stat().st_size,
                "extension": f.suffix.lower(),
                "sha256": self._sha256(f)
            }

        # 3. Check modality completeness
        modalities_present = []
        for mod, file_list in modalities_found.items():
            if file_list:
                modalities_present.append(mod)
            else:
                issues.append(f"Missing modality: {mod}")

        # 4. Determine overall status
        missing_mods = REQUIRED_MODALITIES - set(modalities_present)
        if missing_mods:
            status = "INCOMPLETE"
        elif issues:
            status = "WARNING"
        else:
            status = "VALIDATED"

        # 5. Build metadata checksum (reproducibility)
        meta_str = "|".join(sorted(files.keys()))
        metadata_checksum = hashlib.sha256(meta_str.encode()).hexdigest()[:16]

        return self._build_result(
            pseudo_id, status, issues,
            {mod: files_list for mod, files_list in modalities_found.items()},
            files,
            metadata_checksum
        )

    def ingest_batch(self, pseudo_ids: List[str]) -> List[Dict[str, Any]]:
        """Run ingestion on a list of patient IDs."""
        return [self.ingest(pid) for pid in pseudo_ids]

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_image(path: Path):
        """Check image can be decoded by PIL."""
        try:
            from PIL import Image
            with Image.open(path) as img:
                img.verify()
            return True, None
        except Exception as exc:
            return False, str(exc)

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()[:12]

    @staticmethod
    def _build_result(pseudo_id, status, issues, modalities, files, checksum=""):
        return {
            "pseudo_id": pseudo_id,
            "status": status,
            "issues": issues,
            "modalities_present": list(modalities.keys()) if isinstance(modalities, dict) else [],
            "modality_files": modalities,
            "files": files,
            "metadata_checksum": checksum,
            "human_review_required": status in ("ERROR", "INCOMPLETE")
        }
