"""Module 5 — Anonymization: SHA-256 pseudonymisation with environment salt."""

from __future__ import annotations

import csv
import hashlib
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_SALT_ENV_VAR = "DQ_SALT"
_PSEUDO_PREFIX = "P"
_HASH_CHARS = 10


def _get_salt() -> str:
    """Read salt from environment variable or raise."""
    salt = os.environ.get(_SALT_ENV_VAR, "")
    if not salt:
        # Use a deterministic but non-empty default for development only
        logger.warning(
            "Environment variable %s not set — using default development salt. "
            "Set %s in your .env file for production!",
            _SALT_ENV_VAR,
            _SALT_ENV_VAR,
        )
        salt = "bcmid_dev_salt_change_me"
    return salt


def pseudonymise(patient_id: str, salt: str | None = None) -> str:
    """
    Generate a pseudonym for a patient ID.

    Format: P + first 10 hex chars of SHA-256(salt + patient_id)
    Deterministic: same inputs always produce the same output.
    """
    if salt is None:
        salt = _get_salt()
    h = hashlib.sha256((salt + patient_id).encode("utf-8")).hexdigest()
    return _PSEUDO_PREFIX + h[:_HASH_CHARS]


def build_mapping(patient_ids: list[str], salt: str | None = None) -> dict[str, str]:
    """
    Build a dict mapping real_id -> pseudo_id for all patients.

    Parameters
    ----------
    patient_ids : list of real patient folder names
    salt : override salt (for testing)
    """
    if salt is None:
        salt = _get_salt()

    mapping: dict[str, str] = {}
    seen_pseudos: set[str] = set()

    for pid in patient_ids:
        pseudo = pseudonymise(pid, salt)
        if pseudo in seen_pseudos:
            # Collision (astronomically rare) — extend hash
            h = hashlib.sha256((salt + pid + "extra").encode()).hexdigest()
            pseudo = _PSEUDO_PREFIX + h[:_HASH_CHARS]
        seen_pseudos.add(pseudo)
        mapping[pid] = pseudo

    return mapping


def save_private_mapping(mapping: dict[str, str], out_path: Path) -> None:
    """
    Save the real_id -> pseudo_id mapping to a PRIVATE CSV file.

    This file must NEVER be shared or committed to git.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["real_id", "pseudo_id"])
        for real_id, pseudo_id in sorted(mapping.items()):
            writer.writerow([real_id, pseudo_id])
    logger.info("Private mapping saved to %s (DO NOT SHARE)", out_path)


def load_private_mapping(path: Path) -> dict[str, str]:
    """Load real_id -> pseudo_id mapping from CSV."""
    mapping: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["real_id"]] = row["pseudo_id"]
    return mapping
