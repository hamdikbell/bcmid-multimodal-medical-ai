"""Tests for anonymization module."""

from __future__ import annotations

import pytest

from dq_agent.anonymize import pseudonymise, build_mapping


def test_pseudonymise_deterministic():
    """Same inputs must always produce the same pseudonym."""
    salt = "test_salt_xyz"
    pid = "P001y19"
    assert pseudonymise(pid, salt) == pseudonymise(pid, salt)


def test_pseudonymise_format():
    """Pseudonym must start with 'P' and have 11 chars total."""
    p = pseudonymise("any_id", "salt")
    assert p.startswith("P")
    assert len(p) == 11  # P + 10 hex chars


def test_real_id_not_in_pseudo():
    """Real patient ID must NOT appear in the pseudonym."""
    pid = "P001y19"
    pseudo = pseudonymise(pid, "salt")
    assert pid not in pseudo


def test_mapping_no_collisions():
    """All pseudonyms in the mapping must be unique."""
    pids = [f"patient_{i:04d}" for i in range(100)]
    mapping = build_mapping(pids, salt="unique_salt")
    pseudos = list(mapping.values())
    assert len(pseudos) == len(set(pseudos)), "Collision in pseudonym mapping"


def test_mapping_covers_all_ids():
    """Every input ID must have a pseudonym."""
    pids = ["A", "B", "C"]
    mapping = build_mapping(pids, salt="s")
    assert set(mapping.keys()) == set(pids)


def test_private_mapping_saved(tmp_path):
    """Private mapping file must be created at the right path."""
    from dq_agent.anonymize import save_private_mapping, load_private_mapping

    mapping = {"real_001": "Pabc1234567", "real_002": "Pxyz9876543"}
    path = tmp_path / "id_mapping_PRIVATE.csv"
    save_private_mapping(mapping, path)
    assert path.exists()

    loaded = load_private_mapping(path)
    assert loaded == mapping
