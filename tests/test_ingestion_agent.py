"""
tests/test_ingestion_agent.py
Tests for Bloc 1 — Ingestion Agent.
Uses the real uploads directory where test patients exist.
"""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from ingestion.ingestion_agent import IngestionAgent

UPLOADS = Path(__file__).parent.parent / "bcmid_work" / "uploads"
KNOWN_PATIENTS = [p.name for p in UPLOADS.iterdir() if p.is_dir()] if UPLOADS.exists() else []


@pytest.fixture
def agent():
    return IngestionAgent(uploads_root=UPLOADS)


@pytest.mark.skipif(not UPLOADS.exists(), reason="uploads dir not present")
def test_known_patient_validated(agent):
    """T-ING-01: A complete patient folder must return VALIDATED or WARNING status."""
    if not KNOWN_PATIENTS:
        pytest.skip("No patient folders in uploads/")
    pid = KNOWN_PATIENTS[0]
    result = agent.ingest(pid)
    assert result["status"] in ("VALIDATED", "WARNING"), f"Unexpected status: {result}"
    assert result["pseudo_id"] == pid


def test_missing_patient_returns_error(agent):
    """T-ING-02: Non-existent patient ID must return ERROR status."""
    result = agent.ingest("P_NONEXISTENT_XYZ")
    assert result["status"] == "ERROR"
    assert result["human_review_required"] is True
    assert len(result["issues"]) > 0


@pytest.mark.skipif(not UPLOADS.exists(), reason="uploads dir not present")
def test_batch_ingestion(agent):
    """T-ING-03: Batch ingestion returns one result per patient."""
    pids = KNOWN_PATIENTS[:3] if len(KNOWN_PATIENTS) >= 3 else KNOWN_PATIENTS
    if not pids:
        pytest.skip("No patient folders in uploads/")
    results = agent.ingest_batch(pids)
    assert len(results) == len(pids)
    for r in results:
        assert "status" in r
        assert "pseudo_id" in r
        assert "files" in r


@pytest.mark.skipif(not UPLOADS.exists(), reason="uploads dir not present")
def test_result_has_required_keys(agent):
    """T-ING-04: Ingestion result must contain all required output keys."""
    if not KNOWN_PATIENTS:
        pytest.skip("No patient folders in uploads/")
    result = agent.ingest(KNOWN_PATIENTS[0])
    required_keys = {"pseudo_id", "status", "issues", "modalities_present", "files", "metadata_checksum", "human_review_required"}
    assert required_keys.issubset(result.keys()), f"Missing keys: {required_keys - result.keys()}"
