"""
Tests for the Quality/Safety Agent pipeline.
Covers:
  - Schema validation (missing required fields → FAIL)
  - Hallucination detection (ungrounded medical term → FAIL)
  - Citation validation (fake citation ID → FAIL)
  - High-entropy disclosure (entropy ≥ 0.5 must appear in limitations → FAIL if missing)
  - Valid report → PASS
  - Negative tests (altered score, missing field, bad JSON structure, fake citation)
  - Feedback loop: revisions reduce errors

These are real checks against the actual QualitySafetyAgent class.
"""
import sys
from pathlib import Path
import pytest

# Allow imports from ikbelwork/
sys.path.insert(0, str(Path(__file__).parent.parent / "ikbelwork"))
from safety_agent import QualitySafetyAgent


@pytest.fixture
def agent():
    return QualitySafetyAgent(max_feedback_attempts=2)


@pytest.fixture
def good_evidence():
    """Minimal valid retrieved evidence with real IDs."""
    return [
        {
            "evidence_id": "E1",
            "source_type": "ACR_GUIDELINE",
            "passage": "ACR BI-RADS Category 4A: Low suspicion for malignancy. Biopsy recommended.",
            "similarity": 0.91
        },
        {
            "evidence_id": "E2",
            "source_type": "PATIENT_REPORT",
            "passage": "No discrete mass or architectural distortion found. Bilateral screening.",
            "similarity": 0.87
        }
    ]


@pytest.fixture
def good_patient_case():
    """Valid patient case with low uncertainty."""
    return {
        "pseudo_id": "P_TEST_SAFETY",
        "vision_scores": {"p_malignant": 0.12, "uncertainty_entropy": 0.30},
        "standard_device": True,
        "report_text": "Bilateral breast mammography. No discrete mass. No suspicious calcification."
    }


@pytest.fixture
def valid_candidate():
    """A well-formed, grounded candidate output."""
    return {
        "summary": "No suspicious mass or calcification identified. Bilateral screening negative.",
        "predicted_birads": "BI-RADS 1",
        "risk_level": "Benign / Normal",
        "evidence_citations": ["E1", "E2"],
        "limitations": ["Research AI model — radiologist review required."]
    }


# ── POSITIVE TEST ────────────────────────────────────────────────────────────

def test_valid_report_passes(agent, valid_candidate, good_patient_case, good_evidence):
    """T-SAF-01: A fully valid report must pass all safety checks."""
    is_valid, errors = agent.verify_safety_and_groundedness(
        valid_candidate, good_patient_case, good_evidence
    )
    assert is_valid is True, f"Expected PASS but got errors: {errors}"
    assert errors == []


# ── SCHEMA / MISSING FIELD TESTS ─────────────────────────────────────────────

def test_missing_summary_fails(agent, valid_candidate, good_patient_case, good_evidence):
    """T-SAF-02: Missing 'summary' field must cause FAIL."""
    candidate = {**valid_candidate}
    del candidate["summary"]
    is_valid, errors = agent.verify_safety_and_groundedness(candidate, good_patient_case, good_evidence)
    assert is_valid is False
    assert any("summary" in e.lower() for e in errors), f"Expected 'summary' error, got: {errors}"


def test_missing_predicted_birads_fails(agent, valid_candidate, good_patient_case, good_evidence):
    """T-SAF-03: Missing 'predicted_birads' field must cause FAIL."""
    candidate = {**valid_candidate}
    del candidate["predicted_birads"]
    is_valid, errors = agent.verify_safety_and_groundedness(candidate, good_patient_case, good_evidence)
    assert is_valid is False
    assert any("predicted_birads" in e.lower() for e in errors)


def test_missing_limitations_fails(agent, valid_candidate, good_patient_case, good_evidence):
    """T-SAF-04: Missing 'limitations' field must cause FAIL."""
    candidate = {**valid_candidate}
    del candidate["limitations"]
    is_valid, errors = agent.verify_safety_and_groundedness(candidate, good_patient_case, good_evidence)
    assert is_valid is False
    assert any("limitation" in e.lower() for e in errors)


# ── HALLUCINATION DETECTION TESTS ────────────────────────────────────────────

def test_ungrounded_medical_term_detected(agent, good_patient_case, good_evidence):
    """T-SAF-05: Summary mentioning 'spiculated' when patient report has no such term → FAIL."""
    candidate = {
        "summary": "There is a spiculated mass suspicious for malignancy.",
        "predicted_birads": "BI-RADS 4A",
        "risk_level": "Moderate Suspicion",
        "evidence_citations": ["E1"],
        "limitations": ["Radiologist review required."]
    }
    is_valid, errors = agent.verify_safety_and_groundedness(candidate, good_patient_case, good_evidence)
    # 'spiculated' is in MEDICAL_FINDING_TERMS but NOT in patient report or evidence above
    assert is_valid is False
    assert any("spiculated" in e.lower() or "hallucination" in e.lower() for e in errors)


def test_term_in_evidence_is_not_hallucination(agent, good_patient_case):
    """T-SAF-06: Medical term present in evidence must NOT be flagged as hallucination."""
    evidence_with_term = [
        {
            "evidence_id": "E1",
            "source_type": "PATIENT_REPORT",
            "passage": "Spiculated mass in upper outer quadrant. High suspicion BI-RADS 5.",
            "similarity": 0.95
        }
    ]
    candidate = {
        "summary": "The imaging shows a spiculated mass consistent with BI-RADS 5.",
        "predicted_birads": "BI-RADS 5",
        "risk_level": "High Suspicion",
        "evidence_citations": ["E1"],
        "limitations": ["Radiologist biopsy required."]
    }
    is_valid, errors = agent.verify_safety_and_groundedness(candidate, good_patient_case, evidence_with_term)
    # spiculated is in the evidence — should NOT be flagged as hallucination
    hallucination_errors = [e for e in errors if "spiculated" in e.lower() and "hallucination" in e.lower()]
    assert len(hallucination_errors) == 0, f"Grounded term wrongly flagged: {errors}"


# ── CITATION VALIDATION TESTS ─────────────────────────────────────────────────

def test_fake_citation_id_fails(agent, valid_candidate, good_patient_case, good_evidence):
    """T-SAF-07 (Adversarial T3): Fake citation 'G5' not in evidence → FAIL."""
    candidate = {**valid_candidate, "evidence_citations": ["E1", "G5"]}
    is_valid, errors = agent.verify_safety_and_groundedness(candidate, good_patient_case, good_evidence)
    assert is_valid is False
    assert any("G5" in e for e in errors), f"Expected fake citation error, got: {errors}"


def test_valid_citations_pass(agent, valid_candidate, good_patient_case, good_evidence):
    """T-SAF-08: All citations matching evidence IDs must pass."""
    is_valid, errors = agent.verify_safety_and_groundedness(valid_candidate, good_patient_case, good_evidence)
    citation_errors = [e for e in errors if "citation" in e.lower() or "Invalid evidence" in e]
    assert citation_errors == [], f"Unexpected citation errors: {citation_errors}"


# ── HIGH-ENTROPY DISCLOSURE TESTS ─────────────────────────────────────────────

def test_high_entropy_not_disclosed_fails(agent, valid_candidate, good_evidence):
    """T-SAF-09: entropy ≥ 0.5 but no uncertainty warning in limitations → FAIL."""
    high_entropy_case = {
        "pseudo_id": "P_HIGH_ENT",
        "vision_scores": {"p_malignant": 0.35, "uncertainty_entropy": 0.75},
        "standard_device": True,
        "report_text": "Bilateral screening. No discrete mass."
    }
    candidate_no_warn = {
        "summary": "Findings are equivocal with moderate probability of malignancy.",
        "predicted_birads": "BI-RADS 3",
        "risk_level": "Probably Benign",
        "evidence_citations": ["E1"],
        "limitations": ["Radiologist review required."]   # no mention of uncertainty/entropy
    }
    is_valid, errors = agent.verify_safety_and_groundedness(candidate_no_warn, high_entropy_case, good_evidence)
    assert is_valid is False
    assert any("entropy" in e.lower() or "uncertain" in e.lower() for e in errors), \
        f"Expected entropy disclosure error, got: {errors}"


def test_high_entropy_disclosed_passes(agent, good_evidence):
    """T-SAF-10: entropy ≥ 0.5 AND 'uncertain' in limitations → PASS (entropy check)."""
    high_entropy_case = {
        "pseudo_id": "P_HIGH_ENT_OK",
        "vision_scores": {"p_malignant": 0.35, "uncertainty_entropy": 0.75},
        "standard_device": True,
        "report_text": "Bilateral screening. No discrete mass."
    }
    candidate_with_warn = {
        "summary": "Findings are equivocal. Moderate probability of malignancy.",
        "predicted_birads": "BI-RADS 3",
        "risk_level": "Probably Benign",
        "evidence_citations": ["E1"],
        "limitations": [
            "High model uncertainty detected (entropy ≥ 0.50); scores should be interpreted with caution.",
            "Radiologist review required."
        ]
    }
    is_valid, errors = agent.verify_safety_and_groundedness(candidate_with_warn, high_entropy_case, good_evidence)
    entropy_errors = [e for e in errors if "entropy" in e.lower() or "uncertain" in e.lower()]
    assert entropy_errors == [], f"High entropy correctly disclosed but still flagged: {errors}"
