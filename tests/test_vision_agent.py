"""
tests/test_vision_agent.py
Tests for Bloc 3 — Vision Agent (simulation mode only — no .pth checkpoint present).
Verifies correct simulation disclosure, output schema, and entropy calculation.
"""
import sys
import math
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from vision.vision_agent import VisionAgent, _shannon_entropy, _p_to_birads


@pytest.fixture
def agent():
    """Create VisionAgent — will be in simulation mode (no checkpoint)."""
    return VisionAgent()


# ── Utility function tests ────────────────────────────────────────────────────

def test_shannon_entropy_uniform():
    """T-VIS-01: Uniform distribution over 3 classes has entropy = log2(3) ≈ 1.585."""
    probs = [1/3, 1/3, 1/3]
    e = _shannon_entropy(probs)
    assert abs(e - math.log2(3)) < 0.001

def test_shannon_entropy_certain():
    """T-VIS-02: Certain prediction has entropy ≈ 0."""
    probs = [1.0, 0.0, 0.0]
    e = _shannon_entropy(probs)
    assert e < 0.001

def test_birads_thresholds_coverage():
    """T-VIS-03: _p_to_birads maps all probability ranges to valid BI-RADS categories."""
    valid_categories = {"1", "2", "3", "4A", "4B", "4C", "5"}
    test_probs = [0.0, 0.01, 0.05, 0.15, 0.40, 0.70, 0.88, 0.97, 1.0]
    for p in test_probs:
        cat = _p_to_birads(p)
        assert cat in valid_categories, f"Invalid BI-RADS for p={p}: {cat}"

# ── Simulation mode tests ─────────────────────────────────────────────────────

def test_simulation_mode_disclosure(agent):
    """T-VIS-04: Without checkpoint, simulation_mode must be True."""
    assert agent.simulation_mode is True

def test_predict_suspicious_hint(agent):
    """T-VIS-05: 'spiculated suspect' hint → high p_malignant in simulation."""
    result = agent.predict(clinical_hint="spiculated mass suspected, biopsy recommended")
    assert result["simulation_mode"] is True
    assert result["p_malignant"] >= 0.5, f"Expected high p_mal, got {result['p_malignant']}"

def test_predict_normal_hint(agent):
    """T-VIS-06: Negative hint → low p_malignant in simulation."""
    result = agent.predict(clinical_hint="no discrete mass, bilateral negative examination")
    assert result["simulation_mode"] is True
    assert result["p_malignant"] < 0.15, f"Expected low p_mal, got {result['p_malignant']}"

def test_predict_output_schema(agent):
    """T-VIS-07: Prediction output must contain all required keys."""
    result = agent.predict(clinical_hint="normal screening")
    required_keys = {"simulation_mode", "p_normal", "p_benign", "p_malignant",
                     "uncertainty_entropy", "predicted_birads_heuristic", "birads_note"}
    assert required_keys.issubset(result.keys()), f"Missing keys: {required_keys - result.keys()}"

def test_predict_probabilities_sum_to_one(agent):
    """T-VIS-08: p_normal + p_benign + p_malignant must sum to ~1.0."""
    result = agent.predict(clinical_hint="surveillance follow-up")
    total = result["p_normal"] + result["p_benign"] + result["p_malignant"]
    assert abs(total - 1.0) < 0.01, f"Probs don't sum to 1: {total}"

def test_predict_entropy_positive(agent):
    """T-VIS-09: Uncertainty entropy must be non-negative."""
    result = agent.predict(clinical_hint="")
    assert result["uncertainty_entropy"] >= 0.0

def test_predict_birads_note_present(agent):
    """T-VIS-10: birads_note must warn that mapping is heuristic, not validated."""
    result = agent.predict(clinical_hint="suspicious mass")
    assert "heuristic" in result["birads_note"].lower() or "not" in result["birads_note"].lower()
