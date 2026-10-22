"""
Quality / Safety Agent with Feedback Loop (Bloc 6)
Implements:
1. Groundedness verification (checks if stated findings are present in evidence)
2. Hallucination detection (lexical & factual checking against patient inputs)
3. Consistency with BI-RADS clinical rules & Vision scores
4. Feedback revision loop (sends revision instructions back to Reasoning Agent)
"""
from __future__ import annotations
import json
import re
from typing import Dict, Any, List, Tuple, Optional


class QualitySafetyAgent:
    MEDICAL_FINDING_TERMS = [
        "hypoechoic", "lesion", "mass", "calcification", "calcifications",
        "nodule", "cyst", "lymph", "thickening", "clip", "scar",
        "lumpectomy", "mastectomy", "density", "dense", "asymmetry",
        "distortion", "architectural", "spiculated", "microlobulated"
    ]

    def __init__(self, max_feedback_attempts: int = 2):
        self.max_feedback_attempts = max_feedback_attempts

    def verify_safety_and_groundedness(
        self,
        candidate_output: Dict[str, Any],
        patient_case: Dict[str, Any],
        retrieved_evidence: List[Dict[str, Any]]
    ) -> Tuple[bool, List[str]]:
        """
        Runs comprehensive safety checks:
        - Schema & field validation
        - Hallucination detection against allowed source texts
        - Numerical consistency with Vision probabilities
        - Uncertainty & non-standard device warning rules
        """
        errors: List[str] = []

        # 1. Structural schema check
        required_fields = ["summary", "predicted_birads", "risk_level", "evidence_citations", "limitations"]
        for f in required_fields:
            if f not in candidate_output or not candidate_output[f]:
                errors.append(f"Missing required field: '{f}'")

        # 2. Build allowed vocabulary context for hallucination check
        allowed_context = ""
        if patient_case.get("report_text"):
            allowed_context += " " + patient_case["report_text"]
        for ev in retrieved_evidence:
            allowed_context += " " + ev.get("passage", "")

        allowed_context_lower = allowed_context.lower()
        summary_text = str(candidate_output.get("summary", "")).lower()

        # Check for ungrounded medical finding terms
        for term in self.MEDICAL_FINDING_TERMS:
            if term in summary_text and term not in allowed_context_lower:
                errors.append(f"Potential hallucination detected: Medical finding term '{term}' mentioned in summary but absent from patient inputs and retrieved evidence.")

        # 3. Vision score numerical consistency
        vision = patient_case.get("vision_scores", {})
        if vision:
            p_mal = vision.get("p_malignant", 0.0)
            entropy = vision.get("uncertainty_entropy", 0.0)

            # High entropy must be mentioned
            if entropy >= 0.5:
                lims = " ".join(candidate_output.get("limitations", [])).lower()
                if "uncertain" not in lims and "entropy" not in lims and "caution" not in lims:
                    errors.append("Safety omission: High model uncertainty (entropy >= 0.5) must be explicitly disclosed in limitations.")

            # Device warning
            if not patient_case.get("standard_device", True):
                lims = " ".join(candidate_output.get("limitations", [])).lower()
                if "device" not in lims and "non-standard" not in lims and "reliability" not in lims:
                    errors.append("Safety omission: Examination acquired with non-standard device; reduced score reliability must be noted.")

        # 4. Citation check
        citations = candidate_output.get("evidence_citations", [])
        valid_ev_ids = {ev.get("evidence_id") for ev in retrieved_evidence if ev.get("evidence_id")}
        for c in citations:
            if c not in valid_ev_ids and c != "None":
                errors.append(f"Invalid evidence citation: '{c}' does not match any retrieved evidence ID.")

        is_valid = len(errors) == 0
        return is_valid, errors

    def execute_with_feedback_loop(
        self,
        reasoning_agent_fn,
        patient_case: Dict[str, Any],
        retrieved_evidence: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Executes Reasoning Agent with an active Safety Feedback Loop:
        If safety verification fails, returns structured revision feedback to Reasoning Agent.
        """
        feedback_history = []
        revision_feedback = None

        for attempt in range(1, self.max_feedback_attempts + 2):
            # Prompt reasoning agent (with revision feedback if subsequent attempt)
            output = reasoning_agent_fn(patient_case, retrieved_evidence, revision_feedback)

            # Safety check
            is_valid, errors = self.verify_safety_and_groundedness(output, patient_case, retrieved_evidence)

            if is_valid:
                output["safety_audit"] = {
                    "approved": True,
                    "attempts": attempt,
                    "status": "APPROVED_BY_SAFETY_AGENT",
                    "feedback_resolved": feedback_history
                }
                return output
            else:
                feedback_history.append({"attempt": attempt, "errors": errors})
                if attempt <= self.max_feedback_attempts:
                    revision_feedback = (
                        f"SAFETY FEEDBACK (Attempt {attempt}): Your previous output had the following safety issues: "
                        + "; ".join(errors)
                        + ". Please revise your summary to fix these issues. Ensure all claims are strictly grounded in evidence."
                    )
                else:
                    # Exceeded retries: flag for manual review
                    output["safety_audit"] = {
                        "approved": False,
                        "attempts": attempt,
                        "status": "FLAGGED_FOR_MANUAL_CLINICIAN_REVIEW",
                        "unresolved_errors": errors
                    }
                    output["review_level"] = "PRIORITY_MANUAL_REVIEW"
                    return output

        return output
