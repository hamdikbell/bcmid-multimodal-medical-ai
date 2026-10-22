"""
src/reasoning/reasoning_agent.py — Bloc 5: Reasoning Agent (LLM Interface)
Wraps the LLM-based clinical synthesis step. When a local Qwen2.5-3B-Instruct
model is available (via transformers), runs real LLM inference. Otherwise uses
the documented rule-based fallback from run_orchestrator.py.

SIMULATION NOTE: Without GPU/model, the mock_reasoning_agent_llm() is called.
All outputs are labeled reasoning_mode="llm_succeeded" or "rule_based_fallback".
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Allow ikbelwork imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "ikbelwork"))


class ReasoningAgent:
    """Bloc 5 — Reasoning Agent.

    Attempts real LLM inference with Qwen2.5-3B-Instruct. Falls back to
    rule-based synthesis if model unavailable.
    """

    MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

    def __init__(self, model_id: Optional[str] = None):
        self.model_id = model_id or self.MODEL_ID
        self.pipeline = None
        self.tokenizer = None
        self._try_load_model()

    def _try_load_model(self):
        """Attempt to load Qwen2.5 via transformers pipeline."""
        try:
            from transformers import pipeline, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, local_files_only=True)
            self.pipeline = pipeline("text-generation", model=self.model_id, tokenizer=self.tokenizer,
                                     device_map="auto", torch_dtype="auto", local_files_only=True)
            print(f"[ReasoningAgent] Loaded LLM: {self.model_id}")
        except Exception as exc:
            print(f"[ReasoningAgent] LLM not available ({type(exc).__name__}). Using rule-based fallback.")
            self.pipeline = None

    @property
    def simulation_mode(self) -> bool:
        return self.pipeline is None

    def synthesize(
        self,
        patient_case: Dict[str, Any],
        retrieved_evidence: List[Dict[str, Any]],
        revision_feedback: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate clinical synthesis. Uses LLM if available, else rule-based."""
        if self.pipeline is not None:
            return self._llm_synthesize(patient_case, retrieved_evidence, revision_feedback)
        return self._rule_based_synthesize(patient_case, retrieved_evidence, revision_feedback)

    def _llm_synthesize(self, patient_case, retrieved_evidence, revision_feedback):
        """Real LLM synthesis path via transformers."""
        from ikbelwork.run_orchestrator import _build_prompt_for_llm  # type: ignore
        prompt = _build_prompt_for_llm(patient_case, retrieved_evidence, revision_feedback)
        result = self.pipeline(prompt, max_new_tokens=512, do_sample=False, temperature=1.0)
        raw_output = result[0]["generated_text"]
        return {
            "reasoning_mode": "llm_succeeded",
            "model_id": self.model_id,
            "raw_output": raw_output,
            "simulation_mode": False,
        }

    def _rule_based_synthesize(self, patient_case, retrieved_evidence, revision_feedback):
        """Documented rule-based fallback when LLM unavailable."""
        from run_orchestrator import mock_reasoning_agent_llm  # type: ignore
        output = mock_reasoning_agent_llm(patient_case, retrieved_evidence, revision_feedback)
        output["reasoning_mode"] = "rule_based_fallback"
        output["simulation_mode"] = True
        output["simulation_note"] = (
            "Qwen2.5-3B-Instruct not available in current environment. "
            "Rule-based heuristic synthesis used. NOT LLM inference."
        )
        return output
