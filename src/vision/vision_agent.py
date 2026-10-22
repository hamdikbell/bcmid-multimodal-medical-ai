"""
src/vision/vision_agent.py — Bloc 3: Multimodal Vision Agent
Wraps the ResNet-18 based vision pipeline. When model checkpoints are available,
runs real inference. Otherwise runs in SIMULATION mode with explicit disclosure.

SIMULATION MODE: No .pth checkpoint found → keyword/metadata heuristic fallback.
This is NOT clinical inference. All outputs are labeled simulation_mode=True.
"""
from __future__ import annotations
import math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


# ── BI-RADS mapping from binary malignancy probability ─────────────────────────
# Based on ACR BI-RADS 5th edition thresholds (NOT clinically validated for this model)
# These are heuristic approximations only — documented as simulation.
BIRADS_THRESHOLDS = [
    (0.02, "1"),    # Negative — essentially 0% risk
    (0.10, "2"),    # Benign finding
    (0.35, "3"),    # Probably benign
    (0.65, "4A"),   # Low suspicion
    (0.85, "4B"),   # Moderate suspicion
    (0.95, "4C"),   # High suspicion
    (1.01, "5"),    # Highly suggestive of malignancy
]


def _p_to_birads(p_mal: float) -> str:
    """Map malignancy probability to BI-RADS category (heuristic — not clinically validated)."""
    for threshold, category in BIRADS_THRESHOLDS:
        if p_mal < threshold:
            return category
    return "5"


def _shannon_entropy(probs: list) -> float:
    """Compute Shannon entropy (bits) from a probability vector."""
    return round(-sum(p * math.log2(p) for p in probs if p > 0), 4)


class VisionAgent:
    """Bloc 3 — Multimodal Vision Agent.

    Runs ResNet-18 inference (or simulation) on mammography + ultrasound images.
    Always returns a dict with simulation_mode field to ensure transparency.
    """

    CHECKPOINT_PATHS = [
        Path("bcmid_work/models/resnet18_mammo.pth"),
        Path("bcmid_work/models/resnet18_fusion.pth"),
        Path("models/resnet18_mammo.pth"),
    ]

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None
        self.checkpoint_path = None
        self._try_load_model()

    def _try_load_model(self):
        """Attempt to load ResNet-18 checkpoint. Sets self.model = None if unavailable."""
        for cp in self.CHECKPOINT_PATHS:
            if cp.exists():
                try:
                    import torch
                    import torchvision.models as models
                    model = models.resnet18(weights=None)
                    model.fc = __import__("torch.nn", fromlist=["Linear"]).Linear(512, 3)
                    model.load_state_dict(torch.load(cp, map_location=self.device))
                    model.eval()
                    self.model = model
                    self.checkpoint_path = cp
                    print(f"[VisionAgent] Loaded checkpoint: {cp}")
                    return
                except Exception as exc:
                    print(f"[VisionAgent] Could not load checkpoint {cp}: {exc}")
        print("[VisionAgent] WARNING: No checkpoint found. Running in SIMULATION mode.")

    @property
    def simulation_mode(self) -> bool:
        return self.model is None

    def predict(
        self,
        mammo_path: Optional[Path] = None,
        us_path: Optional[Path] = None,
        clinical_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run inference or simulation. Returns vision scores with full provenance."""
        if self.simulation_mode:
            return self._simulate(mammo_path, us_path, clinical_hint)
        return self._infer(mammo_path, us_path)

    def _infer(self, mammo_path: Optional[Path], us_path: Optional[Path]) -> Dict[str, Any]:
        """Real CNN inference path (only reachable when checkpoint loaded)."""
        import torch
        import torchvision.transforms as T
        from PIL import Image

        transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        def run_model(img_path: Path) -> Tuple[float, float, float]:
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1).squeeze().tolist()
            return tuple(probs)  # (p_normal, p_benign, p_malignant)

        mammo_probs = run_model(mammo_path) if mammo_path and mammo_path.exists() else (0.8, 0.15, 0.05)
        us_probs = run_model(us_path) if us_path and us_path.exists() else None

        # Fusion: weighted average (mammography primary, ultrasound secondary)
        p_norm = mammo_probs[0]
        p_benign = mammo_probs[1]
        p_mal = mammo_probs[2]

        if us_probs:
            # Upweight mammography (0.7) over ultrasound (0.3)
            p_norm = 0.7 * mammo_probs[0] + 0.3 * us_probs[0]
            p_benign = 0.7 * mammo_probs[1] + 0.3 * us_probs[1]
            p_mal = 0.7 * mammo_probs[2] + 0.3 * us_probs[2]

        entropy = _shannon_entropy([p_norm, p_benign, p_mal])

        return {
            "simulation_mode": False,
            "checkpoint": str(self.checkpoint_path),
            "p_normal": round(p_norm, 4),
            "p_benign": round(p_benign, 4),
            "p_malignant": round(p_mal, 4),
            "uncertainty_entropy": entropy,
            "predicted_birads_heuristic": _p_to_birads(p_mal),
            "birads_note": "Heuristic mapping from binary probability — not a validated BI-RADS classifier.",
            "mammo_probs": {"p_normal": round(mammo_probs[0], 4), "p_benign": round(mammo_probs[1], 4), "p_malignant": round(mammo_probs[2], 4)},
            "us_probs": {"p_normal": round(us_probs[0], 4), "p_benign": round(us_probs[1], 4), "p_malignant": round(us_probs[2], 4)} if us_probs else None,
        }

    def _simulate(
        self,
        mammo_path: Optional[Path],
        us_path: Optional[Path],
        clinical_hint: Optional[str],
    ) -> Dict[str, Any]:
        """
        Simulation fallback when no model checkpoint is available.
        Keyword heuristic on clinical hint — clearly labeled simulation_mode=True.
        """
        hint = (clinical_hint or "").lower()
        has_negation = any(neg in hint for neg in ["absence", "sans", "pas de", "aucun", "negatif", "negative"])
        is_suspect = any(w in hint for w in ["spicul", "suspect", "biopsie", "microcalcif"]) and not has_negation
        is_surv = any(w in hint for w in ["kyste", "suivi", "controle", "probably benign"]) and not has_negation

        if is_suspect:
            p_norm, p_benign, p_mal = 0.08, 0.18, 0.74
        elif is_surv:
            p_norm, p_benign, p_mal = 0.20, 0.68, 0.12
        else:
            p_norm, p_benign, p_mal = 0.90, 0.08, 0.02

        entropy = _shannon_entropy([p_norm, p_benign, p_mal])

        return {
            "simulation_mode": True,
            "simulation_note": "No CNN checkpoint found. Keyword-heuristic scores only. NOT clinical inference.",
            "checkpoint": None,
            "p_normal": p_norm,
            "p_benign": p_benign,
            "p_malignant": p_mal,
            "uncertainty_entropy": entropy,
            "predicted_birads_heuristic": _p_to_birads(p_mal),
            "birads_note": "Heuristic simulation — not a validated BI-RADS classifier.",
            "mammo_probs": None,
            "us_probs": None,
        }
