"""
src/config/settings.py — Central configuration for BCMID pipeline.
All paths, thresholds, and environment flags in one place.
"""
from __future__ import annotations
from pathlib import Path
import os

# ── Base paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent.parent
BCMID_WORK = BASE_DIR / "bcmid_work"
UPLOADS_DIR = BCMID_WORK / "uploads"
REPORTS_DIR = BCMID_WORK / "reports"
VALIDATED_DIR = BCMID_WORK / "validated"
MODELS_DIR = BCMID_WORK / "models"
IKBELWORK_DIR = BASE_DIR / "ikbelwork"

# ── Data files ────────────────────────────────────────────────────────────────
MANIFEST_CSV = VALIDATED_DIR / "manifest.csv"
QUALITY_REPORT_JSON = VALIDATED_DIR / "quality_report.json"
SPLITS_JSON = VALIDATED_DIR / "splits.json"
FINAL_REPORTS_JSON = BCMID_WORK / "final_medical_reports.json"
RAG_INDEX_FAISS = BCMID_WORK / "rag_index.faiss"
RAG_CHUNKS_CSV = BCMID_WORK / "rag_chunks.csv"
ACR_GUIDELINES_JSON = IKBELWORK_DIR / "acr_birads_guidelines.json"

# ── Model checkpoints ─────────────────────────────────────────────────────────
MAMMO_CHECKPOINT = MODELS_DIR / "resnet18_mammo.pth"
US_CHECKPOINT = MODELS_DIR / "resnet18_us.pth"
FUSION_CHECKPOINT = MODELS_DIR / "resnet18_fusion.pth"
LLM_MODEL_ID = os.getenv("BCMID_LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")

# ── Vision thresholds (heuristic — not clinically validated) ──────────────────
BIRADS_THRESHOLDS = [
    (0.02, "1"),
    (0.10, "2"),
    (0.35, "3"),
    (0.65, "4A"),
    (0.85, "4B"),
    (0.95, "4C"),
    (1.01, "5"),
]

# ── Safety thresholds ─────────────────────────────────────────────────────────
HIGH_ENTROPY_THRESHOLD = 0.5   # Entropy ≥ 0.5 → must disclose uncertainty
MAX_SAFETY_RETRIES = 2         # QualitySafetyAgent max feedback attempts

# ── API settings ──────────────────────────────────────────────────────────────
API_HOST = os.getenv("BCMID_API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("BCMID_API_PORT", "8000"))
N8N_WEBHOOK_BASE = os.getenv("BCMID_N8N_URL", "http://localhost:5678")

# ── Simulation flags ──────────────────────────────────────────────────────────
# Set to True if no model checkpoints are present (default in current deployment)
VISION_SIMULATION = not MAMMO_CHECKPOINT.exists()
LLM_SIMULATION = True   # Set to False when Qwen2.5-3B-Instruct is deployed locally

# ── Preprocessing parameters (from quality_report.json) ──────────────────────
PREPROCESS = {
    "target_size": 512,
    "mammo_clahe_clip_limit": 2.0,
    "mammo_clahe_tile_grid": 8,
    "us_median_ksize": 3,
    "nlm_h": 5,
    "bit_depth": 8,
}

# ── Privacy ───────────────────────────────────────────────────────────────────
PRIVATE_MAPPING_PATH = BCMID_WORK / "id_mapping_PRIVATE.csv"
SALT_ENV_VAR = "BCMID_ANON_SALT"   # Salt read from environment for reproducible anonymization
