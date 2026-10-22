"""Configuration loader for the Data & Quality Agent."""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

load_dotenv()  # charge .env si présent


@dataclass
class PreprocessingConfig:
    target_size: int = 512
    mammo_clahe_clip_limit: float = 2.0
    mammo_clahe_tile_grid: int = 8
    us_median_ksize: int = 3
    nlm_h: int = 5
    bit_depth: int = 8


@dataclass
class ModalityConfig:
    require_report: bool = False
    min_modalities: int = 2


@dataclass
class IntegrityConfig:
    min_dimension: int = 32
    std_threshold: float = 1.0


@dataclass
class SplitsConfig:
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    rare_class_threshold: int = 5


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str = "dq_agent.log"


@dataclass
class Config:
    zip_path: Path = Path("")
    tsv_path: Path = Path("")
    work_dir: Path = Path("bcmid_work")
    private_dir: Path = Path("bcmid_work")
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    modality_policy: ModalityConfig = field(default_factory=ModalityConfig)
    integrity: IntegrityConfig = field(default_factory=IntegrityConfig)
    splits: SplitsConfig = field(default_factory=SplitsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @property
    def raw_dir(self) -> Path:
        return self.work_dir / "raw"

    @property
    def validated_dir(self) -> Path:
        return self.work_dir / "validated"

    @property
    def mammo_dir(self) -> Path:
        return self.validated_dir / "mammo"

    @property
    def us_dir(self) -> Path:
        return self.validated_dir / "us"

    @property
    def reports_dir(self) -> Path:
        return self.validated_dir / "reports"

    @property
    def plots_dir(self) -> Path:
        return self.validated_dir / "plots"

    @property
    def mapping_file(self) -> Path:
        """Private mapping — outside validated/."""
        return self.private_dir / "id_mapping_PRIVATE.csv"

    @property
    def progress_file(self) -> Path:
        return self.work_dir / ".progress.json"



def load_config(path: str | Path = "config.yaml") -> Config:
    """Load and validate configuration from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = Config(
        zip_path=Path(raw.get("zip_path", "")),
        tsv_path=Path(raw.get("tsv_path", "")),
        work_dir=Path(raw.get("work_dir", "bcmid_work")),
        private_dir=Path(raw.get("private_dir", raw.get("work_dir", "bcmid_work"))),
    )

    if pp := raw.get("preprocessing"):
        cfg.preprocessing = PreprocessingConfig(**pp)

    if mp := raw.get("modality_policy"):
        cfg.modality_policy = ModalityConfig(**mp)

    if ig := raw.get("integrity"):
        cfg.integrity = IntegrityConfig(**ig)

    if sp := raw.get("splits"):
        cfg.splits = SplitsConfig(**sp)

    if lg := raw.get("logging"):
        cfg.logging = LoggingConfig(**lg)

    return cfg


def setup_logging(cfg: Config) -> None:
    """Configure root logger with file + console handlers."""
    level = getattr(logging, cfg.logging.level.upper(), logging.INFO)
    fmt = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    log_path = cfg.work_dir / cfg.logging.file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)
