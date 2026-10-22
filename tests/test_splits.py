"""Tests for splits module — no data leakage between train/val/test."""

from __future__ import annotations

import pytest

from dq_agent.splits import make_splits


def test_no_patient_in_two_splits(config_for_tests):
    """Core invariant: no patient appears in more than one split."""
    multimodal = ["P001", "P002", "P003", "P004", "P005", "P006", "P007", "P008"]
    labels = {p: "4A" for p in multimodal}

    splits = make_splits(config_for_tests, multimodal, labels)

    train = set(splits["train"])
    val = set(splits["val"])
    test = set(splits["test"])

    assert train.isdisjoint(val), f"Patients in train AND val: {train & val}"
    assert train.isdisjoint(test), f"Patients in train AND test: {train & test}"
    assert val.isdisjoint(test), f"Patients in val AND test: {val & test}"


def test_all_patients_assigned(config_for_tests):
    """Every multimodal patient must appear in exactly one split."""
    multimodal = [f"P{i:03d}" for i in range(20)]
    labels = {p: "3" for p in multimodal}

    splits = make_splits(config_for_tests, multimodal, labels)
    all_assigned = set(splits["train"]) | set(splits["val"]) | set(splits["test"])

    assert all_assigned == set(multimodal)


def test_rare_class_grouping():
    """Classes with fewer than threshold patients must be grouped as RARE."""
    from dq_agent.config import Config, SplitsConfig

    cfg = Config.__new__(Config)
    cfg.splits = SplitsConfig(random_seed=42, rare_class_threshold=3)

    # Only 2 patients with "0" → rare
    multimodal = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10"]
    labels = {
        "P1": "0", "P2": "0",           # rare (2 < 3)
        "P3": "3", "P4": "3", "P5": "3",
        "P6": "4A", "P7": "4A", "P8": "4A",
        "P9": "4B", "P10": "4B",         # rare (2 < 3)
    }

    splits = make_splits(cfg, multimodal, labels)
    assert "0" in splits["rare_groups_pooled"] or "4B" in splits["rare_groups_pooled"]


def test_splits_deterministic():
    """Same seed must always produce same splits."""
    from dq_agent.config import Config, SplitsConfig

    cfg = Config.__new__(Config)
    cfg.splits = SplitsConfig(random_seed=99, rare_class_threshold=2)

    multimodal = [f"P{i:03d}" for i in range(30)]
    labels = {p: "3" for p in multimodal}

    s1 = make_splits(cfg, multimodal, labels)
    s2 = make_splits(cfg, multimodal, labels)

    assert s1["train"] == s2["train"]
    assert s1["val"] == s2["val"]
    assert s1["test"] == s2["test"]
