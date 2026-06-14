"""
tests/test_pipelines.py — Tests for dataset loaders using synthetic fallbacks

These tests verify the synthetic data generators work correctly
WITHOUT requiring internet access or an LLM.
Run with: pytest tests/
"""

import sys, json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import dspy
from data.loader import (
    _synthetic_threat_data, _synthetic_cve_data,
    _generate_alert_example, _random_alert,
    train_val_split,
)
from configs.settings import THREAT_CATEGORIES, SEVERITY_LEVELS


def test_synthetic_threat_data():
    examples = _synthetic_threat_data(70)  # 10 per category
    assert len(examples) > 0
    for ex in examples:
        assert isinstance(ex, dspy.Example)
        assert ex.threat_type in THREAT_CATEGORIES
        assert "log_entry" in ex.inputs()


def test_synthetic_cve_data():
    examples = _synthetic_cve_data(20)
    assert len(examples) == 20
    for ex in examples:
        assert isinstance(ex, dspy.Example)
        assert ex.severity in SEVERITY_LEVELS[:4]  # CRITICAL/HIGH/MEDIUM/LOW
        assert ex.cve_id.startswith("CVE-")


def test_random_alert():
    alert = _random_alert(42)
    assert "id" in alert
    assert "title" in alert
    assert "severity" in alert
    assert alert["severity"] in SEVERITY_LEVELS
    assert alert["id"].startswith("ALT-")


def test_generate_alert_example():
    ex = _generate_alert_example(42)
    assert isinstance(ex, dspy.Example)
    alerts = json.loads(ex.alert_queue)
    assert len(alerts) >= 3
    assert len(alerts) <= 6
    # Expected priority order should be valid JSON
    order = json.loads(ex.expected_priority_order)
    assert len(order) == len(alerts)
    assert ex.top_severity in SEVERITY_LEVELS


def test_train_val_split():
    examples = list(range(100))
    train, val = train_val_split(examples, val_ratio=0.2)
    assert len(train) == 80
    assert len(val) == 20
    # No overlap
    assert set(train).isdisjoint(set(val))


def test_alert_example_priority_ordering():
    """Verify that CRITICAL alerts are always sorted first in expected order."""
    for seed in range(10):
        ex = _generate_alert_example(seed)
        alerts = json.loads(ex.alert_queue)
        order  = json.loads(ex.expected_priority_order)

        sev_map = {a["id"]: a["severity"] for a in alerts}
        severity_rank = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFORMATIONAL": 4}

        ranks = [severity_rank[sev_map[aid]] for aid in order]
        assert ranks == sorted(ranks), f"Priority order not sorted for seed {seed}: {ranks}"


if __name__ == "__main__":
    test_synthetic_threat_data()
    test_synthetic_cve_data()
    test_random_alert()
    test_generate_alert_example()
    test_train_val_split()
    test_alert_example_priority_ordering()
    print("All pipeline/dataset tests passed.")
