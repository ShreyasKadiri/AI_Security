"""
tests/test_signatures.py — Sanity checks for DSPy Signatures

These tests verify signature field structure WITHOUT requiring an LLM.
Run with: pytest tests/
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import dspy
from src.signatures.threat import ThreatClassifier, ThreatClassifierWithReasoning
from src.signatures.cve_triage import (
    CVESeverityClassifier, CVEImpactAssessor, CVERemediation,
    AlertPrioritizer, AlertAnalyst,
)


def test_threat_classifier_fields():
    fields = ThreatClassifier.model_fields
    assert "log_entry" in fields
    assert "threat_type" in fields
    assert "confidence" in fields
    assert "key_indicators" in fields


def test_threat_classifier_with_reasoning_fields():
    fields = ThreatClassifierWithReasoning.model_fields
    assert "log_entry" in fields
    assert "threat_type" in fields
    assert "mitre_tactic" in fields
    assert "recommended_action" in fields


def test_cve_severity_classifier_fields():
    fields = CVESeverityClassifier.model_fields
    for f in ["cve_id", "cve_description", "cvss_score", "severity",
              "affected_component", "attack_vector", "requires_auth", "impact_summary"]:
        assert f in fields, f"Missing field: {f}"


def test_cve_impact_assessor_fields():
    fields = CVEImpactAssessor.model_fields
    for f in ["cve_id", "infrastructure_context", "likely_affected",
              "blast_radius", "immediate_actions", "patch_urgency"]:
        assert f in fields, f"Missing field: {f}"


def test_cve_remediation_fields():
    fields = CVERemediation.model_fields
    for f in ["cve_id", "short_term_fix", "long_term_fix", "detection_query", "effort_estimate"]:
        assert f in fields, f"Missing field: {f}"


def test_alert_prioritizer_fields():
    fields = AlertPrioritizer.model_fields
    for f in ["alert_queue", "priority_order", "top_threat_summary",
              "correlated_alerts", "escalate_immediately"]:
        assert f in fields, f"Missing field: {f}"


def test_alert_analyst_fields():
    fields = AlertAnalyst.model_fields
    for f in ["alert_id", "verdict", "attack_stage", "iocs", "containment_steps", "ticket_summary"]:
        assert f in fields, f"Missing field: {f}"


def test_signatures_are_dspy_signatures():
    """All signature classes must subclass dspy.Signature."""
    for sig in [ThreatClassifier, ThreatClassifierWithReasoning,
                CVESeverityClassifier, CVEImpactAssessor, CVERemediation,
                AlertPrioritizer, AlertAnalyst]:
        assert issubclass(sig, dspy.Signature), f"{sig.__name__} is not a dspy.Signature"


if __name__ == "__main__":
    test_threat_classifier_fields()
    test_threat_classifier_with_reasoning_fields()
    test_cve_severity_classifier_fields()
    test_cve_impact_assessor_fields()
    test_cve_remediation_fields()
    test_alert_prioritizer_fields()
    test_alert_analyst_fields()
    test_signatures_are_dspy_signatures()
    print("All signature tests passed.")
