"""
tests/test_modules.py — Tests for module helper functions

These test the deterministic helper functions (parsing, normalization,
heuristic fallbacks) that DON'T require an LLM call.
Run with: pytest tests/
"""

import sys, json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.modules.all_modules import (
    _normalize_threat, _normalize_severity, _parse_float, _parse_bool,
    _parse_json_list, _heuristic_threat, _heuristic_priority,
    ThreatReport, CVEReport, TriageReport,
)


def test_normalize_threat():
    assert _normalize_threat("DDoS") == "DDoS"
    assert _normalize_threat("ddos attack") == "DDoS"
    assert _normalize_threat("Port Scan detected") == "PortScan"
    assert _normalize_threat("unknown_garbage") == "Normal"


def test_normalize_severity():
    assert _normalize_severity("critical") == "CRITICAL"
    assert _normalize_severity("This is HIGH severity") == "HIGH"
    assert _normalize_severity("garbage") == "HIGH"  # default


def test_parse_float():
    assert _parse_float(0.85) == 0.85
    assert _parse_float("0.7") == 0.7
    assert _parse_float("85") == 0.85          # percent → fraction
    assert _parse_float("confidence: 0.92") == 0.92
    assert _parse_float("garbage") == 0.5      # default


def test_parse_bool():
    assert _parse_bool(True) is True
    assert _parse_bool("true") is True
    assert _parse_bool("Yes") is True
    assert _parse_bool("false") is False
    assert _parse_bool("no") is False


def test_parse_json_list():
    assert _parse_json_list('["ALT-001", "ALT-002"]') == ["ALT-001", "ALT-002"]
    assert _parse_json_list('Some text ["ALT-003"] more text') == ["ALT-003"]
    assert _parse_json_list('ALT-005 then ALT-006') == ["ALT-005", "ALT-006"]
    assert _parse_json_list('nothing here') == []


def test_heuristic_threat():
    assert _heuristic_threat("proto=UDP flood=True high_rate") == "DDoS"
    assert _heuristic_threat("nmap portscan syn_only") == "PortScan"
    assert _heuristic_threat("ssh brute login_fail=12") == "BruteForce"
    assert _heuristic_threat("sqli payload detected webattack") == "WebAttack"
    assert _heuristic_threat("c2 beacon irc botnet") == "Botnet"
    assert _heuristic_threat("smb lateral winrm priv_esc") == "Infiltration"
    assert _heuristic_threat("normal http traffic") == "Normal"


def test_heuristic_priority():
    queue = json.dumps([
        {"id": "ALT-001", "severity": "LOW"},
        {"id": "ALT-002", "severity": "CRITICAL"},
        {"id": "ALT-003", "severity": "MEDIUM"},
    ])
    order = _heuristic_priority(queue)
    assert order[0] == "ALT-002"  # CRITICAL first
    assert order[-1] == "ALT-001"  # LOW last


def test_threat_report_dataclass():
    report = ThreatReport(
        log_entry="test log",
        threat_type="DDoS",
        confidence=0.9,
        key_indicators="high packet rate",
    )
    d = report.to_dict()
    assert d["threat_type"] == "DDoS"
    assert d["confidence"] == 0.9
    assert "Threat:" in report.summary()


def test_cve_report_dataclass():
    report = CVEReport(
        cve_id="CVE-2021-44228",
        cve_description="Log4Shell",
        cvss_score="10.0",
        severity="CRITICAL",
        affected_component="Apache Log4j",
        attack_vector="Network",
        requires_auth=False,
        impact_summary="RCE",
    )
    d = report.to_dict()
    assert d["cve_id"] == "CVE-2021-44228"
    assert d["severity"] == "CRITICAL"


def test_triage_report_dataclass():
    report = TriageReport(
        alert_queue_raw="[]",
        priority_order=["ALT-001", "ALT-002"],
        top_threat_summary="Ransomware detected",
        correlated_alerts="None",
        escalate_immediately=True,
    )
    d = report.to_dict()
    assert d["priority_order"] == ["ALT-001", "ALT-002"]
    assert d["escalate_immediately"] is True


if __name__ == "__main__":
    test_normalize_threat()
    test_normalize_severity()
    test_parse_float()
    test_parse_bool()
    test_parse_json_list()
    test_heuristic_threat()
    test_heuristic_priority()
    test_threat_report_dataclass()
    test_cve_report_dataclass()
    test_triage_report_dataclass()
    print("All module tests passed.")
