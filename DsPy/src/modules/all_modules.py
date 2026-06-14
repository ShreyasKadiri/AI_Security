"""
src/modules/ — DSPy Modules for all three pipelines

A Module in DSPy wraps one or more Predictors (LM calls) and
defines a forward() method that orchestrates them. Modules are:
  - Composable: nest inside other modules
  - Optimizable: DSPy optimizers tune their parameters
  - Serializable: save/load optimized programs to disk

Predictors used:
  dspy.Predict           — single LM call, no reasoning trace
  dspy.ChainOfThought    — adds auto 'reasoning' field before outputs
  dspy.Assert            — constrains outputs at runtime

This file contains all three pipeline modules for conciseness.
In production, split into threat_module.py, cve_module.py, triage_module.py.
"""

import sys
import json
import re
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import dspy
from src.signatures.threat import ThreatClassifier, ThreatClassifierWithReasoning
from src.signatures.cve_triage import (
    CVESeverityClassifier, CVEImpactAssessor, CVERemediation,
    AlertPrioritizer, AlertAnalyst,
)
from configs.settings import THREAT_CATEGORIES, SEVERITY_LEVELS


# =============================================================================
# Pipeline 1: Threat Classification Module
# =============================================================================

@dataclass
class ThreatReport:
    log_entry:          str
    threat_type:        str
    confidence:         float
    key_indicators:     str
    reasoning:          str = ""
    mitre_tactic:       str = ""
    recommended_action: str = ""

    def summary(self) -> str:
        bar = "█" * int(self.confidence * 10) + "░" * (10 - int(self.confidence * 10))
        return (
            f"Threat:     {self.threat_type}\n"
            f"Confidence: [{bar}] {self.confidence:.2f}\n"
            f"Indicators: {self.key_indicators}\n"
            f"MITRE:      {self.mitre_tactic or 'N/A'}\n"
            f"Action:     {self.recommended_action or 'N/A'}\n"
        )

    def to_dict(self) -> dict:
        return asdict(self)


class ThreatClassifierModule(dspy.Module):
    """
    Classifies a network log entry into a threat category.

    Uses dspy.ChainOfThought(ThreatClassifierWithReasoning):
    - Automatically adds a 'reasoning' field before outputs
    - Model thinks through protocol, ports, volumes before deciding
    - CoT significantly improves accuracy on ambiguous samples

    Why CoT here?
    Network traffic classification is ambiguous: a high-bandwidth
    connection to port 443 could be DDoS, exfiltration, or normal
    streaming. CoT forces the model to reason through each dimension
    before committing to a verdict.
    """

    def __init__(self):
        super().__init__()
        # ChainOfThought adds 'reasoning' output field automatically
        self.classifier = dspy.ChainOfThought(ThreatClassifierWithReasoning)

    def forward(self, log_entry: str) -> ThreatReport:
        try:
            pred = self.classifier(log_entry=log_entry)

            threat_type = _normalize_threat(getattr(pred, "threat_type", "Normal"))
            confidence  = _parse_float(getattr(pred, "confidence", 0.5))

            # dspy.Assert: constrain output to valid threat categories
            dspy.Assert(
                threat_type in THREAT_CATEGORIES,
                f"threat_type must be one of {THREAT_CATEGORIES}, got '{threat_type}'"
            )

            return ThreatReport(
                log_entry=log_entry,
                threat_type=threat_type,
                confidence=confidence,
                key_indicators=getattr(pred, "key_indicators", ""),
                reasoning=getattr(pred, "reasoning", ""),
                mitre_tactic=getattr(pred, "mitre_tactic", ""),
                recommended_action=getattr(pred, "recommended_action", ""),
            )
        except Exception as e:
            # Heuristic fallback
            threat_type = _heuristic_threat(log_entry)
            return ThreatReport(
                log_entry=log_entry,
                threat_type=threat_type,
                confidence=0.5,
                key_indicators=f"Heuristic fallback: {str(e)[:80]}",
            )


class BatchThreatClassifier(dspy.Module):
    """Classify multiple log entries and return aggregate statistics."""

    def __init__(self):
        super().__init__()
        self.classifier = ThreatClassifierModule()

    def forward(self, log_entries: list[str]) -> tuple[list[ThreatReport], dict]:
        reports = []
        for log in log_entries:
            reports.append(self.classifier(log_entry=log))

        # Aggregate stats
        from collections import Counter
        dist   = Counter(r.threat_type for r in reports)
        avg_conf = sum(r.confidence for r in reports) / max(len(reports), 1)
        threats  = [r for r in reports if r.threat_type != "Normal"]

        stats = {
            "total":         len(reports),
            "threats":       len(threats),
            "normal":        dist.get("Normal", 0),
            "distribution":  dict(dist),
            "avg_confidence": round(avg_conf, 3),
            "top_threat":    dist.most_common(1)[0][0] if dist else "None",
        }
        return reports, stats


# =============================================================================
# Pipeline 2: CVE Analysis Module
# =============================================================================

@dataclass
class CVEReport:
    cve_id:                  str
    cve_description:         str
    cvss_score:              str
    severity:                str
    affected_component:      str
    attack_vector:           str
    requires_auth:           bool
    impact_summary:          str
    # Impact assessment (step 2)
    likely_affected:         bool  = True
    affected_assets:         str   = ""
    blast_radius:            str   = ""
    exploitation_likelihood: str   = ""
    immediate_actions:       str   = ""
    patch_urgency:           str   = ""
    # Remediation (step 3)
    short_term_fix:          str   = ""
    long_term_fix:           str   = ""
    detection_query:         str   = ""
    effort_estimate:         str   = ""
    reasoning:               str   = ""

    def to_dict(self) -> dict:
        return asdict(self)


class CVEAnalystModule(dspy.Module):
    """
    Three-step CVE analysis pipeline:
      1. Classify severity + basic metadata (Predict)
      2. Assess impact on specific infrastructure (ChainOfThought)
      3. Generate remediation plan (Predict)

    This demonstrates MODULE COMPOSITION — three separate LM calls
    each focused on one specific sub-task, rather than one mega-prompt
    trying to do everything at once.

    Decomposing tasks this way:
    - Improves accuracy (each predictor focuses on one thing)
    - Enables independent optimization of each step
    - Makes failures easier to diagnose
    """

    def __init__(self):
        super().__init__()
        self.severity_classifier = dspy.Predict(CVESeverityClassifier)
        self.impact_assessor     = dspy.ChainOfThought(CVEImpactAssessor)  # CoT for complex reasoning
        self.remediator          = dspy.Predict(CVERemediation)

    def forward(
        self,
        cve_id: str,
        cve_description: str,
        cvss_score: str,
        infrastructure_context: str = "General enterprise: Windows AD, Linux web servers, Cisco networking.",
    ) -> CVEReport:
        # ── Step 1: Severity classification ──────────────────────────────────
        try:
            sev_pred = self.severity_classifier(
                cve_id=cve_id,
                cve_description=cve_description,
                cvss_score=cvss_score,
            )
            severity           = _normalize_severity(getattr(sev_pred, "severity", "HIGH"))
            affected_component = getattr(sev_pred, "affected_component", "Unknown")
            attack_vector      = getattr(sev_pred, "attack_vector", "Network")
            requires_auth      = _parse_bool(getattr(sev_pred, "requires_auth", False))
            impact_summary     = getattr(sev_pred, "impact_summary", "")
        except Exception as e:
            severity, affected_component = "HIGH", "Unknown"
            attack_vector, requires_auth = "Network", False
            impact_summary = f"Classification error: {e}"

        # ── Step 2: Infrastructure impact assessment ──────────────────────────
        try:
            imp_pred = self.impact_assessor(
                cve_id=cve_id,
                cve_description=cve_description,
                cvss_score=cvss_score,
                severity=severity,
                infrastructure_context=infrastructure_context,
            )
            likely_affected         = _parse_bool(getattr(imp_pred, "likely_affected", True))
            affected_assets         = getattr(imp_pred, "affected_assets", "")
            blast_radius            = getattr(imp_pred, "blast_radius", "Unknown")
            exploitation_likelihood = getattr(imp_pred, "exploitation_likelihood", "Unknown")
            immediate_actions       = getattr(imp_pred, "immediate_actions", "")
            patch_urgency           = getattr(imp_pred, "patch_urgency", "High")
            reasoning               = getattr(imp_pred, "reasoning", "")
        except Exception as e:
            likely_affected = True
            affected_assets, blast_radius = "Unknown", "Unknown"
            exploitation_likelihood = "Unknown"
            immediate_actions = f"Error in impact assessment: {e}"
            patch_urgency, reasoning = "High", ""

        # ── Step 3: Remediation plan ──────────────────────────────────────────
        try:
            rem_pred     = self.remediator(
                cve_id=cve_id,
                affected_component=affected_component,
                severity=severity,
                blast_radius=blast_radius,
            )
            short_term_fix   = getattr(rem_pred, "short_term_fix", "")
            long_term_fix    = getattr(rem_pred, "long_term_fix", "")
            detection_query  = getattr(rem_pred, "detection_query", "")
            effort_estimate  = getattr(rem_pred, "effort_estimate", "")
        except Exception as e:
            short_term_fix = long_term_fix = detection_query = effort_estimate = ""

        return CVEReport(
            cve_id=cve_id,
            cve_description=cve_description[:300],
            cvss_score=cvss_score,
            severity=severity,
            affected_component=affected_component,
            attack_vector=attack_vector,
            requires_auth=requires_auth,
            impact_summary=impact_summary,
            likely_affected=likely_affected,
            affected_assets=affected_assets,
            blast_radius=blast_radius,
            exploitation_likelihood=exploitation_likelihood,
            immediate_actions=immediate_actions,
            patch_urgency=patch_urgency,
            short_term_fix=short_term_fix,
            long_term_fix=long_term_fix,
            detection_query=detection_query,
            effort_estimate=effort_estimate,
            reasoning=reasoning,
        )


# =============================================================================
# Pipeline 3: Alert Triage Module
# =============================================================================

@dataclass
class TriageReport:
    alert_queue_raw:     str
    priority_order:      list
    top_threat_summary:  str
    correlated_alerts:   str
    escalate_immediately: bool
    # Deep analysis of top alert
    top_alert_verdict:   str   = ""
    attack_stage:        str   = ""
    iocs:                str   = ""
    containment_steps:   str   = ""
    ticket_summary:      str   = ""
    reasoning:           str   = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


class AlertTriageModule(dspy.Module):
    """
    Two-step alert triage pipeline:
      1. Prioritize the full alert queue (ChainOfThought)
      2. Deep-analyze the top-priority alert (ChainOfThought)

    Demonstrates:
    - dspy.Assert for JSON output validation
    - ChainOfThought for complex multi-alert correlation
    - Module composition: prioritizer feeds into analyzer
    """

    def __init__(self):
        super().__init__()
        self.prioritizer = dspy.ChainOfThought(AlertPrioritizer)
        self.analyst     = dspy.ChainOfThought(AlertAnalyst)

    def forward(self, alert_queue: str) -> TriageReport:
        # ── Step 1: Prioritize the queue ──────────────────────────────────────
        try:
            prio_pred = self.prioritizer(alert_queue=alert_queue)
            priority_order_raw   = getattr(prio_pred, "priority_order", "[]")
            top_threat_summary   = getattr(prio_pred, "top_threat_summary", "")
            correlated_alerts    = getattr(prio_pred, "correlated_alerts", "None detected")
            escalate_immediately = _parse_bool(getattr(prio_pred, "escalate_immediately", False))
            reasoning            = getattr(prio_pred, "reasoning", "")

            # dspy.Assert: validate JSON output
            priority_order = _parse_json_list(priority_order_raw)
            dspy.Assert(
                isinstance(priority_order, list) and len(priority_order) > 0,
                "priority_order must be a non-empty JSON list of alert IDs"
            )
        except Exception as e:
            # Fallback: extract IDs from queue and sort by raw severity
            priority_order       = _heuristic_priority(alert_queue)
            top_threat_summary   = f"Triage fallback: {str(e)[:80]}"
            correlated_alerts    = "Error in correlation analysis"
            escalate_immediately = True
            reasoning            = ""

        # ── Step 2: Deep-analyze the top alert ───────────────────────────────
        top_alert = _get_alert_by_id(alert_queue, priority_order[0] if priority_order else None)
        verdict = attack_stage = iocs = containment_steps = ticket_summary = ""

        if top_alert:
            try:
                analysis = self.analyst(
                    alert_id=top_alert.get("id", ""),
                    alert_title=top_alert.get("title", ""),
                    alert_desc=top_alert.get("desc", ""),
                    alert_severity=top_alert.get("severity", "HIGH"),
                    alert_source=top_alert.get("source", "Unknown"),
                )
                verdict           = getattr(analysis, "verdict", "")
                attack_stage      = getattr(analysis, "attack_stage", "")
                iocs              = getattr(analysis, "iocs", "")
                containment_steps = getattr(analysis, "containment_steps", "")
                ticket_summary    = getattr(analysis, "ticket_summary", "")
            except Exception as e:
                verdict = f"Analysis error: {str(e)[:80]}"

        return TriageReport(
            alert_queue_raw=alert_queue,
            priority_order=priority_order,
            top_threat_summary=top_threat_summary,
            correlated_alerts=correlated_alerts,
            escalate_immediately=escalate_immediately,
            top_alert_verdict=verdict,
            attack_stage=attack_stage,
            iocs=iocs,
            containment_steps=containment_steps,
            ticket_summary=ticket_summary,
            reasoning=reasoning,
        )


# =============================================================================
# Utility helpers
# =============================================================================

def _normalize_threat(raw: str) -> str:
    raw_clean = re.sub(r"[\s_-]+", "", str(raw)).lower()
    for cat in THREAT_CATEGORIES:
        cat_clean = re.sub(r"[\s_-]+", "", cat).lower()
        if cat_clean in raw_clean:
            return cat
    return "Normal"


def _normalize_severity(raw: str) -> str:
    raw = str(raw).strip().upper()
    for s in SEVERITY_LEVELS:
        if s in raw:
            return s
    return "HIGH"


def _parse_float(val) -> float:
    if isinstance(val, (int, float)):
        v = float(val)
    else:
        m = re.search(r"0?\.\d+|\d+\.\d+|\d+", str(val))
        v = float(m.group()) if m else 0.5
    if v > 1.0:
        v = v / 100.0
    return max(0.0, min(1.0, v))


def _parse_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("true", "yes", "1")


def _parse_json_list(raw: str) -> list:
    raw = str(raw).strip()
    m   = re.search(r"\[.*?\]", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    ids = re.findall(r'ALT-\d+', raw)
    return ids if ids else []


def _get_alert_by_id(alert_queue_json: str, alert_id: Optional[str]) -> Optional[dict]:
    if not alert_id:
        return None
    try:
        alerts = json.loads(alert_queue_json)
        return next((a for a in alerts if a.get("id") == alert_id), alerts[0] if alerts else None)
    except Exception:
        return None


def _heuristic_threat(log: str) -> str:
    log = log.lower()
    if any(k in log for k in ["syn_flood", "ddos", "flood", "high_rate"]):  return "DDoS"
    if any(k in log for k in ["scan", "portscan", "nmap", "syn_only"]):      return "PortScan"
    if any(k in log for k in ["brute", "ssh", "rdp", "ftp", "login_fail"]): return "BruteForce"
    if any(k in log for k in ["sqli", "xss", "traversal", "webattack"]):    return "WebAttack"
    if any(k in log for k in ["c2", "beacon", "irc", "tunnel", "botnet"]):  return "Botnet"
    if any(k in log for k in ["lateral", "smb", "winrm", "priv_esc"]):      return "Infiltration"
    return "Normal"


def _heuristic_priority(alert_queue_json: str) -> list:
    try:
        alerts = json.loads(alert_queue_json)
        order  = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFORMATIONAL": 4}
        sorted_alerts = sorted(alerts, key=lambda a: order.get(a.get("severity", "LOW"), 5))
        return [a["id"] for a in sorted_alerts]
    except Exception:
        return []
