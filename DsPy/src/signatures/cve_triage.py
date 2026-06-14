"""
src/signatures/cve.py — Signatures for Pipeline 2: CVE Analysis
src/signatures/triage.py — Signatures for Pipeline 3: Alert Triage

Both in one file for conciseness. In a real project, split into separate files.
"""

import dspy


# =============================================================================
# Pipeline 2: CVE Analysis Signatures
# =============================================================================

class CVESeverityClassifier(dspy.Signature):
    """
    You are a vulnerability management specialist.
    Given a CVE description and CVSS score, classify the severity
    and explain the potential impact on enterprise infrastructure.
    """

    cve_id: str = dspy.InputField(
        desc="CVE identifier, e.g. CVE-2021-44228"
    )
    cve_description: str = dspy.InputField(
        desc="Plain-English description of the vulnerability from NVD"
    )
    cvss_score: str = dspy.InputField(
        desc="CVSS base score (0.0–10.0) or 'N/A' if unavailable"
    )

    severity: str = dspy.OutputField(
        desc="One of: CRITICAL, HIGH, MEDIUM, LOW — matching the CVSS score range"
    )
    affected_component: str = dspy.OutputField(
        desc="The specific software component, version range, or system type affected"
    )
    attack_vector: str = dspy.OutputField(
        desc="One of: Network, Adjacent, Local, Physical"
    )
    requires_auth: bool = dspy.OutputField(
        desc="True if exploitation requires prior authentication, False if unauthenticated"
    )
    impact_summary: str = dspy.OutputField(
        desc=(
            "One sentence describing the worst-case impact: "
            "e.g. 'Unauthenticated remote code execution as SYSTEM on all Windows servers.'"
        )
    )


class CVEImpactAssessor(dspy.Signature):
    """
    You are a threat intelligence analyst assessing CVE impact on a specific infrastructure.
    Given a CVE and infrastructure context, determine whether the infrastructure
    is likely affected, estimate the blast radius, and recommend immediate actions.
    """

    cve_id: str = dspy.InputField(desc="CVE identifier")
    cve_description: str = dspy.InputField(desc="CVE description from NVD")
    cvss_score: str = dspy.InputField(desc="CVSS base score")
    severity: str = dspy.InputField(desc="Severity classification: CRITICAL/HIGH/MEDIUM/LOW")
    infrastructure_context: str = dspy.InputField(
        desc=(
            "Brief description of the infrastructure to assess, e.g.: "
            "'500-employee org, Windows Active Directory, Apache web servers, "
            "Cisco ASA firewalls, no WAF, public-facing GitLab instance.'"
        )
    )

    likely_affected: bool = dspy.OutputField(
        desc="True if the described infrastructure is plausibly affected by this CVE"
    )
    affected_assets: str = dspy.OutputField(
        desc="Comma-separated list of asset types likely affected in this infrastructure"
    )
    blast_radius: str = dspy.OutputField(
        desc="One of: Isolated, Department, Organization-wide, Supply-chain"
    )
    exploitation_likelihood: str = dspy.OutputField(
        desc="One of: Active (exploited in wild), High, Medium, Low, Theoretical"
    )
    immediate_actions: str = dspy.OutputField(
        desc=(
            "Numbered list of immediate actions (max 3), most urgent first. "
            "E.g.: '1. Isolate all Log4j-dependent services. "
            "2. Apply emergency patch or WAF rule. "
            "3. Hunt for exploitation indicators using provided IoCs.'"
        )
    )
    patch_urgency: str = dspy.OutputField(
        desc="One of: Emergency (patch now), Urgent (within 24h), High (within 7d), Routine"
    )


class CVERemediation(dspy.Signature):
    """
    Given a CVE and its impact assessment, produce actionable remediation steps
    with effort estimates, suitable for a vulnerability management ticket.
    """

    cve_id: str = dspy.InputField(desc="CVE identifier")
    affected_component: str = dspy.InputField(desc="Affected software/component")
    severity: str = dspy.InputField(desc="CRITICAL/HIGH/MEDIUM/LOW")
    blast_radius: str = dspy.InputField(desc="Scope of impact")

    short_term_fix: str = dspy.OutputField(
        desc="Immediate mitigation (hours): workaround, firewall rule, config change"
    )
    long_term_fix: str = dspy.OutputField(
        desc="Permanent remediation: patch version, upgrade path, architecture change"
    )
    detection_query: str = dspy.OutputField(
        desc=(
            "A SIEM/log query to detect exploitation attempts. "
            "Use generic syntax, e.g.: "
            "'source=web_logs | search uri=*jndi:* OR payload=*${* '"
        )
    )
    effort_estimate: str = dspy.OutputField(
        desc="Estimated effort: e.g. '2h for emergency WAF rule, 1 week for full patch rollout'"
    )


# =============================================================================
# Pipeline 3: Alert Triage Signatures
# =============================================================================

class AlertPrioritizer(dspy.Signature):
    """
    You are a Tier 1 SOC analyst triaging an incoming alert queue.
    Given a list of security alerts in JSON format, prioritize them
    by threat severity, potential business impact, and urgency.
    Consider alert correlations — multiple related alerts may indicate
    a coordinated attack and should be grouped and escalated together.
    """

    alert_queue: str = dspy.InputField(
        desc=(
            "JSON array of security alerts, each with fields: "
            "id, title, severity (CRITICAL/HIGH/MEDIUM/LOW/INFORMATIONAL), "
            "source, desc, timestamp"
        )
    )

    priority_order: str = dspy.OutputField(
        desc=(
            "JSON array of alert IDs in priority order, highest first. "
            "Example: [\"ALT-00042\", \"ALT-00017\", \"ALT-00089\"]"
        )
    )
    top_threat_summary: str = dspy.OutputField(
        desc=(
            "One sentence describing the highest-priority threat: "
            "what it is, what system is affected, and what the attacker may be doing."
        )
    )
    correlated_alerts: str = dspy.OutputField(
        desc=(
            "If any alerts appear related (e.g. phishing → malware → C2), "
            "describe the suspected attack chain. 'None detected' if unrelated."
        )
    )
    escalate_immediately: bool = dspy.OutputField(
        desc="True if any alert requires immediate escalation to Tier 2 or IR team"
    )


class AlertAnalyst(dspy.Signature):
    """
    You are a Tier 2 SOC analyst performing deep analysis on a prioritized alert.
    Given a single high-priority alert and its context, produce a detailed
    analysis suitable for an incident response ticket.
    """

    alert_id: str = dspy.InputField(desc="Alert identifier")
    alert_title: str = dspy.InputField(desc="Alert title")
    alert_desc: str = dspy.InputField(desc="Full alert description")
    alert_severity: str = dspy.InputField(desc="CRITICAL/HIGH/MEDIUM/LOW")
    alert_source: str = dspy.InputField(desc="Detection source: EDR, SIEM, NDR, etc.")

    verdict: str = dspy.OutputField(
        desc="One of: True Positive, False Positive, Benign True Positive"
    )
    attack_stage: str = dspy.OutputField(
        desc=(
            "Likely kill chain stage: Reconnaissance, Weaponization, Delivery, "
            "Exploitation, Installation, C2, Actions-on-Objectives, or Unknown"
        )
    )
    iocs: str = dspy.OutputField(
        desc=(
            "Indicators of Compromise extracted or inferred from the alert. "
            "Format: 'IP: x.x.x.x, Hash: abc123, Domain: evil.com'. "
            "'None extractable' if alert contains no IOCs."
        )
    )
    containment_steps: str = dspy.OutputField(
        desc=(
            "Numbered containment steps in order of execution. "
            "E.g.: '1. Isolate host from network. 2. Preserve memory image. "
            "3. Reset credentials for all accounts on host. 4. Notify stakeholders.'"
        )
    )
    ticket_summary: str = dspy.OutputField(
        desc=(
            "3-sentence incident ticket summary: what happened, "
            "what systems are affected, what actions were taken."
        )
    )
